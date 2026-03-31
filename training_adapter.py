"""
training_adapter.py — Train segmentation / detection adapters.

Trains the seg and det components with their OWN objectives:

  Segmentation loss:
    Masked Patch Modeling — mask 15% of patches, backbone must
    reconstruct them from context.

  Detection loss:
    Coverage + diversity — selected patches must represent the
    full slide and be diverse.

Usage:
    # With real data
    python training_adapter.py \
        --data_path path/to/train.json \
        --seg_backbone transformer --det_backbone abmil

    # With auto-generated mock data (no data_path needed)
    python training_adapter.py --generate_mock \
        --seg_backbone lightweight --det_backbone abmil
"""

import argparse
import csv
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from adapter_cv import CVModel


# ─────────────────────────────────────────────────────────────────
# Mock data generation
# ─────────────────────────────────────────────────────────────────


def generate_mock_data(work_dir, n_slides=6, dim=512):
    """Generate mock WSI feature CSVs + JSON for testing."""
    mock_dir = osp.join(work_dir, "mock_data")
    os.makedirs(mock_dir, exist_ok=True)
    np.random.seed(42)

    slides = [
        ("MOCK-SLIDE-001", 800),
        ("MOCK-SLIDE-002", 1200),
        ("MOCK-SLIDE-003", 600),
        ("MOCK-SLIDE-004", 2000),
        ("MOCK-SLIDE-005", 1500),
        ("MOCK-SLIDE-006", 400),
    ][:n_slides]

    entries = []
    cols = [str(i) for i in range(dim)] + ["patch_name"]
    for name, n_patches in slides:
        feats = np.random.randn(n_patches, dim).astype(np.float32)
        csv_path = osp.join(mock_dir, f"{name}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i in range(n_patches):
                row = [f"{v:.4f}" for v in feats[i]]
                row.append(f"{i}.jpeg")
                w.writerow(row)
        entries.append(
            {
                "id": name,
                "image": [csv_path],
            }
        )

    json_path = osp.join(mock_dir, "train.json")
    with open(json_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Generated {n_slides} mock slides in {mock_dir}")
    return json_path


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────


class FeatureDataset(Dataset):
    """Load pre-computed CONCH patch features from CSV."""

    def __init__(self, data_path, max_patches=10240, feature_dim=512):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.max_patches = max_patches
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.data)

    def _load_csv(self, csv_path):
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                vals = [float(v) for v in row[: self.feature_dim]]
                rows.append(vals)
                if len(rows) >= self.max_patches:
                    break
        total = len(rows)
        if total > self.max_patches:
            idx = np.linspace(0, total - 1, self.max_patches, dtype=int)
            rows = [rows[i] for i in idx]
        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, idx):
        entry = self.data[idx]
        paths = entry.get("image", [])
        if isinstance(paths, str):
            paths = [paths]
        feat_list = []
        for p in paths:
            if p.endswith(".csv") and os.path.exists(p):
                feat_list.append(self._load_csv(p))
        features = (
            torch.cat(feat_list)
            if feat_list
            else torch.zeros(1, self.feature_dim)
        )
        return {
            "features": features,
            "slide_id": entry.get("id", ""),
        }


def collate_fn(batch):
    feats = [b["features"] for b in batch]
    if all(f.shape == feats[0].shape for f in feats):
        stacked = torch.stack(feats)
    else:
        max_n = max(f.shape[0] for f in feats)
        padded = []
        for f in feats:
            if f.shape[0] < max_n:
                pad = torch.zeros(max_n - f.shape[0], f.shape[1])
                f = torch.cat([f, pad])
            padded.append(f)
        stacked = torch.stack(padded)
    return {
        "features": stacked,
        "slide_ids": [b["slide_id"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────
# Losses for segmentation backbone
# ─────────────────────────────────────────────────────────────────


class SegReconHead(nn.Module):
    """Projection head to reconstruct masked patches."""

    def __init__(self, dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.head(x)


def compute_seg_loss(seg_backbone, recon_head, features, mask_ratio=0.15):
    """Masked Patch Modeling loss for segmentation backbone.

    1. Randomly mask 15% of patches (replace with zeros)
    2. Run masked features through seg backbone
    3. Reconstruct original features at masked positions
    4. L2 loss between reconstruction and original
    """
    B, N, D = features.shape
    n_mask = max(int(N * mask_ratio), 1)

    mask = torch.zeros(B, N, dtype=torch.bool, device=features.device)
    for b in range(B):
        idx = torch.randperm(N, device=features.device)[:n_mask]
        mask[b, idx] = True

    masked_features = features.clone()
    masked_features[mask] = 0.0

    refined = seg_backbone(masked_features)

    pred = recon_head(refined[mask])
    target = features[mask].detach()

    loss = F.mse_loss(pred, target)

    with torch.no_grad():
        cosine = F.cosine_similarity(pred, target, dim=-1).mean()

    return loss, {
        "seg_recon_loss": loss.item(),
        "seg_cosine_sim": cosine.item(),
        "n_masked": n_mask,
    }


# ─────────────────────────────────────────────────────────────────
# Losses for detection backbone
# ─────────────────────────────────────────────────────────────────


def compute_det_loss(detector, features, diversity_weight=0.1):
    """Coverage + diversity loss for detection backbone.

    Coverage: selected patches should reconstruct the full-slide
    representation (mean of ALL patches). Measures whether the
    detector picks a representative subset.

    Diversity: selected patches should be diverse (measured by
    mean pairwise cosine distance). Prevents attention collapse.

    CLAM clustering loss is added automatically when applicable.
    """
    B, N, D = features.shape
    full_repr = features.mean(dim=1)

    has_clam = hasattr(detector, "method") and detector.method == "clam"
    if has_clam:
        selected, scores, indices, clam_loss = detector(
            features, return_cluster_loss=True
        )
    else:
        selected, scores, indices = detector(features)
        clam_loss = torch.tensor(0.0, device=features.device)

    sel_repr = selected.mean(dim=1)
    coverage_loss = F.mse_loss(sel_repr, full_repr.detach())

    sel_norm = F.normalize(selected, dim=-1)
    M = selected.shape[1]
    if M > 1:
        sim_matrix = sel_norm @ sel_norm.transpose(-2, -1)
        mask = ~torch.eye(
            M, dtype=torch.bool, device=features.device
        ).unsqueeze(0)
        mean_sim = sim_matrix[mask.expand(B, -1, -1)].mean()
        diversity_loss = mean_sim
    else:
        diversity_loss = torch.tensor(0.0, device=features.device)

    attn_entropy = -(scores * (scores + 1e-8).log()).sum(dim=-1).mean()

    total = coverage_loss + diversity_weight * diversity_loss
    if has_clam:
        total = total + 0.1 * clam_loss

    info = {
        "det_coverage_loss": coverage_loss.item(),
        "det_diversity_loss": diversity_loss.item(),
        "det_attn_entropy": attn_entropy.item(),
        "det_n_selected": selected.shape[1],
        "det_score_mean": scores.mean().item(),
        "det_score_std": scores.std().item(),
    }
    if has_clam:
        info["det_clam_loss"] = clam_loss.item()

    return total, info


# ─────────────────────────────────────────────────────────────────
# Combined forward pass
# ─────────────────────────────────────────────────────────────────


def run_forward_pass(
    cv_model,
    recon_head,
    batch,
    device,
    dtype,
    seg_weight=1.0,
    det_weight=1.0,
    mask_ratio=0.15,
):
    """Compute seg + det losses independently.

    Returns total_loss and per-component diagnostics.
    """
    features = batch["features"]
    if isinstance(features, list):
        features = features[0]
    if features.dim() == 2:
        features = features.unsqueeze(0)
    features = features.to(device=device, dtype=dtype)

    info = {"n_patches": features.shape[1]}
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    if cv_model.seg is not None:
        seg_loss, seg_info = compute_seg_loss(
            cv_model.seg, recon_head, features, mask_ratio
        )
        total_loss = total_loss + seg_weight * seg_loss
        info.update(seg_info)

    if cv_model.det is not None:
        det_loss, det_info = compute_det_loss(cv_model.det, features)
        total_loss = total_loss + det_weight * det_loss
        info.update(det_info)

    info["total_loss"] = total_loss.item()
    return total_loss, info


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────


def evaluate(cv_model, recon_head, dataloader, device, dtype):
    """Evaluate seg/det quality on full dataset."""
    cv_model.eval()
    if recon_head is not None:
        recon_head.eval()

    all_info = []
    with torch.no_grad():
        for batch in dataloader:
            _, info = run_forward_pass(
                cv_model, recon_head, batch, device, dtype
            )
            all_info.append(info)

    avg = {}
    keys = all_info[0].keys()
    for k in keys:
        vals = [d[k] for d in all_info if isinstance(d.get(k), float)]
        if vals:
            avg[k] = sum(vals) / len(vals)

    cv_model.train()
    if recon_head is not None:
        recon_head.train()
    return avg


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Train segmentation / detection adapters"
    )
    p.add_argument(
        "--data_path", default=None, help="JSON with slide feature paths"
    )
    p.add_argument(
        "--generate_mock",
        action="store_true",
        help="Auto-generate mock data for testing",
    )
    p.add_argument("--work_dir", default="./work_dirs/adapter_training")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_patches", type=int, default=10240)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=50)

    p.add_argument(
        "--seg_backbone",
        type=str,
        default="transformer",
        choices=[
            "transformer",
            "lightweight",
            "conv",
            "mamba",
            "nystrom",
            "longmil",
            "graph",
            "none",
        ],
    )
    p.add_argument(
        "--det_backbone",
        type=str,
        default=None,
        choices=["abmil", "clam", "transmil", "tissue_cls"],
    )
    p.add_argument(
        "--fusion_mode",
        type=str,
        default="det_guided",
        choices=["det_guided", "score_weight", "concat", "gated", "add"],
    )
    p.add_argument("--cv_num_layers", type=int, default=4)
    p.add_argument("--cv_num_heads", type=int, default=8)
    p.add_argument("--cv_ff_dim", type=int, default=2048)
    p.add_argument("--cv_dropout", type=float, default=0.1)
    p.add_argument("--det_top_ratio", type=float, default=0.7)
    p.add_argument("--det_top_k", type=int, default=None)

    p.add_argument("--seg_weight", type=float, default=1.0)
    p.add_argument("--det_weight", type=float, default=1.0)
    p.add_argument("--mask_ratio", type=float, default=0.15)

    p.add_argument(
        "--torch_dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    dtype = getattr(torch, args.torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.generate_mock:
        args.data_path = generate_mock_data(args.work_dir)
    if args.data_path is None:
        raise ValueError("Provide --data_path or --generate_mock")

    seg_name = args.seg_backbone if args.seg_backbone != "none" else None
    det_name = args.det_backbone

    if seg_name and det_name:
        mode = (
            f"DUAL: seg={seg_name}, det={det_name}, fusion={args.fusion_mode}"
        )
    elif seg_name:
        mode = f"SEG-ONLY: {seg_name}"
    elif det_name:
        mode = f"DET-ONLY: {det_name}"
    else:
        raise ValueError("At least one of seg or det must be set")

    print(f"Mode: {mode}")

    # ── Build CVModel ────────────────────────────────────────────
    cv_model = CVModel(
        seg_backbone=seg_name,
        det_backbone=det_name,
        fusion_mode=args.fusion_mode,
        cv_input_dim=512,
        cv_num_heads=args.cv_num_heads,
        cv_num_layers=args.cv_num_layers,
        cv_ff_dim=args.cv_ff_dim,
        cv_dropout=args.cv_dropout,
        llm_hidden_size=512,
        proj_depth=1,
        det_top_k=args.det_top_k,
        det_top_ratio=args.det_top_ratio,
    ).to(device=device, dtype=dtype)

    pc = cv_model.param_count()
    print(f"Params: {pc}")

    recon_head = None
    if cv_model.seg is not None:
        recon_head = SegReconHead(dim=512).to(device=device, dtype=dtype)

    # ── Optimizer ────────────────────────────────────────────────
    params = list(cv_model.parameters())
    if recon_head is not None:
        params += list(recon_head.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # ── Dataset ──────────────────────────────────────────────────
    dataset = FeatureDataset(args.data_path, max_patches=args.max_patches)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    total_steps = len(dataloader) * args.epochs
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=1e-6
    )

    print(
        f"Data: {len(dataset)} samples, "
        f"{len(dataloader)} steps/epoch, "
        f"{args.epochs} epochs\n"
    )

    # ── Training ─────────────────────────────────────────────────
    global_step = 0

    for epoch in range(args.epochs):
        cv_model.train()
        if recon_head:
            recon_head.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            loss, info = run_forward_pass(
                cv_model,
                recon_head,
                batch,
                device,
                dtype,
                seg_weight=args.seg_weight,
                det_weight=args.det_weight,
                mask_ratio=args.mask_ratio,
            )
            loss = loss / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * args.grad_accum
            global_step += 1

            if global_step % args.log_interval == 0:
                parts = [f"Step {global_step}"]
                parts.append(f'loss={info["total_loss"]:.4f}')
                if "seg_recon_loss" in info:
                    parts.append(f'seg_recon={info["seg_recon_loss"]:.4f}')
                    parts.append(f'seg_cos={info["seg_cosine_sim"]:.4f}')
                if "det_coverage_loss" in info:
                    parts.append(f'det_cov={info["det_coverage_loss"]:.4f}')
                    parts.append(f'det_div={info["det_diversity_loss"]:.4f}')
                    parts.append(
                        f'sel={info["det_n_selected"]}/' f'{info["n_patches"]}'
                    )
                if "det_clam_loss" in info:
                    parts.append(f'clam={info["det_clam_loss"]:.4f}')
                print(f"  [E{epoch+1}] " + " | ".join(parts))

            if global_step % args.eval_interval == 0:
                avg = evaluate(cv_model, recon_head, dataloader, device, dtype)
                eparts = ["[Eval]"]
                if "seg_recon_loss" in avg:
                    eparts.append(f'recon={avg["seg_recon_loss"]:.4f}')
                    eparts.append(f'cos={avg["seg_cosine_sim"]:.4f}')
                if "det_coverage_loss" in avg:
                    eparts.append(f'cov={avg["det_coverage_loss"]:.4f}')
                    eparts.append(f'div={avg["det_diversity_loss"]:.4f}')
                    eparts.append(f'entropy={avg["det_attn_entropy"]:.4f}')
                print(f"  " + " | ".join(eparts))

            if global_step % args.save_interval == 0:
                sd = osp.join(args.work_dir, f"step_{global_step}")
                cv_model.save(sd)

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"\n  Epoch {epoch+1} done | " f"avg_loss={avg_loss:.4f}\n")

        sd = osp.join(args.work_dir, f"epoch_{epoch+1}")
        cv_model.save(sd)

    # ── Final ────────────────────────────────────────────────────
    final_dir = osp.join(args.work_dir, "final")
    cv_model.save(final_dir)

    print("Final evaluation:")
    avg = evaluate(cv_model, recon_head, dataloader, device, dtype)
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nSaved to: {final_dir}")


if __name__ == "__main__":
    main()
