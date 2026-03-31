# Adapter Training Results — All Backbones

1 epoch, 6 mock slides (400–2000 patches each), batch=1, lr=1e-3.

---

## Commands

### Segmentation only (7 backbones)

```bash
# lightweight — MLP, no cross-patch
python training_adapter.py --generate_mock --seg_backbone lightweight

# transformer — full self-attention
python training_adapter.py --generate_mock --seg_backbone transformer

# conv — depthwise 1-D convolution
python training_adapter.py --generate_mock --seg_backbone conv

# mamba — bidirectional SSM
python training_adapter.py --generate_mock --seg_backbone mamba

# nystrom — Nystrom-approximated attention
python training_adapter.py --generate_mock --seg_backbone nystrom

# longmil — local-window + global-pool hybrid
python training_adapter.py --generate_mock --seg_backbone longmil

# graph — k-NN graph convolution
python training_adapter.py --generate_mock --seg_backbone graph
```

### Detection only (4 backbones)

```bash
# abmil — gated attention top-K
python training_adapter.py --generate_mock --seg_backbone none --det_backbone abmil

# clam — multi-class attention + instance clustering
python training_adapter.py --generate_mock --seg_backbone none --det_backbone clam

# transmil — transformer + PPEG
python training_adapter.py --generate_mock --seg_backbone none --det_backbone transmil

# tissue_cls — MLP classifier + stratified sampling
python training_adapter.py --generate_mock --seg_backbone none --det_backbone tissue_cls
```

---

## Segmentation Results

Loss = Masked Patch Modeling (reconstruct 15% masked patches from context).
Lower recon loss = better context modelling.
Higher cosine sim = better reconstruction quality.

| # | Backbone | Params | Avg Loss | Recon Loss | Cosine Sim |
|---|----------|--------|----------|------------|------------|
| 1 | lightweight | 1.3M | 1.0281 | 1.0069 | 0.0033 |
| 2 | transformer | 6.3M | 1.0344 | 1.0103 | 0.0022 |
| 3 | conv | 2.4M | 1.0237 | 1.0068 | 0.0053 |
| 4 | mamba | 15.5M | 1.0252 | **1.0006** | **0.0123** |
| 5 | nystrom | 6.3M | 1.0357 | 1.0053 | 0.0016 |
| 6 | longmil | 8.3M | 1.0272 | 1.0052 | 0.0008 |
| 7 | graph | 4.5M | 1.0232 | 1.0041 | -0.0006 |

**Best recon:** mamba (1.0006) — bidirectional scan captures
long-range context most effectively for reconstruction.

**Best cosine:** mamba (0.0123) — reconstructed features most
similar to originals.

---

## Detection Results

Coverage loss = how well selected patches represent the full slide.
Diversity loss = mean pairwise cosine similarity of selected patches (lower = more diverse).
Entropy = attention distribution entropy (higher = more spread).

| # | Backbone | Params | Coverage | Diversity | Entropy | Score Mean |
|---|----------|--------|----------|-----------|---------|------------|
| 8 | abmil | 526K | 0.0009 | 0.0004 | 6.838 | 0.0012 |
| 9 | clam | 660K | 0.0009 | 0.0004 | 6.837 | 0.0012 |
| 10 | transmil | 5.6M | **0.0007** | **0.0002** | 6.826 | 0.0012 |
| 11 | tissue_cls | 428K | 0.0048 | 0.0005 | 376.7 | 0.2536 |

**Best coverage:** transmil (0.0007) — transformer context helps
select the most representative subset.

**Best diversity:** transmil (0.0002) — selected patches are most
diverse.

**CLAM extra:** instance clustering loss = 0.538 (auxiliary signal
for separating positive/negative patches).

**tissue_cls:** high entropy (376.7) because it uses class-weighted
scoring rather than attention, producing a different score
distribution.

---

## Notes

- All runs use `--generate_mock` which creates 6 random feature CSVs
  (400–2000 patches × 512-d) inline. No external data needed.
- For real training, replace `--generate_mock` with
  `--data_path path/to/train.json`.
- Add `--epochs 50 --lr 1e-4` for proper convergence.
- Dual-branch (seg + det): add both flags, e.g.
  `--seg_backbone mamba --det_backbone transmil --fusion_mode gated`.
