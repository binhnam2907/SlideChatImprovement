# adapter_cv — Dual-Branch Vision Adapter for SlideChat

## 1. Architecture Overview

`adapter_cv` processes CONCH patch features through two parallel
branches, then fuses their outputs before projecting to the LLM:

```
CONCH features (B, N, 512)
       │
       ├──────────────────────────────────────────┐
       │                                          │
       ▼                                          ▼
┌─────────────────┐                    ┌─────────────────┐
│  SEGMENTATION   │                    │   DETECTION     │
│  (refine ALL    │                    │   (score &      │
│   patches)      │                    │    select M)    │
│                 │                    │                 │
│  transformer    │                    │  abmil          │
│  lightweight    │                    │  clam           │
│  conv           │                    │  transmil       │
│  mamba          │                    │  tissue_cls     │
│  nystrom        │                    │                 │
│  longmil        │                    │                 │
│  graph          │                    │                 │
└────────┬────────┘                    └───┬─────┬──────┘
         │                                 │     │
    seg_out (B,N,D)               det_out (B,M,D)│
         │                          scores (B,N) │
         │                         indices       │
         └──────────┬──────────────────┘         │
                    │                             │
                    ▼                             │
           ┌───────────────┐                      │
           │    FUSION     │◄─────────────────────┘
           │               │
           │  det_guided   │ pick seg features at det indices
           │  score_weight │ multiply seg by det scores
           │  concat       │ [seg; det] → project to D
           │  gated        │ sigmoid gate blends both
           │  add          │ element-wise sum
           └───────┬───────┘
                   │
              fused (B, M, D)
                   │
                   ▼
          ┌────────────────┐
          │ FeatureProjector│
          │  512 → 3584    │
          └────────┬───────┘
                   │
                   ▼
            Qwen2.5-7B LLM
```

---

## 2. Package Layout

```
adapter_cv/
├── __init__.py               all exports
├── cv_model.py               CVModel — dual-branch orchestrator
├── feature_projector.py      MLP (512 → 3584)
├── fusion.py                 5 fusion strategies
├── wsi_detection.py          detector model definitions
│
├── segmentation/             refines all patches
│   ├── __init__.py           SEG_REGISTRY + build_seg_backbone()
│   ├── cv_backbone.py        transformer     O(N²)   ~6 M
│   ├── lightweight_backbone.py  MLP           O(N)    ~1 M
│   ├── conv_backbone.py      depthwise conv  O(N)    ~2.1 M
│   ├── mamba_backbone.py     bidirectional SSM O(N)  ~4.2 M
│   ├── nystrom_backbone.py   Nystrom approx  O(N·m)  ~6 M
│   ├── longmil_backbone.py   local+global    O(N·(w+g)) ~8 M
│   └── graph_backbone.py     k-NN GCN        O(N·K)  ~4.2 M
│
├── detection/                scores & selects patches
│   └── __init__.py           DET_REGISTRY + build_det_backbone()
│       delegates to wsi_detection.py:
│       abmil                 gated attention      ~262 K
│       clam                  multi-class + cluster ~460 K
│       transmil              transformer + PPEG   ~5.3 M
│       tissue_cls            MLP classifier       ~165 K
│
└── backbone/                 backward-compat re-exports
    └── __init__.py
```

---

## 3. Fusion — How Seg & Det Outputs Merge

The key question: the segmentation branch produces rich features
for ALL N patches, and the detection branch identifies which M
patches matter most. How do we combine them?

### `det_guided` (default, recommended)

Use detection indices to select from segmentation output.

```
seg_out (B, N, D)  ──select at det_indices──►  (B, M, D)
```

Result has the detection's selection intelligence AND the
segmentation's contextual representations. Zero extra parameters.

### `score_weight`

Multiply ALL segmentation features by detection scores as soft
attention. Does NOT reduce sequence length — keeps all N patches.

```
output = seg_out * softmax(det_scores)    →  (B, N, D)
```

Use when the LLM context window can handle all N patches.

### `concat`

Concatenate both views of each selected patch and project back.

```
[seg_sel; det_out] → Linear(2D, D)    →  (B, M, D)
```

Gives the model two independent representations to reason over.
Small extra cost (one Linear layer).

### `gated`

Learned sigmoid gate blends segmentation and detection features:

```
gate = σ(W_s · seg_sel + W_d · det_out)
output = gate · seg_sel + (1-gate) · det_out    →  (B, M, D)
```

Most flexible — the model learns per-dimension how much to trust
each branch. Two extra Linear layers.

### `add`

Element-wise addition of seg-selected and det-selected features:

```
output = LayerNorm(seg_sel + det_out)    →  (B, M, D)
```

Simplest learnable-free fusion.

---

## 4. Usage

### Config (xtuner train)

```python
model = dict(
    type=LLaVAModel,
    enable_cv_model=True,
    cv_model_config=dict(
        seg_backbone='transformer',
        det_backbone='abmil',          # set in detector_method too
        fusion_mode='det_guided',
    ),
    detector_method='abmil',
    detector_config=dict(top_ratio=0.7),
    llm=dict(...),
)
```

### training_adapter.py

```bash
# Full dual-branch: transformer seg + abmil det + gated fusion
python training_adapter.py \
    --seg_backbone transformer \
    --det_backbone abmil \
    --fusion_mode gated \
    --epochs 5 --lr 1e-4

# Mamba seg + CLAM det + det_guided fusion
python training_adapter.py \
    --seg_backbone mamba \
    --det_backbone clam \
    --fusion_mode det_guided \
    --det_top_ratio 0.7

# Seg only (no detection, all patches kept)
python training_adapter.py \
    --seg_backbone longmil

# Det only (no segmentation context)
python training_adapter.py \
    --seg_backbone none \
    --det_backbone abmil
```

Note: pass `--seg_backbone none` to disable segmentation. Pass
`--det_backbone` as None (omit the flag) to disable detection.

---

## 5. Single-Branch Fallback

CVModel degrades gracefully when only one branch is enabled:

| seg | det | behaviour |
|-----|-----|-----------|
| on  | on  | dual-branch → fusion → project |
| on  | off | seg → project (all N patches) |
| off | on  | det → project (M selected) |
| off | off | raw → project (passthrough) |

---

## 6. Training

### Stage 1 — alignment (CVModel only)

```
Loss (LLM cross-entropy)
  ↑ gradients pass through frozen LLM
FeatureProjector   ← updated
  ↑
Fusion             ← updated (if gated/concat)
  ↑
Segmentation       ← updated
Detection          ← updated
  ↑
raw features       (leaf)
```

### Stage 2 — instruction tuning

Unfreeze LLM. CVModel can stay trainable or be frozen.

---

## 7. Backbone Comparison

| Backbone | Params | Complexity | Best for |
|----------|--------|------------|----------|
| `transformer` | ~6 M | O(N²) | max quality, small slides |
| `nystrom` | ~6 M | O(N·64) | near-full-attention, cheaper |
| `longmil` | ~8 M | O(N·320) | very large slides |
| `mamba` | ~4.2 M | O(N) | long-range, linear cost |
| `graph` | ~4.2 M | O(N·8) | spatial topology matters |
| `conv` | ~2.1 M | O(N) | local patterns, fast |
| `lightweight` | ~1 M | O(N) | quick baseline |
