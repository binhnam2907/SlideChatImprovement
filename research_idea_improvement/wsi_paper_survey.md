# Comprehensive Survey: WSI Analysis Methods (2018–2025)

A systematic review of every major approach in whole-slide image
analysis, organised by category. Each entry notes the paper, key
idea, and whether it is implemented in `adapter_cv/backbone/`.

---

## 1. Feature Extractors / Foundation Models

These models produce the per-patch embeddings that all downstream
methods consume. Our pipeline uses CONCH (512-d).

| Model | Venue | Architecture | Training Data | Embedding Dim | Notes |
|-------|-------|-------------|---------------|---------------|-------|
| **CONCH** | Nature Medicine 2024 | ViT + CLIP-style VL | 1.17 M histopath image-text pairs | 512 | Highest overall in 2025 benchmark of 19 models on 13 cohorts. Vision-language enables zero-shot classification. |
| **UNI** | Nature Medicine 2024 | ViT-L/16 | 100 M patches, 100 K WSIs | 1024 | General-purpose pathology encoder. |
| **UNI 2** | Jan 2025 | ViT-H/14-reg8 | 200 M patches, 350 K WSIs | 1280 | Pre-extracted TCGA/CPTAC/PANDA embeddings available. |
| **Virchow** | 2023 | ViT-H | 1.5 M WSIs | 1280 | Trained by Paige. |
| **Virchow2** | 2024 | ViT-H | 3.1 M WSIs | 1280 | 632 M params. SOTA on 12 tile-level tasks in 2025 31-model benchmark. |
| **Virchow2G** | 2024 | ViT-G | 3.1 M WSIs | 1536 | 1.9 B params. Distilled mini (22 M) also available. |
| **Atlas 2** | 2025 | ViT | 5.5 M WSIs (3 institutions) | — | SOTA across 80 public benchmarks. |
| **H-optimus-0** | 2024 | ViT | proprietary | — | Competitive with Virchow2. |
| **Phikon** | 2023 | ViT | TCGA | 768 | Smaller-scale, open-weights. |
| **CTransPath** | MICCAI 2022 | Swin Transformer | 15 M patches | 768 | Widely used before UNI/CONCH era. |
| **PLIP** | Nature Medicine 2023 | CLIP fine-tune | 208 K pathology image-text from Twitter | 512 | First pathology vision-language model. |
| **PathFLIP** | 2025 | VL with region decomposition | slide-level captions decomposed to regions | — | Fine-grained WSI captioning + localisation. |
| **BiomedCLIP** | 2023 | CLIP fine-tune | PMC-15M biomedical pairs | 512 | Broader biomedical domain, not pathology-specific. |

**Key insight from benchmarks:** Model scale alone does not predict
performance. Data diversity and domain-specific training matter
more. CONCH and Virchow2 consistently rank highest across diverse
tasks.

---

## 2. Aggregation / MIL Methods

How to combine N patch embeddings into a slide-level decision.

### 2.1 Attention-Based Pooling

| Method | Venue | Key Idea | Status |
|--------|-------|----------|--------|
| **ABMIL** | ICML 2018 | Gated attention (tanh × sigmoid) to weight patches | Implemented in `wsi_detection.py` as `GatedAttentionSelector` |
| **CLAM** | Nature BME 2021 | Multi-class attention branches + instance-level clustering loss | Implemented as `CLAMSelector` |
| **DSMIL** | CVPR 2021 | Dual-stream: max-pooled critical instance + attention-aggregated context | Not yet implemented |
| **DTFD-MIL** | CVPR 2022 | Dual-tier feature distillation: pseudo-bags → distilled aggregation | Not yet implemented |
| **ASMIL** | 2024 | Anchor-stabilised attention, normalised sigmoid, token random dropping; +6.49% F1 | Not yet implemented |
| **DGR-MIL** | ECCV 2024 | Diverse Global Representation via cross-attention + DPP diversification | Not yet implemented |
| **ACMIL** | ECCV 2024 | Multi-branch attention + stochastic top-K instance masking | Not yet implemented |

### 2.2 Transformer-Based

| Method | Venue | Key Idea | Status |
|--------|-------|----------|--------|
| **TransMIL** | NeurIPS 2021 | Transformer + PPEG pyramid positional encoding; CLS token | Implemented in `wsi_detection.py` as `TransMILSelector` |
| **HIPT** | CVPR 2022 | 3-level hierarchical ViT (cell → patch → region → slide), DINO pre-training | Architecture inspiration in backbone design |
| **FALFormer** | 2024 | Nystrom attention with feature-aware landmarks for WSI | Implemented as `NystromBackbone` |
| **LongMIL** | NeurIPS 2024 | Local windowed attention + global pool cross-attention, linear complexity | Implemented as `LongMILBackbone` |

### 2.3 State-Space / Mamba

| Method | Venue | Key Idea | Status |
|--------|-------|----------|--------|
| **MambaMIL** | MICCAI 2024 | Mamba SSM + Sequence Reordering for order-aware scanning | Implemented as `MambaBackbone` |
| **MambaMIL+** | 2025 | Overlapping scan + stripe position encoding + contextual token selection | Scanning improvements applicable |
| **MoEMambaMIL** | 2025 | Region-nested scan + Mixture-of-Experts for multi-scale | Architecture ideas for future extension |
| **LBMamba** | 2025 | Locally bi-directional Mamba scan; +3% AUC vs MambaMIL | Bidirectional idea used in our `MambaBackbone` |
| **EfficientMIL** | 2024 | GRU/LSTM/Mamba + Adaptive Patch Selector (relevance + diversity + uncertainty) | GRU fallback in `MambaBackbone` |

### 2.4 Graph-Based

| Method | Venue | Key Idea | Status |
|--------|-------|----------|--------|
| **PatchGCN** | MICCAI 2021 | WSI as 2D point cloud; k-NN spatial graph + GCN message passing | Implemented as `GraphBackbone` |
| **WiKG** | CVPR 2024 | WSI as knowledge graph; head-to-tail edge embeddings + knowledge-aware attention | Architecture ideas for future |

### 2.5 Tissue-Level Classification

| Method | Key Idea | Status |
|--------|----------|--------|
| **TissueClassifier** | MLP per-patch tissue type prediction + stratified quota sampling | Implemented in `wsi_detection.py` |

---

## 3. WSI-Specific Multimodal LLMs

| Model | Venue | Architecture | Notes |
|-------|-------|-------------|-------|
| **SlideChat** | CVPR 2025 | CONCH features → LongNet encoder → MLP projector → Qwen2.5-7B | Our base system |
| **PathChat** | 2024 | UNI features → adapter → Llama 2 | Patch-level, not whole-slide |
| **PathFLIP** | 2025 | Region-level VL pre-training for WSI captioning | Slide-level with region decomposition |

---

## 4. Implemented Backbones in adapter_cv/backbone/

All backbones share the same interface:
- Input: `(B, N, input_dim)`, Output: `(B, N, input_dim)`
- Expose `self.input_dim`
- Accept `**kwargs` for unused params

| Key | Class | File | Paper Inspiration | Complexity | Params (512-d, 4 layers) |
|-----|-------|------|-------------------|-----------|--------------------------|
| `transformer` | CVBackbone | `cv_backbone.py` | Standard Transformer | O(N^2) | ~6 M |
| `lightweight` | LightweightBackbone | `lightweight_backbone.py` | Baseline MLP | O(N) | ~1 M |
| `conv` | ConvBackbone | `conv_backbone.py` | Depth-wise separable conv | O(N) | ~2.1 M |
| `mamba` | MambaBackbone | `mamba_backbone.py` | MambaMIL (MICCAI 2024) | O(N) | ~4.2 M |
| `nystrom` | NystromBackbone | `nystrom_backbone.py` | Nystromformer + FALFormer | O(N·m) | ~6 M |
| `longmil` | LongMILBackbone | `longmil_backbone.py` | LongMIL (NeurIPS 2024) | O(N·(w+g)) | ~8 M |
| `graph` | GraphBackbone | `graph_backbone.py` | PatchGCN (MICCAI 2021) | O(N·K) | ~4.2 M |

### Complexity Legend

- N = number of patches (typically 1 K – 40 K)
- m = number of Nystrom landmarks (default 64)
- w = local attention window size (default 256)
- g = number of global pool tokens (default 64)
- K = k-NN neighbours (default 8)

### When to Use What

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Small slides (< 5 K patches), max quality | `transformer` | Full attention captures everything |
| Quick experiments / debugging | `lightweight` | Fastest, cheapest |
| Large slides (> 20 K), limited GPU | `conv` or `mamba` | Linear cost, still captures context |
| Very large slides, need global context | `longmil` | Local + global hybrid, linear cost |
| Approximating full attention cheaply | `nystrom` | Nystrom landmarks give near-full-attention quality at O(N·64) |
| Spatial neighbourhood matters | `graph` | k-NN graph captures topology |
| No `mamba_ssm` installed | `mamba` still works | Falls back to bidirectional GRU |

---

## 5. Detection Models in wsi_detection.py

| Key | Class | Paper | Idea |
|-----|-------|-------|------|
| `abmil` | GatedAttentionSelector | Ilse et al. ICML 2018 | Gated attention top-K + random mix |
| `clam` | CLAMSelector | Lu et al. Nature BME 2021 | Multi-class attention + instance clustering |
| `transmil` | TransMILSelector | Shao et al. NeurIPS 2021 | Transformer + PPEG + CLS attention |
| `tissue_cls` | TissueClassifier | — | MLP classifier + stratified sampling |

---

## 6. Full Pipeline

```
Raw WSI (.svs / .tiff)
    │  (offline, run once)
    ▼
CONCH feature extraction → CSV files (N × 512)
    │
    ▼
┌─────────────────────────────────────────────┐
│              adapter_cv.CVModel              │
│                                             │
│  1. PatchDetector (optional)                │
│     abmil / clam / transmil / tissue_cls    │
│     N patches → M selected patches          │
│                                             │
│  2. Backbone (configurable)                 │
│     transformer / lightweight / conv /      │
│     mamba / nystrom / longmil / graph        │
│     context refinement                      │
│     + residual connection                   │
│                                             │
│  3. FeatureProjector                        │
│     MLP: 512 → 3584 (Qwen2.5 dim)          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
    prepare_inputs_labels_for_multimodal()
    replaces <image> token with visual embeddings
                   │
                   ▼
           Qwen2.5-7B-Instruct
           text generation / VQA
```

---

## 7. References

1. Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018
2. Lu et al., "Data-efficient and weakly supervised computational pathology on whole-slide images", Nature BME 2021
3. Shao et al., "TransMIL: Transformer based Correlated MIL for WSI Classification", NeurIPS 2021
4. Chen et al., "Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning" (HIPT), CVPR 2022
5. Chen et al., "Whole Slide Images are 2D Point Clouds" (PatchGCN), MICCAI 2021
6. Xiong et al., "Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention", AAAI 2021
7. Tan et al., "FALFormer: Feature-aware Landmarks Self-Attention for WSI Classification", 2024
8. Qu et al., "Rethinking Transformer for Long Contextual Histopathology WSI Analysis" (LongMIL), NeurIPS 2024
9. Yang et al., "MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering", MICCAI 2024
10. Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
11. Lu et al., "Visual Language Pretrained Multiple Instance Learning for Pathology" (CONCH), Nature Medicine 2024
12. Chen et al., "A General-Purpose Foundation Model for Computational Pathology" (UNI), Nature Medicine 2024
13. Vorontsov et al., "Virchow: A Million-Slide Digital Pathology Foundation Model", 2023
14. Virchow2 / Virchow2G, 2024, 632M / 1.9B params on 3.1M WSIs
15. Atlas 2, 2025, trained on 5.5M WSIs
16. Huang et al., "PLIP: A Visual-Language Foundation Model for Pathology", Nature Medicine 2023
17. SlideChat, "A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding", CVPR 2025
18. Li et al., "Dynamic Graph Representation with Knowledge-aware Attention" (WiKG), CVPR 2024
19. Zhang et al., "DTFD-MIL: Double-Tier Feature Distillation MIL", CVPR 2022
20. Li et al., "Dual-stream MIL Network" (DSMIL), CVPR 2021
21. Qu et al., "DGR-MIL: Diverse Global Representation MIL", ECCV 2024
22. Zhang et al., "ACMIL: Attention-Challenging MIL", ECCV 2024
23. Yang et al., "ASMIL: Attention-Stabilized MIL", 2024
24. MambaMIL+, 2025; MoEMambaMIL, 2025; LBMamba, 2025
25. PathFLIP, 2025, fine-grained language-image pretraining for WSI
26. PathBench, 2025, comprehensive pathology foundation model benchmark
