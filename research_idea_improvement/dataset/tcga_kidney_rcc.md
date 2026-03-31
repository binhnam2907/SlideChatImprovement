# TCGA Kidney RCC — Papers & Models Using This Data

## 1. Dataset Summary

| Cohort | Subtype | Cases | Approx. Slides | Prevalence |
|--------|---------|-------|-----------------|------------|
| TCGA-KIRC | Clear Cell RCC | 537 | ~1,650 | 65–70 % |
| TCGA-KIRP | Papillary RCC | 291 | ~770 | 15–20 % |
| TCGA-KICH | Chromophobe RCC | 113 | ~270 | 5–7 % |
| **Total** | | **941** | **~2,690** | |

Source: GDC Data Portal (portal.gdc.cancer.gov), TCIA.

---

## 2. Subtype Classification (3-class: KIRC vs KIRP vs KICH)

### 2.1 CLAM — Lu et al. (Nature BME, 2021)

- **Method:** Attention-based MIL with instance-level clustering.
  Slide-level labels only, no pixel annotations.
- **Data:** TCGA-RCC (KIRC + KIRP + KICH).
- **Result:** AUC > 0.93 for 3-class subtyping.
  On external validation slides: AUC = 0.97 (95 % CI 0.96–0.98).
- **Key finding:** Attention maps correctly localise diagnostic
  regions (clear cytoplasm for ccRCC, papillary cores for pRCC,
  raisinoid nuclei for chRCC) without spatial annotations.

### 2.2 TransMIL — Shao et al. (NeurIPS, 2021)

- **Method:** Transformer-based MIL with PPEG positional encoding.
- **Data:** TCGA-RCC (3-class subtyping).
- **Result:** **AUC = 98.82 %** — SOTA at publication.
  Also tested on TCGA-NSCLC (96.03 %) and CAMELYON16 (93.09 %).
- **Key finding:** Modelling inter-patch correlations via
  Transformer significantly outperforms independent-instance MIL.

### 2.3 Pan-RCC CNN — Cheng et al. (Scientific Reports, 2019)

- **Method:** Inception-v3 patch classifier → majority vote.
- **Data:** TCGA (KIRC + KIRP + KICH), 941 patients.
- **Result:** **94.07 % accuracy** for 3-class subtyping.
  Also predicted survival from histology features.
- **Key finding:** First large-scale deep learning study on all
  three RCC subtypes simultaneously.

### 2.4 RenalNet — Rashid et al. (Scientific Reports, 2025)

- **Method:** CNN with Multi-Channel Residual Transformation (MCRT)
  + Group Convolutional Deep Localisation (GCDL).
- **Data:** TCGA kidney + 2 additional datasets.
- **Result:** **91.67 – 97.24 % accuracy** across three datasets.
  Significantly fewer parameters than standard architectures.
- **Key finding:** Purpose-built architecture for RCC captures
  subtype-specific morphological features more efficiently than
  generic CNNs.

### 2.5 Multi-Model Comparison — Wessels et al. (npj Dig Med, 2024)

- **Method:** Compared TransMIL, CLAM, InceptionV3, ViT, and
  Prov-GigaPath on RCC subtyping.
- **Data:** 289 external validation RCC slides
  (ccRCC vs papRCC vs chRCC).
- **Result:** All models achieved strong AUROC.
  Prov-GigaPath (slide-level foundation model) competitive with
  task-specific MIL methods.
- **Key finding:** Foundation models can match task-specific
  training with zero/few-shot transfer.

### 2.6 DNN 5-Class Classifier — Zhu et al. (Scientific Reports, 2021)

- **Method:** Deep neural network classifying 5 RCC-related tissue
  classes (3 subtypes + oncocytoma + normal).
- **Data:** Internal resection slides + TCGA external validation.
- **Result:** Internal AUC = 0.98 (95 % CI 0.97–1.00).
  External (TCGA) AUC = **0.97** (95 % CI 0.96–0.98).
- **Key finding:** Model generalises across institutions when
  trained on diverse biopsy + resection slides.

---

## 3. Foundation Model Feature Extractors on Kidney

### 3.1 General Benchmark — Carrillo-Perez et al. (Nature BME, 2025)

- **Models tested:** 19 foundation models including CONCH, UNI,
  Virchow2, CTransPath, Phikon, PLIP, RetCCL.
- **Data:** 13 cohorts, 9,528 slides, 6,818 patients
  (includes TCGA kidney).
- **Result:**
  - **CONCH** = highest overall.
  - **Virchow2** = close second.
  - Ensemble CONCH + Virchow2 outperforms in 55 % of tasks.
- **Key finding:** Data diversity > data volume. Vision-language
  pre-training (CONCH) outperforms vision-only models.

### 3.2 Kidney-Specific Benchmark — Kasireddy et al. (arXiv, Mar 2026)

- **Models tested:** 11 HFMs including UNI, CONCH, Virchow,
  CTransPath, Phikon, RetCCL, REMEDIS.
- **Data:** 11 kidney-specific tasks across PAS, H&E, PASM, IHC
  stains (tile-level and slide-level).
- **Results:**
  - **Moderate–strong** on coarse meso-scale morphology
    (diagnostic classification, structural change detection).
  - **Performance drops significantly** for fine-grained
    microstructural tasks, complex phenotypes, prognosis.
- **Key finding:** Current HFMs encode static meso-scale features
  but lack capacity for subtle renal pathology. Kidney-specific
  foundation models are needed.
- **Code:** `kidney-hfm-eval` Python package released.

---

## 4. Survival Prediction on TCGA-KIRC

### 4.1 PatchGCN — Chen et al. (MICCAI, 2021)

- **Method:** WSI as 2D point cloud → k-NN spatial graph → GCN
  message passing for survival prediction.
- **Data:** 4,370 WSIs from 5 TCGA cancer types including KIRC.
- **Result:** Outperformed all weakly-supervised methods by
  **3.58–9.46 %** in concordance index.
- **Key finding:** Spatial graph structure captures tumour
  microenvironment topology that bag-of-patches MIL misses.

### 4.2 SCMIL — Wang et al. (2024)

- **Method:** Sparse Context-aware MIL for predicting survival
  probability distributions (not just risk scores).
- **Data:** TCGA-KIRC, TCGA-LUAD.
- **Result:** Improved calibration and discrimination over
  standard Cox-based MIL methods.
- **Key finding:** Predicting full survival distributions gives
  clinicians more actionable information than a single risk score.

### 4.3 MPRS Multimodal — Chen et al. (npj Dig Med, 2025)

- **Method:** Fuses WSI features + CT imaging + clinical data
  for recurrence prediction in ccRCC.
- **Data:** Multi-centre ccRCC cohorts.
- **Result:** C-index = **0.886** (internal), **0.838** (external).
- **Key finding:** Multimodal fusion significantly outperforms
  any single modality for prognosis.

---

## 5. Tumor Region Segmentation

### 5.1 RCdpia — Sun et al. (arXiv, 2024)

- **Task:** Pixel-level tumor region annotation on TCGA kidney.
- **Data:** 887 TCGA cases:
  486 KIRC, 292 KIRP, 109 KICH.
  Validation: 682 cases from FAHZU.
- **Annotations:**
  - Characteristic tumor regions (subtype-defining areas)
  - Full tumor outline (tALL)
  - Adjacent normal tissue
- **Patch extraction:** 256 × 256 at 20× magnification.
- **Key finding:** Two independent pathologists annotated each
  slide. Cross-centre evaluation showed performance drops,
  highlighting domain shift.

### 5.2 CPTAC-CCRCC Tumor Annotations (TCIA, 2020)

- **Task:** Radiologist-annotated tumor segmentation (RECIST 1.1).
- **Data:** 60 ccRCC subjects with CT-correlated annotations.
- **Use:** Baseline for tumour extent estimation, not histology.

---

## 6. Nuclei Segmentation & Cell Detection on Kidney

### 6.1 Cell AI Foundation Models — Rosenberg et al. (Comms Med, 2025)

- **Models tested:** CellViT, StarDist, Cellpose.
- **Data:** 2,542 kidney WSIs from multiple centres and species.
- **Results (baseline → fine-tuned):**

| Model | Baseline F1 | Fine-tuned F1 | Training Strategy |
|-------|-------------|---------------|-------------------|
| CellViT | 0.780 | 0.795 | Easy 100 % + Hard 100 % |
| StarDist | 0.738 | **0.820** | Easy 100 % + Hard 100 % |
| Cellpose | 0.675 | 0.753 | Easy 100 % |

- **Key finding:** Human-in-the-loop enrichment (pathologist
  corrects "hard" patches) boosts all models. StarDist benefits
  most from fine-tuning. Kidney nuclei segmentation remains
  an open problem — best F1 is only 0.82.

### 6.2 HoVer-Net — Graham et al. (Med Image Anal, 2019)

- **Method:** Simultaneous nuclei segmentation + classification
  (5 types: epithelial, inflammatory, connective, dead, other).
- **Data:** CoNSeP (colorectal) + applied to kidney tissue.
- **Use on kidney:** Extracts cell-level features (nuclear grade
  proxy, cell density, inflammatory infiltrate quantification).

### 6.3 HoVer-NeXt — Baumann et al. (MICCAI, 2024)

- **Method:** Faster variant of HoVer-Net pipeline.
- **Result:** F1 = 0.84 on PanNuke; 5× faster than CellViT,
  17× faster than HoVer-Net.
- **Use on kidney:** Practical for large-scale kidney WSI
  processing where speed matters.

---

## 7. Grading & Molecular Prediction on KIRC

### 7.1 Fuhrman / ISUP Nuclear Grade Prediction

| Paper | Year | Method | AUC |
|-------|------|--------|-----|
| Chen et al. | 2020 | Patch CNN + aggregation | 0.89 |
| Tabibu et al. | 2019 | InceptionV3 fine-tune | 0.93 |
| Benchmarking (2025) | 2025 | CONCH features + MIL | 0.91–0.96 |

Grade prediction from H&E alone achieves AUC 0.89–0.96, making
it a viable AI-assisted diagnostic tool.

### 7.2 Molecular Mutation Prediction from H&E

| Target | Method | AUC | Reference |
|--------|--------|-----|-----------|
| BAP1 mutation | Attention MIL | 0.80–0.87 | Jang et al. 2023 |
| PBRM1 mutation | Attention MIL | 0.72–0.82 | Jang et al. 2023 |
| VHL mutation | CLAM | 0.70–0.78 | Various |
| Molecular subtype | Foundation features + MIL | 0.70–0.89 | Frontiers review 2025 |

---

## 8. Summary Table — All Models on TCGA Kidney

| Model | Year | Venue | Task | Data | Key Metric |
|-------|------|-------|------|------|------------|
| CLAM | 2021 | Nature BME | 3-class subtyping | KIRC/KIRP/KICH | AUC 0.97 |
| TransMIL | 2021 | NeurIPS | 3-class subtyping | TCGA-RCC | AUC 0.988 |
| Pan-RCC CNN | 2019 | Sci Reports | 3-class subtyping | KIRC/KIRP/KICH | Acc 94.07 % |
| RenalNet | 2025 | Sci Reports | 3-class subtyping | TCGA + 2 external | Acc 97.24 % |
| DNN 5-class | 2021 | Sci Reports | 5-class (incl. oncocytoma) | Internal + TCGA | AUC 0.97 |
| Wessels et al. | 2024 | npj Dig Med | 3-class subtyping | External 289 slides | TransMIL/CLAM/GigaPath compared |
| CONCH benchmark | 2025 | Nature BME | Weakly-sup classification | 13 cohorts inc. kidney | CONCH best overall |
| Kidney HFM bench | 2026 | arXiv | 11 kidney tasks | 11 HFMs | Meso-scale OK, fine-grain poor |
| PatchGCN | 2021 | MICCAI | Survival prediction | 5 TCGA cancers inc. KIRC | +3.58–9.46 % C-index |
| SCMIL | 2024 | — | Survival distribution | TCGA-KIRC | Improved calibration |
| MPRS multimodal | 2025 | npj Dig Med | Recurrence prediction | ccRCC multi-centre | C-index 0.886 |
| RCdpia | 2024 | arXiv | Tumor region annotation | 887 TCGA kidney cases | Patch-level labels |
| Cell FM bench | 2025 | Comms Med | Nuclei segmentation | 2,542 kidney WSIs | StarDist F1 0.82 |
| HoVer-NeXt | 2024 | MICCAI | Nuclei seg + class | PanNuke (incl. kidney) | F1 0.84, 17× faster |
| Fuhrman grading | 2019–2025 | Various | Nuclear grade | TCGA-KIRC | AUC 0.89–0.96 |
| BAP1/PBRM1 | 2023 | Various | Mutation from H&E | TCGA-KIRC | AUC 0.72–0.87 |

---

## 9. Key Takeaways for Our Pipeline

1. **Subtyping is solved** (AUC > 0.97 with TransMIL/CLAM).
   The open challenges are grading, molecular prediction, and
   survival — these require richer feature representations.

2. **CONCH features are the best upstream choice** for kidney.
   The 2025 Nature BME benchmark confirmed CONCH outperforms
   all 18 other extractors including UNI and Virchow2.

3. **RCdpia gives us patch-level tumor/normal labels** for all
   three subtypes (887 cases). This is the ideal pre-training
   data for our `TissueClassifier` detection backbone.

4. **Nuclei segmentation on kidney is still hard** (best F1 = 0.82).
   The kidney-specific HFM benchmark shows current foundation
   models struggle with fine-grained renal structures.

5. **Graph-based methods (PatchGCN) outperform MIL for survival**
   by 3.6–9.5 %, validating our `GraphBackbone` implementation.

6. **Multimodal fusion boosts prognosis** (C-index 0.886 vs
   single-modality). Our dual-branch seg+det fusion aligns
   with this finding — combining complementary views improves
   downstream performance.
