# Report Generation Flow — SlideChat (SlideChatImprovement)

This document traces the complete flow of "report generation" in the SlideChat project. The project does not have a dedicated PDF/document report generator. Instead, **report generation** refers to **free-form pathology text generation** from Whole Slide Image (WSI) patch features using a multimodal LLM pipeline. This happens through two main paths: **training-time evaluation** and **inference-time testing**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WSI Feature CSV                              │
│               (N patches × 512-dim CONCH embeddings)                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LongNet Encoder                                │
│           (2-layer Transformer, 512-dim → 512-dim)                  │
│           Aggregates patch-level features into slide-level          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MLP Projector                                   │
│             (512-dim → LLM hidden size, e.g. 3584)                  │
│             Bridges visual and language feature spaces               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Qwen2.5-7B-Instruct LLM                          │
│     Receives merged multimodal input_ids + image embeddings         │
│     Generates free-form pathology text / answers                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
                   Generated Text Output
          (pathology report / VQA answer / caption)
```

---

## Path A: Training-Time Report Generation (EvaluateChatHook)

This path generates sample pathology reports **during training** to monitor model quality. It runs periodically based on `evaluation_freq`.

### Step 1 — Configuration

**File:** `xtuner/configs/slidechat/stage_1.py` (lines 53–57)

The config defines the evaluation prompt and a reference WSI feature file:

```python
evaluation_freq = 1000
SYSTEM = ''
evaluation_images = './BLCA/TCGA-GV-A40G-01Z-00-DX1.csv'
evaluation_inputs = ['Generate an overview summarizing the principal findings from the pathology examination of the whole slide image.']
```

The hook is registered in `custom_hooks` (lines 154–164):

```python
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
]
```

### Step 2 — Hook Initialization

**File:** `xtuner/engine/hooks/evaluate_chat_hook.py` (lines 20–95)

`EvaluateChatHook.__init__` performs:

1. Wraps `evaluation_inputs` and `evaluation_images` into lists (lines 36–47)
2. Loads each evaluation image via `load_image()` — for `.csv` files this reads the patch features and reshapes to `(1, N, 512)` (lines 48–50, delegated to `xtuner/dataset/utils.py` lines 537–557)
3. Builds the prompt instruction template from `prompt_template` (lines 51–60)
4. Configures generation parameters: `max_new_tokens=600`, `temperature=0.1`, `top_p=0.75`, `top_k=40` (lines 77–88)
5. Sets up stopping criteria (lines 90–93)

### Step 3 — Trigger Points

The hook fires at three moments during training:

| Method | File Location | When | Notes |
|--------|---------------|------|-------|
| `before_train` | line 234 | Before training starts | Short generation (`max_new_tokens=50`) as sanity check |
| `after_train_iter` | line 256 | Every `every_n_iters` iterations | Full generation; also runs at checkpoint save points |
| `after_train` | line 275 | After training ends | Final sample generation |

### Step 4 — Sample Generation

**File:** `xtuner/engine/hooks/evaluate_chat_hook.py`, `_generate_samples` (lines 196–232)

1. Unwraps model from any wrapper (DDP/DeepSpeed) → line 203–204
2. Disables gradient checkpointing and enables KV cache for inference → lines 218–220
3. Switches model to `eval()` mode → line 220
4. Dispatches to `_eval_images()` (if images provided) or `_eval_language()` → lines 221–226
5. Restores training mode after generation → lines 228–232

### Step 5 — WSI Image Evaluation (Core Report Generation)

**File:** `xtuner/engine/hooks/evaluate_chat_hook.py`, `_eval_images` (lines 106–167)

For each `(sample_image, sample_input)` pair:

1. **Check WSI mode** — `runner.cfg.sample_type == 'wsi'` (line 117)
2. **Load features** — Convert numpy array to tensor, move to GPU (lines 118–119)
3. **Build prompt** — Prepend `<image>` token to input text, format with template (lines 121–123)
4. **Tokenize with image placeholder** — Split text on `<image>`, encode each chunk, insert `IMAGE_TOKEN_INDEX` (-200) between them (lines 124–138)
5. **LongNet encoding** — `model.LongNet_encoder(src_tokens=None, token_embeddings=image)` → aggregates patch features (line 142)
6. **Project to LLM space** — `model.projector(image)` → maps 512-dim to LLM hidden size (line 145)
7. **Merge multimodal inputs** — `prepare_inputs_labels_for_multimodal()` replaces the image token placeholder with actual visual embeddings in the input sequence (lines 147–150)
8. **Generate text** — `model.generate()` with the generation config (lines 153–158)
9. **Decode output** — `tokenizer.decode(generation_output[0])` (line 159)
10. **Log and save** — Print to logger (lines 160–161), optionally save to `.txt` file (lines 163–164)

### Step 6 — Output Saving

**File:** `xtuner/engine/hooks/evaluate_chat_hook.py`, `_save_eval_output` (lines 97–104)

Output is saved to: `{runner.log_dir}/vis_data/eval_outputs_iter_{runner.iter}.txt`

```
Eval output 1:
<system_prompt><user_input><generated_report>

Eval output 2:
...
```

---

## Path B: Inference-Time Report/Answer Generation (xtuner test)

This path runs the trained model on a test CSV to generate answers (or report-style text) for each slide.

### Step 1 — CLI Entry Point

**File:** `xtuner/entry_point.py` (lines 157–159, 244–302)

```bash
xtuner test <config> --checkpoint <ckpt> --test_slide_csv <csv> --test_output_csv <output.csv>
```

The `cli()` function dispatches to `xtuner/tools/test.py` via `subprocess.run()`.

### Step 2 — Configuration & Model Loading

**File:** `xtuner/tools/test.py` (lines 85–152)

1. Parse config file via `Config.fromfile()` (line 97)
2. Build MMEngine `Runner` (lines 116–122)
3. Load checkpoint weights into model (lines 129–134)
4. Build tokenizer (`Qwen2.5-7B-Instruct`) (lines 137–141)
5. Extract and prepare sub-models on GPU in eval mode:
   - `LongNet_encoder` → lines 146–148
   - `projector` → lines 150–152
   - `llm` → lines 143–144

### Step 3 — Test Data Loading

**File:** `xtuner/tools/test.py` (lines 155–159)

1. Read test CSV: `pd.read_csv(args.test_slide_csv)` — expects columns: `ID, Slide, Tumor, Broad Category, Narrow Category, Question, A, B, C, D, Answer` (line 155)
2. Initialize empty output DataFrame with an `Output` column (lines 157–159)

### Step 4 — Per-Slide Processing Loop

**File:** `xtuner/tools/test.py` (lines 161–266)

For each row in the test CSV:

#### 4a. Load WSI Features (lines 165–181)

1. Construct path: `TCGA_patch_feat/{Tumor}/{Slide}.csv` (line 166)
2. Read CSV and take first 512 columns (lines 168–169)
3. Subsample to max 38,400 patches using `np.linspace` (lines 170–175)
4. Reshape to `(1, N, 512)` tensor and move to GPU (lines 176–181)

#### 4b. Build Prompt (lines 182–211)

1. Use `PROMPT_TEMPLATE.qwen_chat` for formatting (line 182)
2. Combine question + multiple-choice options (lines 184–191)
3. Prepend `<image>\n` token to input (line 195)
4. Format with instruction template (line 196)
5. Tokenize: split on `<image>`, encode chunks, insert `IMAGE_TOKEN_INDEX` (lines 197–211)

#### 4c. Model Forward Pass (lines 213–221)

1. **LongNet encoding**: `LongNet_encoder(src_tokens=None, token_embeddings=image)` (line 214)
2. **Projection**: `projector(image)` maps to LLM embedding space (line 217)
3. **Multimodal merge**: `prepare_inputs_labels_for_multimodal()` (lines 218–221)

#### 4d. Text Generation (lines 223–249)

1. Configure generation: `max_new_tokens=500`, `do_sample=False` (greedy) (lines 223–230)
2. Set stop words from prompt template (lines 231–236)
3. `llm.generate()` (lines 238–243)
4. Decode output tokens (line 245)
5. Strip trailing `<|im_end|>` token if present (lines 246–247)

#### 4e. Collect Results (lines 251–266)

1. Build output row with all original columns + `Output` (generated text) (lines 251–264)
2. Append to DataFrame (line 265)
3. Save to CSV after each row: `df_test_output.to_csv(args.test_output_csv)` (line 266)

### Output Format

The final output is a CSV file with columns:

```
ID | Slide | Tumor | Broad Category | Narrow Category | Question | A | B | C | D | Answer | Output
```

Where `Output` contains the model-generated text (report summary or VQA answer).

---

## Path C: Extended Test Pipeline (test_full.py)

**File:** `xtuner/tools/test_full.py`

This is an extended version that adds:

1. **On-the-fly WSI processing** — Reads raw `.svs`/`.tiff` slide files directly via OpenSlide
2. **Patch extraction** — `TileWorker` (multiprocessing) extracts tiles from the WSI at the highest resolution, applies edge-detection filtering (lines 61–99)
3. **CONCH feature extraction** — Extracts 512-dim embeddings per patch using the CONCH model (if available)
4. **Caching** — Saves extracted features to `.h5` files for reuse
5. Same downstream pipeline: LongNet → Projector → LLM → text output → CSV

---

## Path D: Training Data Flow (Supervised Learning)

During training, the model learns to generate reports/captions through supervised fine-tuning.

### Data Format

**File:** `dataset/train_data_example.json`

```json
[
    {
        "id": "example_001",
        "image": ["path/to/TCGA-XX-XXXX.csv"],
        "conversations": [
            {"from": "human", "value": "<image>\nGenerate an overview..."},
            {"from": "gpt", "value": "The pathology examination reveals..."}
        ]
    }
]
```

### Data Loading

**File:** `xtuner/dataset/llava.py`, `LLaVADataset.__getitem__` (lines 110–133)

1. For `.csv` image files: read patch features, take first 512 columns (lines 118–121)
2. Subsample to `sample_num` (default 10,240) patches if needed (lines 123–127)
3. Convert to numpy → tensor (lines 129–130)
4. Set as `data_dict['pixel_values']` (line 132)

### Model Forward (Training)

**File:** `xtuner/model/llava.py`, `LLaVAModel.forward` (lines 321–349)

1. Extract `pixel_values` from data batch (line 332)
2. Pass through `LongNet_encoder` (line 334)
3. Pass through `projector` (line 337)
4. Merge with text via `prepare_inputs_labels_for_multimodal()` (line 340)
5. Compute cross-entropy loss via `self.llm(**data)` (line 363)

---

## Key Components Summary

| Component | File | Role |
|-----------|------|------|
| **CLI entry** | `xtuner/entry_point.py` | Routes `xtuner test/train/chat` commands |
| **LLaVAModel** | `xtuner/model/llava.py` | Core multimodal model: LongNet + Projector + Qwen LLM |
| **LongNet Encoder** | `xtuner/model/torchscale/` | 2-layer Transformer for patch feature aggregation |
| **MLP Projector** | `xtuner/model/modules/projector/` | Linear bridge between visual and language spaces |
| **EvaluateChatHook** | `xtuner/engine/hooks/evaluate_chat_hook.py` | Training-time report generation monitor |
| **test.py** | `xtuner/tools/test.py` | Inference-time batch generation on test CSV |
| **test_full.py** | `xtuner/tools/test_full.py` | End-to-end: raw WSI → patches → features → text |
| **LLaVADataset** | `xtuner/dataset/llava.py` | Training data loader for WSI features + conversations |
| **Config (stage 1)** | `xtuner/configs/slidechat/stage_1.py` | Caption/alignment training config |
| **Config (stage 2)** | `xtuner/configs/slidechat/stage_2.py` | VQA/instruction fine-tuning config |
| **prepare_inputs_labels_for_multimodal** | `xtuner/model/utils.py` | Merges image embeddings into token sequence |
| **load_image / load_wsi_feature** | `xtuner/dataset/utils.py` | CSV feature loading and subsampling |

---

## Data Flow Diagram (End-to-End)

```
                     ┌─────────────┐
                     │  Raw WSI    │
                     │ (.svs/.tiff)│
                     └──────┬──────┘
                            │  (test_full.py only)
                            ▼
                   ┌─────────────────┐
                   │  Patch Extract  │
                   │  (TileWorker)   │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ CONCH Encoder   │
                   │ 512-dim embed   │
                   └────────┬────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │    Patch Feature CSV     │
              │   (N rows × 512 cols)    │◄── Pre-computed for test.py
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │  Subsample (if N>38400)  │
              │  np.linspace indexing    │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │   Reshape (1, N, 512)    │
              │   → torch.Tensor → GPU   │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │    LongNet Encoder       │
              │ permute → encoder →      │
              │ permute back             │
              │ Output: (1, 576, 512)    │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │    MLP Projector         │
              │  512 → LLM hidden dim   │
              │  Output: (1, 576, 3584)  │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │ prepare_inputs_labels    │
              │ _for_multimodal()        │
              │ Replaces IMAGE_TOKEN     │
              │ with visual embeddings   │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │  Qwen2.5-7B-Instruct    │
              │  llm.generate()          │
              │  Autoregressive decode   │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │  tokenizer.decode()      │
              │  Strip <|im_end|>        │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │   Output                 │
              │  • .txt (train eval)     │
              │  • .csv (test inference) │
              └──────────────────────────┘
```

---

## Prompt Templates

The project uses Qwen chat format defined in `xtuner/utils/templates.py`:

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
<image>
{question_or_report_prompt}<|im_end|>
<|im_start|>assistant
{generated_report}<|im_end|>
```

For report generation during training eval, the prompt is:
> "Generate an overview summarizing the principal findings from the pathology examination of the whole slide image."

For test inference, prompts come from the test CSV's `Question` column with multiple-choice options appended.
