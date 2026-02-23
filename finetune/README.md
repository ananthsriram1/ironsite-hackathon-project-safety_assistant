# Fine-tune Qwen3-VL on ConstructionSite 10k

This folder contains everything needed to fine-tune **Qwen3-VL-8B-Instruct** on the [ConstructionSite 10k](https://huggingface.co/datasets/LouisChen15/ConstructionSite) dataset using **LoRA** on a GPU instance (e.g. NVIDIA B200).

**→ For a clear order of operations (dependencies → model → dataset → upload & run), see [STEP_BY_STEP.md](STEP_BY_STEP.md).**

---

## Prerequisites

- Ubuntu (or similar) with **NVIDIA drivers** and **CUDA 12.x** installed.
- [Hugging Face token](https://huggingface.co/settings/tokens) with access to:
  - Dataset: `LouisChen15/ConstructionSite` (accept the dataset terms on the Hub if required).
  - Model: `Qwen/Qwen3-VL-8B-Instruct`.

---

## Quick start on a B200 instance

### 1. System prep and dependencies

```bash
# From the project root or from this finetune/ directory
chmod +x finetune/setup_b200.sh
./finetune/setup_b200.sh
# When prompted: huggingface-cli login and paste your HF token
```

Or manually:

```bash
sudo apt update && sudo apt install -y git git-lfs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r finetune/requirements-finetune.txt
huggingface-cli login
```

### 2. Prepare the dataset (required before training)

Training reads from a **pre-formatted cache**, not the raw Hugging Face dataset. Run once:

```bash
cd finetune
python scripts/download_dataset.py   # cache raw dataset
python prepare_data.py               # build ./dataset_cache/ (train + test split)
```

### 3. (Optional) Inspect dataset columns

```bash
python scripts/inspect_dataset.py
```

### 4. Run training

```bash
cd finetune
python train.py
```

For an uninterruptible run (e.g. over SSH):

```bash
nohup python train.py > training_logs.txt 2>&1 &
tail -f training_logs.txt
```

Checkpoints and the final LoRA adapter are saved under `./qwen-safety-finetuned/`.

### 5. Evaluate the adapter (optional)

```bash
python evaluate_adapter.py
# Uses ./qwen-safety-finetuned/final_adapter by default; set ADAPTER_PATH in the script to test another adapter.
```

---

## Full training analysis

This section explains the end-to-end pipeline: data, model and LoRA setup, training configuration, and how results are measured.

### 1. Data pipeline

- **Source:** [LouisChen15/ConstructionSite](https://huggingface.co/datasets/LouisChen15/ConstructionSite) (train split only).
- **Preparation:** `prepare_data.py` runs **once** before training. It:
  - Loads the dataset and strips all columns except `image`.
  - Builds **multi-task** conversation examples in Qwen chat format (system + user + assistant).
  - **Task mix (per example, deterministic by index, seed 42):**
    - **60%** — Safety VQA: user asks for safety rule violations; assistant answer is built from `rule_1_violation` … `rule_4_violation` (reason + bounding boxes in `[x_min, y_min, x_max, y_max]` normalized to 0–1000).
    - **25%** — Captioning: user asks for a description; assistant uses `image_caption`.
    - **15%** — Grounding: user asks to locate excavators, rebar, workers with white hard hats; assistant uses dataset bounding boxes for those categories.
  - Splits into **train** (95%) and **test** (5%) with a fixed seed (42) and saves to `./dataset_cache/`.
- **Training** uses the cached train split; **evaluation** (in `evaluate_adapter.py`) uses a separate holdout from the raw dataset (5% of train, 100 samples by default) for rule/bbox metrics.

**Rule semantics in the data:**

| Rule | Description |
|------|-------------|
| 1 | Basic PPE (hard hats, safety glasses, vests, protective clothing). |
| 2 | Safety harness when working ≥3 m high without edge protection. |
| 3 | Edge protection (guardrails, fences) for underground projects ≥3 m depth. |
| 4 | No workers within excavator blind spots or operating radius. |

### 2. Model and LoRA configuration

- **Base model:** `Qwen/Qwen3-VL-8B-Instruct`, loaded in **bfloat16**.
- **Attention:** Uses **Flash Attention 2** if `flash_attn` is installed (e.g. on B200); otherwise **SDPA**.
- **LoRA (PEFT):**
  - **Rank** `r = 32`, **alpha** `64` (effective scaling 2×).
  - **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all linear layers in the MLP and attention of the language backbone).
  - **Dropout:** 0.05; **bias:** not trained.
  - **Task type:** `CAUSAL_LM`.
- Only the LoRA parameters are trained; the rest of the 8B model is frozen, which keeps memory and compute manageable on a single high-memory GPU.

### 3. Training configuration

- **Framework:** Hugging Face `Trainer` with a **custom data collator** that:
  - Applies the processor’s chat template to each example’s `messages`.
  - Packs images and text, then builds **labels**: padding, image token, and vision start/end tokens are masked with `-100` so loss is computed only on the assistant text.
- **Effective batch size:** `1` per device × `16` gradient accumulation steps = **16** per update (single-GPU). With multi-GPU, effective batch = `world_size × 16`.
- **Epochs:** 3 over the full training set.
- **Optimizer:** AdamW (`adamw_torch_fused`), **learning rate** `2e-5`, **weight decay** `0.01`.
- **Schedule:** Cosine with **warmup ratio** 0.05; **max gradient norm** 1.0.
- **Precision:** `bf16=True`, `tf32=True`.
- **Efficiency:** Gradient checkpointing enabled (reentrant=False); **dataloader** 2 workers, pin memory.
- **Checkpoints:** Saved every 100 steps, keep last 3; **evaluation** every 100 steps; **TensorBoard** logs in `./qwen-safety-finetuned/logs`.

### 4. Evaluation methodology

`evaluate_adapter.py` loads the base model, applies the saved LoRA adapter, and runs on a held-out set (default 100 samples, 5% of train split, seed 42). For each sample it:

1. Runs the model with a fixed safety-inspection user prompt.
2. **Rule detection:** Parses which of rules 1–4 the model mentions (including keyword fallbacks for PPE, harness, edge protection, excavator proximity). Compares to ground truth from `rule_*_violation` to compute:
   - **Precision** — When the model says “violation”, how often it’s correct.
   - **Recall** — Fraction of true violations that the model flags.
   - **F1** — Harmonic mean of precision and recall.
3. **Bounding box quality:** Extracts predicted boxes from the text (supports both [0–1] and [0–1000] coordinates), matches them greedily to ground-truth boxes, and reports **mean IoU** (Intersection over Union). Only samples where the model outputs at least one box contribute to mean IoU.

So the pipeline is optimized for: (1) correctly identifying which of the four safety rules are violated, and (2) localizing violations with bounding boxes where applicable.

### 5. Results achieved

After training, run:

```bash
python evaluate_adapter.py
```

to get metrics for the adapter at `ADAPTER_PATH` (default: `./qwen-safety-finetuned/final_adapter`). Typical outputs:

- **Rule Detection Precision** — When the model reports a violation, how often it matches ground truth.
- **Rule Detection Recall** — How many of the actual violations the model catches.
- **Rule Detection F1** — Overall rule-detection balance.
- **Mean Bounding Box IoU** — Alignment of predicted vs. ground-truth boxes (higher is better).
- **Samples with parsed rules/boxes** — How often the model output could be parsed for rules and bboxes.

Training loss and eval loss can be inspected via TensorBoard:

```bash
tensorboard --logdir ./qwen-safety-finetuned/logs
```

For this project, the fine-tuned VLM is used in the **IronSite** safety dashboard as the compliance verification stage (Stage 3), often with an optional LoRA adapter to reduce hallucination and improve construction-specific understanding (see project DEVPOST and RUN docs).

---

## What the training script does (summary)

- **Model:** `Qwen/Qwen3-VL-8B-Instruct` in bfloat16, with **LoRA** on `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Data:** Pre-formatted dataset in `./dataset_cache/` (from `prepare_data.py`): ConstructionSite 10k train split, multi-task (safety VQA, captioning, grounding).
- **Training:** Hugging Face `Trainer`, custom multimodal collator, bf16, gradient checkpointing, **3 epochs**, effective batch size **16** (per-device batch 1 × 16 grad-accum steps), eval and checkpoints every 100 steps.

---

## Inference after training

A small script is provided to run the fine-tuned model on a single image:

```bash
python inference.py --image /path/to/construction_site.jpg
# Or specify adapter path:
python inference.py --image /path/to/image.jpg --adapter ./qwen-safety-finetuned/final_adapter
```

It loads the base model, applies the LoRA adapter, and prints the model’s safety-inspection response.

---

## References

- [ConstructionSite 10k (Hugging Face)](https://huggingface.co/datasets/LouisChen15/ConstructionSite)
- [ConstructionSite 10k implementation (GitHub)](https://github.com/LouisChen15/ConstructionSite-10k-Implementation)
- [Qwen3-VL (Hugging Face)](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [TRL – Fine-tuning a VLM with SFT](https://huggingface.co/docs/trl/main/en/training_vlm_sft)
