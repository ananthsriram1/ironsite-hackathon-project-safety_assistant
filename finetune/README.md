# Fine-tune Qwen3-VL on ConstructionSite 10k

This folder contains everything needed to fine-tune **Qwen3-VL-8B-Instruct** on the [ConstructionSite 10k](https://huggingface.co/datasets/LouisChen15/ConstructionSite) dataset using **LoRA** on a GPU instance (e.g. NVIDIA B200).

**→ For a clear order of operations (dependencies → model → dataset → upload & run), see [STEP_BY_STEP.md](STEP_BY_STEP.md).**

## Prerequisites

- Ubuntu (or similar) with **NVIDIA drivers** and **CUDA 12.x** installed.
- [Hugging Face token](https://huggingface.co/settings/tokens) with access to:
  - Dataset: `LouisChen15/ConstructionSite` (accept the dataset terms on the Hub if required).
  - Model: `Qwen/Qwen3-VL-8B-Instruct`.

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

### 2. (Optional) Inspect dataset columns

```bash
python finetune/scripts/inspect_dataset.py
```

### 3. Run training

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

## What the training script does

- **Model:** `Qwen/Qwen3-VL-8B-Instruct` in bfloat16, with **LoRA** on `q_proj`, `v_proj`, `k_proj`, `o_proj`.
- **Data:** ConstructionSite 10k **train** split. Each sample is turned into a single user turn (image + safety-inspection question) and one assistant turn (answer text built from `rule_1_violation` … `rule_4_violation` `reason` fields).
- **Training:** TRL `SFTTrainer` with a custom multimodal collate, bf16, gradient checkpointing, 3 epochs, batch size 8 × 2 grad-accum steps.

## Inference after training

A small script is provided to run the fine-tuned model on a single image:

```bash
python inference.py --image /path/to/construction_site.jpg
# Or specify adapter path:
python inference.py --image /path/to/image.jpg --adapter ./qwen-safety-finetuned/final_adapter
```

It loads the base model, applies the LoRA adapter, and prints the model’s safety-inspection response.

## References

- [ConstructionSite 10k (Hugging Face)](https://huggingface.co/datasets/LouisChen15/ConstructionSite)
- [ConstructionSite 10k implementation (GitHub)](https://github.com/LouisChen15/ConstructionSite-10k-Implementation)
- [Qwen3-VL (Hugging Face)](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [TRL – Fine-tuning a VLM with SFT](https://huggingface.co/docs/trl/main/en/training_vlm_sft)
