# Step-by-step: Fine-tune Qwen3-VL on ConstructionSite 10k (B200)

Follow these steps **in order** on your B200 instance (or any Ubuntu machine with NVIDIA drivers + CUDA 12.x).

---

## Before you start

- **Hugging Face token:** Create one at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- **Dataset access:** Open [LouisChen15/ConstructionSite](https://huggingface.co/datasets/LouisChen15/ConstructionSite) and accept the terms if the dataset is gated.
- **Model access:** Ensure you can access [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (usually public).

---

## Step 1: Install all dependencies

On the GPU instance (e.g. after SSH into the B200):

```bash
# System packages
sudo apt-get update
sudo apt-get install -y git git-lfs

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face and training libraries
pip install transformers datasets accelerate peft trl evaluate bitsandbytes pillow qwen-vl-utils
```

**Optional:** Log in to Hugging Face now (you’ll need it for the next steps):

```bash
huggingface-cli login
# Paste your token when prompted.
```

---

## Step 2: Download the model

This caches the model and processor so training doesn’t have to download them later.

```bash
# From the directory that contains the finetune folder (e.g. project root or finetune/)
python finetune/scripts/download_model.py
```

Wait until you see: `Model and processor cached. You can run training next.`

---

## Step 3: Download the dataset

This caches the ConstructionSite 10k train split.

```bash
python finetune/scripts/download_dataset.py
```

If you see a permission or gated-dataset error, accept the dataset terms on the [dataset page](https://huggingface.co/datasets/LouisChen15/ConstructionSite), then run the script again.

Wait until you see: `Dataset cached. You can run training next.`

---

## Step 4: Upload the `finetune/` folder and run the scripts

1. **Upload the whole `finetune/` folder** to your B200 instance (e.g. with `scp`, `rsync`, or your cloud’s file transfer).  
   The folder should contain at least:
   - `train.py`
   - `inference.py`
   - `setup_b200.sh`
   - `requirements-finetune.txt`
   - `scripts/inspect_dataset.py`
   - `scripts/download_model.py`
   - `scripts/download_dataset.py`
   - `README.md`
   - `STEP_BY_STEP.md`

2. **On the instance, go into the finetune directory:**

   ```bash
   cd finetune
   ```

3. **Optional – inspect dataset columns:**

   ```bash
   python scripts/inspect_dataset.py
   ```

4. **Run training:**

   ```bash
   python train.py
   ```

   To run in the background (recommended so it survives SSH disconnect):

   ```bash
   nohup python train.py > training_logs.txt 2>&1 &
   tail -f training_logs.txt
   ```

   Outputs and the final adapter are saved under `./qwen-safety-finetuned/` (e.g. `qwen-safety-finetuned/final_adapter`).

5. **After training – run inference on an image:**

   ```bash
   python inference.py --image /path/to/construction_site.jpg
   ```

   To use a specific adapter path:

   ```bash
   python inference.py --image /path/to/image.jpg --adapter ./qwen-safety-finetuned/final_adapter
   ```

---

## Quick reference (all steps in order)

| Step | What to do |
|------|------------|
| 1 | Install deps: `apt` + PyTorch (cu121) + `transformers`, `datasets`, `peft`, `trl`, etc. |
| 2 | Log in: `huggingface-cli login` |
| 3 | Download model: `python finetune/scripts/download_model.py` |
| 4 | Download dataset: `python finetune/scripts/download_dataset.py` |
| 5 | Upload `finetune/` to the instance |
| 6 | `cd finetune` and run `python train.py` (or with `nohup` for long runs) |
| 7 | Run `python inference.py --image <path>` to test the fine-tuned model |

---

## If you prefer one-shot setup on the instance

If the instance is empty and you’ve already uploaded `finetune/`:

```bash
cd finetune
chmod +x setup_b200.sh
./setup_b200.sh
# When prompted: huggingface-cli login

python scripts/download_model.py
python scripts/download_dataset.py
python train.py
```

This does dependencies (Step 1), then model (Step 2), then dataset (Step 3), then you run training (Step 4) manually in that order.
