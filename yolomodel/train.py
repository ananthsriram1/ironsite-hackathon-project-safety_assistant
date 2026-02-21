"""
IronSite — YOLO11 PPE Training Script
Optimised for B200 GPUs on Vast.ai / yast.ai (high-VRAM, headless server).

Usage (after uploading and unzipping your dataset):
    python train.py

Paths are set for /workspace/ layout on Vast.ai.
Adjust DATASET_ROOT, OUTPUT_ROOT, or MODEL at the top of this file if needed.
"""

import os
import sys
import shutil
import random
import subprocess
from pathlib import Path

import torch

# ─────────────────────────────────────────────
# CONFIGURATION — edit these to match your paths
# ─────────────────────────────────────────────

# Where your unzipped dataset lives (must contain images/ and labels/ subfolders)
DATASET_ROOT = Path("/workspace/Sample_Dataset_25k")

# Where the prepared train/val split and data.yaml will be written
OUTPUT_ROOT = Path("/workspace/PPE_Sample")

# Where training artefacts (weights, logs, plots) are saved
RUNS_DIR = "/workspace/runs/ppe_sample"

# YOLO model to fine-tune from (yolo11n=nano, yolo11s=small, yolo11m=medium, yolo11l=large)
# yolo11s is a good default. Use yolo11m or yolo11l if you want higher accuracy.
MODEL = "yolo11s.pt"

# Training hyperparameters
EPOCHS = 50
IMGSZ = 640
VAL_RATIO = 0.2
SEED = 42
RESUME = True  # Skip files already present (safe to re-run if interrupted)
# Use symlinks instead of copying — saves ~8GB on a 20GB instance (no duplicate dataset)
USE_SYMLINKS = True

# ─────────────────────────────────────────────
# B200 GPU SETTINGS (maximise throughput)
# ─────────────────────────────────────────────
# The B200 has 192 GB VRAM. We use:
#   batch=128   → fills VRAM efficiently for YOLO11s at 640px
#   workers=16  → fast parallel data loading
#   amp=True    → BFloat16 mixed precision (native to Blackwell / B200)
#   cache=True  → cache images in RAM so data loading never bottlenecks the GPU
#   optimizer="AdamW" → often converges faster than SGD on large batches
#   cos_lr=True → cosine LR decay, pairs well with large batches
#   nbs=64      → nominal batch size reference (lr is auto-scaled relative to this)

BATCH = 128
WORKERS = 16
AMP = True
CACHE = True  # set to "disk" if RAM is limited (it usually isn't on B200 machines)
OPTIMIZER = "AdamW"
COS_LR = True

# Dataset — 7 classes matching the notebook
NC = 7
CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "gloves",
    3: "vest",
    4: "no helmet",
    5: "no gloves",
    6: "no vest",
}


def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU found. Training will be very slow on CPU.")
        return
    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}  |  {vram_gb:.1f} GB VRAM")
    if n > 1:
        print(f"  {n} GPUs detected — Ultralytics will use all via DataParallel.")


def prepare_dataset(dataset_root: Path, output_root: Path) -> Path:
    """
    Build a YOLO-format train/val split from a flat images/ + labels/ folder.
    Returns the path to the written data.yaml.
    Mirrors the logic in the Ironsite Hackathon.ipynb notebook exactly.
    """
    img_dir = dataset_root / "images"
    lbl_dir = dataset_root / "labels"

    print("\n[Step 1/7] Checking dataset path...")
    if not dataset_root.exists():
        sys.exit(f"ERROR: Dataset not found at {dataset_root}. Check DATASET_ROOT.")
    print(f"  OK: {dataset_root}")

    print("\n[Step 2/7] Checking images/ and labels/ subfolders...")
    if not img_dir.is_dir():
        sys.exit(f"ERROR: Missing images/ folder at {img_dir}")
    if not lbl_dir.is_dir():
        sys.exit(f"ERROR: Missing labels/ folder at {lbl_dir}")
    print("  OK")

    print("\n[Step 3/7] Scanning for image-label pairs...")
    pairs = []
    for f in img_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if f.name.startswith("."):
            continue
        lbl = lbl_dir / f"{f.stem}.txt"
        if lbl.exists():
            pairs.append((f, lbl))

    if len(pairs) == 0:
        sys.exit(
            "ERROR: No image-label pairs found. Check that images/ and labels/ are not empty "
            "and that each image has a matching .txt label with the same base name."
        )
    print(f"  Found {len(pairs)} matched pairs.")

    print("\n[Step 4/7] Creating train/val split...")
    random.seed(SEED)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_RATIO))
    train_pairs = pairs[:-n_val]
    val_pairs = pairs[-n_val:]
    print(f"  Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    action = "Linking" if USE_SYMLINKS else "Copying"
    print(f"\n[Step 5/7] {action} files to {output_root} (RESUME={RESUME})...")
    for split, pair_list in [("train", train_pairs), ("val", val_pairs)]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
        done = 0
        skipped = 0
        total = len(pair_list)
        for i, (img_path, lbl_path) in enumerate(pair_list):
            dst_img = output_root / split / "images" / img_path.name
            dst_lbl = output_root / split / "labels" / (lbl_path.stem + ".txt")
            if RESUME and dst_img.exists() and dst_lbl.exists():
                skipped += 1
            else:
                src_img = img_path.resolve()
                src_lbl = lbl_path.resolve()
                if USE_SYMLINKS:
                    dst_img.symlink_to(src_img)
                    dst_lbl.symlink_to(src_lbl)
                else:
                    shutil.copy2(img_path, dst_img)
                    shutil.copy2(lbl_path, dst_lbl)
                done += 1
            if (i + 1) % 1000 == 0 or i + 1 == total:
                pct = (i + 1) / total * 100
                print(f"    {split}: {i+1}/{total} ({pct:.0f}%)  done={done}  skipped={skipped}")
        print(f"  [{split}] done.")

    print("\n[Step 6/7] Verifying files...")
    for split in ("train", "val"):
        n_imgs = len(list((output_root / split / "images").iterdir()))
        n_lbls = len(list((output_root / split / "labels").iterdir()))
        status = "OK" if n_imgs == n_lbls else "MISMATCH"
        print(f"  {split}/images: {n_imgs}  {split}/labels: {n_lbls}  [{status}]")

    print("\n[Step 7/7] Writing data.yaml...")
    names_block = "\n".join(f"  {k}: {v}" for k, v in CLASS_NAMES.items())
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        f"# IronSite PPE Dataset — {NC} classes\n"
        f"path: {output_root}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"nc: {NC}\n"
        f"names:\n{names_block}\n"
    )
    print(f"  Written: {data_yaml}")
    return data_yaml


def run_training(data_yaml: Path):
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  IronSite YOLO Training — B200-Optimised")
    print("=" * 60)
    print(f"  Model:     {MODEL}")
    print(f"  Data:      {data_yaml}")
    print(f"  Epochs:    {EPOCHS}")
    print(f"  Image sz:  {IMGSZ}")
    print(f"  Batch:     {BATCH}  (B200-tuned)")
    print(f"  Workers:   {WORKERS}")
    print(f"  AMP:       {AMP}  (BF16 on Blackwell)")
    print(f"  Cache:     {CACHE}")
    print(f"  Optimizer: {OPTIMIZER}")
    print(f"  Output:    {RUNS_DIR}/train/")
    print("=" * 60 + "\n")

    model = YOLO(MODEL)
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        amp=AMP,
        cache=CACHE,
        optimizer=OPTIMIZER,
        cos_lr=COS_LR,
        nbs=64,          # nominal batch size — LR auto-scaled relative to this
        lr0=0.01,        # initial learning rate
        lrf=0.01,        # final LR as fraction of lr0
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        # Augmentation (sensible defaults for construction/PPE)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,  # disable mosaic for final 10 epochs for clean validation
        # Output
        project=RUNS_DIR,
        name="train",
        exist_ok=True,
        plots=True,
    )

    best = Path(RUNS_DIR) / "train" / "weights" / "best.pt"
    last = Path(RUNS_DIR) / "train" / "weights" / "last.pt"
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best weights : {best}")
    print(f"  Last weights : {last}")
    print("=" * 60)
    print("\nTo copy weights back to your Mac:")
    print(f"  scp -P <PORT> root@<IP>:{best} ./best.pt")
    print(f"  scp -P <PORT> root@<IP>:{last} ./last.pt")

    return results


def main():
    print("\n=== IronSite — Train.py ===")
    print(f"Python {sys.version.split()[0]}  |  Torch {torch.__version__}")
    check_gpu()

    data_yaml = prepare_dataset(DATASET_ROOT, OUTPUT_ROOT)
    run_training(data_yaml)


if __name__ == "__main__":
    main()
