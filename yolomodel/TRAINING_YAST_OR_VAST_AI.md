# Training YOLO on yast.ai / Vast.ai with B200 GPUs (40GB+ Container)

This guide gives **exact instructions** to train the IronSite YOLO (PPE/construction hazard) model from `Ironsite Hackathon.ipynb` on a cloud GPU instance with **at least 40GB disk**, using your image dataset.

> **Note:** If you meant **Vast.ai** (vast.ai), these steps apply directly. If you use another provider (e.g. Yotta, RunPod) with B200 or similar GPUs, the same workflow applies: 40GB+ container, same dataset layout, same training commands.

---

## 1. Project and dataset summary

- **Repo:** IronSite — Construction Site Safety (PPE + hazard detection).
- **Notebook:** `Ironsite Hackathon.ipynb` — YOLO11 (Ultralytics) training and inference.
- **Dataset format:** YOLO-style:
  - One folder with `images/` (jpg/png) and `labels/` (`.txt` per image, same stem).
  - Classes in the notebook: **7** — `person`, `helmet`, `gloves`, `vest`, `no helmet`, `no gloves`, `no vest`.
- **Training in notebook:** `YOLO("yolo11n.pt")` → `model.train(data=..., epochs=50, imgsz=640, batch=16)`.

---

## 2. Reserve a 40GB+ container (Vast.ai example)

### 2.1 Create account and add credits

- Sign up at [vast.ai](https://vast.ai).
- Add credits (Billing → Add credits). You’ll be charged for GPU time + **disk** (disk is charged even when the instance is stopped until you destroy it).

### 2.2 Set disk to at least 40GB

- Go to **Create** (new instance).
- In the instance configuration, find the **Storage / Disk size** slider.
- Set it to **at least 40 GB** (e.g. 50 GB if your dataset + runs are large).
- **Important:** Disk size is fixed at creation; you cannot increase it later.

### 2.3 Choose GPU and image

- **GPU:** Filter for **B200** if available, or **H100 / A100 / A6000** (24GB+ VRAM) for YOLO training.
- **Image:** Use a PyTorch + CUDA image so Ultralytics runs with GPU. For example:
  - `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
  - Or a Jupyter template that uses a similar image (e.g. “PyTorch” template).

### 2.4 Launch mode

- **Jupyter:** Easiest if you want to run the notebook in the browser.
- **SSH:** Use if you prefer terminal + `scp`/rsync to copy dataset and run a Python script.

Click **Rent** and wait for the instance to start. Note the SSH host/port or Jupyter URL.

---

## 3. Get your dataset onto the instance

Your dataset must look like this (same as in the notebook):

```text
SomeFolder/
  images/   # .jpg / .jpeg / .png
  labels/   # .txt (one per image, same base name)
```

**Ways to get it into the 40GB container:**

1. **SCP/SFTP (SSH):**  
   From your machine (where the dataset lives):
   ```bash
   scp -P <INSTANCE_PORT> -r /path/to/YourDataset ironsite@<INSTANCE_IP>:~/dataset/
   ```
   So on the instance you have e.g. `~/dataset/YourDataset/images/` and `~/dataset/YourDataset/labels/`.

2. **Google Drive (if you use the notebook’s Drive path):**  
   Mount Drive in the instance and copy from the mounted path to a local folder (e.g. `~/dataset/`) so training doesn’t depend on Drive being mounted later.

3. **Zip upload:**  
   Zip `YourDataset` (images + labels), upload to the instance (e.g. via Jupyter “Upload” or scp), then:
   ```bash
   unzip -q YourDataset.zip -d ~/dataset/
   ```

Ensure the dataset is **inside** the 40GB disk (e.g. under `/root` or `~/`), not on a small system volume.

---

## 4. Prepare environment on the instance

SSH into the instance (or open a Jupyter terminal) and run:

```bash
pip install ultralytics opencv-python numpy pyyaml
```

If you prefer to use the notebook, upload `Ironsite Hackathon.ipynb` to the instance and use the same `pip install` in a cell (as in the notebook).

---

## 5. Prepare `data.yaml` and train/val split

The notebook builds a train/val split and writes `data.yaml`. You can do the same in Python on the instance.

### 5.1 Set paths

- `DATASET_ROOT`: path to your dataset folder that contains `images/` and `labels/` (e.g. `~/dataset/YourDataset` or `/content/drive/MyDrive/dataset/Sample_Dataset_25k`).
- `OUTPUT_ROOT`: where you want the split and `data.yaml` (e.g. `~/PPE_Train` or `/workspace/PPE_Train`). This must be on the 40GB disk.

### 5.2 Run the same logic as the notebook

Run this (adjust `DATASET_ROOT` and `OUTPUT_ROOT`):

```python
import os
import shutil
import random
from pathlib import Path

DATASET_ROOT = Path("/root/dataset/YourDataset")   # <-- your dataset path
OUTPUT_ROOT = Path("/root/PPE_Train")               # <-- output on 40GB disk
VAL_RATIO = 0.2
SEED = 42
RESUME = True

img_dir = DATASET_ROOT / "images"
lbl_dir = DATASET_ROOT / "labels"
assert img_dir.is_dir() and lbl_dir.is_dir(), "Need images/ and labels/"

pairs = []
for f in img_dir.iterdir():
    if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg", ".png") or f.name.startswith("."):
        continue
    stem = f.stem
    lbl = lbl_dir / f"{stem}.txt"
    if lbl.exists():
        pairs.append((f, lbl))

random.seed(SEED)
random.shuffle(pairs)
n_val = max(1, int(len(pairs) * VAL_RATIO))
train_pairs, val_pairs = pairs[:-n_val], pairs[-n_val:]

for split, pair_list in [("train", train_pairs), ("val", val_pairs)]:
    (OUTPUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in pair_list:
        shutil.copy2(img_path, OUTPUT_ROOT / split / "images" / img_path.name)
        shutil.copy2(lbl_path, OUTPUT_ROOT / split / "labels" / (lbl_path.stem + ".txt"))

# data.yaml (7 classes as in notebook)
DATA_YAML = OUTPUT_ROOT / "data.yaml"
DATA_YAML.write_text(f"""# PPE dataset - 7 classes
path: {OUTPUT_ROOT}
train: train/images
val: val/images
nc: 7
names:
  0: person
  1: helmet
  2: gloves
  3: vest
  4: no helmet
  5: no gloves
  6: no vest
""")
print("data.yaml path:", DATA_YAML)
print("Train:", len(train_pairs), "Val:", len(val_pairs))
```

Use the printed `data.yaml` path in the next step.

---

## 6. Run YOLO training (exact commands)

From the **same environment** (same `OUTPUT_ROOT` and `DATA_YAML` path):

```python
from ultralytics import YOLO

# Path to data.yaml (must match what you wrote above)
data_yaml = "/root/PPE_Train/data.yaml"   # or str(OUTPUT_ROOT / "data.yaml")

model = YOLO("yolo11n.pt")
results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/ppe_sample",
    name="train",
)
```

- **Best weights:** `runs/ppe_sample/train/weights/best.pt`
- **Last weights:** `runs/ppe_sample/train/weights/last.pt`

If you run out of GPU memory, reduce `batch` (e.g. `batch=8` or `batch=4`). With a 40GB disk, keep an eye on space: dataset + `runs/` + any caches must fit.

---

## 7. Optional: use a larger model and more epochs

For better accuracy at the cost of time and VRAM:

```python
model = YOLO("yolo11s.pt")   # or yolo11m.pt if GPU has enough VRAM
results = model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,   # reduce to 8 or 4 if OOM
    project="runs/ppe_sample",
    name="train",
)
```

---

## 8. Save results off the instance

Before destroying the instance, copy the trained weights and (optional) `runs/` to your machine or cloud:

```bash
# From your local machine
scp -P <PORT> ironsite@<IP>:~/runs/ppe_sample/train/weights/best.pt ./
scp -P <PORT> ironsite@<IP>:~/runs/ppe_sample/train/weights/last.pt ./
```

Or use the provider’s “Download” or cloud-sync if available.

---

## 9. Checklist (40GB+ B200-style run)

| Step | Action |
|------|--------|
| 1 | Create instance with **≥ 40 GB disk** (slider at create time). |
| 2 | Choose B200 (or H100/A100) and a PyTorch/CUDA image. |
| 3 | Upload/copy dataset so you have `.../images/` and `.../labels/`. |
| 4 | `pip install ultralytics opencv-python numpy pyyaml`. |
| 5 | Run the split script and write `data.yaml` (7 classes). |
| 6 | Run `model = YOLO("yolo11n.pt")` and `model.train(data=..., epochs=50, imgsz=640, batch=16, ...)`. |
| 7 | Download `best.pt` (and optionally `last.pt`) before destroying the instance. |

If your provider is named differently (e.g. “yast.ai”), the same steps apply: ensure the container has at least 40GB disk, put the dataset and `data.yaml` there, then run the same `ultralytics` training commands.
