"""
IronSite — Batch video benchmarking with trained PPE model.
Run on Vast.ai after training. Uses batched inference + frame_skip for speed.

Upload "Benchmarking Vids" folder AFTER training finishes (saves bandwidth/disk
contention; you only have 20GB so avoid uploading 900MB while training).
"""

import os
import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "/workspace/runs/ppe_sample/train/weights/best.pt"
INPUT_DIR = "/workspace/Benchmarking Vids"
OUTPUT_DIR = "/workspace/hazard_results"
BATCH_SIZE = 64   # B200 can handle this easily
FRAME_SKIP = 3    # Infer every Nth frame (3 ≈ 10 fps if source is 30 fps)
CONF_THRESHOLD = 0.25

# Map model class names to hazard taxonomy (matches train.py classes)
CONSTRUCTION_HAZARD_CLASSES = {
    "person": "person",
    "helmet": "hardhat",
    "hardhat": "hardhat",
    "gloves": "gloves",
    "vest": "safety_vest",
    "safety vest": "safety_vest",
    "no helmet": "no_hardhat",
    "no_hardhat": "no_hardhat",
    "no gloves": "no_gloves",
    "no vest": "no_safety_vest",
    "no_safety_vest": "no_safety_vest",
    "machinery": "excavator",
    "vehicle": "dump_truck",
}

HAZARD_RULES = {
    "heavy_equipment_proximity": {
        "description": "Person near heavy machinery",
        "requires": [["excavator", "dump_truck", "forklift"], "person"],
        "severity": "high",
    },
    "ppe_violation_no_hardhat": {
        "description": "Person without hardhat",
        "requires": [["no_hardhat"], "person"],
        "severity": "high",
    },
    "ppe_violation_no_safety_vest": {
        "description": "Person without safety vest",
        "requires": ["no_safety_vest"],
        "severity": "high",
    },
}


# ==========================================
# 2. HAZARD LOGIC
# ==========================================
def calculate_proximity(bbox1, bbox2):
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def map_class_name(class_name):
    c_lower = class_name.lower().strip()
    return CONSTRUCTION_HAZARD_CLASSES.get(c_lower, c_lower.replace(" ", "_"))


def detect_hazards(detections, proximity_threshold=0.3):
    hazards = []
    by_class = {}
    for det in detections:
        c = det["class_name"]
        if c not in by_class:
            by_class[c] = []
        by_class[c].append(det)

    for h_name, rule in HAZARD_RULES.items():
        reqs = rule["requires"]
        if len(reqs) == 2:
            req1_dets = []
            if isinstance(reqs[0], list):
                for c in reqs[0]:
                    req1_dets.extend(by_class.get(c, []))
            else:
                req1_dets = by_class.get(reqs[0], [])
            req2_dets = by_class.get(reqs[1], [])
            for d1 in req1_dets:
                for d2 in req2_dets:
                    dist = calculate_proximity(d1["bbox"], d2["bbox"])
                    if dist < proximity_threshold:
                        hazards.append({"type": h_name, "severity": rule["severity"]})
        elif len(reqs) == 1:
            req = reqs[0]
            if isinstance(req, list):
                for c in req:
                    if by_class.get(c):
                        hazards.append({"type": h_name, "severity": rule["severity"]})
                        break
            elif by_class.get(req):
                hazards.append({"type": h_name, "severity": rule["severity"]})
    return hazards


# ==========================================
# 3. BATCH VIDEO PROCESSOR (one output frame per input frame)
# ==========================================
def _annotate_frame(res):
    dets = []
    if res.boxes is not None:
        for j in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[j].item())
            name = res.names[cls_id]
            mapped = map_class_name(name)
            if mapped:
                dets.append({"class_name": mapped, "bbox": res.boxes.xyxy[j].cpu().numpy()})
    hazards = detect_hazards(dets)
    ann = res.plot()
    if hazards:
        y = 30
        for h in hazards[:5]:
            cv2.putText(
                ann, f"WARNING: {h['type']} ({h['severity']})",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            y += 28
    return ann


def process_video_fast(model, video_path, output_path, batch_size=64, frame_skip=3):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Skipping {video_path.name} (cannot open)")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"  Skipping {video_path.name} (cannot create output)")
        cap.release()
        return

    print(f"  Processing {video_path.name}  |  Frames: {total_frames}  |  Infer every {frame_skip}")

    frame_index = 0
    batch_frames = []
    last_annotated = None
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            batch_frames.append(frame.copy())

        if len(batch_frames) >= batch_size:
            results = model(batch_frames, conf=CONF_THRESHOLD, verbose=False)
            for res in results:
                ann = _annotate_frame(res)
                last_annotated = ann
                for _ in range(frame_skip):
                    out.write(ann)
                    written += 1
            batch_frames = []
            if (frame_index + 1) % 300 < batch_size * frame_skip:
                print(f"    -> {frame_index + 1}/{total_frames}")

        frame_index += 1

    # Last partial batch
    if batch_frames:
        results = model(batch_frames, conf=CONF_THRESHOLD, verbose=False)
        for res in results:
            ann = _annotate_frame(res)
            last_annotated = ann
            n_write = min(frame_skip, frame_index - written)
            for _ in range(n_write):
                out.write(ann)
                written += 1
            if written >= frame_index:
                break

    # Pad to match input frame count (trailing frames reuse last annotated)
    while written < frame_index:
        out.write(last_annotated if last_annotated is not None else frame)
        written += 1

    cap.release()
    out.release()
    print(f"  Saved: {output_path.name}  ({written} frames)\n")


# ==========================================
# 4. MAIN
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Weights not found at {MODEL_PATH}. Train first (python train.py).")
        return

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    input_path = Path(INPUT_DIR)
    if not input_path.is_dir():
        print(f"ERROR: Input dir not found: {INPUT_DIR}")
        return

    videos = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in (".mp4", ".mov", ".avi")
    ]
    if not videos:
        print(f"No videos found in {INPUT_DIR}")
        return

    print(f"Found {len(videos)} videos. Starting batch benchmark...\n" + "=" * 50)
    for vid in sorted(videos):
        out_file = Path(OUTPUT_DIR) / f"{vid.stem}_annotated.mp4"
        process_video_fast(
            model, vid, out_file,
            batch_size=BATCH_SIZE,
            frame_skip=FRAME_SKIP,
        )
    print("All benchmarking videos complete.")


if __name__ == "__main__":
    main()
