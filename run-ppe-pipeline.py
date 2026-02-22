import os
import cv2
import torch
import numpy as np
import time
import sys
from collections import deque, defaultdict
from ultralytics import YOLO, SAM

# ---------------------------------------------------------
# CONFIGURATION (update for your environment)
# ---------------------------------------------------------
WORKSPACE = "/workspace"  # or "/content" for Colab
# YOLO weights on GPU (upload from project root):
#   scp -P 16517 model/yolomodel/weights/best.pt  root@154.57.34.95:/workspace/weights/best.pt
#   scp -P 16517 model/yolomodel/weights/last.pt   root@154.57.34.95:/workspace/weights/last.pt
# On GPU first run: mkdir -p /workspace/weights
YOLO_WEIGHTS_PATH = f"{WORKSPACE}/weights/best.pt"   # trained; use "last.pt" for latest checkpoint
SAM_WEIGHTS_PATH = f"{WORKSPACE}/sam3.pt"
# Video: use file in Benchmarking Vids, or set to a single file path
VIDEO_PATH = f"{WORKSPACE}/Benchmarking Vids/Production Masonry.mp4"
OUTPUT_PATH = f"{WORKSPACE}/yolo_sam3_clean_prompts.mp4"
# Optional: set to a folder path to save sample frames for visual verification (e.g. f"{WORKSPACE}/sample_frames")
SAVE_SAMPLE_FRAMES_DIR = None  # or e.g. f"{WORKSPACE}/sample_frames"

# Detection & compliance
PER_CLASS_CONF = {
    "person": 0.30,
    "helmet": 0.25,
    "gloves": 0.25,
    "vest": 0.25,
    "no helmet": 0.20,
    "no gloves": 0.20,
    "no vest": 0.20,
}
WINDOW_SIZE = 30
VOTE_THRESHOLD = 0.35   # class must appear in >=35% of frames to count
SAM_INTERVAL = 5

# Crop-and-classify: for each tracked person, re-run YOLO on just their crop.
# This turns a ~15px head at distance into a ~100px+ head for YOLO to classify.
CROP_CONF = 0.20       # confidence threshold for crop-level detections
CROP_IMGSZ = 320       # resolution to run YOLO on the person crop
CROP_EXPAND = 0.05     # expand person box by 5% each side before cropping
# Which crop classes count as PPE presence vs violation (crop is from person, so positions shift)
CROP_POSITIVE = {"helmet", "gloves", "vest"}
CROP_NEGATIVE = {"no helmet", "no gloves", "no vest"}

# STRICT VALIDATION MAP (Exactly your 7 trained classes)
VALID_CLASSES = ["person", "helmet", "gloves", "vest", "no helmet", "no gloves", "no vest"]


def box_iou(a, b):
    """IoU between two boxes [x1, y1, x2, y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def expand_box(box, factor=0.15):
    """Expand box by factor (e.g. 0.15 = 15% on all sides). Returns [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * factor, h * factor
    return [x1 - dx, y1 - dy, x2 + dx, y2 + dy]


def extract_person_crop(frame, person_box, img_h, img_w):
    """Crop + expand person box. Returns numpy crop or None if invalid."""
    x1, y1, x2, y2 = expand_box(person_box, CROP_EXPAND)
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(img_w, x2)), int(min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def batch_classify_ppe(person_crop_pairs, yolo_detector, device):
    """
    Run a single batched YOLO inference across all person crops.
    person_crop_pairs: list of (person_id, crop_numpy) — None crops are skipped.
    Returns dict { person_id: (positives: set, negatives: set) }.
    """
    valid = [(pid, crop) for pid, crop in person_crop_pairs if crop is not None]
    if not valid:
        return {}

    person_ids = [pid for pid, _ in valid]
    crop_images = [crop for _, crop in valid]

    # Single batched GPU call — far more efficient than N sequential calls
    results = yolo_detector(
        crop_images,
        conf=CROP_CONF,
        imgsz=CROP_IMGSZ,
        verbose=False,
        device=device,
        half=True,   # FP16: ~2x faster on H100, negligible accuracy difference
        # No augment=True here — the crop itself already solves the scale problem;
        # TTA would triple the inference time for negligible gain.
    )

    output = {}
    crop_names = yolo_detector.names
    for person_id, result in zip(person_ids, results):
        positives, negatives = set(), set()
        if result.boxes is not None:
            for cls_id in result.boxes.cls.cpu().numpy().tolist():
                cls_name = crop_names[int(cls_id)].lower()
                if cls_name in CROP_POSITIVE:
                    positives.add(cls_name)
                elif cls_name in CROP_NEGATIVE:
                    negatives.add(cls_name)
        output[person_id] = (positives, negatives)

    return output


def main():
    # ---------------------------------------------------------
    # 1. Hardware & GPU Setup
    # ---------------------------------------------------------
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This script will run on CPU and be very slow.")
        device = "cpu"
    else:
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda:0"
        # Hardware-level optimizations for H100/A100 class GPUs
        torch.backends.cuda.matmul.allow_tf32 = True   # faster matmul via TF32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True           # auto-tune conv kernels after first batch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU VRAM: {vram_gb:.1f} GB")

    # ---------------------------------------------------------
    # 2. Load Models
    # ---------------------------------------------------------
    print(f"\nLoading custom YOLO detector from {YOLO_WEIGHTS_PATH}...")
    try:
        yolo_detector = YOLO(YOLO_WEIGHTS_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(1)

    sam_masker = None
    if os.path.isfile(SAM_WEIGHTS_PATH):
        print(f"Loading SAM 3 segmentation model from {SAM_WEIGHTS_PATH}...")
        try:
            sam_masker = SAM(SAM_WEIGHTS_PATH)
        except Exception as e:
            print(f"Warning: SAM failed to load ({e}). Running YOLO-only (no masks).")
    else:
        print(f"SAM weights not found at {SAM_WEIGHTS_PATH}. Running YOLO-only (no masks).")

    # ---------------------------------------------------------
    # 3. Video Setup
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"\nStarting Tracking Pipeline on {VIDEO_PATH} ({total_frames} frames)...")

    # Temporal sliding window: per worker, per class -> deque of 0/1 over last WINDOW_SIZE frames
    worker_detection_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=WINDOW_SIZE)))
    cached_annotated = None
    # Verification: per-class detection counts over the whole run
    detection_counts = defaultdict(int)
    if SAVE_SAMPLE_FRAMES_DIR:
        os.makedirs(SAVE_SAMPLE_FRAMES_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 4. Processing Loop
    # ---------------------------------------------------------
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # YOLO Detection & Tracking (global conf low; we post-filter per class)
        # imgsz=1280: far better recall for small PPE items on distant workers
        # half=True: FP16 inference — ~2x throughput on H100 with negligible accuracy loss
        # augment=True: TTA (flip + multi-scale) on the full frame pass
        yolo_results = yolo_detector.track(
            frame,
            conf=0.20,
            imgsz=1280,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
            device=device,
            half=True,
            augment=True,
        )[0]

        raw_boxes = yolo_results.boxes.xyxy.cpu().numpy().tolist()
        raw_classes = yolo_results.boxes.cls.cpu().numpy().tolist()
        raw_confs = yolo_results.boxes.conf.cpu().numpy().tolist()
        names = yolo_detector.names

        if yolo_results.boxes.id is not None:
            raw_track_ids = yolo_results.boxes.id.int().cpu().numpy().tolist()
        else:
            raw_track_ids = [-1] * len(raw_boxes)

        # Post-filter by per-class confidence before SAM and association
        boxes, classes, track_ids = [], [], []
        for b, c, t, conf in zip(raw_boxes, raw_classes, raw_track_ids, raw_confs):
            class_name = names[int(c)].lower()
            if class_name not in VALID_CLASSES:
                continue
            threshold = PER_CLASS_CONF.get(class_name, 0.25)
            if conf >= threshold:
                boxes.append(b)
                classes.append(class_name)
                track_ids.append(t)
                detection_counts[class_name] += 1

        if len(boxes) > 0:
            persons = []
            ppes = []
            for box, class_name, track_id in zip(boxes, classes, track_ids):
                if class_name == "person":
                    persons.append({"box": box, "id": track_id})
                else:
                    ppes.append({"box": box, "class": class_name})

            # --- PASS 1: IoU-based PPE-to-person association (full-frame detections) ---
            person_gear_this_frame = defaultdict(set)
            for ppe in ppes:
                for person in persons:
                    if person["id"] == -1:
                        continue
                    expanded_person = expand_box(person["box"], factor=0.15)
                    if box_iou(ppe["box"], expanded_person) > 0:
                        person_gear_this_frame[person["id"]].add(ppe["class"])

            # --- PASS 2: Batched crop-and-classify for all tracked persons ---
            # All crops are sent to the GPU in a single batched call instead of N sequential calls.
            person_crop_pairs = [
                (person["id"], extract_person_crop(frame, person["box"], h, w))
                for person in persons if person["id"] != -1
            ]
            crop_results_map = batch_classify_ppe(person_crop_pairs, yolo_detector, device)

            for person_id, (crop_pos, crop_neg) in crop_results_map.items():
                for cls in crop_pos:
                    person_gear_this_frame[person_id].add(cls)
                for cls in crop_neg:
                    if cls.replace("no ", "") not in crop_pos:
                        person_gear_this_frame[person_id].add(cls)

            # One vote per (person, class) per frame: 1 if associated this frame, else 0
            for person in persons:
                if person["id"] == -1:
                    continue
                for class_name in VALID_CLASSES:
                    if class_name == "person":
                        continue
                    hit = 1 if class_name in person_gear_this_frame[person["id"]] else 0
                    worker_detection_history[person["id"]][class_name].append(hit)

            # SAM: run every SAM_INTERVAL frames or on first detections (if SAM is loaded)
            if sam_masker is not None:
                run_sam_this_frame = (frame_count % SAM_INTERVAL == 0) or (cached_annotated is None)
                sam_points = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
                sam_labels = [1] * len(boxes)
                if run_sam_this_frame:
                    sam_results = sam_masker(
                        frame,
                        bboxes=boxes,
                        points=sam_points,
                        point_labels=sam_labels,
                        verbose=False,
                        device=device,
                    )[0]
                    annotated_frame = sam_results.plot()
                    cached_annotated = annotated_frame.copy()
                else:
                    annotated_frame = frame.copy()
            else:
                annotated_frame = frame.copy()

            # Visual overlays on current frame (boxes/labels)
            def gear_from_history(worker_id):
                gear = set()
                for cls in VALID_CLASSES:
                    if cls == "person":
                        continue
                    hist = worker_detection_history[worker_id][cls]
                    if len(hist) > 0 and sum(hist) / len(hist) >= VOTE_THRESHOLD:
                        gear.add(cls)
                return gear

            PPE_ITEMS = ["helmet", "vest", "gloves"]

            for box, class_name, track_id in zip(boxes, classes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 255)
                label = class_name

                if class_name == "person":
                    if track_id != -1:
                        gear = gear_from_history(track_id)
                        # Build per-item status: True=has it, False=missing, None=uncertain
                        status_parts = []
                        has_violation = False
                        for item in PPE_ITEMS:
                            has_item = item in gear
                            missing_item = f"no {item}" in gear
                            if missing_item and not has_item:
                                status_parts.append(f"NO {item.upper()}")
                                has_violation = True
                            elif has_item:
                                status_parts.append(f"{item[0].upper()}{item[1:]}✓")
                            else:
                                status_parts.append(f"{item[0].upper()}{item[1:]}?")
                        status_str = " | ".join(status_parts)
                        if has_violation:
                            label = f"#{track_id}: {status_str}"
                            color = (0, 0, 255)
                        elif any("?" in p for p in status_parts):
                            label = f"#{track_id}: {status_str}"
                            color = (0, 165, 255)  # orange = uncertain
                        else:
                            label = f"#{track_id}: {status_str}"
                            color = (0, 200, 0)
                    else:
                        label = "Worker?"
                        color = (200, 200, 0)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 18), (x1 + text_w + 4, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Save sample frames for visual verification (at start, 25%, 50%, 75%)
            if SAVE_SAMPLE_FRAMES_DIR and total_frames > 0:
                target_frames = [1] + [max(1, int(total_frames * p / 100)) for p in (25, 50, 75)]
                if frame_count in target_frames:
                    pct = 0 if frame_count == 1 else int(100 * frame_count / total_frames)
                    path = os.path.join(SAVE_SAMPLE_FRAMES_DIR, f"frame_{frame_count:05d}_pct{pct}.jpg")
                    cv2.imwrite(path, annotated_frame)

        else:
            annotated_frame = frame.copy()

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed
            vram_used = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
            print(f"Frame {frame_count}/{total_frames} | {fps_current:.1f} FPS | VRAM {vram_used:.1f} GB")

        # Flush GPU allocator cache every 500 frames to prevent memory fragmentation
        if frame_count % 500 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        out.write(annotated_frame)

    # ---------------------------------------------------------
    # 5. Cleanup & Results
    # ---------------------------------------------------------
    cap.release()
    out.release()
    print(f"\nPipeline complete! Video saved to {OUTPUT_PATH}")

    # Verification: detection summary (healthy run has person + PPE classes)
    print("\n--- DETECTION SUMMARY (total counts per class) ---")
    for cls in VALID_CLASSES:
        print(f"  {cls}: {detection_counts[cls]}")
    total_detections = sum(detection_counts.values())
    print(f"  Total detections: {total_detections}")
    if total_detections == 0:
        print("  WARNING: No detections. Check video path, weights, and PER_CLASS_CONF thresholds.")
    elif detection_counts["person"] == 0:
        print("  WARNING: No person detections. Model or confidence may need adjustment.")

    print("\n--- FINAL WORKER COMPLIANCE RECORD (sliding-window vote) ---")
    for worker_id in sorted(worker_detection_history.keys()):
        gear = set()
        for cls in VALID_CLASSES:
            if cls == "person":
                continue
            hist = worker_detection_history[worker_id][cls]
            if len(hist) > 0 and sum(hist) / len(hist) >= VOTE_THRESHOLD:
                gear.add(cls)
        print(f"Worker #{worker_id}: Detected attributes -> {list(gear) if gear else ['None']}")

if __name__ == "__main__":
    main()