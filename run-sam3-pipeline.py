"""
SAM3 PPE Detection Pipeline
============================
Strategy:
  1. YOLO (person class only) → worker bounding boxes + persistent tracking IDs
  2. Per worker: decompose the person box into 4 anatomical regions
       head    → top ~20% of person box  → detect helmet
       torso   → mid ~40% of person box  → detect high-vis vest
       l_hand  → lower-left region       → detect gloves vs bare hands
       r_hand  → lower-right region      → detect gloves vs bare hands
  3. SAM3 receives all region boxes + foreground/background points in ONE batched call
  4. Color analysis on each returned mask:
       helmet  → hard-hat HSV ranges (yellow / orange / white / blue / red)
       vest    → fluorescent high-vis HSV ranges (yellow / orange)
       hands   → skin-tone fraction → "bare hands" vs "gloves"
  5. Temporal sliding-window vote (30 frames, 35% threshold)
  6. Annotate output video with per-worker PPE status

Upload:
  scp -P 16517 run-sam3-pipeline.py root@154.57.34.95:/workspace/
Run:
  python /workspace/run-sam3-pipeline.py
"""

import os
import cv2
import torch
import numpy as np
import time
import sys
from collections import deque, defaultdict
from ultralytics import YOLO, SAM

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
WORKSPACE = "/workspace"
YOLO_WEIGHTS_PATH = f"{WORKSPACE}/weights/best.pt"   # used ONLY for person detection + tracking
# SAM model: auto-downloaded if not present.
# "sam2_b.pt" (SAM2 base, ~80 MB) or "sam2_l.pt" (large, better accuracy)
SAM_MODEL_NAME = "sam2_b.pt"
VIDEO_PATH = f"{WORKSPACE}/Benchmarking Vids/Dataset video.mp4"
OUTPUT_PATH = f"{WORKSPACE}/sam3_ppe_output.mp4"
SAVE_SAMPLE_FRAMES_DIR = None  # e.g. f"{WORKSPACE}/sam3_frames"

WINDOW_SIZE = 30
VOTE_THRESHOLD = 0.35

# ---------------------------------------------------------
# BODY REGION PROPORTIONS (relative to person box)
# ---------------------------------------------------------
# Head: top 22% height, 10% horizontal padding each side
HEAD_TOP_FRAC   = 0.00;  HEAD_BOT_FRAC  = 0.22
HEAD_HPAD_FRAC  = 0.10

# Torso: 18% to 60% height, full width
TORSO_TOP_FRAC  = 0.18;  TORSO_BOT_FRAC = 0.60

# Hands: 60% to 105% height (allows slightly below foot line for low arms)
LHAND_X1_FRAC = -0.12;  LHAND_X2_FRAC = 0.38
RHAND_X1_FRAC =  0.62;  RHAND_X2_FRAC = 1.12
HAND_TOP_FRAC  = 0.60;  HAND_BOT_FRAC = 1.05

# ---------------------------------------------------------
# COLOR ANALYSIS THRESHOLDS (all HSV, uint8 scale 0-179/255/255)
# ---------------------------------------------------------
# Hard hat colors: yellow, orange, white, blue, red
HARDHAT_RANGES = [
    ((20, 120, 100), (35, 255, 255)),   # safety yellow
    ((5,  150, 100), (20, 255, 255)),   # safety orange
    ((0,   0,  200), (180, 50, 255)),   # white
    ((100, 80,  80), (130, 255, 255)),  # blue
    ((0,  120,  80), (10,  255, 255)),  # red (low hue)
    ((170, 120, 80), (180, 255, 255)),  # red (high hue)
]
# High-vis safety vest: fluorescent yellow/orange
HIVIS_RANGES = [
    ((18, 150, 150), (38, 255, 255)),   # fluorescent yellow
    ((4,  150, 150), (18, 255, 255)),   # fluorescent orange
]
# Skin tone (HSV) — covers many skin tones; hand is "bare" if >=30% of pixels are skin
SKIN_LOWER = np.array([0,   20, 60],  dtype=np.uint8)
SKIN_UPPER = np.array([25, 170, 235], dtype=np.uint8)
BARE_HAND_SKIN_THRESH = 0.30   # fraction of mask pixels that must be skin-tone
MIN_MASK_COVERAGE     = 0.05   # mask must cover at least 5% of its region to be valid

# Mask overlay colors per body region (BGR)
REGION_MASK_COLORS = {
    "head":   (255, 180,   0),   # blue
    "torso":  (0,   220,  60),   # green
    "l_hand": (0,   140, 255),   # orange
    "r_hand": (0,   100, 200),   # dark orange
}


# ---------------------------------------------------------
# HELPERS: geometry
# ---------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get_body_regions(person_box, frame_h, frame_w):
    """
    Decompose a person bounding box into 4 anatomical sub-regions.
    Returns dict:  { region_name: (sub_box, fg_point, bg_points) }
      sub_box   = [x1, y1, x2, y2] (clamped to frame)
      fg_point  = [cx, cy] foreground prompt (label=1) — "segment this"
      bg_points = list of [cx, cy] background prompts (label=0) — "not these"
    """
    x1, y1, x2, y2 = person_box
    pw = x2 - x1
    ph = y2 - y1

    def clip(bx1, by1, bx2, by2):
        return [
            clamp(int(bx1), 0, frame_w - 1), clamp(int(by1), 0, frame_h - 1),
            clamp(int(bx2), 1, frame_w),     clamp(int(by2), 1, frame_h),
        ]

    cx = (x1 + x2) / 2
    torso_cy = y1 + ph * ((TORSO_TOP_FRAC + TORSO_BOT_FRAC) / 2)

    head_box  = clip(x1 - pw*HEAD_HPAD_FRAC, y1 + ph*HEAD_TOP_FRAC,
                     x2 + pw*HEAD_HPAD_FRAC, y1 + ph*HEAD_BOT_FRAC)
    head_fg   = [cx, y1 + ph * 0.10]

    torso_box = clip(x1, y1 + ph*TORSO_TOP_FRAC, x2, y1 + ph*TORSO_BOT_FRAC)
    torso_fg  = [cx, torso_cy]

    lhand_box = clip(x1 + pw*LHAND_X1_FRAC, y1 + ph*HAND_TOP_FRAC,
                     x1 + pw*LHAND_X2_FRAC, y1 + ph*HAND_BOT_FRAC)
    lhand_fg  = [x1 + pw * 0.12, y1 + ph * 0.82]

    rhand_box = clip(x1 + pw*RHAND_X1_FRAC, y1 + ph*HAND_TOP_FRAC,
                     x1 + pw*RHAND_X2_FRAC, y1 + ph*HAND_BOT_FRAC)
    rhand_fg  = [x1 + pw * 0.88, y1 + ph * 0.82]

    return {
        "head":    (head_box,  head_fg,  [[cx, torso_cy]]),            # bg = torso
        "torso":   (torso_box, torso_fg, [head_fg]),                   # bg = head
        "l_hand":  (lhand_box, lhand_fg, [[cx, torso_cy]]),            # bg = torso
        "r_hand":  (rhand_box, rhand_fg, [[cx, torso_cy]]),            # bg = torso
    }


# ---------------------------------------------------------
# HELPERS: color analysis on segmentation masks
# ---------------------------------------------------------
def mask_color_check(frame_bgr, mask_bool, color_ranges, min_fraction=0.12):
    """
    Return True if any of the HSV color ranges accounts for >= min_fraction of mask pixels.
    """
    if not np.any(mask_bool):
        return False
    pixels = frame_bgr[mask_bool]
    if len(pixels) < 30:
        return False
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    n = len(hsv)
    for lo, hi in color_ranges:
        lo_a = np.array(lo, dtype=np.uint8)
        hi_a = np.array(hi, dtype=np.uint8)
        in_range = np.all((hsv >= lo_a) & (hsv <= hi_a), axis=1)
        if in_range.sum() / n >= min_fraction:
            return True
    return False


def classify_hand(frame_bgr, mask_bool):
    """
    Returns 'gloves', 'bare', or 'uncertain' for a hand-region mask.
    Logic:
      - If >=30% of mask pixels are skin-tone → bare hands
      - Otherwise → gloves (or some non-skin covering)
    """
    if not np.any(mask_bool):
        return "uncertain"
    pixels = frame_bgr[mask_bool]
    if len(pixels) < 40:
        return "uncertain"
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    skin = np.all((hsv >= SKIN_LOWER) & (hsv <= SKIN_UPPER), axis=1)
    if skin.sum() / len(hsv) >= BARE_HAND_SKIN_THRESH:
        return "bare"
    return "gloves"


# ---------------------------------------------------------
# HELPERS: sliding-window vote
# ---------------------------------------------------------
def voted_ppe(history, worker_id):
    """
    Returns dict { 'helmet': bool, 'vest': bool, 'gloves': bool, 'bare_hands': bool }
    based on the 30-frame sliding window.
    """
    result = {}
    for key in ("helmet", "vest", "gloves", "bare_hands"):
        hist = history[worker_id][key]
        result[key] = len(hist) > 0 and (sum(hist) / len(hist)) >= VOTE_THRESHOLD
    return result


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    # GPU setup
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — this will be slow on CPU.")
        device = "cpu"
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda:0"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram_gb:.1f} GB")

    # Load YOLO (person detection + tracking only)
    print(f"\nLoading YOLO person detector from {YOLO_WEIGHTS_PATH}...")
    try:
        yolo_detector = YOLO(YOLO_WEIGHTS_PATH)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Resolve person class index dynamically
    person_class_idx = None
    for idx, name in yolo_detector.names.items():
        if name.lower() == "person":
            person_class_idx = idx
            break
    if person_class_idx is None:
        print("ERROR: 'person' class not found in model. Check weights.")
        sys.exit(1)
    print(f"  'person' is class index {person_class_idx}")

    # Load SAM3 / SAM2 (auto-downloads if not cached)
    print(f"\nLoading SAM model ({SAM_MODEL_NAME})...")
    try:
        sam_masker = SAM(SAM_MODEL_NAME)
    except Exception as e:
        print(f"Error loading SAM: {e}")
        sys.exit(1)

    # Video I/O
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open {VIDEO_PATH}")
        sys.exit(1)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    vid_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))
    print(f"\nProcessing {VIDEO_PATH}  ({total_frames} frames @ {fps:.1f} fps)")
    print(f"Output: {OUTPUT_PATH}")

    if SAVE_SAMPLE_FRAMES_DIR:
        os.makedirs(SAVE_SAMPLE_FRAMES_DIR, exist_ok=True)

    # State
    history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=WINDOW_SIZE)))
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # --- STEP 1: YOLO — detect & track persons only ---
        yolo_results = yolo_detector.track(
            frame,
            classes=[person_class_idx],
            conf=0.30,
            imgsz=1280,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
            device=device,
            half=True,
        )[0]

        persons = []
        if yolo_results.boxes is not None:
            boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy().tolist()
            if yolo_results.boxes.id is not None:
                track_ids = yolo_results.boxes.id.int().cpu().numpy().tolist()
            else:
                track_ids = [-1] * len(boxes_xyxy)
            for box, tid in zip(boxes_xyxy, track_ids):
                persons.append({"box": box, "id": tid})

        # --- STEP 2: build SAM3 prompts for all body regions of all persons ---
        all_bboxes        = []
        all_points        = []
        all_point_labels  = []
        prompt_meta       = []   # (person_id, region_name)

        for person in persons:
            if person["id"] == -1:
                continue
            regions = get_body_regions(person["box"], vid_h, vid_w)
            for region_name, (sub_box, fg_pt, bg_pts) in regions.items():
                # Skip degenerate boxes
                if sub_box[2] <= sub_box[0] or sub_box[3] <= sub_box[1]:
                    continue
                pts    = [fg_pt]  + bg_pts
                labels = [1]      + [0] * len(bg_pts)
                all_bboxes.append(sub_box)
                all_points.append(pts)
                all_point_labels.append(labels)
                prompt_meta.append((person["id"], region_name))

        # --- STEP 3: single batched SAM3 call for all prompts ---
        person_region_results = {}   # (person_id, region_name) -> (bool | str)
        annotated_frame = frame.copy()

        if all_bboxes:
            try:
                sam_results = sam_masker(
                    frame,
                    bboxes=all_bboxes,
                    points=all_points,
                    point_labels=all_point_labels,
                    verbose=False,
                    device=device,
                )

                for idx, (result, (person_id, region_name)) in enumerate(zip(sam_results, prompt_meta)):
                    if result.masks is None or len(result.masks.data) == 0:
                        continue
                    # SAM returns mask in original frame coords
                    mask_bool = result.masks.data[0].cpu().numpy().astype(bool)

                    # Guard: mask must be non-trivially large
                    region_box = all_bboxes[idx]
                    region_area = max(1, (region_box[2]-region_box[0]) * (region_box[3]-region_box[1]))
                    if mask_bool.sum() < region_area * MIN_MASK_COVERAGE:
                        continue

                    if region_name == "head":
                        person_region_results[(person_id, "helmet")] = mask_color_check(
                            frame, mask_bool, HARDHAT_RANGES, min_fraction=0.12
                        )
                    elif region_name == "torso":
                        person_region_results[(person_id, "vest")] = mask_color_check(
                            frame, mask_bool, HIVIS_RANGES, min_fraction=0.12
                        )
                    elif region_name in ("l_hand", "r_hand"):
                        hand_status = classify_hand(frame, mask_bool)
                        key_gloves     = (person_id, "gloves")
                        key_bare       = (person_id, "bare_hands")
                        prev_gloves    = person_region_results.get(key_gloves, False)
                        prev_bare      = person_region_results.get(key_bare,   False)
                        # Either hand wearing gloves is enough to call it gloved
                        person_region_results[key_gloves] = prev_gloves or (hand_status == "gloves")
                        person_region_results[key_bare]   = (
                            not person_region_results[key_gloves]
                            and (prev_bare or hand_status == "bare")
                        )

                # Draw SAM masks on frame (color = region type)
                for result, (_, region_name) in zip(sam_results, prompt_meta):
                    if result.masks is None:
                        continue
                    mask_bool = result.masks.data[0].cpu().numpy().astype(bool)
                    color = REGION_MASK_COLORS.get(region_name, (180, 180, 180))
                    # 55% tint blend: keep original texture visible under colour overlay
                    for ch, c in enumerate(color):
                        annotated_frame[:, :, ch][mask_bool] = (
                            annotated_frame[:, :, ch][mask_bool] * 0.45 + c * 0.55
                        ).astype(np.uint8)

            except Exception as e:
                print(f"  SAM error on frame {frame_count}: {e}")

        # --- STEP 4: update temporal voting ---
        for person in persons:
            pid = person["id"]
            if pid == -1:
                continue
            history[pid]["helmet"].append(
                1 if person_region_results.get((pid, "helmet"), False) else 0
            )
            history[pid]["vest"].append(
                1 if person_region_results.get((pid, "vest"), False) else 0
            )
            history[pid]["gloves"].append(
                1 if person_region_results.get((pid, "gloves"), False) else 0
            )
            history[pid]["bare_hands"].append(
                1 if person_region_results.get((pid, "bare_hands"), False) else 0
            )

        # --- STEP 5: draw overlays ---
        for person in persons:
            pid = person["id"]
            x1, y1, x2, y2 = map(int, person["box"])

            if pid == -1:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (180, 180, 0), 2)
                cv2.putText(annotated_frame, "Worker?", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1)
                continue

            ppe = voted_ppe(history, pid)

            # Per-item status strings
            helm_str  = "Helmet✓"    if ppe["helmet"]     else ("NO HELMET" if history[pid]["helmet"] else "Helmet?")
            vest_str  = "Vest✓"      if ppe["vest"]       else ("NO VEST"   if history[pid]["vest"]   else "Vest?")
            if ppe["gloves"]:
                hand_str = "Gloves✓"
            elif ppe["bare_hands"]:
                hand_str = "BARE HANDS"
            else:
                hand_str = "Hands?"

            has_violation = "NO " in helm_str or "NO " in vest_str or "BARE" in hand_str
            color = (0, 0, 220) if has_violation else (
                (0, 165, 255) if "?" in helm_str + vest_str + hand_str else (0, 200, 0)
            )

            label = f"#{pid}: {helm_str} | {vest_str} | {hand_str}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - 18), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

                # Draw thin body-region prompt boxes for debuggability
            regions = get_body_regions(person["box"], vid_h, vid_w)
            for rbox, _, _ in regions.values():
                cv2.rectangle(annotated_frame,
                              (rbox[0], rbox[1]), (rbox[2], rbox[3]),
                              (80, 80, 80), 1)

        # Sample frame saving
        if SAVE_SAMPLE_FRAMES_DIR and total_frames > 0:
            targets = {1} | {max(1, int(total_frames * p / 100)) for p in (10, 25, 50, 75, 90)}
            if frame_count in targets:
                pct = int(100 * frame_count / total_frames)
                cv2.imwrite(
                    os.path.join(SAVE_SAMPLE_FRAMES_DIR, f"frame_{frame_count:05d}_pct{pct}.jpg"),
                    annotated_frame
                )

        out.write(annotated_frame)

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_now = frame_count / elapsed
            vram = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
            print(f"Frame {frame_count}/{total_frames} | {fps_now:.1f} FPS | VRAM {vram:.1f} GB")

        if frame_count % 500 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    cap.release()
    out.release()
    print(f"\nDone. Output saved to {OUTPUT_PATH}")

    # Final compliance report
    print("\n--- FINAL WORKER PPE REPORT (SAM3 + color analysis) ---")
    for worker_id in sorted(history.keys()):
        ppe = voted_ppe(history, worker_id)
        helm  = "Helmet: OK"       if ppe["helmet"]     else "Helmet: MISSING"
        vest  = "Vest: OK"         if ppe["vest"]       else "Vest: MISSING"
        if ppe["gloves"]:
            hand = "Hands: Gloved"
        elif ppe["bare_hands"]:
            hand = "Hands: BARE (violation)"
        else:
            hand = "Hands: uncertain"
        print(f"Worker #{worker_id}: {helm} | {vest} | {hand}")


if __name__ == "__main__":
    main()
