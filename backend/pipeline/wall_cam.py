import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Class mappings (from notebook)
# ---------------------------------------------------------------------------

CONSTRUCTION_HAZARD_CLASSES = {
    "person": "person",
    "machinery": "excavator",
    "vehicle": "dump_truck",
    "hardhat": "hardhat",
    "mask": "mask",
    "no-hardhat": "no_hardhat",
    "no-mask": "no_mask",
    "no-safety vest": "no_safety_vest",
    "safety cone": "safety_cone",
    "safety vest": "safety_vest",
}

CONSTRUCTION_CLASSES = {
    "person": "person",
    "excavator": "excavator",
    "crane": "crane",
    "ladder": "ladder",
    "scaffold": "scaffold",
    "dump_truck": "dump_truck",
    "forklift": "forklift",
    "truck": "dump_truck",
    "car": "dump_truck",
    "machinery": "excavator",
    "vehicle": "dump_truck",
}

COCO_TO_HAZARD = {
    "person": "person",
    "truck": "dump_truck",
    "car": "dump_truck",
    "bus": "dump_truck",
}

HAZARD_RULES = {
    "suspended_load_hazard": {
        "description": "Crane/lifting equipment with person nearby",
        "requires": [["crane"], "person"],
        "severity": "HIGH",
        "violation_type": "PROXIMITY_HAZARD",
    },
    "heavy_equipment_proximity": {
        "description": "Person near heavy machinery",
        "requires": [["excavator", "dump_truck", "forklift"], "person"],
        "severity": "HIGH",
        "violation_type": "PROXIMITY_HAZARD",
    },
    "person_on_ladder": {
        "description": "Person on ladder — potential fall hazard",
        "requires": [["ladder"], "person"],
        "severity": "MEDIUM",
        "violation_type": "LADDER_MISUSE",
    },
    "person_on_scaffold": {
        "description": "Person on scaffold — height safety concern",
        "requires": [["scaffold"], "person"],
        "severity": "MEDIUM",
        "violation_type": "SCAFFOLD_VIOLATION",
    },
    "ppe_violation_no_hardhat": {
        "description": "Person without hardhat",
        "requires": [["no_hardhat"], "person"],
        "severity": "HIGH",
        "violation_type": "PPE_MISSING",
    },
    "ppe_violation_no_safety_vest": {
        "description": "Person without safety vest",
        "requires": ["no_safety_vest"],
        "severity": "HIGH",
        "violation_type": "PPE_MISSING",
    },
    "ppe_violation_no_mask": {
        "description": "Person without mask",
        "requires": [["no_mask"], "person"],
        "severity": "MEDIUM",
        "violation_type": "PPE_MISSING",
    },
}

MODEL_PATH = Path(__file__).parent.parent / "models" / "yolo11s_construction.pt"

# Singleton — loaded once per process
_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _map_class(raw: str) -> Optional[str]:
    c = raw.lower()
    if c in CONSTRUCTION_HAZARD_CLASSES:
        return CONSTRUCTION_HAZARD_CLASSES[c]
    if c in ("hardhat", "helmet"):
        return "hardhat"
    if c in ("no-hardhat", "no hardhat", "no_hardhat"):
        return "no_hardhat"
    if c in ("mask", "face mask"):
        return "mask"
    if c in ("no-mask", "no mask", "no_mask"):
        return "no_mask"
    if c in ("safety vest", "vest", "safety_vest"):
        return "safety_vest"
    if c in ("no-safety vest", "no safety vest", "no_safety_vest", "no-safety-vest"):
        return "no_safety_vest"
    if c in ("safety cone", "cone", "safety_cone"):
        return "safety_cone"
    if c in ("person", "people", "worker"):
        return "person"
    if c in ("machinery", "machine", "equipment", "excavator", "digger"):
        return "excavator"
    if c in ("vehicle", "truck", "car", "dump truck", "dump_truck"):
        return "dump_truck"
    if c in CONSTRUCTION_CLASSES:
        return CONSTRUCTION_CLASSES[c]
    if c in COCO_TO_HAZARD:
        return COCO_TO_HAZARD[c]
    if "crane" in c or "hook" in c:
        return "crane"
    if "ladder" in c:
        return "ladder"
    if "scaffold" in c:
        return "scaffold"
    if "forklift" in c or "fork" in c:
        return "forklift"
    return None


def _proximity(b1: list, b2: list) -> float:
    cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
    cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def _detect_hazards(detections: List[Dict], proximity_threshold: float = 0.3) -> List[Dict]:
    hazards = []
    by_class: Dict[str, list] = {}
    for det in detections:
        by_class.setdefault(det["class_name"], []).append(det)

    for hazard_name, rule in HAZARD_RULES.items():
        required = rule["requires"]

        if len(required) == 2:
            req1, req2 = required
            req1_dets = []
            if isinstance(req1, list):
                for c in req1:
                    req1_dets.extend(by_class.get(c, []))
            else:
                req1_dets = by_class.get(req1, [])
            req2_dets = by_class.get(req2, []) if isinstance(req2, str) else []

            for d1 in req1_dets:
                for d2 in req2_dets:
                    dist = _proximity(d1["bbox"], d2["bbox"])
                    if dist < proximity_threshold:
                        hazards.append({
                            "hazard_type": hazard_name,
                            "description": rule["description"],
                            "severity": rule["severity"],
                            "violation_type": rule["violation_type"],
                            "objects": [
                                {"class": d1["class_name"], "confidence": d1["confidence"]},
                                {"class": d2["class_name"], "confidence": d2["confidence"]},
                            ],
                            "proximity": round(dist, 4),
                        })

        elif len(required) == 1:
            req = required[0]
            classes = req if isinstance(req, list) else [req]
            for c in classes:
                for det in by_class.get(c, []):
                    hazards.append({
                        "hazard_type": hazard_name,
                        "description": rule["description"],
                        "severity": rule["severity"],
                        "violation_type": rule["violation_type"],
                        "objects": [{"class": det["class_name"], "confidence": det["confidence"]}],
                    })
                if by_class.get(c):
                    break

    return hazards


def _save_person_crops(
    frame: np.ndarray,
    detections: List[Dict],
    frame_idx: int,
    crops_dir: Path,
    pad: float = 0.25,
) -> List[Dict]:
    h, w = frame.shape[:2]
    crops = []
    for i, det in enumerate(d for d in detections if d["class_name"] == "person"):
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        px1 = max(0, int(x1 - pad * bw))
        py1 = max(0, int(y1 - pad * bh))
        px2 = min(w, int(x2 + pad * bw))
        py2 = min(h, int(y2 + pad * bh))
        crop = frame[py1:py2, px1:px2]
        crop_name = f"frame_{frame_idx:06d}_person_{i}.jpg"
        cv2.imwrite(str(crops_dir / crop_name), crop)
        crops.append({
            "path": crop_name,
            "person_index": i,
            "bbox_original": det["bbox"],
            "bbox_padded": [px1, py1, px2, py2],
            "confidence": det["confidence"],
        })
    return crops


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_wall_cam_pipeline(
    video_path: Path,
    job_dir: Path,
    site_id: str,
    date: str,
    sample_every_n: int = 5,
    conf_threshold: float = 0.30,
    proximity_threshold: float = 0.3,
) -> dict:
    """
    Run YOLO detection on wall-cam video.

    Writes to job_dir:
      detections.jsonl     — one JSON record per sampled frame
      hazard_events.json   — only frames where hazards were detected (+ crops metadata)
      summary.json         — aggregate counts and class list

    Flagged frames saved as JPEG in job_dir/frames/.
    Person crops (padded) saved in job_dir/crops/ for VLM step.

    Returns the summary dict.
    """
    frames_dir = job_dir / "frames"
    crops_dir = job_dir / "crops"
    frames_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    model = _get_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hazard_events: List[dict] = []
    hazard_counts: Dict[str, int] = {}
    all_classes: set = set()
    frames_processed = 0
    frame_idx = 0

    with open(job_dir / "detections.jsonl", "w") as log:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n != 0:
                frame_idx += 1
                continue

            timestamp = round(frame_idx / fps, 3)
            results = model(frame, conf=conf_threshold, verbose=False)
            result = results[0]

            detections: List[Dict] = []
            if result.boxes is not None:
                for i in range(len(result.boxes)):
                    bbox = result.boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(result.boxes.conf[i].cpu().numpy())
                    class_id = int(result.boxes.cls[i].cpu().numpy())
                    mapped = _map_class(model.names[class_id])
                    if mapped:
                        detections.append({
                            "class_name": mapped,
                            "original_class": model.names[class_id],
                            "bbox": bbox,
                            "confidence": round(conf, 4),
                        })
                        all_classes.add(mapped)

            hazards = _detect_hazards(detections, proximity_threshold)

            frame_record: Dict = {
                "frame_idx": frame_idx,
                "timestamp_seconds": timestamp,
                "detections": detections,
                "hazards": hazards,
            }

            if hazards:
                # Save full frame
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frames_dir / frame_filename), frame)

                # Save padded person crops for VLM
                crops = _save_person_crops(frame, detections, frame_idx, crops_dir)

                frame_record["frame_path"] = frame_filename
                frame_record["person_crops"] = crops
                hazard_events.append(frame_record)

                for h in hazards:
                    hazard_counts[h["hazard_type"]] = hazard_counts.get(h["hazard_type"], 0) + 1

            log.write(json.dumps(frame_record) + "\n")
            frames_processed += 1
            frame_idx += 1

    cap.release()

    summary = {
        "site_id": site_id,
        "date": date,
        "video_fps": fps,
        "total_frames": total_frames,
        "frames_processed": frames_processed,
        "frames_with_hazards": len(hazard_events),
        "hazard_counts": hazard_counts,
        "detected_classes": sorted(all_classes),
    }

    (job_dir / "hazard_events.json").write_text(json.dumps(hazard_events, indent=2))
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary
