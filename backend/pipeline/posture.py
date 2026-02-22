"""
posture.py — posture risk analysis using YOLO pose + inlined ergonomic heuristics

This runs in addition to the PPE/YOLO+SAM pipeline. It samples video frames, runs a pose
model, computes joint angles, derives a simple REBA-inspired risk score, and emits posture
events plus per-worker summaries.
"""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pipeline.vlm_step import run_vlm_on_video

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POSE_MODEL_PATH = Path(os.getenv("POSE_MODEL_PATH", "/workspace/models/yolo26l-pose.pt"))
DEFAULT_FPS = 25.0
MIN_CONF = 0.35
MIN_RISK_CONFIDENCE = 0.55
MIN_OVERREACH_CONFIDENCE = 0.55
TRUNK_GOOD_MAX_DEG = 28
TRUNK_DEG_KNEELING = 20
KNEE_ANGLE_SQUAT_DEG = 105
TRUNK_DEG_AWKWARD = 38
TRUNK_DEG_MSD_HIGH = 58
ARM_RAISE_OVERREACH_DEG = 52
ARM_RAISE_MAX_OVERREACH_DEG = 102
COMBINED_BORDERLINE = 2
GOOD_REQUIRES_CONFIDENCE = 0.50

# COCO 17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

_pose_model: YOLO | None = None


def _get_pose_model() -> YOLO:
    global _pose_model
    if _pose_model is None:
        device = os.getenv("POSTURE_CUDA_DEVICE", None)
        device_arg = device if device else None
        print(f"[posture] loading pose model from {POSE_MODEL_PATH} (device={device_arg or 'auto'})")
        _pose_model = YOLO(str(POSE_MODEL_PATH), task="pose", device=device_arg)
    return _pose_model


# ---------------------------------------------------------------------------
# Ergonomic helpers (from notebook)
# ---------------------------------------------------------------------------

def angle_deg(v1, v2):
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def _reba_group_a(angles):
    trunk = angles.get("trunk_flexion_deg")
    neck = angles.get("neck_flexion_deg")
    knee = angles.get("knee_angle_deg")
    lateral = angles.get("trunk_lateral_deg")
    score = 1
    if trunk is not None:
        if trunk > 55:
            score = 4
        elif trunk > 18:
            score = 3
        elif trunk > 5:
            score = 2
    if neck is not None and (neck > 22 or neck < 8):
        score = min(score + 1, 5)
    if lateral is not None and lateral > 12:
        score = min(score + 1, 6)
    if knee is not None and knee < KNEE_ANGLE_SQUAT_DEG:
        score = min(score + 1, 7)
    return score


def _reba_group_b(angles):
    arm_raise = angles.get("arm_raise_deg")
    elbow = angles.get("elbow_flexion_deg")
    score = 1
    if arm_raise is not None:
        if arm_raise > ARM_RAISE_MAX_OVERREACH_DEG:
            score = 1
        elif arm_raise >= 90:
            score = 4
        elif arm_raise >= 45:
            score = 3
        elif arm_raise >= 20:
            score = 2
    if elbow is not None:
        if elbow < 60 or elbow > 100:
            score = min(score + 1, 6)
        if elbow < 30 or elbow > 120:
            score = min(score + 1, 7)
    return score


def _reba_combined(score_a, score_b):
    base = max(score_a, score_b)
    if score_a >= 3 and score_b >= 3:
        base = min(base + 2, 15)
    elif score_a >= 2 or score_b >= 2:
        base = min(base + 1, 15)
    return base


def _arm_is_elevated(arm_deg):
    if arm_deg is None:
        return False
    return ARM_RAISE_OVERREACH_DEG <= arm_deg <= ARM_RAISE_MAX_OVERREACH_DEG


def _would_allow_good(trunk, knee, arm, combined):
    if combined >= COMBINED_BORDERLINE:
        return False
    if trunk is not None and trunk > TRUNK_GOOD_MAX_DEG:
        return False
    if knee is not None and trunk is not None:
        if knee < KNEE_ANGLE_SQUAT_DEG and trunk > TRUNK_DEG_KNEELING:
            return False
    if _arm_is_elevated(arm):
        return False
    return True


def angles_from_keypoints(kp):
    c = kp[:, 2]
    out = {}
    used_conf = []
    if c[L_SHOULDER] <= MIN_CONF or c[R_SHOULDER] <= MIN_CONF:
        return out
    mid_shoulder = (kp[L_SHOULDER, :2] + kp[R_SHOULDER, :2]) / 2
    used_conf.extend([c[L_SHOULDER], c[R_SHOULDER]])
    if c[L_HIP] <= MIN_CONF or c[R_HIP] <= MIN_CONF:
        return out
    mid_hip = (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2
    used_conf.extend([c[L_HIP], c[R_HIP]])
    vertical = np.array([0.0, -1.0])
    horizontal = np.array([1.0, 0.0])
    trunk_vec = mid_shoulder - mid_hip
    out["trunk_flexion_deg"] = angle_deg(trunk_vec, vertical)
    shoulder_line = kp[R_SHOULDER, :2] - kp[L_SHOULDER, :2]
    out["trunk_lateral_deg"] = abs(angle_deg(shoulder_line, horizontal))
    head_pt = None
    if c[NOSE] > MIN_CONF:
        head_pt = kp[NOSE, :2]
        used_conf.append(c[NOSE])
    elif c[L_EAR] > MIN_CONF and c[R_EAR] > MIN_CONF:
        head_pt = (kp[L_EAR, :2] + kp[R_EAR, :2]) / 2
        used_conf.extend([c[L_EAR], c[R_EAR]])
    if head_pt is not None:
        out["neck_flexion_deg"] = angle_deg(mid_shoulder - head_pt, trunk_vec)
    if (
        c[L_HIP] > MIN_CONF
        and c[L_KNEE] > MIN_CONF
        and c[L_ANKLE] > MIN_CONF
    ):
        a, b = kp[L_HIP, :2] - kp[L_KNEE, :2], kp[L_ANKLE, :2] - kp[L_KNEE, :2]
        out["knee_angle_deg"] = angle_deg(a, b)
        used_conf.extend([c[L_KNEE], c[L_ANKLE]])
    elif (
        c[R_HIP] > MIN_CONF
        and c[R_KNEE] > MIN_CONF
        and c[R_ANKLE] > MIN_CONF
    ):
        a, b = kp[R_HIP, :2] - kp[R_KNEE, :2], kp[R_ANKLE, :2] - kp[R_KNEE, :2]
        out["knee_angle_deg"] = angle_deg(a, b)
        used_conf.extend([c[R_KNEE], c[R_ANKLE]])
    arm_raise_list = []
    if c[R_SHOULDER] > MIN_CONF and c[R_ELBOW] > MIN_CONF:
        arm_raise_list.append(
            angle_deg(kp[R_ELBOW, :2] - kp[R_SHOULDER, :2], horizontal)
        )
    if c[L_SHOULDER] > MIN_CONF and c[L_ELBOW] > MIN_CONF:
        arm_raise_list.append(
            angle_deg(kp[L_ELBOW, :2] - kp[L_SHOULDER, :2], horizontal)
        )
    if arm_raise_list:
        out["arm_raise_deg"] = max(arm_raise_list)
    elbow_list = []
    if (
        c[R_SHOULDER] > MIN_CONF
        and c[R_ELBOW] > MIN_CONF
        and c[R_WRIST] > MIN_CONF
    ):
        u = kp[R_ELBOW, :2] - kp[R_SHOULDER, :2]
        v = kp[R_WRIST, :2] - kp[R_ELBOW, :2]
        elbow_list.append(angle_deg(u, v))
        used_conf.extend([c[R_ELBOW], c[R_WRIST]])
    if (
        c[L_SHOULDER] > MIN_CONF
        and c[L_ELBOW] > MIN_CONF
        and c[L_WRIST] > MIN_CONF
    ):
        u = kp[L_ELBOW, :2] - kp[L_SHOULDER, :2]
        v = kp[L_WRIST, :2] - kp[L_ELBOW, :2]
        elbow_list.append(angle_deg(u, v))
        used_conf.extend([c[L_ELBOW], c[L_WRIST]])
    if elbow_list:
        out["elbow_flexion_deg"] = min(elbow_list)
    if used_conf:
        out["keypoint_confidence"] = float(np.mean(used_conf))
    return out


def risk_from_angles(angles):
    trunk = angles.get("trunk_flexion_deg")
    arm = angles.get("arm_raise_deg")
    knee = angles.get("knee_angle_deg")
    confidence = angles.get("keypoint_confidence", 0.6)
    violation_type = "COMPLIANT_BEHAVIOR"
    risk_level = 1
    good_posture = True
    reason_short = "ok"

    if trunk is None and arm is None and knee is None:
        return {
            "good_posture": True,
            "risk_level": 1,
            "violation_type": "COMPLIANT_BEHAVIOR",
            "confidence": 0.0,
            "angles": angles,
            "reason_short": "no_angles",
            "weak_good": True,
        }

    score_a = _reba_group_a(angles)
    score_b = _reba_group_b(angles)
    combined = _reba_combined(score_a, score_b)

    if combined >= 8:
        risk_level, good_posture, reason_short = 4, False, f"reba{combined}"
    elif combined >= 5:
        risk_level, good_posture, reason_short = 3, False, f"reba{combined}"
    elif combined >= COMBINED_BORDERLINE:
        risk_level, good_posture, reason_short = 2, False, f"reba{combined}"

    if _arm_is_elevated(arm) and confidence >= MIN_OVERREACH_CONFIDENCE:
        risk_level, good_posture, violation_type, reason_short = (
            max(risk_level, 3),
            False,
            "OVERREACH",
            f"arm{arm:.0f}",
        )

    if good_posture and not _would_allow_good(trunk, knee, arm, combined):
        good_posture, reason_short = False, "good_gate_fail"
        if risk_level < 2:
            risk_level = 2

    if not good_posture:
        arms_not_elevated = not _arm_is_elevated(arm)
        if (
            knee is not None
            and knee < KNEE_ANGLE_SQUAT_DEG
            and trunk is not None
            and trunk > TRUNK_DEG_KNEELING
            and arms_not_elevated
        ):
            violation_type, reason_short = (
                "KNEELING_SQUATTING_LOW",
                f"knee{knee:.0f}_trunk{trunk:.0f}",
            )
        elif _arm_is_elevated(arm) and confidence >= MIN_OVERREACH_CONFIDENCE:
            violation_type = "OVERREACH"
            reason_short = (
                reason_short if "arm" in reason_short else f"arm{arm:.0f}"
            )
        elif (
            trunk is not None
            and trunk >= TRUNK_DEG_MSD_HIGH
            and risk_level == 4
        ):
            violation_type, reason_short = "MSD_HIGH_RISK", f"trunk{trunk:.0f}"
        elif trunk is not None and trunk >= TRUNK_DEG_AWKWARD:
            violation_type, reason_short = "AWKWARD_POSTURE", f"trunk{trunk:.0f}"
        else:
            violation_type, reason_short = (
                "AWKWARD_POSTURE",
                reason_short or "reba",
            )
    if not good_posture and confidence < MIN_RISK_CONFIDENCE:
        good_posture, risk_level, violation_type, reason_short = (
            True,
            1,
            "COMPLIANT_BEHAVIOR",
            "low_conf",
        )
    weak_good = bool(good_posture and confidence < GOOD_REQUIRES_CONFIDENCE)
    return {
        "good_posture": good_posture,
        "risk_level": risk_level,
        "violation_type": violation_type,
        "confidence": confidence,
        "angles": angles,
        "reason_short": reason_short,
        "weak_good": weak_good,
    }


# ---------------------------------------------------------------------------
# Main posture pipeline
# ---------------------------------------------------------------------------

def run_posture_pipeline(
    video_path: Path,
    job_dir: Path,
    sample_every_n: int = 3,
    conf_threshold: float = 0.5,
) -> tuple[dict, list[dict]]:
    """
    Runs YOLO pose on sampled frames to produce per-worker ergonomic risk events.
    Returns (summary, posture_events).
    """
    pose_model = _get_pose_model()

    posture_frames_dir = job_dir / "posture_frames"
    posture_frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for posture analysis: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    posture_events: list[dict] = []
    worker_log: Dict[int, list[dict]] = {}

    frame_idx = 0
    frames_sampled = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n != 0:
            frame_idx += 1
            continue

        frames_sampled += 1
        timestamp = round(frame_idx / fps, 3)

        try:
            results = pose_model(frame_bgr, conf=conf_threshold, imgsz=640, verbose=False)[0]
        except Exception as e:
            print(f"[posture] frame {frame_idx} error: {e}")
            frame_idx += 1
            continue

        if results.keypoints is None or results.keypoints.data is None:
            frame_idx += 1
            continue

        kp_data = results.keypoints.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.zeros((kp_data.shape[0], 4))
        n_pose = kp_data.shape[0]

        for j in range(n_pose):
            kp = kp_data[j]
            angles = angles_from_keypoints(kp)
            if not angles:
                continue
            risk = risk_from_angles(angles)
            track_id = int(results.boxes.id[j].item()) if results.boxes.id is not None else j
            worker_log.setdefault(track_id, []).append(risk)

            if not risk["good_posture"]:
                frame_name = f"posture_{frame_idx:06d}_worker_{track_id}.jpg"
                cv2.imwrite(str(posture_frames_dir / frame_name), frame_bgr)
                posture_events.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_seconds": timestamp,
                        "track_id": track_id,
                        "bbox": boxes[j].tolist() if len(boxes) > j else None,
                        "risk_level": risk["risk_level"],
                        "violation_type": risk["violation_type"],
                        "confidence": risk["confidence"],
                        "angles": risk["angles"],
                        "reason_short": risk["reason_short"],
                        "frame_path": frame_name,
                    }
                )

        frame_idx += 1

    cap.release()

    summary = {
        "fps": fps,
        "total_frames": total_frames,
        "frames_sampled": frames_sampled,
        "events": len(posture_events),
        "workers": len(worker_log),
    }

    (job_dir / "posture_summary.json").write_text(json.dumps(summary, indent=2))
    (job_dir / "posture_events.json").write_text(json.dumps(posture_events, indent=2))
    print(f"[posture] done — {len(posture_events)} risk frames across {summary['workers']} workers")
    return summary, posture_events


# ---------------------------------------------------------------------------
# Optional VLM pass over posture events
# ---------------------------------------------------------------------------

def run_posture_vlm(
    video_path: Path,
    job_dir: Path,
    posture_events: list[dict],
    posture_summary: dict,
    camera_source: str = "POSTURE",
) -> list[dict]:
    """
    Reuse the VLM pipeline to summarize posture hazards.
    Writes posture_vlm.json in job_dir.
    """
    # Build hazard_frames compatible with VLM input
    hazard_frames: list[dict] = []
    for pe in posture_events:
        hazard_frames.append(
            {
                "frame_idx": pe.get("frame_idx"),
                "timestamp_seconds": pe.get("timestamp_seconds"),
                "frame_path": pe.get("frame_path"),
                "detections": [],
                "hazards": [
                    {
                        "hazard_type": pe.get("violation_type"),
                        "description": pe.get("reason_short"),
                        "severity": "HIGH" if pe.get("risk_level", 1) >= 3 else "MEDIUM",
                        "violation_type": pe.get("violation_type"),
                        "track_id": pe.get("track_id"),
                        "objects": [{"class": "worker", "confidence": pe.get("confidence", 0.6)}],
                    }
                ],
            }
        )

    out_file = job_dir / "posture_vlm.json"
    if not hazard_frames:
        out_file.write_text("[]")
        return []

    assessment, annotated_paths = run_vlm_on_video(
        video_path=video_path,
        hazard_frames=hazard_frames,
        job_dir=job_dir,
        summary=posture_summary,
        camera_source=camera_source,
    )

    # Mirror structure of assess_all_events return for consistency
    first = hazard_frames[0]
    last = hazard_frames[-1]
    result = [{
        "event": {
            "start_ts": first.get("timestamp_seconds", 0.0),
            "end_ts": last.get("timestamp_seconds", posture_summary.get("total_frames", 0) / max(posture_summary.get("fps", 25.0), 1)),
            "duration_seconds": round(last.get("timestamp_seconds", 0.0) - first.get("timestamp_seconds", 0.0), 3),
            "frame_count": len(hazard_frames),
            "peak_yolo_severity": "HIGH",
            "yolo_hazard_types": list({h["hazard_type"] for f in hazard_frames for h in f.get("hazards", [])}),
            "representative_frame_path": first.get("frame_path"),
            "representative_timestamp": first.get("timestamp_seconds", 0.0),
            "chunk_index": 0,
        },
        "vlm": assessment.model_dump(),
        "error": None,
    }]

    out_file.write_text(json.dumps(result, indent=2))
    print(f"[posture] VLM done — {len(assessment.hazards)} hazards")
    return result
