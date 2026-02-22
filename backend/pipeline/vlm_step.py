"""
vlm_step.py — VLM hazard assessment stage

Runs after YOLO. Takes temporal event windows produced by group_into_events(),
sends the most representative frame + crop to Llama 3.2 Vision via Ollama,
and returns structured VLMAssessment objects ready for DB ingestion.

Security: all file access is sandboxed to job_dir via _safe_path().
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Literal, Optional

import ollama
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")

MAX_IMAGE_BYTES = 10 * 1024 * 1024    # 10 MB — reject anything larger

# ---------------------------------------------------------------------------
# Output schema — maps directly to safety_events + metadata_json
# ---------------------------------------------------------------------------

class PPEStatus(BaseModel):
    helmet:  Literal["PRESENT", "ABSENT", "UNCLEAR"]
    vest:    Literal["PRESENT", "ABSENT", "UNCLEAR"]
    gloves:  Literal["PRESENT", "ABSENT", "UNCLEAR"]
    # Additional PPE — included if visible
    harness:         Optional[Literal["PRESENT", "ABSENT", "UNCLEAR"]] = None
    eye_protection:  Optional[Literal["PRESENT", "ABSENT", "UNCLEAR"]] = None
    safety_boots:    Optional[Literal["PRESENT", "ABSENT", "UNCLEAR"]] = None


class HazardDetail(BaseModel):
    violation_type: Literal[
        "PPE_MISSING",
        "FALL_PROTECTION_MISSING",
        "PROXIMITY_HAZARD",
        "ZONE_BREACH",
        "LADDER_MISUSE",
        "SCAFFOLD_VIOLATION",
        "BEHAVIORAL_UNSAFE",
    ]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    explanation: str        # what the model sees — specific and brief
    recommendation: str     # one actionable instruction

    @field_validator("explanation", "recommendation")
    @classmethod
    def no_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class VLMAssessment(BaseModel):
    # Scene context
    scene_summary:   str        # ≤ 2 sentences describing the overall scene
    worker_count:    int        # number of workers visible
    equipment_present: list[str]  # e.g. ["excavator", "scaffold"]

    # Per-worker PPE (best effort — one entry per visible worker)
    ppe_per_worker: list[PPEStatus]

    # Reasoning trace — stored for audit, not shown to end users
    reasoning: str

    # Findings
    hazards:    list[HazardDetail]   # empty = no violations found
    no_hazards: bool                 # explicit clean-bill flag

    # Model confidence in this assessment
    confidence: Literal["LOW", "MEDIUM", "HIGH"]

    @field_validator("worker_count")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("worker_count cannot be negative")
        return v


# ---------------------------------------------------------------------------
# System prompt
# (prompt repetition is intentional — reinforces key rules across the prompt)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Marcus Chen, a senior construction safety manager with 22 years of experience on \
heavy civil, commercial, and industrial construction projects. You are certified in OSHA 1926 Construction \
Standards and hold a NEBOSH Construction Certificate.

You are reviewing flagged footage from the IronSite wall-mounted safety camera. Your YOLO pre-detection \
system has already flagged this frame as a potential violation. Your job is to confirm, correct, or expand \
on that detection with your expert eye.

═══════════════════════════════════════════════════════════
MANDATORY PPE REQUIREMENTS — OSHA 1926 CONSTRUCTION STANDARD
Every worker visible on site MUST be wearing ALL THREE of the following at all times:

  1. HARD HAT / SAFETY HELMET — OSHA 1926.100
     Hard hat must be worn correctly: brim forward, chinstrap engaged if at height.
     A bare head, hair net only, bump cap, or backwards hat IS a violation.

  2. HIGH-VISIBILITY SAFETY VEST — OSHA 1926.201
     Must be a Class 2 or Class 3 reflective vest, fully zipped or fastened.
     An open vest, vest worn as a jacket, or no vest at all IS a violation.

  3. PROTECTIVE GLOVES — OSHA 1926.28
     Must be appropriate for the task. Bare hands while handling materials,
     tools, or equipment IS a violation. Gloves hanging from a belt IS a violation.

Additionally check for:
  4. FALL PROTECTION HARNESS — required when working at any height > 6 feet (OSHA 1926.502)
  5. EYE PROTECTION — required near cutting, grinding, chemical, or debris hazards (OSHA 1926.102)
  6. SAFETY BOOTS — required on all active construction sites (OSHA 1926.96)
═══════════════════════════════════════════════════════════

SCENE HAZARDS TO ASSESS — beyond PPE:

  PROXIMITY: Is any worker within the swing radius or operating envelope of an excavator, \
crane, forklift, or dump truck? This is HIGH-CRITICAL severity.

  FALL RISK: Is any worker on a scaffold, ladder, elevated platform, or roof edge without a \
visible harness or guardrail? HIGH severity. Use violation_type FALL_PROTECTION_MISSING.

  PHONE USE: Is any worker visibly holding or looking at a mobile phone while in an active \
work zone or near operating machinery? MEDIUM severity. Use violation_type BEHAVIORAL_UNSAFE.

  RUNNING: Is any worker running on the site? Running is prohibited on all active construction \
sites due to collision and trip risk. MEDIUM severity. Use violation_type BEHAVIORAL_UNSAFE.

  ZONE BREACH: Is any worker in a restricted or exclusion zone (indicated by cones, barriers, \
or signage)? HIGH severity. Use violation_type ZONE_BREACH.

═══════════════════════════════════════════════════════════
IMPORTANT — REMEMBER THESE RULES:
- Every worker must have: HARD HAT + SAFETY VEST + GLOVES. Missing any one = PPE_MISSING violation.
- Assess each visible worker individually. Do not assume PPE is present if you cannot see it clearly.
- If unsure whether PPE is present, mark as ABSENT — safety-first.
- Flag phone use and running as BEHAVIORAL_UNSAFE — these are real and detectable violations.
- Flag missing harness on scaffold/ladder as FALL_PROTECTION_MISSING.
- YOLO context is provided as a hint. You may confirm, correct, or add to it.
- Output ONLY valid JSON. No extra explanation outside the JSON.
═══════════════════════════════════════════════════════════

Your reasoning field must follow this structure:
SCENE SCAN: [what is in the frame — describe environment, workers, equipment]
WORKER-BY-WORKER: [for each worker: what PPE is visible, what is missing]
EQUIPMENT ASSESSMENT: [any machinery present, is anyone in proximity]
REGULATORY VERDICT: [which OSHA rules are being violated and why]
CONFIDENCE NOTE: [why you are HIGH/MEDIUM/LOW confidence — lighting, occlusion, distance]"""


USER_TEMPLATE = """\
YOLO PRE-DETECTION CONTEXT:
  Flagged hazards : {yolo_hazards}
  Detected classes: {yolo_classes}
  Frame timestamp : {timestamp}s into video
  Event window    : {event_start}s – {event_end}s ({frame_count} flagged frames in this window)

REMEMBER: Check for hard hat, safety vest, AND gloves on every visible worker. \
Missing any one of these is a PPE_MISSING violation. Also assess proximity to equipment, \
fall risks, and any other site hazards you observe.

Analyze this construction site image and return your structured assessment."""


# ---------------------------------------------------------------------------
# Security — path sandboxing
# ---------------------------------------------------------------------------

def _safe_path(job_dir: Path, relative: str) -> Path:
    """
    Resolves a relative path against job_dir and asserts it stays within job_dir.
    Raises ValueError on path traversal attempts.
    """
    resolved = (job_dir / relative).resolve()
    try:
        resolved.relative_to(job_dir.resolve())
    except ValueError:
        raise ValueError(f"Path traversal blocked: {relative!r} escapes job_dir")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Not a regular file: {resolved}")
    return resolved


def _load_image_b64(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({size} bytes) — max {MAX_IMAGE_BYTES}")
    return base64.b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------------------
# Temporal event grouping
# ---------------------------------------------------------------------------

def group_into_events(hazard_frames: list[dict], gap_seconds: float = 3.0) -> list[dict]:
    """
    Collapses consecutive hazard frames within gap_seconds into a single event window.
    Selects the representative frame = highest total YOLO detection confidence in the window.
    """
    if not hazard_frames:
        return []

    events: list[dict] = []
    current: list[dict] = [hazard_frames[0]]

    for frame in hazard_frames[1:]:
        if frame["timestamp_seconds"] - current[-1]["timestamp_seconds"] <= gap_seconds:
            current.append(frame)
        else:
            events.append(_build_event(current))
            current = [frame]
    events.append(_build_event(current))
    return events


def _build_event(frames: list[dict]) -> dict:
    # Representative frame = highest sum of YOLO detection confidences
    def total_conf(f: dict) -> float:
        return sum(d["confidence"] for d in f.get("detections", []))

    rep = max(frames, key=total_conf)

    # Worst severity across the window
    sev_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    peak_sev = "LOW"
    for f in frames:
        for h in f.get("hazards", []):
            if sev_rank.get(h.get("severity", "LOW"), 0) > sev_rank[peak_sev]:
                peak_sev = h["severity"]

    return {
        "start_ts":            frames[0]["timestamp_seconds"],
        "end_ts":              frames[-1]["timestamp_seconds"],
        "duration_seconds":    round(frames[-1]["timestamp_seconds"] - frames[0]["timestamp_seconds"], 3),
        "frame_count":         len(frames),
        "peak_yolo_severity":  peak_sev,
        "representative_frame": rep,          # best frame to send to VLM
        "all_frames":          frames,
        # Aggregate YOLO hazard types across the window
        "yolo_hazard_types":   list({h["hazard_type"] for f in frames for h in f.get("hazards", [])}),
        "yolo_classes_seen":   list({d["class_name"] for f in frames for d in f.get("detections", [])}),
    }


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def run_vlm_on_event(event: dict, job_dir: Path) -> VLMAssessment:
    """
    Runs the VLM on a single temporal event window.
    Sends the representative frame (full frame preferred, person crop as fallback).
    """
    rep = event["representative_frame"]

    # Prefer the full flagged frame — more context for the VLM
    image_path: Optional[Path] = None
    if rep.get("frame_path"):
        try:
            image_path = _safe_path(job_dir / "frames", rep["frame_path"])
        except (ValueError, FileNotFoundError):
            pass

    # Fallback: first person crop
    if image_path is None and rep.get("person_crops"):
        crop_rel = rep["person_crops"][0]["path"]
        try:
            image_path = _safe_path(job_dir / "crops", crop_rel)
        except (ValueError, FileNotFoundError):
            pass

    if image_path is None:
        raise FileNotFoundError(f"No accessible image for event at {rep['timestamp_seconds']}s")

    img_b64 = _load_image_b64(image_path)

    user_msg = USER_TEMPLATE.format(
        yolo_hazards  = json.dumps(event["yolo_hazard_types"]),
        yolo_classes  = json.dumps(event["yolo_classes_seen"]),
        timestamp     = rep["timestamp_seconds"],
        event_start   = event["start_ts"],
        event_end     = event["end_ts"],
        frame_count   = event["frame_count"],
    )

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg, "images": [img_b64]},
        ],
        format=VLMAssessment.model_json_schema(),
        options={"temperature": 0.05, "num_predict": 1024},
    )

    raw = response["message"]["content"]
    return VLMAssessment.model_validate_json(raw)


# ---------------------------------------------------------------------------
# Convenience: run VLM over all events from a pipeline run
# ---------------------------------------------------------------------------

def assess_all_events(
    hazard_frames: list[dict],
    job_dir: Path,
    gap_seconds: float = 3.0,
) -> list[dict]:
    """
    Groups hazard frames into events, runs VLM on each, returns enriched event dicts.
    Writes vlm_assessments.json to job_dir.
    """
    events = group_into_events(hazard_frames, gap_seconds)
    results: list[dict] = []

    for i, event in enumerate(events):
        print(f"[vlm] event {i+1}/{len(events)} — {event['start_ts']}s–{event['end_ts']}s")
        try:
            assessment = run_vlm_on_event(event, job_dir)
            results.append({
                "event": {
                    "start_ts":          event["start_ts"],
                    "end_ts":            event["end_ts"],
                    "duration_seconds":  event["duration_seconds"],
                    "frame_count":       event["frame_count"],
                    "peak_yolo_severity": event["peak_yolo_severity"],
                    "yolo_hazard_types": event["yolo_hazard_types"],
                    "representative_frame_path": event["representative_frame"].get("frame_path"),
                    "representative_timestamp":  event["representative_frame"]["timestamp_seconds"],
                },
                "vlm": assessment.model_dump(),
                "error": None,
            })
        except Exception as exc:
            print(f"[vlm] error on event {i+1}: {exc}")
            results.append({
                "event": event,
                "vlm":   None,
                "error": str(exc),
            })

    (job_dir / "vlm_assessments.json").write_text(json.dumps(results, indent=2))
    print(f"[vlm] done — {len(results)} events assessed, written to vlm_assessments.json")
    return results
