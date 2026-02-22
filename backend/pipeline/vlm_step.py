"""
vlm_step.py — VLM hazard assessment using Qwen3-VL-8B-Instruct via HuggingFace transformers

Per-event flow:
  1. group_into_events()     → cluster consecutive hazard frames within 3s gaps
  2. run_vlm_on_event()      → send representative frame(s) to Qwen3-VL
  3. _extract_json()         → parse structured VLMAssessment from model output
  4. assess_all_events()     → run over all events, write vlm_assessments.json
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Literal, Optional

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID       = os.getenv("VLM_MODEL_ID",     "Qwen/Qwen3-VL-8B-Instruct")
MAX_PIXELS     = int(os.getenv("VLM_MAX_PIXELS",    "100352"))   # ~317x317 — tune for VRAM
MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_NEW_TOKENS", "1024"))
MAX_FRAMES_PER_EVENT = 3      # send up to 3 frames per event window for temporal context
MAX_IMAGE_BYTES = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# Model singleton — loaded once, shared across all inference calls
# ---------------------------------------------------------------------------

_model     = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        print(f"[vlm] loading {MODEL_ID} ...")
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",            # splits across available GPUs automatically
        )
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("[vlm] model loaded")
    return _model, _processor


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class PPEStatus(BaseModel):
    helmet:         Literal["PRESENT", "ABSENT", "UNCLEAR"]
    vest:           Literal["PRESENT", "ABSENT", "UNCLEAR"]
    gloves:         Literal["PRESENT", "ABSENT", "UNCLEAR"]
    harness:        Optional[Literal["PRESENT", "ABSENT", "UNCLEAR"]] = None
    eye_protection: Optional[Literal["PRESENT", "ABSENT", "UNCLEAR"]] = None
    safety_boots:   Optional[Literal["PRESENT", "ABSENT", "UNCLEAR"]] = None


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
    severity:       Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    explanation:    str
    recommendation: str

    @field_validator("explanation", "recommendation")
    @classmethod
    def no_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class VLMAssessment(BaseModel):
    scene_summary:     str
    worker_count:      int
    equipment_present: list[str]
    ppe_per_worker:    list[PPEStatus]
    reasoning:         str
    hazards:           list[HazardDetail]
    no_hazards:        bool
    confidence:        Literal["LOW", "MEDIUM", "HIGH"]

    @field_validator("worker_count")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("worker_count cannot be negative")
        return v


# ---------------------------------------------------------------------------
# Prompts — fill in SYSTEM_PROMPT with your construction safety context
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Marcus Chen. You have 24 years as a senior construction safety manager — 11 of them as \
Chief Safety Officer on high-risk civil infrastructure projects across South-East Asia and the \
Middle East. You hold NEBOSH Construction, IOSH Managing Safely, and OSHA 1926 certifications. \
You have personally witnessed three fatal incidents in your career. Each one was preventable. \
You carry that weight every single day. You do not flag violations to make numbers — you flag them \
because you know exactly what happens when someone doesn't.

You are now reviewing flagged footage from an AI-assisted wall-mounted safety camera. Your \
YOLO detection system has already identified potential violations in these frames. Your job is \
to confirm, expand, or reject those findings with your expert judgment. You are the last \
human-equivalent checkpoint before this footage is logged as an official safety incident.

═══════════════════════════════════════════════════════════════════════
MANDATORY SAFETY RULES — YOU MUST CHECK EVERY WORKER AGAINST ALL FOUR
═══════════════════════════════════════════════════════════════════════

RULE 1 — BASIC PPE (applies to every worker on foot at all times):
  • Hard hat — must be worn correctly: brim forward, chinstrap used at height.
    A bare head, backwards cap, bump cap, or hair net = VIOLATION.
  • High-visibility retroreflective safety vest — must be worn and fully fastened.
    Open vest, vest used as a jacket, or no vest = VIOLATION.
  • Clothing — shoulders and legs must be covered. Short sleeves, shorts = VIOLATION.
  • Safety footwear — closed-toe boots covering the foot. Trainers, sandals = VIOLATION.
  • Eye/face protection — required when cutting, welding, grinding, or drilling.
    No goggles/face shield during these activities = VIOLATION.
  VIOLATION TYPE → PPE_MISSING

RULE 2 — FALL PROTECTION (harness):
  • Any worker at height ≥ 3 metres where the edge has NO guardrail or edge protection \
must wear a safety harness, visibly connected to an anchor point.
  • Visible scaffold, ladder, elevated platform, or roof edge with no harness = VIOLATION.
  VIOLATION TYPE → FALL_PROTECTION_MISSING

RULE 3 — EDGE PROTECTION:
  • Underground excavations ≥ 3 metres deep with steep retaining walls require \
guardrails, fences, or clearly marked edge warning barriers.
  • A worker standing at or near the edge of such a trench or pit without visible edge \
protection = VIOLATION.
  VIOLATION TYPE → SCAFFOLD_VIOLATION (use this for unguarded excavation edges)

RULE 4 — EXCAVATOR BLIND SPOTS AND OPERATING RADIUS:
  • No worker may be within the operating radius or in the blind spots of an excavator \
that is in operation or has an operator inside.
  • Any worker visible within the swing arc or close proximity to an operating excavator \
= CRITICAL violation.
  VIOLATION TYPE → PROXIMITY_HAZARD

ADDITIONAL HAZARDS TO ASSESS:
  • PHONE USE — worker visibly holding/looking at phone in active work zone near machinery.
    BEHAVIORAL_UNSAFE, MEDIUM severity.
  • RUNNING — worker running on site. Prohibited due to trip/collision risk.
    BEHAVIORAL_UNSAFE, MEDIUM severity.
  • ZONE BREACH — worker inside a restricted/exclusion zone marked by cones, barriers, or signage.
    ZONE_BREACH, HIGH severity.
  • LADDER MISUSE — ladder not secured, worker climbing without 3-point contact, overreaching.
    LADDER_MISUSE, HIGH severity.

═══════════════════════════════════════════════════════════════════════
REMEMBER — THESE RULES APPLY TO EVERY WORKER IN EVERY FRAME:
  Rule 1: Hard hat + hi-vis vest + covered clothing + closed footwear. Missing ANY = PPE_MISSING.
  Rule 2: Harness required at ≥ 3m height without edge protection. Missing = FALL_PROTECTION_MISSING.
  Rule 3: Guardrails required at open excavation edges ≥ 3m deep. Missing = SCAFFOLD_VIOLATION.
  Rule 4: No workers in excavator blind spots or operating radius. Breached = PROXIMITY_HAZARD.
═══════════════════════════════════════════════════════════════════════

YOUR COGNITIVE PROCESS — follow this exact sequence in your reasoning field:

  STAGE 1 — SCENE SCAN:
    Describe the environment. What type of construction site is this? What is happening?
    What equipment, machinery, and hazards are present? What is the lighting and visibility?

  STAGE 2 — WORKER INVENTORY:
    Count every visible worker. Assign them labels (Worker A, Worker B, etc.).
    Note their position, activity, and proximity to equipment or edges.

  STAGE 3 — RULE-BY-RULE AUDIT (for each worker):
    Check Rule 1: Is the hard hat present and correct? Is the vest worn and fastened?
      Are shoulders and legs covered? Is footwear appropriate?
    Check Rule 2: Are they at height? Is a harness visible and connected?
    Check Rule 3: Are they near an unguarded excavation edge?
    Check Rule 4: Are they within the operating radius or blind spot of an excavator?
    Check additional: Phone use? Running? Zone breach? Ladder misuse?

  STAGE 4 — CROSS-VERIFICATION:
    Before logging any violation, ask yourself: "Can I clearly see evidence of this in at least
    one of the frames provided?" If the violation is ambiguous due to occlusion, distance, or
    poor lighting — do NOT flag it. Set confidence to LOW and note the uncertainty.
    If you are uncertain whether PPE is present, default to ABSENT (safety-first).

  STAGE 5 — DUPLICATION CHECK:
    Review the already-flagged violations list provided in the user context.
    Do NOT re-flag a violation for the same worker (track ID) if it has already been logged
    for this session. A worker flagged for PPE_MISSING (no hard hat) in a previous event
    window should NOT be flagged again for the same item unless their situation has changed.
    New violations (different type, different worker) should always be logged.

  STAGE 6 — VERDICT:
    Produce your final structured output. Every hazard entry must correspond to a real,
    visually confirmed violation you identified in Stage 3 and verified in Stage 4.

═══════════════════════════════════════════════════════════════════════
CRITICAL DIRECTIVES — READ THESE AND APPLY THEM EVERY TIME:
  1. You MUST check every visible worker against all four rules. No exceptions.
  2. You MUST NOT flag a violation you cannot visually confirm in the frames provided.
  3. You MUST NOT duplicate a violation already logged for the same worker in this session.
  4. If unsure whether PPE is present, mark it ABSENT. Safety-first is not optional.
  5. Your reasoning field is a mandatory audit log — write it as if a court will read it.
  6. Output ONLY valid JSON. Zero text outside the JSON object.
═══════════════════════════════════════════════════════════════════════"""


USER_TEMPLATE = """\
YOLO PRE-DETECTION CONTEXT:
  Flagged hazards : {yolo_hazards}
  Detected classes: {yolo_classes}
  Frame timestamp : {timestamp}s into video
  Event window    : {event_start}s – {event_end}s ({frame_count} flagged frames)

ALREADY FLAGGED THIS SESSION (do NOT re-log these — deduplication required):
{already_flagged}

REMINDER BEFORE YOU START:
  Rule 1 — Every worker needs: hard hat + hi-vis vest + covered clothing + closed footwear.
  Rule 2 — Harness required at ≥ 3m height without edge protection.
  Rule 3 — Guardrails required at open excavation edges ≥ 3m deep.
  Rule 4 — No workers within excavator operating radius or blind spots.
  Cross-verify each violation visually before flagging. No duplicates. No assumptions.

Return a JSON object matching this exact schema:
{schema}

Output ONLY valid JSON. No explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Security — sandboxes file access to job_dir
# ---------------------------------------------------------------------------

def _safe_path(base_dir: Path, relative: str) -> Path:
    resolved = (base_dir / relative).resolve()
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError:
        raise ValueError(f"Path traversal blocked: {relative!r}")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Not a regular file: {resolved}")
    if resolved.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Temporal event grouping
# ---------------------------------------------------------------------------

def group_into_events(hazard_frames: list[dict], gap_seconds: float = 3.0) -> list[dict]:
    """Collapses consecutive hazard frames within gap_seconds into a single event window."""
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
    def total_conf(f: dict) -> float:
        return sum(d["confidence"] for d in f.get("detections", []))

    rep = max(frames, key=total_conf)

    sev_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    peak_sev = "LOW"
    for f in frames:
        for h in f.get("hazards", []):
            if sev_rank.get(h.get("severity", "LOW"), 0) > sev_rank[peak_sev]:
                peak_sev = h["severity"]

    return {
        "start_ts":           frames[0]["timestamp_seconds"],
        "end_ts":             frames[-1]["timestamp_seconds"],
        "duration_seconds":   round(frames[-1]["timestamp_seconds"] - frames[0]["timestamp_seconds"], 3),
        "frame_count":        len(frames),
        "peak_yolo_severity": peak_sev,
        "representative_frame": rep,
        "all_frames":         frames,
        "yolo_hazard_types":  list({h["hazard_type"] for f in frames for h in f.get("hazards", [])}),
        "yolo_classes_seen":  list({d["class_name"] for f in frames for d in f.get("detections", [])}),
    }


# ---------------------------------------------------------------------------
# JSON extraction from raw model output
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Handles markdown code blocks and raw JSON objects in model output."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in model output: {text[:300]}")


# ---------------------------------------------------------------------------
# VLM inference — one event window
# ---------------------------------------------------------------------------

def run_vlm_on_event(event: dict, job_dir: Path, already_flagged: list[dict] | None = None) -> VLMAssessment:
    model, processor = _get_model()

    rep = event["representative_frame"]
    frames_dir = job_dir / "frames"

    # Collect up to MAX_FRAMES_PER_EVENT image paths from this event window
    image_paths: list[Path] = []

    # Representative frame first
    if rep.get("frame_path"):
        try:
            image_paths.append(_safe_path(frames_dir, rep["frame_path"]))
        except (ValueError, FileNotFoundError):
            pass

    # Fill remaining slots from other frames in the window
    for f in event.get("all_frames", []):
        if len(image_paths) >= MAX_FRAMES_PER_EVENT:
            break
        fp = f.get("frame_path")
        if fp and fp != rep.get("frame_path"):
            try:
                image_paths.append(_safe_path(frames_dir, fp))
            except (ValueError, FileNotFoundError):
                continue

    if not image_paths:
        raise FileNotFoundError(f"No accessible frames for event at {rep['timestamp_seconds']}s")

    flagged_summary = (
        "\n".join(
            f"  - Worker track_id={f['track_id']}: {f['violation_type']}"
            for f in (already_flagged or [])
        ) or "  None yet — this is the first event in this session."
    )

    user_text = USER_TEMPLATE.format(
        yolo_hazards    = json.dumps(event["yolo_hazard_types"]),
        yolo_classes    = json.dumps(event["yolo_classes_seen"]),
        timestamp       = rep["timestamp_seconds"],
        event_start     = event["start_ts"],
        event_end       = event["end_ts"],
        frame_count     = event["frame_count"],
        already_flagged = flagged_summary,
        schema          = json.dumps(VLMAssessment.model_json_schema(), indent=2),
    )

    # Build message — images first, text last (Qwen3-VL convention)
    content = [
        {"type": "image", "image": str(p), "max_pixels": MAX_PIXELS}
        for p in image_paths
    ]
    content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    raw = _extract_json(output_text)
    return VLMAssessment.model_validate(raw)


# ---------------------------------------------------------------------------
# Convenience: run VLM over all events from a pipeline run
# ---------------------------------------------------------------------------

def assess_all_events(
    hazard_frames: list[dict],
    job_dir: Path,
    gap_seconds: float = 3.0,
) -> list[dict]:
    """Groups hazard frames into events, runs VLM on each, writes vlm_assessments.json."""
    events  = group_into_events(hazard_frames, gap_seconds)
    results: list[dict] = []

    # Session-level deduplication tracker — {(track_id, violation_type)} already logged
    session_flagged: list[dict] = []

    for i, event in enumerate(events):
        print(f"[vlm] event {i+1}/{len(events)} — {event['start_ts']}s–{event['end_ts']}s")
        try:
            assessment = run_vlm_on_event(event, job_dir, already_flagged=session_flagged)

            # Update session tracker with newly confirmed violations
            for hazard in assessment.hazards:
                track_id = next(
                    (f.get("track_id") for f in event["representative_frame"].get("hazards", [])
                     if f.get("violation_type") == hazard.violation_type),
                    None,
                )
                session_flagged.append({
                    "track_id":      track_id,
                    "violation_type": hazard.violation_type,
                    "event_ts":       event["start_ts"],
                })

            results.append({
                "event": {
                    "start_ts":                  event["start_ts"],
                    "end_ts":                    event["end_ts"],
                    "duration_seconds":          event["duration_seconds"],
                    "frame_count":               event["frame_count"],
                    "peak_yolo_severity":        event["peak_yolo_severity"],
                    "yolo_hazard_types":         event["yolo_hazard_types"],
                    "representative_frame_path": event["representative_frame"].get("frame_path"),
                    "representative_timestamp":  event["representative_frame"]["timestamp_seconds"],
                },
                "vlm":   assessment.model_dump(),
                "error": None,
            })
        except Exception as exc:
            print(f"[vlm] error on event {i+1}: {exc}")
            results.append({"event": event, "vlm": None, "error": str(exc)})

    (job_dir / "vlm_assessments.json").write_text(json.dumps(results, indent=2))
    print(f"[vlm] done — {len(results)} events assessed")
    return results
