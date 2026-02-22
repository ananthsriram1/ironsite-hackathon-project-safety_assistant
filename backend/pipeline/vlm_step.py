"""
vlm_step.py — VLM hazard assessment using Qwen3-VL-8B-Instruct via HuggingFace transformers

Three-pass adversarial Chain-of-Thought (adv-CoT) flow:
  Pass 1 — Jamie Reyes (Generator)   : raw video only, no machine data, field inspection notes
  Pass 2 — Marcus Chen (Discriminator): raw video + annotated frames, no Jamie's notes
                                        → independent expert assessment (catches what Jamie missed)
  Pass 3 — Marcus Chen (Reconciler)  : Jamie's notes + Marcus's notes + YOLO data + annotated frames
                                        → explicit conflict resolution → final structured JSON

Inspired by: "Chain-of-Thought Prompt Optimization via Adversarial Learning" (Yang et al. 2025)
  Jamie   = Generator  (unbiased raw visual observation)
  Marcus  = Discriminator (machine-anchored independent assessment)
  Pass 3  = Modifier/Verifier (reconciles disagreements, produces authoritative verdict)
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
from peft import PeftModel
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID       = os.getenv("VLM_MODEL_ID",      "/workspace/models/qwen3-vl")
ADAPTER_PATH   = os.getenv("VLM_ADAPTER_PATH",  "/workspace/models/adapter1")
MAX_PIXELS          = int(os.getenv("VLM_MAX_PIXELS",          "100352"))  # per-frame ~317x317
MAX_NEW_TOKENS      = int(os.getenv("VLM_MAX_NEW_TOKENS",       "1024"))   # pass-3 final JSON
MAX_NEW_TOKENS_P1   = int(os.getenv("VLM_MAX_NEW_TOKENS_PASS1", "768"))    # pass-1 Jamie notes
MAX_NEW_TOKENS_P2   = int(os.getenv("VLM_MAX_NEW_TOKENS_PASS2", "768"))    # pass-2 Marcus notes
VIDEO_FPS           = float(os.getenv("VLM_VIDEO_FPS",          "1.0"))    # frames/sec to sample

# ---------------------------------------------------------------------------
# Model singleton — loaded once, shared across all inference calls
# ---------------------------------------------------------------------------

_model     = None
_processor = None


def _reset_model():
    """Destroy model singleton and free CUDA memory. Forces fresh allocation on next call."""
    global _model, _processor
    _model = None
    _processor = None
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[vlm] model reset — will reload on next call")


def _get_model():
    global _model, _processor
    if _model is None:
        print(f"[vlm] loading base model from {MODEL_ID} ...")
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        adapter = Path(ADAPTER_PATH)
        if adapter.exists():
            print(f"[vlm] applying adapter from {ADAPTER_PATH} ...")
            _model = PeftModel.from_pretrained(_model, str(adapter))
            _processor = AutoProcessor.from_pretrained(str(adapter))
        else:
            print(f"[vlm] no adapter found at {ADAPTER_PATH}, using base model")
            _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model.eval()
        print("[vlm] model ready")
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
    severity:         Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    explanation:      str
    recommendation:   str
    best_frame_index: Optional[int] = None

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
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Marcus Chen. You have 24 years as a senior construction safety manager — 11 of them as \
Chief Safety Officer on high-risk civil infrastructure projects across South-East Asia and the \
Middle East. You hold NEBOSH Construction, IOSH Managing Safely, and OSHA 1926 certifications. \
You have personally witnessed three fatal incidents in your career. Each one was preventable. \
You carry that weight every single day. You do not flag violations to make numbers — you flag them \
because you know exactly what happens when someone doesn't.

You are now reviewing a construction site safety video from an AI-assisted camera system. A \
YOLO detection model and SAM segmentation pipeline have already pre-analyzed the footage and \
identified potential violations. Their findings are provided as structured data alongside the \
video. Your job is to watch the full video, cross-reference with the pre-analysis data, and \
produce a single definitive safety assessment covering the entire clip.

═══════════════════════════════════════════════════════════════════════
MANDATORY SAFETY RULES — CHECK EVERY WORKER AGAINST ALL FOUR
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
REMEMBER — THESE RULES APPLY TO EVERY WORKER THROUGHOUT THE FULL VIDEO:
  Rule 1: Hard hat + hi-vis vest + covered clothing + closed footwear. Missing ANY = PPE_MISSING.
  Rule 2: Harness required at ≥ 3m height without edge protection. Missing = FALL_PROTECTION_MISSING.
  Rule 3: Guardrails required at open excavation edges ≥ 3m deep. Missing = SCAFFOLD_VIOLATION.
  Rule 4: No workers in excavator blind spots or operating radius. Breached = PROXIMITY_HAZARD.
═══════════════════════════════════════════════════════════════════════

YOUR REASONING PROCESS — execute every stage before producing output:

  STAGE 1 — SCENE SCAN (fine-grained, action-centric):
    Do not ask "is something wrong?" Ask specific questions:
      • How many workers are visible? What are they each doing?
      • Is there any moving machinery? Any excavators, vehicles, cranes in operation?
      • What is the lighting quality? Can PPE colours be reliably distinguished?
      • Are there any elevated surfaces, open edges, or excavations visible?

  STAGE 2 — DETECTION DATA AUDIT:
    Review the YOLO/SAM data. For every flagged detection, note its confidence score.
    Apply this calibration scale:
      conf ≥ 0.70  → HIGH signal. Strong prior that the detection is real.
      conf 0.40–0.69 → MODERATE signal. Treat as a lead to investigate in the video.
      conf 0.15–0.39 → WEAK signal. Requires clear visual confirmation before flagging.
      conf < 0.15  → NOISE. Discard unless you can independently confirm in the video.
    Cross-reference everything against what you actually observe in the video.

  STAGE 3 — PER-WORKER VIDEO AUDIT:
    For each confirmed worker (label them A, B, C…), ask these specific questions in order:
      PPE CHECK:
        Q1: Can I see a hard hat on this worker's head? Correct fit, brim forward?
        Q2: Is a hi-vis vest visible, fastened, and covering the torso?
        Q3: Are gloves present on both hands? Look for a coloured protective covering over the \
fingers and palm. Bare skin or knuckles without a glove material = NO GLOVES = violation. \
"Maybe gloves" or "hands visible" does not satisfy this check — you must see the glove itself.
        Q4: Is footwear visible? Are they boots or something inappropriate?
      HEIGHT CHECK:
        Q5: Is this worker at ≥3m height? Is edge protection or a harness visible?
      PROXIMITY CHECK:
        Q6: Is there operating machinery within close range of this worker?
      BEHAVIOUR CHECK:
        Q7: Is the worker on a phone, running, or inside a restricted zone?

  STAGE 4 — CONFIDENCE CALIBRATION (cross-reference video vs YOLO):
    For each potential violation you identified in Stage 3:
      • Does the YOLO data support this, and at what confidence?
      • Can you directly observe it in the video without ambiguity?
      • Weigh both signals together. Strong visual evidence overrides weak YOLO signal.
        Strong YOLO signal on something hard to see visually warrants closer inspection.
        Neither weak YOLO nor visual ambiguity alone is sufficient to log a violation.
      • Do NOT assume PPE is absent because YOLO flagged something at low confidence.
        Insufficient evidence means no violation — err toward precision, not recall.

  STAGE 5 — DEDUPLICATION:
    Produce ONE violation entry per unique (worker, violation_type) across the full video.
    A worker missing a hard hat for 30 seconds = one PPE_MISSING entry, not thirty.
    Review your list and eliminate any duplicates before writing the final output.

  STAGE 6 — VERDICT:
    Produce your final structured output. Every hazard entry must be backed by
    clear visual evidence and/or strong YOLO signal. No guesses, no assumptions.

═══════════════════════════════════════════════════════════════════════
CRITICAL DIRECTIVES:
  1. Assess the FULL VIDEO — not just a single frame.
  2. One entry per unique (worker, violation_type). Strictly no duplicates.
  3. Do NOT flag violations you cannot visually confirm or that lack meaningful YOLO support.
  4. Do NOT default to "absent" for PPE. Uncertain = uncertain. Unconfirmed ≠ violation.
  5. Weight YOLO confidence scores using the calibration scale. conf < 0.15 = noise.
  6. USE DISCRETION — not every flagged frame is a genuine safety incident. A worker
     briefly adjusting their hat is not a violation. A glove momentarily off-screen is
     not proof it's missing. Apply experienced professional judgment, not box-ticking.
  7. Your reasoning field is a mandatory audit log — write it as if a court will read it.
  8. Output ONLY valid JSON. Zero text outside the JSON object.
═══════════════════════════════════════════════════════════════════════"""


_CAMERA_GUIDANCE = {
    "POV": """\
CAMERA TYPE: POINT-OF-VIEW (body/helmet-worn camera)
  This footage is shot from the perspective of a worker wearing the camera.

  ▶ GLOVE CHECK — THIS IS YOUR PRIMARY TASK FOR THIS VIDEO:
  The camera wearer's hands will appear in frame, typically at the bottom or sides.
  Gloves look like a coloured protective covering over the hand — usually orange, yellow, \
grey, or black. They completely cover the fingers and palm.
  Bare skin means NO GLOVES. There is no middle ground.

  HOW TO ASSESS:
    Step 1 — Do you see the wearer's hands in any frame? If yes, proceed.
    Step 2 — Are those hands covered by a visible glove material (coloured, textured covering)?
             YES → gloves present.
             NO (skin, fingers, knuckles visible without a glove covering) → PPE_MISSING (gloves). Flag it.
    Step 3 — "I can see hands but I'm not sure if there are gloves" is NOT acceptable.
             If the skin is visible at all without a glove covering over it, that IS a violation.
             Do not write "potential glove use" or "hands visible indicating possible gloves". \
Either you see gloves or you see bare skin. One of these is always true when hands are in frame.

  OTHER WORKERS IN FRAME:
  • The wearer's own hard hat and vest are NOT in frame — do not assess those for the wearer.
  • Any other worker who is close enough to clearly see their PPE should be assessed fully \
(hard hat, vest, gloves, footwear). "Close enough" means you can distinguish their face, \
clothing colour, and whether items are present or absent without ambiguity.
  • Workers in the far background who are too small or blurry to assess reliably — skip them. \
Do not flag what you cannot confirm.
  • If hands are never visible in any frame, note that gloves could not be assessed for the wearer.""",

    "WALL_CAM": """\
CAMERA TYPE: FIXED WALL / OVERHEAD CAMERA
  This footage is shot from a static elevated position covering a work zone.
  Key implications:
  • All workers visible in the frame must be assessed — do not focus on just one person.
  • Gloves may be difficult to resolve at distance; rely on YOLO confidence + visual \
confirmation. Do not flag glove absence if the worker's hands are too small or too far \
to see clearly.
  • Hard hats and hi-vis vests are the most reliably visible PPE from this angle — \
treat confirmed absence as a high-confidence violation.
  • Watch for workers entering excavation zones, machinery proximity, or areas marked \
with cones/barriers.""",
}

# ---------------------------------------------------------------------------
# Pass 1 prompts — fresh-eyes observer (raw video only, no detection data)
# ---------------------------------------------------------------------------

PASS1_SYSTEM = """\
You are Jamie Reyes, a field safety inspector with 6 years of on-site construction experience. \
You are conducting an initial walkthrough review of this site camera footage and filing a \
written inspection report.

Your report will be reviewed and audited by Marcus Chen — Chief Safety Officer with 24 years \
of experience. He will be comparing your findings against machine-detection data and annotated \
frames from an AI system. If you miss something obvious or write vague non-observations, \
he will catch it and it will reflect on your competence.

Be thorough, honest, and specific. Name what you see. If something concerns you, write it down \
clearly — do not hedge into uselessness. If something looks fine, say so and say why. \
Note approximate timestamps where helpful. Write in plain English, not JSON. \
This is your inspection report, not a final verdict — Marcus will have the final say."""

PASS1_USER_TEMPLATE = """\
VIDEO OVERVIEW:
  Duration   : ~{duration:.1f}s
  Sampled at : {fps}fps
  Camera     : {camera_type}

Watch this video and write your inspection report. Marcus Chen will be auditing it \
against machine-detection data — be specific and thorough or he will catch gaps.

Cover the following:

  WORKERS:
  — How many workers are visible? Label them A, B, C...
  — What is each worker doing? Are they stationary or moving?
  — Approximate location in frame for each.

  PPE (for each worker):
  — Head: hard hat visible? Correct fit?
  — Torso: hi-vis vest visible and fastened?
  — Hands: any gloves visible? What do their hands look like?
  — Feet: are boots visible?
{pov_glove_note}
  ENVIRONMENT & HAZARDS:
  — Is any heavy machinery operating? Excavators, vehicles, cranes?
  — Any elevated surfaces, scaffolding, open edges, or excavations?
  — Any workers in proximity to operating machinery?
  — Any running, phone use, or restricted zone breaches observed?

  CONCERNS:
  — List anything that gave you pause, even if you are not certain it is a violation.
  — Note moments where visibility was poor or PPE was hard to assess.

Write freely. These notes feed into a second adversarial analysis pass. Do NOT write JSON."""

PASS1_POV_GLOVE_NOTE = """\
  ▶ POV CAMERA — PRIORITY CHECK:
  — The wearer's hands appear in frame (bottom/sides). Look carefully at every moment hands \
are visible.
  — Describe exactly what the hands look like: bare skin? coloured glove material? unclear?
  — Be explicit — "hands appear bare" or "orange glove covering visible" not "possibly gloves".\
"""

# ---------------------------------------------------------------------------
# Pass 2 user template — adversarial discriminator+modifier pass
# ---------------------------------------------------------------------------

PASS2_USER_TEMPLATE = """\
VIDEO OVERVIEW:
  Duration   : ~{duration:.1f}s  |  Camera: {camera_type}

{camera_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD INSPECTOR'S REPORT (Jamie Reyes — initial walkthrough, raw video only):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pass1_notes}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR AUDIT OF JAMIE'S REPORT — execute this before writing your final output:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are Marcus Chen. Jamie has filed the initial inspection report above. You now have \
what Jamie did not: YOLO+SAM machine-detection data and annotated frames. Your job is to \
audit Jamie's report critically and produce the authoritative final safety assessment.

  AUDIT STEP — evaluate Jamie's work:
    • What did Jamie get right? What is confirmed by the detection data and annotated frames?
    • What did Jamie miss or under-report? Check the YOLO flags and annotated frames carefully —
      Jamie only had the raw video, so missed detections are expected.
    • Where did Jamie make assumptions? ("looks like gloves" is not good enough. \
      Either they are there or they are not.)
    • Did Jamie flag anything that the detection data shows was low-confidence noise?
      Do not carry forward Jamie's false positives.

  RECONCILIATION STEP — produce your final position:
    • For each concern in Jamie's report: does the machine data confirm, weaken, or contradict it?
    • For each YOLO detection: does Jamie's walkthrough observation support it?
    • The final truth is the INTERSECTION of strong visual evidence AND detection signal.
      Strong YOLO + Jamie confirmed visually = flag it.
      Weak YOLO + Jamie uncertain = discard it.
      Jamie observed it clearly + YOLO missed it = still flag it — machines miss things too.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOLO + SAM PRE-ANALYSIS DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{yolo_sam_data}

{annotated_frame_note}

CONFIDENCE CALIBRATION:
  conf ≥ 0.70 = strong  |  0.40–0.69 = moderate, verify visually
  0.15–0.39 = weak, must confirm in video  |  < 0.15 = noise, discard

ONE entry per unique (worker, violation_type). No duplicates.

RULES REMINDER:
  Rule 1 — Every confirmed worker: hard hat + hi-vis vest + covered clothing + closed footwear.
  Rule 2 — Harness at ≥3m height without edge protection.
  Rule 3 — Guardrails at open excavation edges ≥3m deep.
  Rule 4 — No workers in excavator operating radius or blind spots.

Return a JSON object matching this exact schema:
{schema}

Output ONLY valid JSON. No explanation outside the JSON."""

PASS2_ANNOTATED_FRAME_NOTE = """\
ANNOTATED FRAMES ({n} images follow — YOLO+SAM bounding boxes and segmentation masks):
  Frames are numbered [Annotated frame 0] to [Annotated frame {n_minus_1}].
  For each hazard you report, set best_frame_index to the 0-based index of the annotated
  frame that most clearly shows that specific violation. Each hazard gets its own index.
  If no clear frame exists for a violation, set best_frame_index to null."""

# ---------------------------------------------------------------------------
# Pass 2 — Marcus's independent annotated-frame assessment (no Jamie, no raw video)
# ---------------------------------------------------------------------------

PASS2_MARCUS_SYSTEM = """\
You are Marcus Chen. You have 24 years as a senior construction safety manager and Chief \
Safety Officer. You are conducting your own independent review of a site camera video.

You have NOT seen any other inspector's report. You are watching the raw video yourself, \
and you also have the machine-annotated frames from the AI detection system (YOLO + SAM) \
showing bounding boxes and segmentation masks over flagged moments.

Watch the raw video with your own eyes first. Then cross-reference with the annotated frames.
Your job: write your independent professional assessment.
  • What do you observe in the raw video? Any workers, PPE, hazards, unsafe behaviour?
  • For each machine-flagged detection: do you agree based on what you see?
  • Did the machine miss anything you can see clearly in the raw footage?
  • Note detections that look like noise or false positives.

Write in plain English. Be direct. Do NOT produce JSON — a third pass will do that."""

PASS2_MARCUS_USER_TEMPLATE = """\
VIDEO OVERVIEW:
  Duration   : ~{duration:.1f}s  |  Camera: {camera_type}

{camera_guidance}

The raw video is above. Watch it with your own eyes first.
Then review the YOLO + SAM detection data and annotated frames below.

YOLO + SAM DETECTION SUMMARY:
{yolo_sam_data}

CONFIDENCE CALIBRATION:
  conf ≥ 0.70 = strong  |  0.40–0.69 = moderate  |  0.15–0.39 = weak  |  < 0.15 = noise

{annotated_frame_note}

Write your independent professional assessment — what you observed in the raw video, \
cross-referenced with what the machine flagged. Do NOT produce JSON — just your expert notes."""

# ---------------------------------------------------------------------------
# Pass 3 — Reconciler: Jamie's notes + Marcus's notes + YOLO data → final JSON
# ---------------------------------------------------------------------------

PASS3_SYSTEM = SYSTEM_PROMPT  # Marcus makes the final call using his full system prompt

PASS3_USER_TEMPLATE = """\
VIDEO OVERVIEW:
  Duration   : ~{duration:.1f}s  |  Camera: {camera_type}

{camera_guidance}

You have three sources of evidence. Reconcile them and produce the final safety assessment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 1 — JAMIE'S FIELD INSPECTION REPORT (raw video, no machine data):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pass1_notes}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 2 — YOUR OWN ASSESSMENT OF THE ANNOTATED FRAMES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pass2_notes}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 3 — YOLO + SAM DETECTION DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{yolo_sam_data}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECONCILIATION RULES — apply before writing your final output:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AGREEMENT (Jamie + you both flagged it, YOLO confirms) → high confidence, flag it.
  SPLIT (Jamie flagged, you did not, or vice versa) → state why one source is more reliable.
    Jamie saw the raw video — trust him on things visible only in motion or POV context.
    You saw the annotated frames — trust yourself on machine-flagged spatial violations.
  MACHINE ONLY (YOLO flagged, neither observer noted it) → conf ≥ 0.70 = flag with note;
    conf < 0.40 = discard unless you have a strong reason.
  OBSERVER ONLY (Jamie or you flagged it, YOLO missed it) → flag if visually clear.
    Machines miss things. Human observation of a clear violation is sufficient.
  CONFLICT (Jamie said present, you said absent, or vice versa) → state your final ruling
    and the reason. You are the CSO. Your word is final.

{annotated_frame_note}

CONFIDENCE CALIBRATION:
  conf ≥ 0.70 = strong  |  0.40–0.69 = moderate  |  0.15–0.39 = weak  |  < 0.15 = noise
  ONE entry per unique (worker, violation_type). No duplicates.

RULES REMINDER:
  Rule 1 — Every confirmed worker: hard hat + hi-vis vest + covered clothing + closed footwear.
  Rule 2 — Harness at ≥3m height without edge protection.
  Rule 3 — Guardrails at open excavation edges ≥3m deep.
  Rule 4 — No workers in excavator operating radius or blind spots.

Return a JSON object matching this exact schema:
{schema}

Output ONLY valid JSON. No explanation outside the JSON."""


# ---------------------------------------------------------------------------
# YOLO/SAM data formatter
# ---------------------------------------------------------------------------

def _format_yolo_sam_data(hazard_frames: list[dict], summary: dict) -> str:
    lines = []

    lines.append(f"  Frames analyzed : {summary.get('frames_processed', '?')}")
    lines.append(f"  Hazard frames   : {summary.get('frames_with_hazards', '?')}")
    lines.append(f"  Workers tracked : {summary.get('workers_tracked', '?')}")
    lines.append(f"  Classes seen    : {', '.join(summary.get('detected_classes', []))}")
    lines.append("")

    compliance = summary.get("worker_compliance", {})
    if compliance:
        lines.append("  Per-worker PPE status (accumulated across full video):")
        for tid, status in compliance.items():
            present = ", ".join(status["ppe_confirmed"]) or "none"
            missing = ", ".join(status["ppe_missing"]) or "none"
            tag = "COMPLIANT" if status["compliant"] else "VIOLATION"
            lines.append(f"    Worker #{tid} [{tag}]  present=[{present}]  missing=[{missing}]")
        lines.append("")

    hazard_counts = summary.get("hazard_counts", {})
    if hazard_counts:
        lines.append("  Hazard type counts (from YOLO):")
        for htype, count in hazard_counts.items():
            lines.append(f"    {htype}: {count} occurrences")
        lines.append("")

    if hazard_frames:
        lines.append("  Flagged frame timestamps (with YOLO confidence scores):")
        for frame in hazard_frames:
            ts = frame["timestamp_seconds"]
            # Include object confidences from raw detections for this frame
            det_by_class: dict = {}
            for d in frame.get("detections", []):
                det_by_class.setdefault(d["class_name"], []).append(d["confidence"])
            det_summary = "  ".join(
                f"{cls}={max(confs):.2f}" for cls, confs in det_by_class.items()
            )
            hazard_parts = []
            for h in frame.get("hazards", []):
                tid = h.get("track_id")
                w = f"(worker #{tid})" if tid is not None else ""
                # Pull the confidence of the objects driving this hazard
                obj_confs = [
                    f"{o['class']}={o['confidence']:.2f}"
                    for o in h.get("objects", [])
                ]
                conf_str = f"[{', '.join(obj_confs)}]" if obj_confs else ""
                hazard_parts.append(f"{h['hazard_type']} {w} {conf_str}".strip())
            lines.append(f"    t={ts:.2f}s | detections: {det_summary}")
            lines.append(f"             hazards: {', '.join(hazard_parts)}")

    return "\n".join(lines)


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
# Shared inference helper
# ---------------------------------------------------------------------------

def _run_inference(model, processor, messages: list[dict], max_new_tokens: int, label: str) -> str:
    """Tokenise messages, run generate, decode and return raw output text."""
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
    print(f"[vlm:{label}] input tokens: {inputs['input_ids'].shape[-1]}, max_new_tokens={max_new_tokens}")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"[vlm:{label}] output ({len(output)} chars): {output[:400]}{'...' if len(output) > 400 else ''}")
    return output


# ---------------------------------------------------------------------------
# Pass 1 — Generator: raw video, fresh eyes, free-form notes
# ---------------------------------------------------------------------------

def _run_pass1(
    video_path: Path,
    duration: float,
    cam_key: str,
    model,
    processor,
) -> str:
    pov_note = PASS1_POV_GLOVE_NOTE if cam_key == "POV" else ""
    user_text = PASS1_USER_TEMPLATE.format(
        duration    = duration,
        fps         = VIDEO_FPS,
        camera_type = cam_key,
        pov_glove_note = pov_note,
    )
    messages = [
        {"role": "system", "content": PASS1_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type":       "video",
                    "video":      video_path.as_uri(),
                    "fps":        VIDEO_FPS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]
    print(f"[vlm:pass1] running — raw video only, fresh-eyes observation")
    notes = _run_inference(model, processor, messages, MAX_NEW_TOKENS_P1, "pass1")
    print(f"[vlm:pass1] notes captured ({len(notes)} chars)")
    return notes


# ---------------------------------------------------------------------------
# Pass 2 — Marcus's independent annotated-frame assessment (no Jamie, no raw video)
# ---------------------------------------------------------------------------

def _run_pass2_marcus(
    video_path: Path,
    annotated_paths: list[Path],
    hazard_frames: list[dict],
    summary: dict,
    duration: float,
    cam_key: str,
    model,
    processor,
) -> str:
    annotated_blocks: list[dict] = []
    for i, ap in enumerate(annotated_paths):
        annotated_blocks.append({"type": "text", "text": f"[Annotated frame {i}]"})
        annotated_blocks.append({"type": "image", "image": ap.as_uri(), "max_pixels": MAX_PIXELS})

    annotated_note = PASS2_ANNOTATED_FRAME_NOTE.format(
        n=len(annotated_paths),
        n_minus_1=max(len(annotated_paths) - 1, 0),
    ) if annotated_paths else ""

    user_text = PASS2_MARCUS_USER_TEMPLATE.format(
        duration            = duration,
        camera_type         = cam_key,
        camera_guidance     = _CAMERA_GUIDANCE[cam_key],
        yolo_sam_data       = _format_yolo_sam_data(hazard_frames, summary),
        annotated_frame_note= annotated_note,
    )

    messages = [
        {"role": "system", "content": PASS2_MARCUS_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type":       "video",
                    "video":      video_path.as_uri(),
                    "fps":        VIDEO_FPS,
                    "max_pixels": MAX_PIXELS,
                },
                *annotated_blocks,
                {"type": "text", "text": user_text},
            ],
        },
    ]

    print(f"[vlm:pass2] Marcus reviewing raw video + annotated frames independently ({len(annotated_paths)} frames, no Jamie's notes)")
    notes = _run_inference(model, processor, messages, MAX_NEW_TOKENS_P2, "pass2")
    print(f"[vlm:pass2] Marcus notes captured ({len(notes)} chars)")
    return notes


# ---------------------------------------------------------------------------
# Pass 3 — Reconciler: Jamie + Marcus + YOLO → final JSON
# ---------------------------------------------------------------------------

def _run_pass3_reconciler(
    annotated_paths: list[Path],
    hazard_frames: list[dict],
    summary: dict,
    pass1_notes: str,
    pass2_notes: str,
    duration: float,
    cam_key: str,
    model,
    processor,
) -> VLMAssessment:
    annotated_blocks: list[dict] = []
    for i, ap in enumerate(annotated_paths):
        annotated_blocks.append({"type": "text", "text": f"[Annotated frame {i}]"})
        annotated_blocks.append({"type": "image", "image": ap.as_uri(), "max_pixels": MAX_PIXELS})

    annotated_note = PASS2_ANNOTATED_FRAME_NOTE.format(
        n=len(annotated_paths),
        n_minus_1=max(len(annotated_paths) - 1, 0),
    ) if annotated_paths else ""

    user_text = PASS3_USER_TEMPLATE.format(
        duration            = duration,
        camera_type         = cam_key,
        camera_guidance     = _CAMERA_GUIDANCE[cam_key],
        pass1_notes         = pass1_notes,
        pass2_notes         = pass2_notes,
        yolo_sam_data       = _format_yolo_sam_data(hazard_frames, summary),
        annotated_frame_note= annotated_note,
        schema              = json.dumps(VLMAssessment.model_json_schema(), indent=2),
    )

    messages = [
        {"role": "system", "content": PASS3_SYSTEM},
        {
            "role": "user",
            "content": [
                *annotated_blocks,
                {"type": "text", "text": user_text},
            ],
        },
    ]

    print(f"[vlm:pass3] Marcus reconciling all three sources → final JSON")
    raw_output = _run_inference(model, processor, messages, MAX_NEW_TOKENS, "pass3")

    print(f"[vlm:pass3] parsing JSON ...")
    raw = _extract_json(raw_output)
    assessment = VLMAssessment.model_validate(raw)
    print(f"[vlm:pass3] final: {len(assessment.hazards)} hazards, confidence={assessment.confidence}, no_hazards={assessment.no_hazards}")
    return assessment


# ---------------------------------------------------------------------------
# VLM inference orchestrator — three-pass adv-CoT
# ---------------------------------------------------------------------------

def run_vlm_on_video(
    video_path: Path,
    hazard_frames: list[dict],
    job_dir: Path,
    summary: dict,
    camera_source: str = "WALL_CAM",
) -> tuple[VLMAssessment, list[Path]]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[vlm] loading model ...")
    model, processor = _get_model()
    print(f"[vlm] model on device: {next(model.parameters()).device}")

    total_frames = (summary or {}).get("total_frames", 0)
    video_fps    = (summary or {}).get("video_fps", 25.0)
    duration     = hazard_frames[-1]["timestamp_seconds"] if hazard_frames else round(total_frames / max(video_fps, 1), 1)
    cam_key      = "POV" if "POV" in camera_source.upper() else "WALL_CAM"
    print(f"[vlm] video: {video_path} ({duration:.1f}s, {len(hazard_frames)} hazard frames, camera={cam_key})")

    # Collect annotated frames saved by the YOLO+SAM pipeline pass
    frames_dir = job_dir / "frames"
    annotated_paths: list[Path] = []
    for hf in hazard_frames:
        fp = hf.get("frame_path")
        if fp:
            full = frames_dir / fp
            if full.exists():
                annotated_paths.append(full)
    print(f"[vlm] {len(annotated_paths)} annotated frames available")

    # ── Pass 1: Jamie — fresh eyes on raw video ───────────────────────────────
    pass1_notes = _run_pass1(video_path, duration, cam_key, model, processor)

    # ── Pass 2: Marcus — independent review of raw video + annotated frames ──
    pass2_notes = _run_pass2_marcus(
        video_path, annotated_paths, hazard_frames, summary or {},
        duration, cam_key, model, processor,
    )

    # ── Pass 3: Marcus — reconciles all three sources → final JSON ────────────
    assessment = _run_pass3_reconciler(
        annotated_paths, hazard_frames, summary or {},
        pass1_notes, pass2_notes, duration, cam_key, model, processor,
    )

    return assessment, annotated_paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def assess_all_events(
    hazard_frames: list[dict],
    job_dir: Path,
    video_path: Path | None = None,
    summary: dict | None = None,
    gap_seconds: float = 3.0,   # kept for API compat, unused
    camera_source: str = "WALL_CAM",
) -> list[dict]:
    """Runs VLM on the full video with YOLO+SAM context, writes vlm_assessments.json."""
    out_file = job_dir / "vlm_assessments.json"

    if video_path is None or not video_path.exists():
        print("[vlm] video not available — skipping VLM")
        out_file.write_text("[]")
        return []

    print(f"[vlm] running full-video assessment ({len(hazard_frames)} hazard frames from YOLO, {VIDEO_FPS}fps sample — VLM always runs)")

    annotated_paths: list[Path] = []
    try:
        assessment, annotated_paths = run_vlm_on_video(video_path, hazard_frames, job_dir, summary or {}, camera_source)
    except Exception as exc:
        err_str = str(exc)
        if "ECC" in err_str or "uncorrectable" in err_str or "AcceleratorError" in err_str:
            print(f"[vlm] ECC error — resetting model and retrying ...")
            _reset_model()
            try:
                assessment, annotated_paths = run_vlm_on_video(video_path, hazard_frames, job_dir, summary or {}, camera_source)
            except Exception as retry_exc:
                print(f"[vlm] retry failed: {retry_exc}")
                out_file.write_text("[]")
                return []
        else:
            print(f"[vlm] error: {exc}")
            out_file.write_text("[]")
            return []

    _s = summary or {}
    _total_frames = _s.get("total_frames", 0)
    _video_fps    = _s.get("video_fps", 25.0)
    _fallback_dur = round(_total_frames / max(_video_fps, 1), 1)

    # Resolve each hazard's best_frame_index to an actual filename
    assessment_dict = assessment.model_dump()
    for hazard in assessment_dict["hazards"]:
        idx = hazard.get("best_frame_index")
        if idx is not None and 0 <= idx < len(annotated_paths):
            hazard["best_frame_path"] = annotated_paths[idx].name
            print(f"[vlm] hazard '{hazard['violation_type']}' best frame: index={idx}, file={hazard['best_frame_path']}")
        else:
            hazard["best_frame_path"] = None

    first = hazard_frames[0] if hazard_frames else {}
    result = [{
        "event": {
            "start_ts":                  hazard_frames[0]["timestamp_seconds"] if hazard_frames else 0.0,
            "end_ts":                    hazard_frames[-1]["timestamp_seconds"] if hazard_frames else _fallback_dur,
            "duration_seconds":          round(hazard_frames[-1]["timestamp_seconds"] - hazard_frames[0]["timestamp_seconds"], 3) if hazard_frames else _fallback_dur,
            "frame_count":               len(hazard_frames),
            "peak_yolo_severity":        "HIGH" if hazard_frames else "NONE",
            "yolo_hazard_types":         list({h["hazard_type"] for f in hazard_frames for h in f.get("hazards", [])}),
            "representative_frame_path": first.get("frame_path"),
            "representative_timestamp":  first.get("timestamp_seconds", 0.0),
        },
        "vlm":   assessment_dict,
        "error": None,
    }]

    out_file.write_text(json.dumps(result, indent=2))
    print(f"[vlm] done — {len(assessment.hazards)} hazards identified across full video")
    return result
