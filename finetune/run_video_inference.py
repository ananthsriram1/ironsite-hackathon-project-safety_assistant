"""
Run base and/or fine-tuned model on a video clip with the Marcus Chen safety prompt.
Frames are sampled evenly across the video. Default is 8 frames; use --num-frames or
--every-n-seconds for fuller coverage (e.g. full video at 1 frame per second).
Compare: --compare runs the same clip with base model then fine-tuned adapter.

Usage:
  python run_video_inference.py --video "/workspace/Benchmarking Vids/clip_handsgloved.mov" --compare
  python run_video_inference.py --video "/workspace/Benchmarking Vids/clip_handsgloved.mov" --adapter /workspace/adapter1
  python run_video_inference.py --video long_video.mp4 --every-n-seconds 1   # 1 frame per second (full video)
  python run_video_inference.py --video clip.mov --num-frames 32             # 32 evenly spaced frames
"""
import argparse
import os
import sys
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

ADAPTER_PATH = "/workspace/adapter1"
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
NUM_FRAMES = 8
MAX_NEW_TOKENS = 512

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
        Q3: Are gloves present on both hands during manual work?
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

USER_QUESTION = (
    "Review this frame from the construction site video. "
    "1) How many persons do you detect? "
    "2) Are the workers wearing gloves? "
    "3) Note any safety violations you can confirm (PPE, fall protection, edge protection, excavator proximity). "
    "Respond concisely; you may use your reasoning stages internally."
)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ms for easy seeking (e.g. 0:01:23.45)."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:06.3f}"


def get_video_duration_sec(video_path: str) -> float:
    """Return video duration in seconds (OpenCV or ffprobe)."""
    try:
        import cv2
    except ImportError:
        pass
    else:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if total > 0 and fps > 0:
                return total / fps
    import subprocess
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    return 10.0


def _extract_frames_ffmpeg(video_path: str, num_frames: int) -> tuple[list[Image.Image], list[float]]:
    """Fallback when OpenCV cannot open the file (e.g. .mov on Linux). Uses ffmpeg."""
    import subprocess
    import tempfile
    import os

    # Get duration in seconds
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr or result.stdout}")
    try:
        duration_sec = float(result.stdout.strip())
    except ValueError:
        duration_sec = 10.0
    if duration_sec <= 0:
        duration_sec = 10.0

    # Evenly spaced timestamps
    if num_frames <= 1:
        timestamps = [0.0]
    else:
        timestamps = np.linspace(0, max(0, duration_sec - 0.1), num_frames).tolist()

    frames = []
    used_timestamps = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, t in enumerate(timestamps):
            out_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            cmd = [
                "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                "-frames:v", "1", "-q:v", "2", out_path,
            ]
            subprocess.run(cmd, capture_output=True, check=False)
            if os.path.exists(out_path):
                frames.append(Image.open(out_path).convert("RGB"))
                used_timestamps.append(t)

    if not frames:
        raise RuntimeError(f"ffmpeg could not extract frames from: {video_path}")
    return frames, used_timestamps


def extract_frames(video_path: str, num_frames: int = NUM_FRAMES) -> tuple[list[Image.Image], list[float]]:
    # Try OpenCV first (works for many formats)
    try:
        import cv2
    except ImportError:
        pass
    else:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if total <= 0:
                total = 1
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frames = []
            timestamps = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, bgr = cap.read()
                if not ret:
                    continue
                rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                frames.append(rgb)
                timestamps.append(i / fps)
            cap.release()
            if frames:
                return frames, timestamps

    # Fallback: ffmpeg (handles .mov and other codecs OpenCV may not support)
    try:
        return _extract_frames_ffmpeg(video_path, num_frames)
    except FileNotFoundError:
        raise RuntimeError(
            f"Cannot open video: {video_path}. "
            "OpenCV failed and ffmpeg is not installed. Install with: apt install ffmpeg"
        )


def run_frames(processor, model, frames, max_new_tokens=MAX_NEW_TOKENS):
    """Run model on each frame; return list of response strings."""
    messages_template = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_QUESTION},
            ],
        },
    ]
    results = []
    for i, frame in enumerate(frames):
        text = processor.apply_chat_template(messages_template, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[[frame]], text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        results.append(response)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="Path to video (e.g. .../clip.mov). Can also use --video.",
    )
    p.add_argument("--video", default=None, help="Path to video (alternative to positional video_path)")
    p.add_argument("--adapter", default=ADAPTER_PATH, help="Path to fine-tuned adapter folder")
    p.add_argument(
        "--num-frames",
        type=int,
        default=NUM_FRAMES,
        help="Number of frames to sample evenly across the video (default %(default)s). Increase for longer videos or fuller coverage.",
    )
    p.add_argument(
        "--every-n-seconds",
        type=float,
        default=None,
        metavar="N",
        help="Sample 1 frame every N seconds (overrides --num-frames). E.g. 1.0 = 1 fps, 2.0 = one every 2s. Use for full-video coverage.",
    )
    p.add_argument("--compare", action="store_true", help="Run with base model AND adapter, print both")
    p.add_argument("--output", "-o", default=None, help="Save results to this txt file (e.g. /workspace/results.txt)")
    args = p.parse_args()

    video = args.video or args.video_path
    if not video:
        p.error("Provide video path: pass it as first argument or use --video PATH")

    if not os.path.isfile(video):
        print(f"Error: Video file not found: {video}", file=sys.stderr)
        parent = os.path.dirname(video)
        if os.path.isdir(parent):
            print(f"\nContents of {parent}:", file=sys.stderr)
            for name in sorted(os.listdir(parent)):
                path = os.path.join(parent, name)
                kind = "dir " if os.path.isdir(path) else "file"
                print(f"  {kind}  {name}", file=sys.stderr)
        else:
            workspace = "/workspace"
            if os.path.isdir(workspace):
                print(f"\nContents of {workspace}:", file=sys.stderr)
                for name in sorted(os.listdir(workspace)):
                    path = os.path.join(workspace, name)
                    kind = "dir " if os.path.isdir(path) else "file"
                    print(f"  {kind}  {name}", file=sys.stderr)
        print("\nUpload the video to the instance or use the correct path.", file=sys.stderr)
        sys.exit(1)

    num_frames = args.num_frames
    if args.every_n_seconds is not None:
        duration_sec = get_video_duration_sec(video)
        num_frames = max(1, int(duration_sec / args.every_n_seconds))
        print(f"Video duration ~{duration_sec:.1f}s; sampling 1 frame every {args.every_n_seconds}s -> {num_frames} frames")

    print("Extracting frames...")
    frames, timestamps = extract_frames(video, num_frames)
    print(f"  Sampled {len(frames)} frames\n")

    # Processor: use base model for both runs so system prompt is handled the same
    print("Loading processor (base model)...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    processor.tokenizer.padding_side = "right"

    print("Loading base model...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.compare:
        # ---- Run 1: Base model only ----
        print("\n" + "=" * 60)
        print("RUN 1 — BASE MODEL (no fine-tuned adapter)")
        print("=" * 60)
        base_model.eval()
        base_results = run_frames(processor, base_model, frames)
        for i, resp in enumerate(base_results):
            ts = _format_timestamp(timestamps[i]) if i < len(timestamps) else ""
            print(f"\n--- Frame {i + 1}/{len(frames)} ({ts}) ---")
            print(resp)
        del base_results

        # ---- Run 2: Base + adapter ----
        print("\n" + "=" * 60)
        print("RUN 2 — FINE-TUNED ADAPTER")
        print("=" * 60)
        print("Loading adapter...")
        model_ft = PeftModel.from_pretrained(base_model, args.adapter)
        model_ft.eval()
        # Use processor from adapter so tokenizer matches training
        processor_ft = AutoProcessor.from_pretrained(args.adapter)
        processor_ft.tokenizer.padding_side = "right"
        ft_results = run_frames(processor_ft, model_ft, frames)
        for i, resp in enumerate(ft_results):
            ts = _format_timestamp(timestamps[i]) if i < len(timestamps) else ""
            print(f"\n--- Frame {i + 1}/{len(frames)} ({ts}) ---")
            print(resp)
        print("\nDone. Compare RUN 1 (base) vs RUN 2 (fine-tuned) above.")
    else:
        # Single run: with adapter
        print("Loading adapter...")
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model.eval()
        processor = AutoProcessor.from_pretrained(args.adapter)
        processor.tokenizer.padding_side = "right"
        print("Running on each frame (fine-tuned model)...\n")
        results = run_frames(processor, model, frames)

        lines = [
            f"Video: {video}",
            f"Model: fine-tuned adapter ({args.adapter})",
            f"Frames: {len(frames)}",
            "",
        ]
        for i, resp in enumerate(results):
            ts = _format_timestamp(timestamps[i]) if i < len(timestamps) else ""
            lines.append(f"--- Frame {i + 1}/{len(frames)} ({ts}) ---")
            lines.append(resp)
            lines.append("")
        lines.append("Done.")

        for line in lines:
            print(line)
        if args.output:
            out_path = args.output
            with open(out_path, "w") as f:
                f.write("\n".join(lines))
            print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
