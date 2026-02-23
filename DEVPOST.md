# Construction Site Safety Intelligence Dashboard

## Inspiration

In 2023, construction workers in the U.S. suffered approximately 1,075 fatal injuries on the job — a rate of 9.6 deaths per 100,000 full-time workers, making construction the deadliest industry sector in the country. The majority of these deaths are preventable: fall protection alone is the single largest violation category cited by OSHA year after year.

What struck us most is the gap between *knowing* what's dangerous and *catching* it as it happens. Supervisors can't watch every worker on a sprawling site simultaneously. Existing monitoring solutions are either prohibitively expensive, require dedicated real-time operators, or only cover a narrow slice of the problem (e.g., just hard hat detection). There's no affordable, passive, continuous system that can watch multiple workers across a site, resolve their identities over time, and produce an actionable safety report without requiring a human to sit behind a screen all day.

We wanted to build something that a site foreman could deploy with commodity cameras and review at the end of a shift — a system that turns raw footage into structured, OSHA-aligned safety intelligence.

## What it does

The Construction Site Safety Intelligence Dashboard is an ML-powered end-of-shift safety reporting system for construction sites. It ingests video from two independent camera sources — **POV body-worn cameras** and **fixed wall-mounted cameras** — and runs them through a unified three-stage AI pipeline to produce per-worker safety reports.

Here's what the system catches:

- **PPE violations**: missing hard hats, safety vests, gloves, eye protection, and respirators, flagged per-worker with frame-level evidence
- **Ergonomic and musculoskeletal risks**: improper lifting, sustained awkward posture, overreach, and deep kneeling/squatting, scored using REBA-inspired joint angle analysis from pose estimation
- **Proximity hazards**: workers operating too close to heavy equipment like excavators, cranes, and forklifts
- **Behavioral violations**: zone breaches, ladder misuse, scaffolding violations, and unsafe behaviors
- **OSHA-aligned compliance checks**: violations are mapped to specific OSHA standards (1926.501, 1926.451, 1926.102, 1910.212, etc.) and scored by severity

Each worker gets a persistent identity across shifts. The dashboard generates per-worker reports with timestamped violation logs, compliant behavior logs, ergonomic risk scores, and PPE compliance summaries — all backed by annotated frame evidence that a supervisor can review.

## How we built it

The core architecture is a **three-stage pipeline** shared by both the posture/ergonomics stream and the PPE violation stream:

**Stage 1 — Primary Detection (Fine-tuned YOLO)**
We fine-tuned YOLO11 on construction-specific PPE datasets (Ultralytics Construction-PPE, Roboflow, SH17, and others) to detect workers, PPE items, missing PPE, heavy equipment, ladders, and scaffolding. For posture analysis, we use a YOLO pose model that extracts 17 COCO keypoints per worker. On the wall-cam side, we run YOLO with BoT-SORT tracking to maintain persistent track IDs across frames.

**Stage 2 — Refinement & Deduplication (SAM 3)**
YOLO bounding boxes are passed to SAM 3 (Segment Anything Model 3) for pixel-accurate segmentation masks. This stage refines noisy bounding boxes, generates worker-level masks for better visual evidence, and enables PPE-to-worker association by checking which worker's bounding box contains each PPE item's center point. Worker safety logs accumulate PPE detections over time — if a tracked worker hasn't been seen with a required item after sufficient frames, the system flags it.

**Stage 3 — Compliance Verification (Fine-tuned Qwen3-VL-8B-Instruct)**
This is where we tackled hallucination. We designed a **three-pass adversarial chain-of-thought** protocol:

1. *Pass 1 (Jamie — Generator)*: A "field inspector" persona reviews raw video only, no machine data, and writes free-form inspection notes. This provides a fresh-eyes observation signal.
2. *Pass 2 (Marcus — Discriminator)*: A "Chief Safety Officer" persona independently reviews the video plus YOLO/SAM annotated frames. No access to Pass 1 notes. This validates machine detections from a different angle.
3. *Pass 3 (Marcus — Reconciler)*: The same persona reconciles all three information sources — Jamie's notes, Marcus's independent assessment, and YOLO/SAM data — into a structured JSON assessment with hazard types, severity levels, explanations, and recommendations.

The VLM also performs **confidence calibration** using YOLO detection scores: high-confidence detections ($\geq 0.70$) are treated as strong priors, moderate ones ($0.40$–$0.69$) require visual verification, and weak signals ($< 0.40$) are only retained if independently confirmed. We optionally fine-tuned the VLM with a LoRA adapter for construction-specific understanding.

**Posture & Ergonomics Pipeline**
For POV footage, we extract joint angles from YOLO pose keypoints — trunk flexion, lateral lean, neck flexion, knee angle, arm raise, and elbow flexion — and compute a REBA-inspired combined risk score. Violations are typed as `AWKWARD_POSTURE` (trunk flexion $> 48°$), `OVERREACH` (arm elevation $> 65°$ with body twist), `MSD_HIGH_RISK` (combined score $\geq 8$), or `KNEELING_SQUATTING_LOW` (deep knee bend with forward lean). A minimum keypoint confidence gate of $0.65$ prevents false positives from poorly-detected poses.

**Backend & Database**
FastAPI serves the pipeline with async job processing. SQLite stores persistent worker identities, shifts, and safety events. Events are deduplicated per (worker, violation type) per shift to avoid over-counting. The backend pre-warms all models on startup to avoid cold-start OOM errors on GPU.

**Frontend Dashboard**
Built with Next.js 14 and TypeScript. The dashboard shows real-time job progress with percentage and stage indicators, per-worker PPE compliance cards, expandable violation logs with frame evidence, posture analysis summaries, and hazard breakdowns. The ingest page supports drag-and-drop video upload for both camera modes, with live polling of job status.

**Infrastructure**
Tested on 4× RTX PRO 6000 Blackwell GPUs. Long videos are chunked into 60-second segments for VLM processing to prevent context overflow. The system processes every 3rd frame by default, balancing coverage with throughput.

## Challenges we ran into

**VLM hallucination was our biggest enemy.** Early runs of the vision-language model would confidently report violations that didn't exist — a worker wearing a hard hat would be flagged as missing one, or a safe distance from equipment would be called a proximity hazard. We iterated through multiple prompting strategies before landing on the three-pass adversarial approach. The key insight was that having two independent "inspectors" assess the same scene and then reconciling their findings dramatically reduced false positives compared to a single-pass prompt.

**PPE-to-worker association is harder than it sounds.** A hard hat detection floating above a worker's bounding box doesn't automatically belong to that worker — especially when workers are clustered together. We had to implement center-point containment logic and temporal smoothing to reliably associate PPE items with the correct tracked worker.

**Posture threshold tuning was an iterative grind.** What counts as "awkward posture" versus normal construction work? A 45-degree trunk lean might be perfectly safe for someone picking up a tool, but dangerous if sustained. We spent significant time with our testing datasets (compliant vs. non-compliant clips) dialing in thresholds — trunk good max at $38°$, overreach at $65°$ arm raise, minimum confidence at $0.65$ — and still consider these works in progress.

**Re-identification in crowded scenes.** Maintaining consistent worker identity when people walk behind pillars, crouch behind equipment, or leave and re-enter frame is fundamentally hard. BoT-SORT handles short occlusions well, but long disappearances require appearance embedding matching, which we designed but haven't fully stress-tested in dense multi-worker scenarios.

**GPU memory management across three large models.** Running YOLO, SAM 3, and an 8B-parameter VLM concurrently required careful choreography — pre-warming models on startup, sequencing inference stages, and chunking video to keep memory within bounds. Early versions would OOM on the first job.

## Accomplishments that we're proud of

**The three-pass adversarial VLM architecture actually works.** Giving the model two independent "inspector" personas and then a reconciliation pass measurably reduced hallucinated violations compared to naive single-pass prompting. The structured JSON output with confidence levels, explanations, and recommendations makes the results auditable — a supervisor can see *why* the system flagged something, not just *that* it did.

**End-to-end pipeline from raw video to structured OSHA-mapped safety report.** You can upload a 20-minute construction video and get back per-worker violation logs with frame evidence, PPE compliance status, ergonomic risk scores, and severity-ranked findings — all mapped to specific OSHA standards. That's a complete workflow, not a demo.

**The dual-camera architecture covers complementary blind spots.** Wall cameras catch site-wide spatial hazards (proximity to equipment, zone breaches, missing PPE visible from a distance), while POV cameras catch what wall cameras can't — glove usage, hand safety, first-person posture, and ergonomic strain. Running both through the same three-stage pipeline keeps the codebase unified.

**REBA-inspired ergonomic scoring from pose estimation.** Translating raw YOLO keypoints into clinically-meaningful joint angles and then into typed musculoskeletal risk categories (overreach, awkward posture, MSD high risk) bridges the gap between computer vision output and occupational health language.

**A working, polished dashboard.** Not just a notebook — a real Next.js application with live job tracking, expandable event details, frame evidence rendering, and PPE compliance cards. A site supervisor could use this.

## What we learned

**Hallucination control is a first-class engineering problem, not an afterthought.** You can't just throw a VLM at safety-critical data and trust the output. The adversarial multi-pass approach — independent assessment, then reconciliation — is a pattern we'll carry forward to any high-stakes AI system.

**OSHA standards are surprisingly well-structured for machine consumption.** The violation categories map cleanly to visual detection tasks. Fall protection, PPE requirements, machine guarding, scaffolding rules — each one translates to a combination of object detection, spatial reasoning, and temporal analysis that a pipeline can implement.

**Pose estimation is powerful but noisy.** YOLO keypoints give you joint angles, but confidence varies wildly with camera angle, occlusion, and worker clothing. Gating on minimum keypoint confidence ($0.65$) was essential — without it, a partially-visible worker in a baggy jacket would trigger false posture violations constantly.

**End-of-shift processing is a pragmatic sweet spot.** Real-time monitoring sounds impressive but requires streaming infrastructure, edge compute, and low-latency networking. Processing video post-shift lets you use higher-quality models (like an 8B VLM), do multi-pass verification, and run on shared GPU instances — making it actually deployable on real construction sites without massive infrastructure investment.

**Building for construction means building for trust.** Every flagged violation needs to come with evidence — the annotated frame, the reasoning, the confidence level. Without that, a foreman will dismiss the system after the first false positive. The metadata JSON blobs we attach to every safety event (frame paths, YOLO detections, VLM explanations, PPE status) exist because auditability is a feature, not overhead.

## What's next for Construction Site Safety Intelligence Dashboard

**Full worker re-identification deployment.** The appearance embedding infrastructure is built — base64 embeddings stored per worker, cosine similarity matching, object permanence with state holding — but it hasn't been stress-tested in dense, real-world multi-worker footage. Getting this robust is the highest-leverage improvement: it's what makes violation *history* meaningful.

**Real-time streaming mode.** The pipeline currently processes uploaded video files. Adding RTSP/WebRTC ingestion would allow live monitoring with periodic batch analysis, keeping the high-quality VLM verification while reducing the feedback delay from hours to minutes.

**Site heatmaps and spatial analytics.** We have worker positions per frame from the wall-cam tracker. Aggregating these into occupancy heatmaps overlaid on site floorplans would show supervisors where workers spend time relative to hazard zones — information that's valuable for site layout planning, not just incident review.

**Expanded OSHA violation taxonomy.** The current system covers the major categories (PPE, fall protection, ergonomics, proximity, scaffolding, ladders), but OSHA's full cited standards list includes electrical, trenching, hazard communication, and more. Each one maps to a detection task we can implement.

**Integration with existing site management systems.** Construction sites already use tools like Procore, Autodesk Construction Cloud, and various EHS platforms. Exporting safety reports in standardized formats — or pushing events directly into these systems — would make adoption frictionless.

**Fine-tuning on site-specific data.** Every construction site has different lighting, worker uniforms, equipment types, and spatial layouts. Offering a calibration workflow where a site uploads a day of footage and the system auto-tunes detection thresholds would dramatically improve accuracy on deployment.
