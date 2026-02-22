# IronSite — Construction Site Safety Intelligence Dashboard

An ML-powered end-of-day safety reporting dashboard for construction sites. Pulls from two independent camera data sources — POV body-worn footage and fixed wall-mounted cameras — to tag workers, track OSHA violations, and surface per-worker safety reports.

---

## Problem Statement

- In 2023, U.S. construction workers experienced approximately **1,075 fatal injuries** on the job.
- The BLS measured a fatal work injury rate of **9.6 deaths per 100,000** full-time equivalent workers.
- Construction had the **most fatalities of any industry sector** in 2023.
- The majority of those who died were men (91.5%); women accounted for 8.5%.
- Most incidents are **preventable** — fall protection alone accounts for the single largest violation category cited by OSHA year over year.

The core gap: no affordable, passive, continuous monitoring system exists that can watch multiple workers across a site, resolve their identities over time, and produce an actionable report without requiring real-time human review.

---

## Approach

IronSite processes video at the end of a shift rather than in real-time, making it practical to deploy on commodity hardware. Two entirely separate data sources feed independently into the dashboard:

---

### Data Source 1 — POV (Body-Worn Cameras)

**Research basis:** [CWPV Dataset](https://figshare.com/articles/dataset/CWPV_A_Working_Postures_of_the_Construction_Working_Postures_Videos_dataset/27907818)

Workers wear cameras that capture first-person footage of their tasks throughout the shift. This footage is processed for musculoskeletal and ergonomic analysis:

- Pose estimation on first-person video to extract joint angles and body mechanics
- Detection of high-risk postures — improper lifting, sustained bending, overreach
- MSD (musculoskeletal disorder) risk scoring per worker per shift
- Flagging repetitive strain patterns over time

```
POV footage (per worker)
        │
        ▼
  Pose Estimation
  (joint angles, body mechanics)
        │
        ▼
  Ergonomic Risk Scoring
  (MSD risk, posture violations)
        │
        ▼
  Per-worker posture event log
```

---

### Data Source 2 — Fixed Wall-Mounted Cameras

**Research basis:** [PMC11367630](https://pmc.ncbi.nlm.nih.gov/articles/PMC11367630/)

Stationary IP cameras mounted around the site (~4m height, 1920×1080 at 24fps) provide a persistent spatial view of the work environment. This footage is processed for behavioral and spatial safety analysis:

- Worker detection and re-identification across camera angles and occlusions
- Zone mapping — who enters restricted or hazard areas
- Proximity detection — workers too close to machinery, edges, or moving equipment
- PPE detection — presence/absence of helmets, vests, eye protection
- Behavioral classification — safe vs. unsafe actions (unauthorized interventions, improper walkway use, etc.)

```
Wall-cam footage (site-wide)
        │
        ▼
  Person Detection + Re-ID
  (persistent worker identity across frames)
        │
        ▼
  Spatial Reasoning
  (zone mapping, proximity, occupancy)
        │
        ▼
  Behavioral Classification
  (PPE, safe/unsafe actions)
        │
        ▼
  Per-worker spatial + behavioral event log
```

---

### Worker Tagging & Object Permanence

Maintaining a consistent worker identity across an entire shift is a core technical problem — especially in the wall-cam system where workers move in and out of frame, get occluded, and reappear later. We address this with:

- **Lightweight identity DB** — stores per-worker appearance embeddings with a unique persistent ID per worker per site
- **Object permanence module** — when a worker disappears from frame, the system holds their last known state and re-associates them on reappearance using embedding similarity rather than requiring continuous tracking
- **Worker tagging** — each worker is assigned an ID at shift start; the DB persists across days so violation history accumulates over time

---

### Dashboard Aggregation

At end of shift, both pipelines write to a shared event store keyed by worker ID. The dashboard reads from this to generate reports — it does not need to know which camera source each event came from.

---

## OSHA Violation Coverage

The system flags detectable violations drawn from OSHA's top cited standards. Coverage depends on what each camera modality can observe:

| OSHA Standard | Violation | Detectable Via |
|---|---|---|
| 1926.501 | Fall protection — unguarded edges, open holes | Wall cam (zone + proximity) |
| 1926.503 | Fall protection training compliance | Worker history DB |
| 1926.451 | Scaffolding — improper use, missing guardrails | Wall cam (spatial) |
| 1926.102 | Eye/face protection (PPE absent) | Wall cam |
| 1910.212 | Machine guarding — worker too close to unguarded machinery | Wall cam (proximity) |
| Ergonomic / MSD risk | Improper lifting, sustained awkward posture | POV |
| Ladder safety | Improper ladder use, overreach | Wall cam |
| Respiratory protection | Missing respirator in flagged zones | Wall cam (PPE detection) |

Violations are scored by severity and frequency. Workers with repeated violations are surfaced prominently in the report.

---

## Dashboard Output

At the end of a shift, the dashboard generates a per-worker report:

- **Worker card** — persistent ID, camera coverage hours, thumbnail
- **Violation log** — timestamped list of flagged events with clip preview
- **Compliant behavior log** — instances of correct PPE use, safe zone adherence, proper posture
- **Ergonomic risk score** — aggregate MSD risk based on posture analysis from POV footage
- **Site heatmap** — spatial occupancy map showing where each worker spent time relative to hazard zones

---

## Data & Datasets

We used a substantial amount of video and derived data to develop and validate the posture and PPE pipelines.

**Research & reference datasets**
- **CWPV Dataset** — Working Postures of Construction Workers Videos. POV footage of construction workers performing real tasks, annotated for musculoskeletal posture analysis. [Figshare](https://figshare.com/articles/dataset/CWPV_A_Working_Postures_of_the_Construction_Working_Postures_Videos_dataset/27907818)
- **Video dataset for safe/unsafe behaviours (Önal & Dandıl)** — High-resolution video dataset (691 clips, 8 behaviour classes) from fixed IP cameras in a production facility; used as reference for wall-cam behavioural classification. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11367630/) · [Mendeley Data](https://data.mendeley.com/datasets/xjmtb22pff/1)
- **OSHA Top Violations 2024** — [osha.gov](https://www.osha.gov/top10citedstandards/)

**PPE & construction safety image/video datasets (training & reference)**
- **Ultralytics Construction-PPE** — 11 classes (helmet, vest, gloves, boots, goggles, person + no_helmet, no_goggles, no_gloves, no_boots). Used for PPE detector training and pipeline design. [Docs](https://docs.ultralytics.com/datasets/detect/construction-ppe/#dataset-structure)
- **Construction site safety (Roboflow)** — [Kaggle](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow/data)
- **PPE kit detection — construction site workers** — [Kaggle](https://www.kaggle.com/datasets/ketakichalke/ppe-kit-detection-construction-site-workers)
- **PPE dataset (YOLOv8)** — [Kaggle](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8)
- **PPE detection v1** — [Kaggle](https://www.kaggle.com/datasets/beyzakucuk/ppe-detection-v1)
- **SH17 dataset for PPE detection** — [Kaggle](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection/data)

**POV video used in this project**
- **Ironsite Hackathon / production dataset** — Long-form POV videos (e.g. 15–20+ minutes per clip) from construction and production scenarios (masonry, prep, standby, transit). Used for posture analysis, PPE analysis, and pipeline testing.
- **Annotated runs** — Posture- and PPE-tracked outputs (per-worker labels, compliant/non-compliant segments) generated from the above for validation and tuning.

**Derived datasets for validation**
- **Testing_Data_Posture** — Per-video folders of compliant and non-compliant clips (1–2+ minutes each) exported from the posture pipeline for human review and threshold tuning.
- **Testing_Data_PPE** — Same structure for PPE (compliant vs missing hard hat/vest/gloves).
- **Sample frames** — Frame-level exports (e.g. every 8 frames) from selected videos for frame-by-frame review and agreement/disagreement analysis.

**Other video**
- Short clips (youtube) used for demos and additional posture/PPE testing.

Overall, the pipeline has been run on many hours of POV footage across multiple videos; the testing datasets and sample frames support iterative refinement of posture and PPE logic.

---

## Running the Application

**Prerequisites:** Python 3.10+, [Bun](https://bun.sh), a GPU instance with CUDA (tested on 4× RTX PRO 6000 Blackwell)

**Model weights required** (not committed — place before running):
- `backend/models/last.pt` — fine-tuned YOLO11 PPE detector
- `/workspace/models/sam3.pt` — SAM3 segmentation model
- `/workspace/models/qwen3-vl` — Qwen3-VL-8B-Instruct base model
- `/workspace/models/adapter1` — fine-tuned LoRA adapter (optional, used if present)

**Terminal 1 — backend (one-time setup):**
```bash
cd backend
bash setup.sh
```

**Terminal 1 — backend (start server):**
```bash
cd backend
source venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,3 uvicorn main:app --host 0.0.0.0 --port 8000
```

Starts FastAPI at `http://localhost:8000` with a SQLite database.

Confirm it's up: `GET http://localhost:8000/health` → `{ "status": "ok" }`

**Terminal 2 — frontend:**
```bash
cd code
bun dev
```

Frontend at `http://localhost:3000`. The dashboard shows a green indicator when the backend is reachable.

**For help running the repo,** contact: [asriram2@terpmail.umd.edu](mailto:asriram2@terpmail.umd.edu), [rajveer@terpmai.umd.edu](mailto:rajveer@umd.edu), [nmokaria@terpmail.umd.edu](mailto:nmokaria@terpmail.umd.edu).

---

## Open Problems

- Exact violation taxonomy — finalized after both pipelines are prototyped
- Re-ID accuracy in crowded scenes within the wall-cam system
- False positive rate for PPE detection under variable lighting
- Defining thresholds for "good safety behavior" vs. neutral behavior
- How much signal POV footage gives when camera angle is obstructed mid-task
