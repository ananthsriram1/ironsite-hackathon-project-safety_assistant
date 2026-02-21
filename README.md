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

## Data Sources

- **CWPV Dataset** — Working Postures of Construction Workers Videos. POV footage of construction workers performing real tasks, annotated for musculoskeletal posture analysis. [Figshare](https://figshare.com/articles/dataset/CWPV_A_Working_Postures_of_the_Construction_Working_Postures_Videos_dataset/27907818)
- **Wall-camera safety research** — IP camera surveillance methodology for classifying safe/unsafe worker behaviors from fixed overhead cameras. [PMC11367630](https://pmc.ncbi.nlm.nih.gov/articles/PMC11367630/)
- **OSHA Top Violations 2024** — [osha.gov](https://www.osha.gov/top10citedstandards/)

---

## Running the Application

No cloud credentials needed. Everything runs locally.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) + [Bun](https://bun.sh)

```bash
# one-time setup
cp code/.env.example code/.env.local
```

**Terminal 1 — backend (API + ML + database):**
```bash
cd backend
docker compose up
```

Starts FastAPI at `http://localhost:8000` with a SQLite database. Edits to backend files hot-reload automatically.

Confirm it's up: `GET http://localhost:8000/health` → `{ "status": "ok" }`

**Terminal 2 — frontend:**
```bash
cd code
bun dev
```

Frontend at `http://localhost:3000`. The dashboard shows a green indicator when the backend is reachable.

---

## Open Problems

- Exact violation taxonomy — finalized after both pipelines are prototyped
- Re-ID accuracy in crowded scenes within the wall-cam system
- False positive rate for PPE detection under variable lighting
- Defining thresholds for "good safety behavior" vs. neutral behavior
- How much signal POV footage gives when camera angle is obstructed mid-task
