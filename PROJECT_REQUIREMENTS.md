# IronSite — Project Requirements

Tracks schema, API contracts, and frontend spec. Update as the ML pipeline matures and violation taxonomy gets finalized.

---

## Status

| Area | Status |
|---|---|
| DB Schema (SQLite) | Done |
| Enums | Done |
| FastAPI Backend | Done (stubs) |
| Next.js Frontend | Done (shell) |
| Worker Identification — POV | Not started |
| Worker Identification — Wall Cam | Not started |
| Object Permanence (Wall Cam) | Not started |
| ML Pipeline (POV) | Not started |
| ML Pipeline (Wall Cam) | Not started |

---

## Tech Stack

| Layer | Choice |
|---|---|
| Database | SQLite via SQLAlchemy (`backend/data/ironsite.db`) |
| Backend | FastAPI + Uvicorn, running in Docker |
| ML Pipeline | Python — added to `backend/main.py` as pipelines are built |
| Frontend | Next.js + Tailwind (`/code`) |
| Frontend → Backend | `fetch` via `code/lib/api.ts` → `http://localhost:8000` |

**Local ports:**
```
Next.js frontend   →  http://localhost:3000   (bun dev)
FastAPI backend    →  http://localhost:8000   (docker compose up)
```

No cloud credentials required. Clone and run.

---

## Running

```bash
# Terminal 1
cd backend && docker compose up

# Terminal 2
cp code/.env.example code/.env.local   # first time only
cd code && bun dev
```

---

## Enums

Defined in `backend/database.py` as Python enums, stored as strings in SQLite.

### `CameraSource`
```
POV         # body-worn camera → musculoskeletal/posture pipeline
WALL_CAM    # fixed mounted camera → spatial/behavioral pipeline
```

### `EventCategory`
```
VIOLATION   # unsafe behavior or posture detected
COMPLIANT   # confirmed safe behavior
```

### `ViolationType`
Stored as plain string in `safety_events.violation_type`. TBD as ML pipeline is built — starting set:
```
# Wall cam
FALL_PROTECTION_MISSING
PPE_MISSING
ZONE_BREACH
PROXIMITY_HAZARD
SCAFFOLD_VIOLATION
LADDER_MISUSE
BEHAVIORAL_UNSAFE

# POV
IMPROPER_LIFT
AWKWARD_POSTURE
MSD_HIGH_RISK
REPETITIVE_STRAIN

# Either
COMPLIANT_BEHAVIOR
```

### `Severity`
```
LOW | MEDIUM | HIGH | CRITICAL
```

### `WorkerStatus`
```
ACTIVE | INACTIVE
```

### `ShiftStatus`
```
IN_PROGRESS | COMPLETED
```

---

## Database Schema

Defined in `backend/database.py`. SQLite file lives at `backend/data/ironsite.db` (Docker volume, persists between restarts).

### `sites`
```
id           TEXT  PK
name         TEXT
location     TEXT
created_at   DATETIME
```

### `devices`
Maps physical POV camera units to a site. The per-shift worker binding is on `shifts.pov_device_id`.
```
id           TEXT  PK
label        TEXT              # e.g. "Cam-03"
site_id      TEXT  FK → sites
created_at   DATETIME
```

### `workers`
Persistent worker identity. Survives across shifts and days.
```
id                    TEXT  PK
site_id               TEXT  FK → sites
display_name          TEXT  nullable
appearance_embedding  TEXT  nullable   # base64 — wall-cam re-ID
face_embedding        TEXT  nullable   # base64 — optional face recognition
status                TEXT             # WorkerStatus enum
created_at            DATETIME
last_seen_at          DATETIME  nullable
total_violations      INT  default 0   # denormalized
total_compliant       INT  default 0   # denormalized
avg_msd_risk          FLOAT  default 0.0  # rolling avg from POV shifts
```

### `shifts`
One row per worker per day.
```
id               TEXT  PK
worker_id        TEXT  FK → workers
site_id          TEXT  FK → sites
date             TEXT              # "YYYY-MM-DD"
started_at       DATETIME
ended_at         DATETIME  nullable
status           TEXT              # ShiftStatus enum
pov_device_id    TEXT  FK → devices  nullable  # which POV cam this worker wore today
msd_risk_score   FLOAT  nullable   # set when POV processing completes
violation_count  INT  default 0    # updated on each event write
compliant_count  INT  default 0    # updated on each event write
```

### `safety_events`
Core event log. One row per flagged moment from either pipeline.
```
id               TEXT  PK
shift_id         TEXT  FK → shifts
worker_id        TEXT  FK → workers
camera_source    TEXT              # CameraSource enum
event_category   TEXT              # EventCategory enum
violation_type   TEXT  nullable    # ViolationType string
severity         TEXT  nullable    # Severity enum, null for COMPLIANT
video_timestamp  FLOAT             # seconds into source video
clip_path        TEXT  nullable    # local path to extracted clip
metadata_json    TEXT  nullable    # JSON blob for pipeline-specific extras
created_at       DATETIME
```

---

## Worker Identification & Tagging

### POV — Device-to-Worker Binding

The camera is worn by one worker. Identity problem: which worker is wearing which camera today?

**Flow:**
1. Worker checks in via dashboard → selects their name + POV device label
2. `POST /workers/checkin` creates a shift with `pov_device_id` set
3. POV pipeline receives video labeled with device ID → queries `GET /devices?site_id=` to resolve `device_id → worker_id + shift_id`
4. All events from that video are tagged against that worker

### Wall Cam — Appearance-Based Re-ID + Object Permanence

Multiple workers visible simultaneously, none pre-labeled.

**At video start:**
- `GET /workers?site_id=` fetches all known workers with embeddings → seeds local in-memory registry
- All matching is done in-memory (cosine similarity) — no per-frame DB calls

**Per detection:**
```
Bounding box crop → re-ID encoder → embedding
        │
        ├── similarity > threshold → existing worker_id
        └── no match → temp ID → POST /workers/register at end of video
```

**Object permanence:**
- When a track disappears, hold last known state in memory (worker_id + embedding + position)
- On reappearance: re-ID check against held state → re-assign same worker_id if match
- Time window TBD (~5–10 min before giving up on a lost track)

**DB is read once at video start, written once at video end** — no per-frame Firestore/DB calls.

---

## API Endpoints

FastAPI at `http://localhost:8000`. Defined in `backend/main.py`.

### Health
```
GET  /health
→ { "status": "ok" }
```

### Sites
```
POST /sites               create a site
GET  /sites               list all sites
```

### Devices
```
POST /devices             register a POV device
GET  /devices?site_id=    list devices for a site
```

### Workers
```
POST /workers/register    register new worker (called by wall-cam re-ID)
POST /workers/checkin     link worker to POV device, open shift
GET  /workers?site_id=    list workers (also seeds wall-cam re-ID registry)
GET  /workers/{id}        get single worker with stats
```

### Shifts
```
GET   /shifts/{id}          get shift + all its events
PATCH /shifts/{id}/close    mark completed, set msd_risk_score
```

### Events
```
POST /events/ingest         write a safety event (called by ML pipeline)
GET  /events                list events, filterable by worker_id / shift_id / event_category
```

### Video Processing
```
POST /process/pov           submit POV video → starts processing job
POST /process/wall-cam      submit wall-cam video → starts processing job
GET  /jobs/{job_id}         poll job status
```

---

## Frontend Pages

Defined in `code/app/`. All data via `code/lib/api.ts` → `http://localhost:8000`.

### `/` — Site Dashboard
- Per-worker cards: name, violations today, MSD score
- Top violations of the day
- Backend connection status indicator

### `/workers` — Worker Registry
- Table: name, status, lifetime violations, last seen

### `/workers/[workerId]` — Worker Detail
- Identity card + stats
- Shift history table
- Violation breakdown by type
- MSD risk trend over time

### `/shifts/[shiftId]` — Shift Report
- Summary: date, worker, duration, MSD score
- Violation log: timestamped, type, severity, camera source
- Compliant behavior log
- POV vs. wall-cam event breakdown

### `/ingest` — Video Upload
- Upload form: POV or wall-cam, file picker, worker/device/site fields
- Recent jobs list with status polling

---

## Open Items

- [ ] Finalize `ViolationType` values once ML pipeline is scoped
- [ ] Choose re-ID model for wall-cam (OSNet / torchreid) and fix embedding dimensions
- [ ] Tune cosine similarity threshold (~0.75 starting point) in wall-cam pipeline
- [ ] Define object permanence time window for lost tracks
- [ ] Site heatmap data format (zones, coordinate system)
- [ ] `clip_path` — local filesystem path for now, define folder structure
