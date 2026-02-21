# IronSite — Project Requirements

Tracks schema, API contracts, and frontend spec. Update as the ML pipeline matures and violation taxonomy gets finalized.

---

## Status

| Area | Status |
|---|---|
| Firestore Schema | Draft |
| Enums | Draft |
| Cloud Functions | Draft |
| Frontend Pages | Draft |
| ML Pipeline (POV) | Not started |
| ML Pipeline (Wall Cam) | Not started |
| Worker Re-ID / Object Permanence | Not started |

---

## Firebase Project

```
Project ID:     ironsite-hackathon
Auth Domain:    ironsite-hackathon.firebaseapp.com
Storage Bucket: ironsite-hackathon.firebasestorage.app
App ID:         1:1616514847:web:a8f51c3c4227a514242a2a
```

**Services used:**
- Firestore — primary database (workers, shifts, events)
- Firebase Storage — video uploads and extracted clips
- Cloud Functions — ML pipeline ingest endpoints, report generation
- Analytics — already initialized

---

## Tech Stack

| Layer | Choice |
|---|---|
| Database | Firestore |
| File Storage | Firebase Storage |
| Backend Logic | Cloud Functions (Python or Node.js) |
| ML Pipeline | Python scripts → calls Cloud Functions to write events |
| Frontend | Next.js + Tailwind |
| Firebase SDK | firebase-js-sdk (web) |

---

## Enums

Stored as string literals in Firestore documents.

### `CameraSource`
```
POV         # body-worn camera, musculoskeletal/posture pipeline
WALL_CAM    # fixed mounted camera, spatial/behavioral pipeline
```

### `EventCategory`
```
VIOLATION   # unsafe behavior or posture detected
COMPLIANT   # confirmed safe behavior (good practice logging)
```

### `ViolationType`
TBD as ML pipeline is built — starting set:
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
LOW
MEDIUM
HIGH
CRITICAL
```

### `WorkerStatus`
```
ACTIVE
INACTIVE
```

### `ShiftStatus`
```
IN_PROGRESS
COMPLETED
```

---

## Firestore Schema

### Collection: `sites`

```
sites/{siteId}
├── name:        string
├── location:    string
└── createdAt:   timestamp
```

---

### Collection: `workers`

Persistent worker identity. One document per worker, survives across shifts and days.

```
workers/{workerId}
├── siteId:         string (ref → sites)
├── displayName:    string | null
├── embedding:      string | null   # base64 or Storage path — for re-ID
├── status:         WorkerStatus
├── createdAt:      timestamp
└── stats:          map             # denormalized, updated on event write
    ├── totalViolations:    number
    ├── totalCompliant:     number
    └── avgMsdRisk:         number  # rolling average from POV shifts
```

> `embedding` may move to Firebase Storage if size is a concern. The `stats` map is denormalized so the dashboard can read worker cards without aggregating events.

---

### Collection: `shifts`

One document per worker per day.

```
shifts/{shiftId}
├── workerId:         string (ref → workers)
├── siteId:           string (ref → sites)
├── date:             string          # "YYYY-MM-DD"
├── startedAt:        timestamp
├── endedAt:          timestamp | null
├── status:           ShiftStatus
├── msdRiskScore:     number | null   # 0.0–1.0, set when POV processing completes
├── violationCount:   number          # updated on each event write
└── compliantCount:   number          # updated on each event write
```

---

### Collection: `safetyEvents`

Core event log. One document per flagged moment from either pipeline.

```
safetyEvents/{eventId}
├── shiftId:          string (ref → shifts)
├── workerId:         string (ref → workers)
├── cameraSource:     CameraSource
├── eventCategory:    EventCategory
├── violationType:    ViolationType
├── severity:         Severity | null   # null for COMPLIANT events
├── videoTimestamp:   number            # seconds into source video
├── clipPath:         string | null     # Firebase Storage path to extracted clip
└── metadata:         map | null        # pipeline-specific extras
    # wall cam examples:
    #   confidence: 0.91
    #   zone: "scaffold_level_2"
    # pov examples:
    #   jointAngles: { spine: 47, knee: 12 }
    #   riskContribution: 0.34
└── createdAt:        timestamp
```

**Indexes needed:**
- `workerId` + `createdAt` — worker history queries
- `shiftId` + `eventCategory` — shift report breakdown
- `siteId` + `date` (via shift join) — daily site reports

---

## Cloud Functions

These are the backend callable surfaces. ML pipeline Python scripts call the HTTP functions to write data. The frontend calls Firestore SDK directly for most reads.

---

### `registerWorker` — HTTP POST
Called by the re-ID module when a new worker is detected for the first time.

Request:
```json
{
  "siteId": "string",
  "displayName": "string | null",
  "embedding": "base64 string"
}
```

Response:
```json
{ "workerId": "string" }
```

---

### `openShift` — HTTP POST
Called at start of processing for a given worker+day.

Request:
```json
{
  "workerId": "string",
  "siteId": "string",
  "date": "YYYY-MM-DD"
}
```

Response:
```json
{ "shiftId": "string" }
```

---

### `closeShift` — HTTP POST
Called when all video for a shift has been processed.

Request:
```json
{
  "shiftId": "string",
  "msdRiskScore": "number | null"
}
```

Sets `status: COMPLETED`, writes `endedAt`, sets `msdRiskScore`.

---

### `ingestEvent` — HTTP POST
Called by ML pipeline scripts (POV or wall-cam) to write a single safety event.

Request:
```json
{
  "shiftId": "string",
  "workerId": "string",
  "cameraSource": "POV | WALL_CAM",
  "eventCategory": "VIOLATION | COMPLIANT",
  "violationType": "string",
  "severity": "string | null",
  "videoTimestamp": 142.5,
  "clipPath": "storage/path/to/clip.mp4 | null",
  "metadata": {}
}
```

Side effects:
- Writes to `safetyEvents`
- Increments `violationCount` or `compliantCount` on the parent shift
- Updates `workers/{workerId}/stats`

Response:
```json
{ "eventId": "string" }
```

---

### `processVideoUpload` — Storage Trigger
Fires automatically when a video is uploaded to Storage.

Trigger path: `uploads/{siteId}/{type}/{filename}`
- `type` is `pov` or `wall-cam`

Behavior:
- Kicks off the appropriate ML pipeline (TBD — likely a Cloud Run job or Pub/Sub message)
- Creates a processing job record

---

### `getShiftReport` — HTTP GET (or Firestore client-side)
Assembles the end-of-day report for a shift. Can be done client-side with two Firestore reads (shift doc + events query) — only needs to be a Cloud Function if report generation becomes expensive.

---

## Frontend Pages

### `/` — Site Dashboard
- Per-worker cards: name/ID, today's shift status, violation count, risk score badge
- Top violations of the day (aggregated across all workers)
- Site heatmap placeholder (wall-cam spatial output, TBD)
- Filter by date

### `/workers` — Worker Registry
- Table: display name, ID, status, lifetime violations, last active
- Sort by violation count, filter by status

### `/workers/[workerId]` — Worker Detail
- Identity card: name, ID, first seen, current status
- Shift history table: date, violations, compliant count, MSD score
- Lifetime violation breakdown by type (bar chart)
- MSD risk trend over time (line chart, POV data)

### `/shifts/[shiftId]` — Shift Report
- Summary: date, worker, duration, MSD score
- Violation log: timestamped, type, severity, camera source, clip preview
- Compliant behavior log
- Source breakdown: how many events from POV vs. wall cam

### `/ingest` — Video Upload
- Upload form: POV or wall-cam toggle, file picker, worker/site assignment
- Recent uploads list with processing status (polling Storage trigger job state)
- Error display for failed jobs

---

## Open Items

- [ ] Finalize `ViolationType` enum once ML pipeline is scoped
- [ ] Decide: embedding stored in Firestore doc (base64) or Firebase Storage path
- [ ] Decide: `getShiftReport` as Cloud Function or assembled client-side
- [ ] Define Storage folder structure for raw uploads vs. extracted clips
- [ ] Firestore security rules (open for hackathon, lock down after)
- [ ] Auth — any login needed for hackathon scope?
- [ ] Site heatmap data format (zones, coordinate system)
- [ ] Cloud Run vs. direct Python execution for ML pipeline jobs
