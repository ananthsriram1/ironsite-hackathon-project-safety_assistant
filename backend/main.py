import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Persistent directory for job outputs (frames, crops, detections)
# Defaults to ./jobs/ next to main.py so frames survive server restarts
JOBS_DIR = Path(os.getenv("JOBS_DIR", Path(__file__).parent / "jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import (
    CameraSource, EventCategory, SafetyEvent, SessionLocal, Severity,
    Shift, ShiftStatus, Site, Worker, init_db,
)
from pipeline.wall_cam import run_wall_cam_pipeline


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("ML service started — SQLite ready")
    yield

app = FastAPI(title="IronSite ML Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
jobs: dict[str, dict] = {}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

@app.post("/sites", status_code=201)
def create_site(name: str, location: str = "", db: Session = Depends(get_db)):
    site = Site(name=name, location=location)
    db.add(site)
    db.commit()
    db.refresh(site)
    return {"id": site.id, "name": site.name}


@app.get("/sites")
def list_sites(db: Session = Depends(get_db)):
    return [{"id": s.id, "name": s.name, "location": s.location} for s in db.query(Site).all()]


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

@app.post("/workers/register", status_code=201)
def register_worker(
    site_id: str,
    display_name: str = "",
    appearance_embedding: str = "",
    db: Session = Depends(get_db),
):
    worker = Worker(
        site_id=site_id,
        display_name=display_name or None,
        appearance_embedding=appearance_embedding or None,
    )
    db.add(worker)
    db.commit()
    db.refresh(worker)
    return {"worker_id": worker.id}



@app.get("/workers")
def list_workers(site_id: str, db: Session = Depends(get_db)):
    workers = db.query(Worker).filter(Worker.site_id == site_id).all()
    return [
        {
            "worker_id": w.id,
            "display_name": w.display_name,
            "status": w.status,
            "total_violations": w.total_violations,
            "total_compliant": w.total_compliant,
            "avg_msd_risk": w.avg_msd_risk,
            "last_seen_at": w.last_seen_at,
        }
        for w in workers
    ]


@app.get("/workers/{worker_id}")
def get_worker(worker_id: str, db: Session = Depends(get_db)):
    w = db.query(Worker).filter(Worker.id == worker_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Worker not found")
    return {
        "worker_id": w.id,
        "display_name": w.display_name,
        "site_id": w.site_id,
        "status": w.status,
        "total_violations": w.total_violations,
        "total_compliant": w.total_compliant,
        "avg_msd_risk": w.avg_msd_risk,
        "created_at": w.created_at,
        "last_seen_at": w.last_seen_at,
    }


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------

@app.get("/shifts/{shift_id}")
def get_shift(shift_id: str, db: Session = Depends(get_db)):
    shift = db.query(Shift).filter(Shift.id == shift_id).first()
    if not shift:
        raise HTTPException(status_code=404, detail="Shift not found")
    events = db.query(SafetyEvent).filter(SafetyEvent.shift_id == shift_id).all()
    return {
        "shift_id": shift.id,
        "worker_id": shift.worker_id,
        "date": shift.date,
        "status": shift.status,
        "msd_risk_score": shift.msd_risk_score,
        "violation_count": shift.violation_count,
        "compliant_count": shift.compliant_count,
        "events": [_fmt_event(e) for e in events],
    }



# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@app.get("/events")
def list_events(
    worker_id: str = None,
    shift_id: str = None,
    event_category: EventCategory = None,
    db: Session = Depends(get_db),
):
    q = db.query(SafetyEvent)
    if worker_id:
        q = q.filter(SafetyEvent.worker_id == worker_id)
    if shift_id:
        q = q.filter(SafetyEvent.shift_id == shift_id)
    if event_category:
        q = q.filter(SafetyEvent.event_category == event_category)
    return [_fmt_event(e) for e in q.all()]


# ---------------------------------------------------------------------------
# Video processing jobs
# ---------------------------------------------------------------------------

@app.post("/process/pov")
async def process_pov(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    worker_id: str = Form(...),
    site_id: str = Form(...),
    date: str = Form(...),
):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "source.mp4"
    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    jobs[job_id] = {"status": "PROCESSING", "events_written": 0, "error": None}
    background_tasks.add_task(_run_detection_job, job_id, job_dir, video_path, site_id, date, worker_id, CameraSource.POV)
    print(f"[POV] job={job_id} worker={worker_id} site={site_id} date={date}")
    return {"job_id": job_id, "status": "PROCESSING"}


@app.post("/process/wall-cam")
async def process_wall_cam(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    worker_id: str = Form(...),
    site_id: str = Form(...),
    date: str = Form(...),
):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "source.mp4"
    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    jobs[job_id] = {"status": "PROCESSING", "events_written": 0, "error": None}
    background_tasks.add_task(_run_detection_job, job_id, job_dir, video_path, site_id, date, worker_id, CameraSource.WALL_CAM)
    print(f"[WALL-CAM] job={job_id} worker={worker_id} site={site_id} date={date}")
    return {"job_id": job_id, "status": "PROCESSING"}


def _run_detection_job(
    job_id: str,
    job_dir: Path,
    video_path: Path,
    site_id: str,
    date: str,
    worker_id: str,
    camera_source: CameraSource,
):
    import traceback
    print(f"[{camera_source}] {job_id} — start, video={video_path}, size={video_path.stat().st_size if video_path.exists() else 0}")
    db = SessionLocal()
    try:
        # Find or create worker
        worker = db.query(Worker).filter(Worker.id == worker_id).first()
        if not worker:
            # Try by display_name for convenience
            worker = db.query(Worker).filter(Worker.display_name == worker_id).first()
        if not worker:
            worker = Worker(site_id=site_id, display_name=worker_id)
            db.add(worker)
            db.commit()
            db.refresh(worker)
            print(f"[{camera_source}] {job_id} — auto-created worker {worker.id} ({worker_id})")

        # Create shift — use resolved worker.id, not the raw form string
        shift = Shift(worker_id=worker.id, site_id=site_id, date=date)
        db.add(shift)
        db.commit()
        db.refresh(shift)
        print(f"[{camera_source}] {job_id} — shift created: {shift.id}")

        # Warm up models before pipeline (loads into GPU memory once)
        print(f"[{camera_source}] {job_id} — loading models")
        from pipeline.wall_cam import _get_yolo, _get_sam
        _get_yolo(); _get_sam()
        print(f"[{camera_source}] {job_id} — models loaded, running pipeline")

        summary, hazard_events, vlm_results = run_wall_cam_pipeline(
            video_path=video_path,
            job_dir=job_dir,
            site_id=site_id,
            date=date,
        )
        print(f"[{camera_source}] {job_id} — pipeline done: {summary}")

        # Write events to DB
        # If VLM ran: use VLM-confirmed hazards (higher quality)
        # If VLM is disabled (vlm_results=[]): fall back to raw YOLO hazard frames
        events_written = 0

        if vlm_results:
            for result in vlm_results:
                if result["error"] or not result["vlm"]:
                    continue
                ev   = result["event"]
                vlm  = result["vlm"]
                for hazard in vlm["hazards"]:
                    sev_str  = hazard.get("severity", "MEDIUM")
                    severity = Severity[sev_str] if sev_str in Severity.__members__ else Severity.MEDIUM
                    event = SafetyEvent(
                        shift_id=shift.id,
                        worker_id=worker.id,
                        camera_source=camera_source,
                        event_category=EventCategory.VIOLATION,
                        violation_type=hazard.get("violation_type"),
                        severity=severity,
                        video_timestamp=ev["representative_timestamp"],
                        clip_path=ev.get("representative_frame_path"),
                        metadata_json=json.dumps({
                            "explanation":       hazard.get("explanation"),
                            "recommendation":    hazard.get("recommendation"),
                            "scene_summary":     vlm.get("scene_summary"),
                            "reasoning":         vlm.get("reasoning"),
                            "confidence":        vlm.get("confidence"),
                            "ppe_per_worker":    vlm.get("ppe_per_worker"),
                            "event_start_ts":    ev["start_ts"],
                            "event_end_ts":      ev["end_ts"],
                            "event_duration_s":  ev["duration_seconds"],
                            "event_frame_count": ev["frame_count"],
                            "yolo_hazard_types": ev.get("yolo_hazard_types"),
                        }),
                    )
                    db.add(event)
                    shift.violation_count += 1
                    worker.total_violations += 1
                    events_written += 1
        else:
            # VLM disabled — write one SafetyEvent per YOLO hazard frame directly
            for frame in hazard_events:
                for hazard in frame.get("hazards", []):
                    sev_str  = hazard.get("severity", "MEDIUM")
                    severity = Severity[sev_str] if sev_str in Severity.__members__ else Severity.MEDIUM
                    event = SafetyEvent(
                        shift_id=shift.id,
                        worker_id=worker.id,
                        camera_source=camera_source,
                        event_category=EventCategory.VIOLATION,
                        violation_type=hazard.get("violation_type", "PPE_MISSING"),
                        severity=severity,
                        video_timestamp=frame["timestamp_seconds"],
                        clip_path=frame.get("frame_path"),
                        metadata_json=json.dumps({
                            "hazard_type":     hazard.get("hazard_type"),
                            "description":     hazard.get("description"),
                            "track_id":        hazard.get("track_id"),
                            "ppe_worn":        hazard.get("ppe_worn"),
                            "ppe_missing":     hazard.get("ppe_missing"),
                            "yolo_hazard_types": [hazard.get("hazard_type")],
                        }),
                    )
                    db.add(event)
                    shift.violation_count += 1
                    worker.total_violations += 1
                    events_written += 1

        shift.status = ShiftStatus.COMPLETED
        shift.ended_at = datetime.now(timezone.utc)
        worker.last_seen_at = datetime.now(timezone.utc)
        db.commit()
        print(f"[{camera_source}] {job_id} — {events_written} events written to SQLite")

        jobs[job_id] = {
            "status": "COMPLETED",
            "events_written": events_written,
            "error": None,
            "summary": summary,
            "job_dir": str(job_dir),
            "shift_id": shift.id,
            "camera_source": camera_source.value,
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{camera_source}] {job_id} — FAILED: {e}\n{tb}")
        db.rollback()
        jobs[job_id] = {"status": "FAILED", "events_written": 0, "error": str(e)}
    finally:
        db.close()
        video_path.unlink(missing_ok=True)
        print(f"[{camera_source}] {job_id} — source video cleaned up")


@app.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    """List all jobs with their SQLite shift + event counts."""
    result = []
    for job_id, job in jobs.items():
        entry = {
            "job_id": job_id,
            "status": job["status"],
            "events_written": job["events_written"],
            "error": job.get("error"),
            "camera_source": job.get("camera_source"),
        }
        shift_id = job.get("shift_id")
        if shift_id:
            shift = db.query(Shift).filter(Shift.id == shift_id).first()
            if shift:
                worker = db.query(Worker).filter(Worker.id == shift.worker_id).first()
                site   = db.query(Site).filter(Site.id == shift.site_id).first()
                event_count = db.query(SafetyEvent).filter(SafetyEvent.shift_id == shift.id).count()
                entry["shift"] = {
                    "shift_id":       shift.id,
                    "date":           shift.date,
                    "worker_id":      shift.worker_id,
                    "worker":         worker.display_name or worker.id if worker else shift.worker_id,
                    "site":           site.name if site else shift.site_id,
                    "violation_count": shift.violation_count,
                    "event_count":    event_count,
                    "status":         shift.status,
                    "started_at":     str(shift.started_at),
                    "ended_at":       str(shift.ended_at) if shift.ended_at else None,
                }
        result.append(entry)
    return result


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}



@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str):
    """Full structured output from the pipeline: summary + all hazard events."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']}")

    job_dir = Path(job["job_dir"])

    summary_file = job_dir / "summary.json"
    vlm_file     = job_dir / "vlm_assessments.json"

    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Results not found on disk")

    summary      = json.loads(summary_file.read_text())
    vlm_results  = json.loads(vlm_file.read_text()) if vlm_file.exists() else []

    return {
        "job_id":      job_id,
        "summary":     summary,
        "vlm_results": vlm_results,
    }


@app.get("/jobs/{job_id}/vlm")
def get_job_vlm(job_id: str):
    """Returns the full VLM assessment for each temporal event window in this job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']}")

    vlm_file = Path(job["job_dir"]) / "vlm_assessments.json"
    if not vlm_file.exists():
        raise HTTPException(status_code=404, detail="VLM assessments not found")

    return json.loads(vlm_file.read_text())


@app.get("/jobs/{job_id}/frames/{filename}")
def get_job_frame(job_id: str, filename: str):
    """Serve a flagged frame or person crop image by filename."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = Path(job.get("job_dir", str(JOBS_DIR / job_id)))

    # Check frames/ then crops/ subdirectory
    for subdir in ("frames", "crops"):
        img_path = job_dir / subdir / filename
        if img_path.exists():
            return FileResponse(str(img_path), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Frame not found")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_event(e: SafetyEvent) -> dict:
    return {
        "event_id": e.id,
        "shift_id": e.shift_id,
        "worker_id": e.worker_id,
        "camera_source": e.camera_source,
        "event_category": e.event_category,
        "violation_type": e.violation_type,
        "severity": e.severity,
        "video_timestamp": e.video_timestamp,
        "clip_path": e.clip_path,
        "metadata": json.loads(e.metadata_json) if e.metadata_json else None,
        "created_at": str(e.created_at),
    }
