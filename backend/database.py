from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Enum as SAEnum,
    create_engine, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import enum
import os
import uuid
from pathlib import Path

_DEFAULT_DB = Path(__file__).parent / "data" / "ironsite.db"
_DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WorkerStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"

class ShiftStatus(str, enum.Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"

class CameraSource(str, enum.Enum):
    POV = "POV"
    WALL_CAM = "WALL_CAM"

class EventCategory(str, enum.Enum):
    VIOLATION = "VIOLATION"
    COMPLIANT = "COMPLIANT"

class Severity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Site(Base):
    __tablename__ = "sites"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name         = Column(String, nullable=False)
    location     = Column(String)
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Worker(Base):
    __tablename__ = "workers"
    id                   = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    site_id              = Column(String, ForeignKey("sites.id"), nullable=False)
    display_name         = Column(String)
    appearance_embedding = Column(String)   # base64 — wall-cam re-ID
    status               = Column(SAEnum(WorkerStatus), default=WorkerStatus.ACTIVE)
    created_at           = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen_at         = Column(DateTime)
    total_violations     = Column(Integer, default=0)
    total_compliant      = Column(Integer, default=0)
    avg_msd_risk         = Column(Float, default=0.0)


class Shift(Base):
    __tablename__ = "shifts"
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    worker_id        = Column(String, ForeignKey("workers.id"), nullable=False)
    site_id          = Column(String, ForeignKey("sites.id"), nullable=False)
    date             = Column(String, nullable=False)   # YYYY-MM-DD
    started_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at         = Column(DateTime)
    status           = Column(SAEnum(ShiftStatus), default=ShiftStatus.IN_PROGRESS)
    msd_risk_score   = Column(Float)
    violation_count  = Column(Integer, default=0)
    compliant_count  = Column(Integer, default=0)


class SafetyEvent(Base):
    __tablename__ = "safety_events"
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    shift_id         = Column(String, ForeignKey("shifts.id"), nullable=False)
    worker_id        = Column(String, ForeignKey("workers.id"), nullable=False)
    camera_source    = Column(SAEnum(CameraSource), nullable=False)
    event_category   = Column(SAEnum(EventCategory), nullable=False)
    violation_type   = Column(String)
    severity         = Column(SAEnum(Severity))
    video_timestamp  = Column(Float)
    clip_path        = Column(String)
    metadata_json    = Column(String)   # JSON blob
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)
