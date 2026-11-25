from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from datetime import datetime, timezone

from app.database import Base


class Detection(Base):
    __tablename__ = "detections"

    # Indexes on camera_id / plate_text / timestamp support the
    # duplicate guard in POST /api/detections (same plate + camera
    # within 30 seconds) and the newest-first paged list in
    # GET /api/detections. They only take effect on a fresh table
    # created by create_tables.py; an existing table needs them
    # added manually.
    __table_args__ = (
        Index(
            "ix_detections_camera_plate_timestamp",
            "camera_id",
            "plate_text",
            "timestamp",
        ),
    )

    id = Column(Integer, primary_key=True)

    event_id = Column(String, unique=True)

    camera_id = Column(String, nullable=False, index=True)

    plate_text = Column(String, nullable=False, index=True)

    confidence = Column(Float, nullable=False)

    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    plate_text = Column(String, nullable=True)
    camera_id = Column(String, nullable=True)
    detection_event_id = Column(String, nullable=True)

    status = Column(
        String,
        nullable=False,
        default="active"
    )

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    acknowledged_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    resolved_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
