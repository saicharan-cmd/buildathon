from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from uuid import uuid4
from datetime import datetime, timezone, date, timedelta
from zoneinfo import ZoneInfo

from app.schemas import DetectionCreate, AlertCreate, AlertStatusUpdate
from app.data import cameras
from app.database import SessionLocal
from app.models import Detection, Alert
from app.pipeline_control import router as pipeline_router

app = FastAPI()
app.include_router(pipeline_router)

# The dashboard is one static HTML file, opened from anywhere
# (double-clicked locally, or hosted). Without this the browser
# blocks its cross-origin calls to this API, so the dashboard
# can never show live data. No auth on this API, so allowing
# any origin matches what the API already allows.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# How long a repeat sighting counts as a duplicate (seconds).
# The pipeline also dedups in memory for the same window, so a
# plate seen again within 30 seconds is stored only once.
DUPLICATE_WINDOW_SECONDS = 30

# Known cameras, from the hardcoded list in app/data.py.
# POST /api/detections used to accept ANY camera_id, even one
# that does not exist. An unknown camera is now a 400.
# An offline camera (HYD03 Kukatpally) is NOT rejected — it is
# a known camera, it just happens to be offline right now.
KNOWN_CAMERA_IDS = {camera["camera_id"] for camera in cameras}


# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Convert database record into JSON-compatible dictionary
def detection_to_dict(detection):
    return {
        "event_id": detection.event_id,
        "camera_id": detection.camera_id,
        "plate_text": detection.plate_text,
        "confidence": detection.confidence,
        "timestamp": detection.timestamp.isoformat()
        if detection.timestamp else None
    }


@app.get("/")
def home():
    return {
        "message": "Welcome to ANPR Backend!",
        "status": "Running"
    }


# POST: Save a detection in PostgreSQL
@app.post("/api/detections")
def create_detection(
    detection: DetectionCreate,
    db: Session = Depends(get_db)
):
    # Validate the camera before anything is stored. The pipeline
    # default HYD01 passes; a typo or a camera that does not exist
    # in app/data.py is a 400, not a row in the database.
    camera_id = detection.camera_id.strip()
    if camera_id not in KNOWN_CAMERA_IDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown camera_id: {detection.camera_id!r}. "
                f"Known cameras: {', '.join(sorted(KNOWN_CAMERA_IDS))}"
            ),
        )

    # Normalise the plate so "ts09ex1234" and "TS09EX1234" dedup
    # against each other instead of being stored twice.
    plate_text = detection.plate_text.strip().upper()
    if not plate_text:
        raise HTTPException(
            status_code=400,
            detail="plate_text is required",
        )
    now = datetime.now(timezone.utc)

    # Duplicate guard: the same plate from the same camera stored
    # very recently is a repeat sighting, not a new detection.
    # Return the existing row instead of storing a duplicate.
    cutoff = now - timedelta(seconds=DUPLICATE_WINDOW_SECONDS)
    existing = (
        db.query(Detection)
        .filter(
            Detection.camera_id == camera_id,
            Detection.plate_text == plate_text,
            Detection.timestamp >= cutoff,
        )
        .order_by(Detection.timestamp.desc())
        .first()
    )
    if existing is not None:
        return {
            "message": "Duplicate detection ignored",
            "duplicate": True,
            "data": detection_to_dict(existing)
        }

    new_detection = Detection(
        event_id=str(uuid4()),
        camera_id=camera_id,
        plate_text=plate_text,
        confidence=detection.confidence,
        timestamp=now
    )

    try:
        db.add(new_detection)
        db.commit()
        db.refresh(new_detection)
    except SQLAlchemyError:
        # A failed commit must not leave the session broken for the
        # next request, and must not crash the server.
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not store detection"
        )

    return {
        "message": "Detection stored successfully",
        "data": detection_to_dict(new_detection)
    }


# GET: Retrieve all detections or filter by camera ID
@app.get("/api/detections")
def get_detections(
    camera_id: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(Detection)

    if camera_id:
        query = query.filter(
            Detection.camera_id == camera_id
        )

    # total_detections is the filtered total, not the page size,
    # so callers can page through all results.
    total = query.count()
    results = (
        query.order_by(Detection.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total_detections": total,
        "detections": [
            detection_to_dict(item) for item in results
        ]
    }


# GET: Retrieve one detection using event ID
@app.get("/api/detections/{event_id}")
def get_detection(
    event_id: str,
    db: Session = Depends(get_db)
):
    detection = db.query(Detection).filter(
        Detection.event_id == event_id
    ).first()

    if detection is None:
        raise HTTPException(
            status_code=404,
            detail="Detection not found"
        )

    return {
        "message": "Detection found",
        "data": detection_to_dict(detection)
    }


# GET: Retrieve all cameras
@app.get("/api/cameras")
def get_cameras():
    return {
        "total_cameras": len(cameras),
        "cameras": cameras
    }


# GET: Retrieve one camera by camera ID
@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str):
    for camera in cameras:
        if camera["camera_id"] == camera_id:
            return {
                "message": "Camera found",
                "data": camera
            }

    raise HTTPException(
        status_code=404,
        detail="Camera not found"
    )


# GET: Traffic analytics summary
@app.get("/api/analytics/summary")
def get_analytics_summary(
    db: Session = Depends(get_db)
):
    # 1. Count total detections
    total_detections = db.query(Detection).count()

    # 2. Calculate average confidence
    average_confidence = db.query(
        func.avg(Detection.confidence)
    ).scalar()

    # 3. Count detections camera-wise
    camera_counts = db.query(
    Detection.camera_id,
    func.count(Detection.id)
    ).group_by(
    Detection.camera_id
    ).order_by(
    Detection.camera_id
    ).all()
    # 4. Prepare camera-wise results
    camera_wise_detections = []

    for camera in cameras:
        count = 0
        for camera_id, detection_count in camera_counts:
            if camera_id == camera["camera_id"]:
                count = detection_count
                break

        camera_wise_detections.append({
            "camera_id": camera["camera_id"],
            "camera_name": camera["name"],
            "detection_count": count
        })

    return {
        "total_detections": total_detections,
        "total_cameras": len(cameras),
        "average_confidence": round(average_confidence, 2)
            if average_confidence is not None else 0,
        "camera_wise_detections": camera_wise_detections
    }

# GET: Hourly traffic analytics
@app.get("/api/analytics/hourly")
def get_hourly_analytics(
    start_date: date | None = None,
    end_date: date | None = None,
    db: Session = Depends(get_db)
):
    # Indian Standard Time
    ist = ZoneInfo("Asia/Kolkata")

    # Get today's date if no dates are provided
    today = datetime.now(ist).date()

    if start_date is None and end_date is None:
        start_date = today
        end_date = today

    elif start_date is None or end_date is None:
        raise HTTPException(
            status_code=400,
            detail="Please provide both start_date and end_date"
        )

    # Validate date range
    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date cannot be after end_date"
        )

    # Convert dates into timezone-aware datetime boundaries
    start_datetime = datetime.combine(
        start_date,
        datetime.min.time(),
        tzinfo=ist
    )

    end_datetime = datetime.combine(
        end_date,
        datetime.min.time(),
        tzinfo=ist
    ) + timedelta(days=1)

    # Get detection counts grouped by hour
    ist_hour = func.extract(
    "hour",
    func.timezone("Asia/Kolkata", Detection.timestamp)
)
    results = db.query(
        ist_hour,
        func.count(Detection.id)
    ).filter(
    Detection.timestamp >= start_datetime,
    Detection.timestamp < end_datetime
    ).group_by(
        ist_hour
    ).order_by(
        ist_hour
    ).all()

    # Prepare all 24 hours, including hours with zero detections
    hourly_counts = {int(hour): count for hour, count in results}

    hourly_traffic = []

    for hour in range(24):
        hourly_traffic.append({
            "hour": f"{hour:02d}:00",
            "detection_count": hourly_counts.get(hour, 0)
        })

    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "Asia/Kolkata",
        "total_detections": sum(hourly_counts.values()),
        "hourly_traffic": hourly_traffic
    }
# GET: Daily traffic analytics
@app.get("/api/analytics/daily")
def get_daily_analytics(
    start_date: date | None = None,
    end_date: date | None = None,
    db: Session = Depends(get_db)
):
    ist = ZoneInfo("Asia/Kolkata")
    today = datetime.now(ist).date()

    if start_date is None and end_date is None:
        start_date = today
        end_date = today

    elif start_date is None or end_date is None:
        raise HTTPException(
            status_code=400,
            detail="Please provide both start_date and end_date"
        )

    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date cannot be after end_date"
        )

    start_datetime = datetime.combine(
        start_date,
        datetime.min.time(),
        tzinfo=ist
    )

    end_datetime = datetime.combine(
        end_date,
        datetime.min.time(),
        tzinfo=ist
    ) + timedelta(days=1)

    ist_date = func.date(
        func.timezone("Asia/Kolkata", Detection.timestamp)
    )

    results = db.query(
        ist_date,
        func.count(Detection.id)
    ).filter(
        Detection.timestamp >= start_datetime,
        Detection.timestamp < end_datetime
    ).group_by(
        ist_date
    ).order_by(
        ist_date
    ).all()

    daily_counts = {
        str(day): count for day, count in results
    }

    daily_traffic = []

    current_date = start_date

    while current_date <= end_date:
        daily_traffic.append({
            "date": str(current_date),
            "detection_count": daily_counts.get(
                str(current_date), 0
            )
        })

        current_date += timedelta(days=1)

    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "Asia/Kolkata",
        "total_detections": sum(daily_counts.values()),
        "daily_traffic": daily_traffic
    }
# GET: Camera-wise traffic analytics
@app.get("/api/analytics/cameras")
def get_camera_analytics(
    db: Session = Depends(get_db)
):
    results = db.query(
        Detection.camera_id,
        func.count(Detection.id),
        func.avg(Detection.confidence)
    ).group_by(
        Detection.camera_id
    ).all()

    analytics = []

    for camera in cameras:
        detection_count = 0
        average_confidence = 0

        for camera_id, count, avg_confidence in results:
            if camera_id == camera["camera_id"]:
                detection_count = count
                average_confidence = avg_confidence or 0
                break

        analytics.append({
            "camera_id": camera["camera_id"],
            "camera_name": camera["name"],
            "total_detections": detection_count,
            "average_confidence": round(
                average_confidence, 2
            )
        })

    return {
        "total_cameras": len(cameras),
        "camera_analytics": analytics
    }


# GET: Retrieve vehicle sightings across cameras
@app.get("/api/vehicles/{plate_text}/trajectory")
def get_vehicle_trajectory(
    plate_text: str,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    # Normalize vehicle registration number
    plate_text = plate_text.strip().upper()

    if not plate_text:
        raise HTTPException(
            status_code=400,
            detail="Vehicle plate number is required"
        )

    # Indian Standard Time
    ist = ZoneInfo("Asia/Kolkata")

    # Validate date parameters
    if start_date is None and end_date is not None:
        raise HTTPException(
            status_code=400,
            detail="Please provide both start_date and end_date"
        )

    if end_date is None and start_date is not None:
        raise HTTPException(
            status_code=400,
            detail="Please provide both start_date and end_date"
        )

    if start_date and end_date:
        if start_date > end_date:
            raise HTTPException(
                status_code=400,
                detail="start_date cannot be after end_date"
            )

    # Search for the vehicle
    query = db.query(Detection).filter(
        Detection.plate_text == plate_text
    )

    # Apply optional date range
    if start_date and end_date:
        start_datetime = datetime.combine(
            start_date,
            datetime.min.time(),
            tzinfo=ist
        )

        end_datetime = datetime.combine(
            end_date,
            datetime.min.time(),
            tzinfo=ist
        ) + timedelta(days=1)

        query = query.filter(
            Detection.timestamp >= start_datetime,
            Detection.timestamp < end_datetime
        )

    # Retrieve sightings in chronological order
    results = (
        query.order_by(Detection.timestamp.asc())
        .limit(limit)
        .all()
    )

    # Match camera IDs with camera names
    camera_names = {
        camera["camera_id"]: camera["name"]
        for camera in cameras
    }

    trajectory = []

    for detection in results:
        trajectory.append({
            "event_id": detection.event_id,
            "camera_id": detection.camera_id,
            "camera_name": camera_names.get(
                detection.camera_id,
                "Unknown camera"
            ),
            "confidence": detection.confidence,
            "timestamp": (
                detection.timestamp.isoformat()
                if detection.timestamp else None
            )
        })

    return {
        "plate_text": plate_text,
        "total_sightings": len(trajectory),
        "timezone": "Asia/Kolkata",
        "trajectory": trajectory
    }
@app.post("/api/alerts")
def create_alert(
    alert_data: AlertCreate,
    db: Session = Depends(get_db)
):
    if not alert_data.alert_type.strip():
        raise HTTPException(status_code=400, detail="Alert type cannot be empty")

    if not alert_data.message.strip():
        raise HTTPException(status_code=400, detail="Alert message cannot be empty")

    alert = Alert(
        alert_type=alert_data.alert_type.strip(),
        message=alert_data.message.strip(),
        plate_text=(
            alert_data.plate_text.strip().upper()
            if alert_data.plate_text else None
        ),
        camera_id=alert_data.camera_id,
        detection_event_id=alert_data.detection_event_id,
        status="active"
    )

    try:
        db.add(alert)
        db.commit()
        db.refresh(alert)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to create alert"
        )

    return {
        "id": alert.id,
        "alert_type": alert.alert_type,
        "message": alert.message,
        "plate_text": alert.plate_text,
        "camera_id": alert.camera_id,
        "status": alert.status,
        "created_at": alert.created_at
    }

@app.get("/api/alerts")
def get_alerts(
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    if status and status not in ["active", "acknowledged", "resolved"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid alert status"
        )

    query = db.query(Alert)

    if status:
        query = query.filter(Alert.status == status)

    total = query.count()

    alerts = (
        query.order_by(Alert.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "alerts": [
            {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "message": alert.message,
                "plate_text": alert.plate_text,
                "camera_id": alert.camera_id,
                "status": alert.status,
                "created_at": alert.created_at,
            }
            for alert in alerts
        ]
    }
@app.patch("/api/alerts/{alert_id}")
def update_alert_status(
    alert_id: int,
    status_data: AlertStatusUpdate,
    db: Session = Depends(get_db)
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()

    if not alert:
        raise HTTPException(
            status_code=404,
            detail="Alert not found"
        )

    if alert.status == "resolved":
        raise HTTPException(
            status_code=400,
            detail="Resolved alerts cannot be updated"
        )

    if status_data.status == "acknowledged":
        if alert.status != "active":
            raise HTTPException(
                status_code=400,
                detail="Only active alerts can be acknowledged"
            )

        alert.status = "acknowledged"
        alert.acknowledged_at = datetime.now(timezone.utc)

    elif status_data.status == "resolved":
        if alert.status != "acknowledged":
            raise HTTPException(
                status_code=400,
                detail="Acknowledge the alert before resolving it"
            )

        alert.status = "resolved"
        alert.resolved_at = datetime.now(timezone.utc)

    try:
        db.commit()
        db.refresh(alert)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to update alert"
        )

    return {
        "message": "Alert status updated successfully",
        "id": alert.id,
        "status": alert.status,
        "acknowledged_at": alert.acknowledged_at,
        "resolved_at": alert.resolved_at
    }