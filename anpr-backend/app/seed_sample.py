"""Seed SAMPLE Hyderabad detections + the stolen-plate alert.

Run once from anpr-backend:

    python -m app.seed_sample

Only runs when the detections table is empty, so running it twice
adds nothing. Every row here is labelled sample: the same 14
sightings the dashboard shows in its sample feed, for Hyderabad
plates — TS09EX1234, TS08AB7777 (stolen), TS10AB1234,
TS11CD5678, TS07XY9999, TS13PQ2468, TS15MN1357. No other plates.
Timestamps are spread over the last ~6 hours (IST), newest 40
minutes ago: all today so the hourly chart shows them, and all
older than 30 seconds so the backend's duplicate guard is not
affected. (Fixed times later today would be in the future and
would make the same plate+camera look like a duplicate for hours.)
"""

from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from app.database import SessionLocal, engine, Base
from app.models import Detection, Alert  # noqa: F401 - registers models

IST = ZoneInfo("Asia/Kolkata")

# (plate, camera_id, confidence) — sample only.
SAMPLE_DETECTIONS = [
    ("TS09EX1234", "HYD06", 0.89),
    ("TS08AB7777", "HYD06", 0.82),
    ("TS10AB1234", "HYD01", 0.92),
    ("TS09EX1234", "HYD02", 0.94),
    ("TS11CD5678", "HYD07", 0.87),
    ("TS07XY9999", "HYD04", 0.78),
    ("TS13PQ2468", "HYD05", 0.90),
    ("TS15MN1357", "HYD08", 0.85),
    ("TS09EX1234", "HYD01", 0.91),
    ("TS08AB7777", "HYD07", 0.76),
    ("TS10AB1234", "HYD02", 0.88),
    ("TS11CD5678", "HYD08", 0.81),
    ("TS07XY9999", "HYD01", 0.93),
    ("TS13PQ2468", "HYD06", 0.84),
]

STOLEN_PLATE = "TS08AB7777"


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        existing = db.query(Detection).count()
        if existing:
            print(
                f"Sample seed skipped: detections table already has "
                f"{existing} row(s)."
            )
            return
        stolen_event_id = None
        now_ist = datetime.now(IST)
        for i, (plate, camera_id, confidence) in enumerate(SAMPLE_DETECTIONS):
            event_id = str(uuid4())
            if plate == STOLEN_PLATE and camera_id == "HYD06":
                stolen_event_id = event_id
            when_ist = now_ist - timedelta(minutes=40 + 25 * i)
            db.add(Detection(
                event_id=event_id,
                camera_id=camera_id,
                plate_text=plate,
                confidence=confidence,
                timestamp=when_ist.astimezone(ZoneInfo("UTC")),
            ))
        db.add(Alert(
            alert_type="stolen",
            message=(
                "Stolen vehicle TS08AB7777 seen in Hyderabad "
                "(sample alert)"
            ),
            plate_text=STOLEN_PLATE,
            camera_id="HYD06",
            detection_event_id=stolen_event_id,
            status="active",
        ))
        db.commit()
        print(
            f"Seeded {len(SAMPLE_DETECTIONS)} sample Hyderabad "
            f"detections and 1 stolen-vehicle alert for {STOLEN_PLATE}."
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
