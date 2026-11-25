"""Buildathon — Streamlit-only core (no Streamlit import here).

Everything the Streamlit app needs, in one testable module:
- SQLite database (created on first run, no Postgres, no .env)
- the 11 cameras (copied from anpr-backend/app/data.py — same ids,
  names, cities, regions, lat/lon and online/offline status)
- Indian plate validation + the OCR fix from
  anpr-backend/app/yolo_pipeline.py (upscale 2x + grayscale retry,
  allowlist A-Z0-9, 10-char position-aware fix, drop one leading /
  trailing junk char on an 11-char read, never invent a digit from a
  9-char read)
- YOLOv8 + EasyOCR detection (model auto-downloaded on first run)
- saving detections + watchlist alerts (TS08AB7777 = stolen)

Confidence rule (as specified for this app): 0.4 * YOLO + 0.6 * OCR,
minimum 0.5 to save. (The FastAPI pipeline uses a plain average;
this app uses the weighted rule everywhere, so what you see here is
what gets saved here.)
"""

import hashlib
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "buildathon.db"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "yolov8-plate.pt"
MODEL_URL = (
    "https://huggingface.co/joker5914/yolov8n-license-plate"
    "/resolve/main/best.pt"
)
MODEL_SHA256 = (
    "75c9c5487e69d96a3076eaf6660b8dc4f84896055f786ee28fd90bdc3b48c1ec"
)

MIN_CONFIDENCE = 0.5
DEDUP_SECONDS = 30

# ---------------------------------------------------------------- cameras
# Same 11 cameras as the FastAPI backend (anpr-backend/app/data.py).
CAMERAS = [
    {"camera_id": "CAM01", "name": "Patia Junction", "city": "Bhubaneswar",
     "region": "odisha", "latitude": 20.3548, "longitude": 85.8197,
     "status": "online"},
    {"camera_id": "CAM02", "name": "Jaydev Vihar", "city": "Bhubaneswar",
     "region": "odisha", "latitude": 20.3008, "longitude": 85.8238,
     "status": "online"},
    {"camera_id": "CAM03", "name": "KIIT Square", "city": "Bhubaneswar",
     "region": "odisha", "latitude": 20.3539, "longitude": 85.8260,
     "status": "offline"},
    {"camera_id": "HYD01", "name": "Hitech City", "city": "Hyderabad",
     "region": "telangana", "latitude": 17.4435, "longitude": 78.3772,
     "status": "online"},
    {"camera_id": "HYD02", "name": "Gachibowli", "city": "Hyderabad",
     "region": "telangana", "latitude": 17.4401, "longitude": 78.3489,
     "status": "online"},
    {"camera_id": "HYD03", "name": "Kukatpally", "city": "Hyderabad",
     "region": "telangana", "latitude": 17.4849, "longitude": 78.4138,
     "status": "offline"},
    {"camera_id": "HYD04", "name": "Banjara Hills", "city": "Hyderabad",
     "region": "telangana", "latitude": 17.4156, "longitude": 78.4347,
     "status": "online"},
    {"camera_id": "HYD05", "name": "Begumpet", "city": "Hyderabad",
     "region": "telangana", "latitude": 17.4447, "longitude": 78.4664,
     "status": "online"},
    {"camera_id": "HYD06", "name": "Secunderabad", "city": "Hyderabad",
     "region": "telangana", "latitude": 17.4399, "longitude": 78.5017,
     "status": "online"},
    {"camera_id": "HYD07", "name": "Charminar", "city": "Hyderabad",
     "region": "telangana", "latitude": 17.3616, "longitude": 78.4747,
     "status": "online"},
    {"camera_id": "HYD08", "name": "LB Nagar", "city": "Hyderabad",
     "region": "telangana", "latitude": 17.3456, "longitude": 78.5522,
     "status": "online"},
]
CAMERA_IDS = {c["camera_id"] for c in CAMERAS}
CAMERA_BY_ID = {c["camera_id"]: c for c in CAMERAS}

# Stolen / watchlist plates. A detection of one of these always
# creates an alert (if there is not already an active one for it).
WATCHLIST = {
    "TS08AB7777": "Stolen vehicle — watchlist plate",
}

# Demo sightings seeded on a fresh database only. They are labelled
# source='demo' in the app and are never presented as real captures.
# Times are spread over today so the hourly chart is not empty.
DEMO_DETECTIONS = [
    # (plate, camera_id, confidence, minutes_ago)
    ("TS09EX1234", "HYD02", 0.91, 540),
    ("TS09EX1234", "HYD06", 0.88, 335),
    ("TS09EX1234", "HYD01", 0.85, 210),
    ("TS08AB7777", "HYD07", 0.82, 420),
    ("TS08AB7777", "HYD06", 0.76, 300),
    ("OD02AB1234", "CAM01", 0.89, 480),
    ("OD02AB1234", "CAM02", 0.84, 390),
    ("TS10AB1234", "HYD04", 0.87, 150),
    ("TS11CD5678", "HYD05", 0.83, 95),
    ("TS07XY9999", "HYD08", 0.81, 60),
]

# ------------------------------------------------------------------ plates
INDIAN_PLATE_RE = re.compile(r"^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})$")
_DIGIT_FIX = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5",
              "B": "8", "Z": "2", "G": "6"}
_LETTER_FIX = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
_OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _position_fix(cleaned):
    """Fix misread characters in a 10-char read, by position.

    Positions 2,3,6,7,8,9 are digits; the rest are letters.
    Returns (fixed_plate, True) or ("", False).
    """
    chars, corrected = list(cleaned), False
    for i, ch in enumerate(chars):
        digit_pos = i in (2, 3, 6, 7, 8, 9)
        if digit_pos and ch in _DIGIT_FIX:
            chars[i], corrected = _DIGIT_FIX[ch], True
        elif not digit_pos and ch in _LETTER_FIX:
            chars[i], corrected = _LETTER_FIX[ch], True
    fixed = "".join(chars)
    if corrected and INDIAN_PLATE_RE.match(fixed):
        return fixed, True
    return "", False


def validate_plate_text(raw):
    """Validate raw OCR text as an Indian plate.

    Returns (plate, valid, corrected, reason).
    - 10 chars: try the position-aware fix (e.g. ODOzAB1234 -> OD02AB1234).
    - 11 chars: drop one leading/trailing junk char, then retry.
    - 9 chars: NEVER repaired — that would invent a missing digit.
    """
    if not raw:
        return "", False, False, "empty OCR result"
    cleaned = re.sub(r"[^A-Z0-9]", "", str(raw).upper())
    if not cleaned:
        return "", False, False, "empty OCR result"
    if INDIAN_PLATE_RE.match(cleaned):
        return cleaned, True, False, "matches Indian plate format"
    if len(cleaned) == 10:
        fixed, ok = _position_fix(cleaned)
        if ok:
            return fixed, True, True, "position-aware correction applied"
    if len(cleaned) == 11:
        for candidate in (cleaned[1:], cleaned[:-1]):
            if INDIAN_PLATE_RE.match(candidate):
                return (candidate, True, True,
                        "dropped one leading/trailing junk character")
            fixed, ok = _position_fix(candidate)
            if ok:
                return (fixed, True, True,
                        "dropped one leading/trailing junk character "
                        "and applied position-aware correction")
    return cleaned, False, False, "does not match Indian plate format"


def combined_confidence(yolo_conf, ocr_conf):
    """0.4 * YOLO + 0.6 * OCR, clamped to 0..1, 3 decimals."""
    try:
        value = 0.4 * float(yolo_conf) + 0.6 * float(ocr_conf)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, value)), 3)


# ------------------------------------------------------------------- model
def model_status():
    """Return (exists, size_ok, sha_ok) for the local model file."""
    if not MODEL_PATH.is_file():
        return False, False, False
    size_ok = MODEL_PATH.stat().st_size > 1_000_000
    sha_ok = False
    try:
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        sha_ok = digest == MODEL_SHA256
    except OSError:
        sha_ok = False
    return True, size_ok, sha_ok


def ensure_model(progress=None):
    """Download the plate model if missing or corrupt.

    Returns (path_or_None, message). Never raises: on failure it
    returns None and a plain-English message so the app can show it
    instead of crashing. `progress` is an optional callable taking
    (downloaded_bytes, total_bytes_or_None).
    """
    exists, size_ok, sha_ok = model_status()
    if exists and size_ok and sha_ok:
        return str(MODEL_PATH), "Model already downloaded."
    if exists and size_ok and not sha_ok:
        # A file is there but it is not the model we expect (wrong
        # file, truncated download). Remove it and re-download.
        try:
            MODEL_PATH.unlink()
        except OSError:
            return None, f"Model file is corrupt and could not be removed: {MODEL_PATH}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_DIR / "yolov8-plate.pt.download"
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Buildathon-Streamlit/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            downloaded = 0
            with open(tmp_path, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        try:
                            progress(downloaded, total)
                        except Exception:  # noqa: BLE001 - progress is best-effort
                            pass
        tmp_path.replace(MODEL_PATH)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return None, (
            "Could not download the plate model. Check your internet "
            f"and try again. ({exc})"
        )
    exists, size_ok, sha_ok = model_status()
    if not (exists and size_ok):
        return None, "Model download finished but the file looks incomplete. Delete models/yolov8-plate.pt and run again."
    if not sha_ok:
        return None, "Model downloaded but its checksum did not match. Delete models/yolov8-plate.pt and run again."
    return str(MODEL_PATH), "Model downloaded."


# ---------------------------------------------------------------- database
def _connect(db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    """Create the DB + tables if missing; seed cameras and demo data
    only on a fresh database. Safe to call on every app start: a
    second run never duplicates anything.
    Returns a dict with what was (re)created, for the UI to show.
    """
    fresh_db = not Path(db_path).is_file()
    conn = _connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                region TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                camera_id TEXT NOT NULL,
                plate_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'captured',
                timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS
                idx_detections_camera_plate_time
                ON detections (camera_id, plate_text, timestamp);
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT,
                camera_id TEXT,
                detection_event_id TEXT,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()

        seeded_cameras = 0
        if conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO cameras VALUES (?,?,?,?,?,?,?)",
                [(c["camera_id"], c["name"], c["city"], c["region"],
                  c["latitude"], c["longitude"], c["status"])
                 for c in CAMERAS],
            )
            seeded_cameras = len(CAMERAS)

        seeded_detections = 0
        if conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0:
            now = datetime.now(timezone.utc)
            for plate, camera_id, conf, mins_ago in DEMO_DETECTIONS:
                ts = (now - timedelta(minutes=mins_ago)).isoformat()
                event_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO detections "
                    "(event_id, camera_id, plate_text, confidence, source, timestamp) "
                    "VALUES (?,?,?,?,?,?)",
                    (event_id, camera_id, plate, conf, "demo", ts),
                )
                seeded_detections += 1
                # One active alert per watchlist plate, not one per
                # sighting — otherwise the demo seeds two alerts for
                # TS08AB7777 and the Alerts page looks duplicated.
                if plate in WATCHLIST and not _active_alert_exists(conn, plate):
                    conn.execute(
                        "INSERT INTO alerts "
                        "(plate_text, camera_id, detection_event_id, "
                        " alert_type, message, status, created_at) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (plate, camera_id, event_id, "watchlist",
                         WATCHLIST[plate], "active", ts),
                    )
        conn.commit()
        return {
            "fresh_db": fresh_db,
            "seeded_cameras": seeded_cameras,
            "seeded_detections": seeded_detections,
        }
    finally:
        conn.close()


def list_cameras(db_path=DB_PATH, region=None):
    conn = _connect(db_path)
    try:
        if region in ("odisha", "telangana"):
            rows = conn.execute(
                "SELECT * FROM cameras WHERE region = ? ORDER BY camera_id",
                (region,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cameras ORDER BY camera_id"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_detections(db_path=DB_PATH, region=None, camera_id=None,
                    plate_query=None, limit=200):
    """Newest first. plate_query is a plain substring match (upper-cased)."""
    conn = _connect(db_path)
    try:
        query = (
            "SELECT d.*, c.name AS camera_name, c.city, c.region "
            "FROM detections d LEFT JOIN cameras c "
            "ON c.camera_id = d.camera_id WHERE 1=1"
        )
        params = []
        if region in ("odisha", "telangana"):
            query += " AND c.region = ?"
            params.append(region)
        if camera_id:
            query += " AND d.camera_id = ?"
            params.append(camera_id)
        if plate_query:
            query += " AND d.plate_text LIKE ?"
            params.append("%" + plate_query.strip().upper() + "%")
        query += " ORDER BY d.timestamp DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def track_plate(plate_text, db_path=DB_PATH):
    """All sightings of one plate (newest first) + its alerts."""
    plate = (plate_text or "").strip().upper()
    if not plate:
        return [], []
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT d.*, c.name AS camera_name, c.city, c.region, "
            "c.latitude, c.longitude "
            "FROM detections d LEFT JOIN cameras c "
            "ON c.camera_id = d.camera_id "
            "WHERE d.plate_text = ? ORDER BY d.timestamp ASC",
            (plate,),
        ).fetchall()
        alerts = conn.execute(
            "SELECT * FROM alerts WHERE plate_text = ? "
            "ORDER BY created_at DESC",
            (plate,),
        ).fetchall()
        return [dict(r) for r in rows], [dict(a) for a in alerts]
    finally:
        conn.close()


def list_alerts(db_path=DB_PATH, status=None):
    conn = _connect(db_path)
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE status = ? "
                "ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def acknowledge_alert(alert_id, db_path=DB_PATH):
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE alerts SET status = 'acknowledged' WHERE id = ?",
            (int(alert_id),),
        )
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def _active_alert_exists(conn, plate_text):
    row = conn.execute(
        "SELECT COUNT(*) FROM alerts "
        "WHERE plate_text = ? AND status = 'active'",
        (plate_text,),
    ).fetchone()
    return row[0] > 0


def save_detection(camera_id, plate_text, confidence, db_path=DB_PATH,
                   source="captured", timestamp=None):
    """Validate + dedup + save one detection.

    Returns (ok, info_dict). ok=False when the plate is invalid, the
    camera is unknown, confidence is below 0.5, or it is a duplicate
    (same plate + camera within 30 seconds). On success info has
    event_id, plate_text, confidence and alert_created (bool).
    """
    plate, valid, corrected, reason = validate_plate_text(plate_text)
    if not valid:
        return False, {"reason": reason, "cleaned": plate}
    if camera_id not in CAMERA_IDS:
        return False, {"reason": f"unknown camera_id: {camera_id}"}
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return False, {"reason": "confidence is not a number"}
    if confidence < MIN_CONFIDENCE:
        return False, {
            "reason": f"confidence {confidence} is below {MIN_CONFIDENCE}",
            "plate_text": plate,
        }
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        # Duplicate guard: same plate + camera within 30 seconds.
        row = conn.execute(
            "SELECT timestamp FROM detections "
            "WHERE camera_id = ? AND plate_text = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (camera_id, plate),
        ).fetchone()
        if row is not None:
            try:
                last = datetime.fromisoformat(row["timestamp"])
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                now = datetime.fromisoformat(ts)
                if now.tzinfo is None:
                    now = now.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < DEDUP_SECONDS:
                    return False, {
                        "reason": "duplicate: same plate + camera "
                                  "within 30 seconds",
                        "plate_text": plate,
                    }
            except ValueError:
                pass  # unreadable old timestamp: do not block the save
        event_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO detections "
            "(event_id, camera_id, plate_text, confidence, source, timestamp) "
            "VALUES (?,?,?,?,?,?)",
            (event_id, camera_id, plate, round(confidence, 3), source, ts),
        )
        alert_created = False
        if plate in WATCHLIST and not _active_alert_exists(conn, plate):
            conn.execute(
                "INSERT INTO alerts "
                "(plate_text, camera_id, detection_event_id, "
                " alert_type, message, status, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (plate, camera_id, event_id, "watchlist",
                 WATCHLIST[plate], "active", ts),
            )
            alert_created = True
        conn.commit()
        return True, {
            "event_id": event_id,
            "plate_text": plate,
            "confidence": round(confidence, 3),
            "corrected": corrected,
            "alert_created": alert_created,
        }
    finally:
        conn.close()


def overview_stats(db_path=DB_PATH, region=None):
    """KPIs + per-camera / per-hour breakdowns. Empty DB safe."""
    detections = list_detections(db_path, region=region, limit=10000)
    cameras = list_cameras(db_path, region=region)
    alerts = list_alerts(db_path, status="active")
    total = len(detections)
    unique = len({d["plate_text"] for d in detections})
    online = sum(1 for c in cameras if c["status"] == "online")
    per_camera = {}
    per_hour = {h: 0 for h in range(24)}
    for d in detections:
        per_camera[d["camera_id"]] = per_camera.get(d["camera_id"], 0) + 1
        try:
            ts = datetime.fromisoformat(d["timestamp"])
            per_hour[ts.hour] = per_hour.get(ts.hour, 0) + 1
        except ValueError:
            continue
    return {
        "total_detections": total,
        "unique_vehicles": unique,
        "cameras_online": online,
        "cameras_total": len(cameras),
        "active_alerts": len(alerts),
        "per_camera": per_camera,
        "per_hour": per_hour,
    }


# ------------------------------------------------------------- detection
# Loaded lazily and cached in module globals so the dashboard pages
# never pay the EasyOCR start-up cost (~20-30 s first time).
_YOLO_MODEL = None
_OCR_READER = None


def _load_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    from ultralytics import YOLO
    path, message = ensure_model()
    if path is None:
        raise RuntimeError(message)
    try:
        _YOLO_MODEL = YOLO(path)
    except Exception as first_exc:  # noqa: BLE001
        # PyTorch 2.6+ defaults torch.load(weights_only=True), which
        # ultralytics 8.2.51 does not handle for .pt checkpoints
        # ("WeightsUnpickler: DetectionModel not an allowed global").
        # This model file is SHA256-verified in ensure_model(), so
        # loading it with weights_only=False is safe. Retry once
        # with that default, then restore torch.load.
        import torch

        orig_load = torch.load

        def _load_no_weights_only(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)

        torch.load = _load_no_weights_only
        try:
            _YOLO_MODEL = YOLO(path)
        except Exception:  # noqa: BLE001
            raise first_exc
        finally:
            torch.load = orig_load
        print(
            "[detect] YOLO loaded with the weights_only=False "
            "fallback (torch 2.6+ default blocks ultralytics 8.2.51)",
            file=sys.stderr,
        )
    return _YOLO_MODEL


def _load_ocr():
    global _OCR_READER
    if _OCR_READER is not None:
        return _OCR_READER
    import easyocr
    _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR_READER


def _ocr_once(reader, plate_image):
    """One EasyOCR pass on a crop. Returns (text, confidence)."""
    if plate_image is None or getattr(plate_image, "size", 0) == 0:
        return "", 0.0
    try:
        results = reader.readtext(
            plate_image, allowlist=list(_OCR_ALLOWLIST)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[detect] EasyOCR failed: {exc}", file=sys.stderr)
        return "", 0.0
    if not results:
        return "", 0.0
    try:
        best = max(results, key=lambda item: float(item[2]))
        return str(best[1]).replace(" ", "").upper(), round(float(best[2]), 3)
    except (IndexError, TypeError, ValueError):
        return "", 0.0


def recognize_plate_text(reader, plate_image):
    """OCR a plate crop, retrying once on an upscaled grayscale crop
    (the fix for EasyOCR misreading 0 as O and 2 as z)."""
    import cv2
    if plate_image is None or getattr(plate_image, "size", 0) == 0:
        return "", 0.0
    text, confidence = _ocr_once(reader, plate_image)
    _, valid, _, _ = validate_plate_text(text)
    if valid and confidence >= MIN_CONFIDENCE:
        return text, confidence
    try:
        upscaled = cv2.resize(
            plate_image, None, fx=2.0, fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        if len(upscaled.shape) == 3:
            upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    except Exception as exc:  # noqa: BLE001
        print(f"[detect] OCR upscale failed: {exc}", file=sys.stderr)
        return text, confidence
    retry_text, retry_confidence = _ocr_once(reader, upscaled)
    _, retry_valid, _, _ = validate_plate_text(retry_text)
    if retry_valid and not valid:
        return retry_text, retry_confidence
    if valid and not retry_valid:
        return text, confidence
    if retry_confidence > confidence:
        return retry_text, retry_confidence
    return text, confidence


def detect_plates_in_image(image_bgr):
    """Run YOLO + EasyOCR on one BGR image (a webcam photo or an
    uploaded picture). Returns a list of dicts, one per plate box
    found, each with: box, yolo_conf, raw_text, ocr_conf, plate_text,
    valid, corrected, reason, confidence, saved (bool, set by the
    caller after save_detection), skip_log (raw OCR log lines).
    Never raises for a bad image: returns [] and logs to stderr.
    """
    results_out = []
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        print("[detect] empty image, nothing to detect", file=sys.stderr)
        return results_out
    try:
        model = _load_yolo()
    except Exception as exc:  # noqa: BLE001
        print(f"[detect] could not load YOLO model: {exc}", file=sys.stderr)
        return results_out
    try:
        reader = _load_ocr()
    except Exception as exc:  # noqa: BLE001
        print(f"[detect] could not start EasyOCR: {exc}", file=sys.stderr)
        return results_out
    try:
        results = model(image_bgr)
    except Exception as exc:  # noqa: BLE001
        print(f"[detect] YOLO detection failed: {exc}", file=sys.stderr)
        return results_out
    if not results:
        return results_out
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return results_out
    height, width = image_bgr.shape[0], image_bgr.shape[1]
    for box in boxes:
        try:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            yolo_conf = round(float(box.conf[0]), 3)
        except (IndexError, TypeError, ValueError):
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 20 or y2 - y1 < 8:
            continue  # too small to read
        crop = image_bgr[y1:y2, x1:x2]
        raw_text, ocr_conf = recognize_plate_text(reader, crop)
        plate, valid, corrected, reason = validate_plate_text(raw_text)
        confidence = combined_confidence(yolo_conf, ocr_conf)
        # Log the raw OCR on every skip, so a failed read is visible
        # instead of silently vanishing (the old 7-found / 0-sent bug).
        skip_log = []
        if not valid:
            skip_log.append(
                f"skipped invalid: raw='{raw_text or '(empty)'}' "
                f"cleaned='{plate or '(empty)'}' ocr_conf={ocr_conf} "
                f"yolo_conf={yolo_conf} ({reason})"
            )
        elif confidence < MIN_CONFIDENCE:
            skip_log.append(
                f"skipped low confidence {confidence}: plate='{plate}' "
                f"raw='{raw_text or '(empty)'}' ocr_conf={ocr_conf} "
                f"yolo_conf={yolo_conf}"
            )
        for line in skip_log:
            print(f"[detect] {line}", file=sys.stderr)
        results_out.append({
            "box": (x1, y1, x2, y2),
            "yolo_conf": yolo_conf,
            "raw_text": raw_text,
            "ocr_conf": ocr_conf,
            "plate_text": plate,
            "valid": valid,
            "corrected": corrected,
            "reason": reason,
            "confidence": confidence,
            "skip_log": skip_log,
        })
    return results_out


def detect_and_save(image_bgr, camera_id, db_path=DB_PATH):
    """Detect plates in one image and save the valid ones.

    Returns (detections_list, saved_count). Each detection dict
    gains 'saved' and 'save_info' keys.
    """
    found = detect_plates_in_image(image_bgr)
    saved = 0
    for det in found:
        det["saved"] = False
        det["save_info"] = {}
        if not det["valid"]:
            continue
        if det["confidence"] < MIN_CONFIDENCE:
            continue
        ok, info = save_detection(
            camera_id, det["plate_text"], det["confidence"],
            db_path=db_path, source="captured",
        )
        det["saved"] = ok
        det["save_info"] = info
        if ok:
            saved += 1
    return found, saved
