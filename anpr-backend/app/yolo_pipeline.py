# Buildathon — YOLOv8 + EasyOCR + OpenCV pipeline.
# Sends {camera_id, plate_text, confidence} to POST /api/detections.
# Needs a plate-trained model at models/yolov8-plate.pt (NOT in the
# repo). yolov8n.pt has no plate class and will not work.

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

_BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is in requirements.txt
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(_BASE_DIR / ".env", override=False)
    load_dotenv(Path.cwd() / ".env", override=False)


def _env_number(name, default, cast):
    """Read a numeric setting, falling back to the default on a bad value."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return cast(str(raw).strip())
    except ValueError:
        print(
            f"[pipeline] warning: {name}={raw!r} is not a number, "
            f"using default {default}",
            file=sys.stderr,
        )
        return default


def _env_int(name, default):
    return _env_number(name, default, int)


def _env_float(name, default):
    return _env_number(name, default, float)


def _resolve_relative(path_str, default_base=_BASE_DIR):
    """Resolve a relative path: current folder first, then anpr-backend."""
    if not path_str:
        return path_str
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    cwd_path = Path.cwd() / path
    if cwd_path.is_file():
        return str(cwd_path)
    base_path = default_base / path
    if base_path.is_file():
        return str(base_path)
    return str(base_path)


CAMERA_ID = os.getenv("CAMERA_ID", "HYD01") or "HYD01"
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0") or "0"
FASTAPI_URL = os.getenv(
    "FASTAPI_URL", "http://localhost:8000/api/detections"
) or "http://localhost:8000/api/detections"
MODEL_PATH = _resolve_relative(
    os.getenv("MODEL_PATH", "models/yolov8-plate.pt")
    or "models/yolov8-plate.pt"
)
FRAME_SKIP = _env_int("FRAME_SKIP", 5)
DEDUP_SECONDS = _env_int("DEDUP_SECONDS", 30)
MIN_CONFIDENCE = _env_float("MIN_CONFIDENCE", 0.5)
API_TIMEOUT_SECONDS = _env_float("API_TIMEOUT_SECONDS", 5)
if API_TIMEOUT_SECONDS <= 0:
    print(
        "[pipeline] warning: API_TIMEOUT_SECONDS must be > 0, "
        "using default 5",
        file=sys.stderr,
    )
    API_TIMEOUT_SECONDS = 5.0
_failed_log_raw = os.getenv("FAILED_LOG", "failed_detections.jsonl") or "failed_detections.jsonl"
FAILED_LOG = (
    _failed_log_raw
    if Path(_failed_log_raw).is_absolute()
    else str(_BASE_DIR / _failed_log_raw)
)

INDIAN_PLATE_RE = re.compile(r"^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})$")
_DIGIT_FIX = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
_LETTER_FIX = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
_OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_recently_sent = {}


def _log(message):
    print(f"[pipeline] {message}", flush=True)


def _clamp_confidence(value):
    """Clamp a confidence value to 0..1, rounded to 3 decimals."""
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return 0.0


def _position_fix(cleaned):
    """Fix misread characters in a 10-character read, by position."""
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
    """Validate plate text as an Indian plate.

    Returns (plate, valid, corrected, reason). An 11-character read
    may drop one leading/trailing character; a 9-character read is
    never repaired — that would invent a missing digit.
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
                return (
                    candidate,
                    True,
                    True,
                    "dropped one leading/trailing junk character",
                )
            fixed, ok = _position_fix(candidate)
            if ok:
                return (
                    fixed,
                    True,
                    True,
                    "dropped one leading/trailing junk character "
                    "and applied position-aware correction",
                )
    return cleaned, False, False, "does not match Indian plate format"


def is_duplicate(camera_id, plate_text, dedup_seconds=DEDUP_SECONDS):
    """True if this plate+camera was already sent within the window."""
    key = (camera_id, plate_text)
    last_sent = _recently_sent.get(key)
    if last_sent is None:
        return False
    return (time.time() - last_sent) < dedup_seconds


def mark_sent(camera_id, plate_text, dedup_seconds=DEDUP_SECONDS):
    now = time.time()
    for key, sent_at in list(_recently_sent.items()):
        if now - sent_at >= dedup_seconds:
            del _recently_sent[key]
    _recently_sent[(camera_id, plate_text)] = now


def log_failed_detection(detection, error):
    """Append a failed send to the local log so it is not lost."""
    record = dict(detection)
    record["error"] = str(error)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(FAILED_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        _log(f"could not write failed-detection log: {exc}")


def process_camera_video_input(source):
    """Step 1: open a camera / video file / RTSP stream."""
    if isinstance(source, str) and source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def extract_video_frames(capture):
    """Step 2: yield frames until the video ends or a read fails."""
    while True:
        try:
            success, frame = capture.read()
        except Exception as exc:  # noqa: BLE001
            _log(f"frame read failed: {exc}")
            break
        if not success or frame is None:
            break
        yield frame


def detect_number_plates(model, frame):
    """Step 3: detect plates with YOLOv8; returns (x1, y1, x2, y2, confidence)."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    try:
        results = model(frame)
    except Exception as exc:  # noqa: BLE001
        _log(f"YOLO detection failed: {exc}")
        return []
    if not results:
        return []
    try:
        boxes = results[0].boxes
    except (IndexError, AttributeError):
        return []
    if boxes is None:
        return []

    height, width = frame.shape[0], frame.shape[1]
    plates = []
    for box in boxes:
        try:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
            confidence = float(box.conf[0])
        except (IndexError, TypeError, ValueError):
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 20 or y2 - y1 < 8:
            continue  # too small to read
        plates.append((x1, y1, x2, y2, _clamp_confidence(confidence)))
    return plates


def _ocr_once(reader, plate_image):
    """One EasyOCR pass; returns (plate_text, confidence)."""
    if plate_image is None or getattr(plate_image, "size", 0) == 0:
        return "", 0.0
    try:
        results = reader.readtext(
            plate_image, allowlist=list(_OCR_ALLOWLIST)
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"EasyOCR failed: {exc}")
        return "", 0.0
    if not results:
        return "", 0.0
    try:
        best = max(results, key=lambda item: float(item[2]))
        text, recognition_confidence = str(best[1]), float(best[2])
    except (IndexError, TypeError, ValueError):
        return "", 0.0
    return text.replace(" ", "").upper(), _clamp_confidence(recognition_confidence)


def recognize_plate_text(reader, plate_image):
    """Step 4: recognize plate text with EasyOCR, retrying once on an upscaled crop."""
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
        _log(f"OCR upscale failed: {exc}")
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


def calculate_detection_confidence(detection_confidence, recognition_confidence):
    """Step 5: average of YOLO and OCR confidence, clamped 0..1."""
    return _clamp_confidence(
        (float(detection_confidence) + float(recognition_confidence)) / 2
    )


def generate_structured_detection_results(camera_id, plate_text, confidence):
    """Step 6: the structured result sent to the API."""
    return {
        "camera_id": camera_id,
        "plate_text": plate_text,
        "confidence": _clamp_confidence(confidence),
    }


def send_detection_results(detection, endpoint):
    """Step 7: POST the detection; retries once; never raises."""
    data = json.dumps(detection).encode("utf-8")
    last_error = None
    for attempt in (1, 2):
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
                try:
                    payload = json.loads(response.read().decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    payload = {}
                return {"ok": True, "status": response.status, "data": payload, "error": None}
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:  # noqa: BLE001
                detail = ""
            last_error = f"API returned {exc.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        if attempt == 1:
            _log(f"send failed (attempt 1), retrying once: {last_error}")
            time.sleep(1)
    return {"ok": False, "status": None, "data": None, "error": last_error}


if __name__ == "__main__":
    camera_id = CAMERA_ID
    source = VIDEO_SOURCE
    if (
        isinstance(source, str)
        and not source.isdigit()
        and "://" not in source
    ):
        source = _resolve_relative(source)
    endpoint = FASTAPI_URL
    frame_skip = max(1, FRAME_SKIP)
    dedup_seconds = max(0, DEDUP_SECONDS)
    min_confidence = _clamp_confidence(MIN_CONFIDENCE)

    if not os.path.isfile(MODEL_PATH):
        print(
            f"[pipeline] error: YOLO model file not found: {MODEL_PATH}\n"
            "[pipeline] A plate-trained YOLOv8 .pt file is required here.\n"
            "[pipeline] yolov8n.pt has no plate class. See anpr-backend/PIPELINE.md.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        model = YOLO(MODEL_PATH)
    except Exception as first_exc:  # noqa: BLE001
        # PyTorch 2.6+ defaults torch.load(weights_only=True), which
        # older ultralytics does not handle for .pt checkpoints
        # ("WeightsUnpickler: DetectionModel not an allowed global").
        # The model file comes from the Hugging Face URL in SETUP.md,
        # so retry once with weights_only=False, then restore.
        try:
            import torch
            _orig_load = torch.load
            def _load_no_weights_only(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_load(*args, **kwargs)
            torch.load = _load_no_weights_only
            try:
                model = YOLO(MODEL_PATH)
            finally:
                torch.load = _orig_load
            print("[pipeline] YOLO loaded with the weights_only=False "
                  "fallback (torch 2.6+ default blocks older ultralytics)",
                  file=sys.stderr)
        except Exception:  # noqa: BLE001
            print(f"[pipeline] error: could not load YOLO model {MODEL_PATH}: {first_exc}", file=sys.stderr)
            sys.exit(1)
    try:
        reader = easyocr.Reader(["en"])
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] error: could not start EasyOCR: {exc}", file=sys.stderr)
        sys.exit(1)

    capture = process_camera_video_input(source)
    if not capture.isOpened():
        print(f"[pipeline] error: could not open video source: {source!r}", file=sys.stderr)
        sys.exit(1)

    stats = {
        "frames_read": 0,
        "frames_processed": 0,
        "plates_found": 0,
        "valid": 0,
        "sent": 0,
        "failed_send": 0,
        "skipped_duplicate": 0,
        "skipped_invalid": 0,
        "skipped_low_confidence": 0,
    }

    _log(f"camera {camera_id} | source {source!r} | sending to {endpoint}")
    _log(
        f"frame_skip={frame_skip} dedup_seconds={dedup_seconds} "
        f"min_confidence={min_confidence} model={MODEL_PATH}"
    )

    try:
        for frame in extract_video_frames(capture):
            stats["frames_read"] += 1
            if stats["frames_read"] % frame_skip != 0:
                continue
            stats["frames_processed"] += 1
            try:
                plates = detect_number_plates(model, frame)
            except Exception as exc:  # noqa: BLE001
                _log(f"frame {stats['frames_read']} failed: {exc}")
                continue
            for x1, y1, x2, y2, detection_confidence in plates:
                stats["plates_found"] += 1
                try:
                    crop = frame[y1:y2, x1:x2]
                    raw_text, recognition_confidence = recognize_plate_text(reader, crop)
                    confidence = calculate_detection_confidence(
                        detection_confidence, recognition_confidence
                    )
                except Exception as exc:  # noqa: BLE001
                    _log(f"plate in frame {stats['frames_read']} failed: {exc}")
                    continue

                plate_text, valid, corrected, reason = validate_plate_text(raw_text)
                if not valid:
                    stats["skipped_invalid"] += 1
                    _log(
                        f"invalid plate, not sent: cleaned='{plate_text or '(empty)'}' "
                        f"raw='{raw_text or '(empty)'}' ocr_conf={recognition_confidence} "
                        f"yolo_conf={detection_confidence} ({reason})"
                    )
                    continue
                if corrected:
                    _log(f"plate corrected, verify manually: {plate_text}")
                if confidence < min_confidence:
                    stats["skipped_low_confidence"] += 1
                    _log(
                        f"low confidence {confidence}, not sent: plate='{plate_text}' "
                        f"raw='{raw_text or '(empty)'}' ocr_conf={recognition_confidence} "
                        f"yolo_conf={detection_confidence}"
                    )
                    continue
                stats["valid"] += 1

                if is_duplicate(camera_id, plate_text, dedup_seconds):
                    stats["skipped_duplicate"] += 1
                    continue

                detection = generate_structured_detection_results(
                    camera_id, plate_text, confidence
                )
                result = send_detection_results(detection, endpoint)
                if result["ok"]:
                    mark_sent(camera_id, plate_text, dedup_seconds)
                    stats["sent"] += 1
                    _log(f"sent {plate_text} conf={confidence}")
                else:
                    stats["failed_send"] += 1
                    _log(f"API send failed for {plate_text}: {result['error']}")
                    log_failed_detection(detection, result["error"])
    except KeyboardInterrupt:
        _log("stopped by user.")
    finally:
        capture.release()

    _log(f"done: {stats}")
