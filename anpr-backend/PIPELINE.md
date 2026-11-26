# Buildathon — YOLOv8 + EasyOCR + OpenCV pipeline

Adds camera/video plate detection to the Buildathon FastAPI backend. Hyderabad cameras
(HYD01-HYD08), and it talks to this repo's PostgreSQL backend.

## What it does — the 7 steps

1. **Process camera/video input** — OpenCV `VideoCapture`: a video
   file, an RTSP stream, or a webcam (`VIDEO_SOURCE=0`).
2. **Extract video frames** — reads frames, processes every
   `FRAME_SKIP`-th frame (default 5) so it keeps up on CPU.
3. **Detect number plates using YOLOv8** — a plate-trained YOLOv8
   model returns plate boxes + YOLO confidence. See the warning
   below about weights.
4. **Recognize plate text using EasyOCR** — EasyOCR reads each
   plate crop (A-Z0-9 allowlist only). If the first read fails
   validation or is low-confidence, the crop is upscaled 2x +
   grayscaled and read again; the best valid read is kept. An
   empty result returns no text instead of crashing.
5. **Calculate detection confidence** — simple average of the YOLO
   and OCR confidences, clamped to 0–1. Only valid plates at/above
   `MIN_CONFIDENCE` (default 0.5) are sent.
6. **Generate structured detection results** — exactly
   `{"camera_id", "plate_text", "confidence"}`.
7. **Send detection results to your FastAPI endpoint** —
   `POST /api/detections`. The backend makes up `event_id` and
   `timestamp` itself.

## Error handling

A failure in one frame or one plate never crashes the run:

- Camera/video that will not open → clear error, exit, no silent
  empty loop.
- Missing or unloadable YOLO model / EasyOCR → clear error at
  startup, exit before any frames are processed.
- Empty OCR result, empty or out-of-bounds plate crop, YOLO failure
  on one frame → that frame/plate is logged and skipped.
- API send failure → retried once, then the detection is appended
  to `failed_detections.jsonl` (with the error and a timestamp)
  instead of being lost silently, and the loop continues.
- The camera is always released (`finally`), even on Ctrl+C.
- A summary is printed at the end: frames read/processed, plates
  found, valid, sent, failed sends, skipped duplicates, skipped
  invalid, skipped low-confidence.

## Duplicate detection

Two layers, both 30 seconds:

- **Pipeline (in memory):** the same plate from the same camera
  is not re-sent within `DEDUP_SECONDS` (default 30).
- **Backend (`POST /api/detections`):** if the same `plate_text` +
  `camera_id` was already stored within the last 30 seconds, the
  existing row is returned with `"duplicate": true` and no new row
  is created. A failed database commit is rolled back and returns
  HTTP 500 instead of crashing.

## Plate validation

Only text matching the Indian plate format is sent:
2 letters + 2 digits + 1–2 letters + 4 digits, e.g. `TS09EX1234`.
A position-aware fix (e.g. `O` → `0` where a digit is required)
is only applied to 10-character text, and the result is flagged
as corrected in the log line. An 11-character read may drop one
leading/trailing junk character and retry the fix; a 9-character
read is never repaired (that would invent a missing digit).
Skipped plates are logged with the raw OCR text and both
confidences. Anything else is skipped as invalid.

## Camera validation

`POST /api/detections` only accepts a known `camera_id` from
`app/data.py` — HYD01-HYD08 only. Any other value is a
HTTP 400 and nothing is stored. An empty `plate_text` is also a
400. An offline camera (HYD03 Kukatpally) is a known camera,
so it is not rejected — it just happens to be offline right now.

## Pagination

`GET /api/detections` now takes `limit` (default 50, max 200) and
`offset` (default 0), still newest first, still supports
`?camera_id=HYD01`. The response shape is unchanged:
`{"total_detections": <filtered total>, "detections": [...]}` —
`total_detections` is the filtered total, not the page size.

## YOLO weights — read this first

`yolov8n.pt` (COCO) has **no plate class**, so it cannot detect
plates. You need a plate-trained YOLOv8 `.pt` file (class 0 =
plate), placed at `anpr-backend/models/yolov8-plate.pt`, and set
`MODEL_PATH` to it if it is elsewhere. **The model file is NOT in
the repo and must NOT be committed** — it is ignored by
`.gitignore`.

## Setup

Run everything from the `anpr-backend` folder — that is where
`requirements.txt`, `.env`, `app/` and `models/` live. The
commands below are Windows (Cherry's machine); on macOS / Linux
use `python3`, `source venv/bin/activate` and `cp` instead of
`copy`.

```bash
cd anpr-backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env         # edit DATABASE_URL / camera if needed
python -m app.test_db          # check PostgreSQL connects
python -m app.create_tables    # create the detections table (once)
```

Do not run `python app/create_tables.py` or
`python app/test_db.py` as scripts — they import `app.database`,
so they only work as modules (`python -m app.create_tables`,
`python -m app.test_db`) from the `anpr-backend` folder.

`.env.example` documents every setting: `DATABASE_URL`,
`CAMERA_ID`, `VIDEO_SOURCE`, `FASTAPI_URL`, `MODEL_PATH`,
`FRAME_SKIP`, `DEDUP_SECONDS`, `MIN_CONFIDENCE`,
`API_TIMEOUT_SECONDS`, `FAILED_LOG`.

Settings: the backend and the pipeline both load `.env` from the
`anpr-backend` folder at startup. A real environment variable
always wins over `.env`. A bad numeric setting in `.env`
(e.g. `FRAME_SKIP=abc`) falls back to its default with a warning
instead of crashing the pipeline.

## Run it

Two terminals, both in the `anpr-backend` folder.

Terminal 1 — backend:

```bash
cd anpr-backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs — that is the backend.

Terminal 2 — pipeline:

```bash
cd anpr-backend
venv\Scripts\activate
python app\yolo_pipeline.py
```

The pipeline reads its settings from `.env` in this folder, so
there is nothing to export by hand. `MODEL_PATH` and a video-file
`VIDEO_SOURCE` are resolved relative to this folder as a
fallback, and failed sends are logged to
`failed_detections.jsonl` in this folder.

Cameras must match the backend (`app/data.py`): HYD01 Hitech City,
HYD02 Gachibowli, HYD03 Kukatpally (offline), HYD04 Banjara Hills,
HYD05 Begumpet, HYD06 Secunderabad, HYD07 Charminar, HYD08 LB Nagar
(all online except HYD03).

## What it does NOT do

- No vehicle tracking / trajectory, no alerts, no watchlist —
  the backend has none of those yet either.
- No auth — anyone can POST to the backend, same as before.
- No frontend / map.

## Notes

- Python 3.10+ (Cherry's machine is Python 3.12.15).
  `requirements.txt` is unpinned in this repo, so no specific
  ultralytics / NumPy version is claimed here — install from
  `requirements.txt` and use the versions it pulls in.
- EasyOCR downloads its models on first use (internet needed once).
- Tested in a VM on Hyderabad plates. Clean Telangana plates
  (`TS09EX1234`, `TS08AB7777`) send fine. Known hard-case limit: on
  hard TS videos (200 px plate, skewed + blurred + dimmed) YOLO finds
  no plate at all, even at confidence 0.10 — that is a model limit,
  not a pipeline bug.
- The indexes in `models.py` only take effect on a fresh table
  created by `create_tables.py`; an existing `detections` table
  needs them added manually.
