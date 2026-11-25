# Buildathon — Hyderabad full stack (FastAPI + PostgreSQL + YOLOv8 + EasyOCR)

Branch: `feature/full-stack-hyd` — everything here is on that branch.
Hyderabad only. No other city, no other cameras.

Live backend: not deployed from this branch yet — run it on your
laptop first (see [`SETUP.md`](SETUP.md)). The Render deploy files
are included in `anpr-backend/` (`requirements-render.txt` and the
`DATABASE_URL` fix in `app/database.py`) for later.

Cameras: 8, all Hyderabad (Telangana), TS plates. 7 online, 1 offline
(HYD03 Kukatpally).

| Camera | Area | Status |
| --- | --- | --- |
| HYD01 | Hitech City | online |
| HYD02 | Gachibowli | online |
| HYD03 | Kukatpally | offline |
| HYD04 | Banjara Hills | online |
| HYD05 | Begumpet | online |
| HYD06 | Secunderabad | online |
| HYD07 | Charminar | online |
| HYD08 | LB Nagar | online |

Sample data: the seed script (`python -m app.seed_sample`) adds
sample detections for Hyderabad plates only — `TS09EX1234`,
`TS08AB7777` (stolen, creates an alert), `TS10AB1234`, `TS11CD5678`,
`TS07XY9999`, `TS13PQ2468`, `TS15MN1357` — labelled as sample. It
only runs when the detections table is empty, so it never duplicates.

## Layout

- `anpr-backend/` — FastAPI app in `app/`, pipeline in
  `app/yolo_pipeline.py`, seed script in `app/seed_sample.py`,
  setup and run docs in `README.md` and `PIPELINE.md`,
  dependencies in `requirements.txt`.
- `dashboard/dashboard.html` — the dashboard: one static file, no
  build step. Hyderabad only, 8 cameras with map, stats + hourly
  chart, "Track any plate" (track + alerts together). Opens against
  your laptop backend at `http://localhost:8000` by default.

## Quick start

First time on a new laptop? Follow [`SETUP.md`](SETUP.md) — 7 steps,
Windows / VS Code. Already set up? Just the 2-command rerun:

```bash
# Terminal 1 — backend (from anpr-backend)
uvicorn app.main:app --reload

# Terminal 2 — pipeline (from anpr-backend, backend running first)
python app\yolo_pipeline.py
```

See [`anpr-backend/README.md`](anpr-backend/README.md) for the backend
and [`anpr-backend/PIPELINE.md`](anpr-backend/PIPELINE.md) for the
detection pipeline.

## Endpoints (all 14)

Detections: `POST /api/detections`, `GET /api/detections`,
`GET /api/detections/{event_id}`. Cameras: `GET /api/cameras`,
`GET /api/cameras/{camera_id}`. Analytics: `GET /api/analytics/summary`,
`/hourly`, `/daily`, `/cameras`. Vehicle: `GET /api/vehicles/{plate_text}/trajectory`.
Alerts: `POST /api/alerts`, `GET /api/alerts`, `PATCH /api/alerts/{alert_id}`.
