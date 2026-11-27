# Buildathon — Hyderabad full stack, first-time setup (7 steps, Windows + VS Code)

Hyderabad only: 8 cameras (HYD01–HYD08), TS plates.

Paste every command as-is, unless a step says **type** — pasting a
password does not work, the characters are hidden while you type.

## Before you start

Install these once: Python 3.12 (tick "Add python to PATH" during
install), PostgreSQL 17, and Git. In VS Code, open a terminal
( Terminal menu > New Terminal ) — every command below runs there.

## Step 1 — Get the code

```powershell
cd D:\
git clone https://github.com/saicharan-cmd/buildathon.git
cd buildathon
```

## Step 2 — Make the database

```powershell
psql -U postgres
```

**Type, don't paste:** at the `Password:` prompt, type the password
you set when you installed PostgreSQL. Nothing shows while you type —
that is normal. Then, at the `postgres=#` prompt, paste these two lines one at a
time:

```sql
CREATE USER buildathon WITH PASSWORD 'buildathon';
CREATE DATABASE buildathon OWNER buildathon;
```

Then type `\q` and press Enter to leave psql.

If `psql` is not found, close VS Code, add
`C:\Program Files\PostgreSQL\17\bin` to your system PATH, and reopen
VS Code.

## Step 3 — Virtual environment + install

```powershell
cd anpr-backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

This installs FastAPI, the YOLOv8 + EasyOCR pipeline, and PostgreSQL
support. It takes a few minutes the first time.

## Step 4 — Point the backend at your database + seed sample plates

```powershell
copy .env.example .env
```

Open `.env` in VS Code and check this one line (change the password
only if your Step 2 password was different):

```text
DATABASE_URL=postgresql+psycopg2://buildathon:buildathon@localhost:5432/buildathon
```

Then create the tables and seed the Hyderabad sample data (this also
creates the `alerts` table):

```powershell
python -m app.test_db
python -m app.create_tables
python -m app.seed_sample
```

`test_db` must print OK / exit 0. If it says `DATABASE_URL is
missing`, you are not in the `anpr-backend` folder. `seed_sample`
adds sample detections for Hyderabad plates only — `TS09EX1234`,
`TS08AB7777` (stolen, makes an alert), `TS10AB1234`, `TS11CD5678`,
`TS07XY9999`, `TS13PQ2468`, `TS15MN1357` — and only runs when the
detections table is empty, so running it twice adds nothing.

## Step 5 — Start the backend (Terminal 1)

This is Terminal 1. Paste only this, then leave it alone:

```powershell
uvicorn app.main:app --reload
```

This terminal is now busy running the backend. Do NOT paste
anything else in this terminal — Step 6 needs a different terminal.
Leave Terminal 1 running. Open http://127.0.0.1:8000/ — you
should see `{"message":"Welcome to ANPR Backend!","status":"Running"}`.
http://127.0.0.1:8000/docs lists every endpoint.
http://127.0.0.1:8000/api/cameras should list exactly 8 Hyderabad
cameras, 7 online (HYD03 Kukatpally offline).

## Step 6 — Get the plate model + run the pipeline (Terminal 2)

Open a **second** terminal in VS Code ( Terminal menu > New
Terminal ). This is Terminal 2 — Terminal 1 is still busy running
the backend, so do not paste these commands there. Then paste:

```powershell
cd D:\buildathon\anpr-backend
venv\Scripts\activate
New-Item -ItemType Directory -Force models
Invoke-WebRequest https://huggingface.co/joker5914/yolov8n-license-plate/resolve/main/best.pt -OutFile models\yolov8-plate.pt
python app\yolo_pipeline.py
```

The `models` folder is **not** in the repo, so you must create it
first with the `New-Item` line above. Without it, the download
fails with `Could not find a part of the path
'...\models\yolov8-plate.pt'` and the pipeline then says
`YOLO model file not found`.

The model is ~6 MB and is **not** in the repo — this download is the
only way to get it. The pipeline uses your laptop webcam by default
(`VIDEO_SOURCE=0` in `.env`); to use a video file instead, set
`VIDEO_SOURCE` in `.env` to the file path, e.g.
`VIDEO_SOURCE=D:\plate.mp4`.

Hold a fake plate up to the camera (for example `TS09EX1234` written
on paper — no real plates needed). The pipeline only sends valid
plates at confidence 0.5 or higher. Check it stored — open
http://127.0.0.1:8000/api/detections?limit=5 in your browser.

## Step 7 — Open the dashboard

Double-click `dashboard\dashboard.html` in the repo. Tap **Connect to
backend** — it tries your laptop backend at `http://localhost:8000`
(make sure Step 5 is still running). The dashboard is Hyderabad
only: 8 cameras, stats + hourly chart, and **Track any plate** shows
that plate's track and its alerts together. Try `TS09EX1234` and
`TS08AB7777` (stolen — it has an alert).

## Every time after this (2-command rerun)

Terminal 1 (backend, from `anpr-backend`):

```powershell
venv\Scripts\activate
uvicorn app.main:app --reload
```

Terminal 2 (pipeline, from `anpr-backend`, backend running first):

```powershell
venv\Scripts\activate
python app\yolo_pipeline.py
```

The browser / dashboard only shows live data while the backend is
running. Stop a terminal with `Ctrl + C`.

## If something fails

- `psycopg2` / `No module named 'psycopg'` — you are not in the venv,
  or you installed with plain `postgresql://` in `.env`. The URL must
  start `postgresql+psycopg2://`.
- `pip install ultralytics` stalls — it is downloading the CUDA
  torch wheel. Cancel it, install CPU torch first (see the PyTorch
  site, CPU wheel), then `pip install -r requirements.txt` again.
- `Could not find a part of the path '...\models\yolov8-plate.pt'`
  — the `models` folder does not exist yet. In Terminal 2, run
  `New-Item -ItemType Directory -Force models` first (Step 6),
  then run the `Invoke-WebRequest` line again.
- `yolov8n.pt has no plate class` — the model file is wrong. You
  need the `best.pt` from Step 6 at `models\yolov8-plate.pt`.
- Dashboard says "Couldn't reach" the backend — the backend is not
  running (Step 5), or you typed the wrong address. Sample data
  shows until it connects.
- `total_detections` is 0 — run `python -m app.seed_sample` (Step 4)
  for the sample plates, or send a plate with the pipeline (Step 6).
  The pipeline only sends valid plates at 0.5+ confidence; blurry or
  angled plates get skipped (that is by design, not a crash).
- Hard / blurred small plates are often missed — that is a limit of
  the model, not a crash. See `PIPELINE.md`.
