# Buildathon — Streamlit-only app (local only)

One app, one command. No Postgres, no FastAPI / uvicorn, no second
terminal, no `.env`. Just clone, install, run Streamlit — the app
creates its own SQLite database (`buildathon.db`), seeds the 11
cameras and a few labelled demo detections, and downloads the plate
model (~6 MB) the first time you use the camera page.

## First time (Windows + VS Code)

Install once: Python 3.12 (tick "Add python to PATH") and Git.
In VS Code, open a terminal (Terminal menu > New Terminal).

```powershell
git clone https://github.com/saicharan-cmd/buildathon.git
cd buildathon
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser. Leave the terminal
running while you use the app. Stop it with `Ctrl + C`.

## Every time after this (2 commands)

```powershell
venv\Scripts\activate
streamlit run app.py
```

## What is in the app

- **Overview** — totals, detections per camera, detections by hour.
- **Detections** — searchable table (filter by camera / plate).
  `DEMO` = seeded demo sightings, `CAPTURED` = plates this app read.
- **Live Capture** — pick a camera, take a photo with your laptop
  webcam (or upload a photo). YOLOv8 + EasyOCR reads the plate and
  saves it if it is a valid Indian plate at confidence 0.5 or higher.
- **Track any plate** — enter any plate, see its track AND its alerts
  together, across Odisha and Hyderabad.
- **Cameras** — all 11 cameras (3 Odisha, 8 Hyderabad) with a map.
- **Alerts** — watchlist. `TS08AB7777` is stolen: any detection of
  it creates an alert here automatically.

Try these plates: `TS09EX1234`, `TS08AB7777`, `OD02AB1234`.

## If something fails

- `pip install ultralytics` stalls — it is downloading the CUDA
  torch wheel. Cancel it, install CPU torch first, then install
  again:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
  ```
- First camera use is slow — the model downloads once (~6 MB), then
  EasyOCR starts (about 20–30 seconds the first time). That is
  normal. After that, each photo takes a few seconds.
- `No plate found` on a photo — hold the plate closer, straight on,
  in good light. Small, angled or blurry plates are often missed;
  that is a limit of the model, not a crash. The app shows the raw
  OCR read so you can see what it saw.
- `streamlit: command not found` — you are not in the venv. Run
  `venv\Scripts\activate` first (from the `buildathon` folder).
- To start fresh, delete `buildathon.db` and run the app again —
  it re-creates and re-seeds itself. Never commit `buildathon.db`
  or `models/*.pt` (both are gitignored).
