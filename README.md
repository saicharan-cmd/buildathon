# Buildathon — Hyderabad full stack

Hyderabad-only ANPR: FastAPI + PostgreSQL + YOLOv8 + EasyOCR.

8 cameras: HYD01 Hitech City, HYD02 Gachibowli, HYD03 Kukatpally (offline), HYD04 Banjara Hills, HYD05 Begumpet, HYD06 Secunderabad, HYD07 Charminar, HYD08 LB Nagar.

Sample plates: TS09EX1234, TS08AB7777 (stolen), TS10AB1234, TS11CD5678, TS07XY9999, TS13PQ2468, TS15MN1357.

## Run

First time: see SETUP.md.

Rerun:
```powershell
# Terminal 1, from anpr-backend
uvicorn app.main:app --reload

# Terminal 2, from anpr-backend
python app\yolo_pipeline.py
```

Then open dashboard/dashboard.html.
