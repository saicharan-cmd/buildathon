Buildathon - pipeline control
==========================================

Where each file goes (project folder = D:\buildathon):
  run.bat               -> D:\buildathon\run.bat
  stop.bat              -> D:\buildathon\stop.bat
  pipeline_control.py   -> D:\buildathon\anpr-backend\app\pipeline_control.py
  this file             -> read it, no need to copy it

The dashboard on this repo (dashboard\dashboard.html) stays as it is.
It has no camera button. Nothing here starts the camera on its own.

1. Backend already has the pipeline control
--------------------------------------------
pipeline_control.py is in anpr-backend\app\, and main.py already
has the 2 lines for it:

    from app.pipeline_control import router as pipeline_router
    app.include_router(pipeline_router)

No change to main.py is needed after a fresh clone.

2. Restart the backend
----------------------
Double-click stop.bat (or close the black Backend window),
then double-click run.bat. run.bat starts ONLY the backend.
The pipeline does NOT start with run.bat.

3. How the pipeline starts now
------------------------------
Start it from Terminal 2, as in SETUP.md Step 6:

    python app\yolo_pipeline.py

Stop it with Ctrl+C in Terminal 2. The backend also has these
endpoints, for checking only:

    POST http://localhost:8000/api/pipeline/start
    POST http://localhost:8000/api/pipeline/stop
    GET  http://localhost:8000/api/pipeline/status

Nothing starts the camera or the pipeline until you run one of
those. Stopping the backend (stop.bat) also stops a pipeline
that was started through the backend.

4. If the pipeline will not start
---------------------------------
  - Check the black Backend terminal window for the error.
  - Check models\yolov8-plate.pt exists in anpr-backend\models.
  - Check the venv has cv2, easyocr and ultralytics installed.
