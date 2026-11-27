@echo off
REM Buildathon - local only. Double-click to start the backend.
REM Starts ONLY the backend (uvicorn). The pipeline does NOT
REM start here - it starts only when you tap Start Camera in
REM dashboard.html (POST /api/pipeline/start).
cd /d "%~dp0"
start "Backend" cmd /k "cd anpr-backend && venv\Scripts\activate && uvicorn app.main:app --reload"
timeout /t 5 /nobreak >nul
start "" "dashboard\dashboard.html"
exit
