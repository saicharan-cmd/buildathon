# pipeline_control.py — local only, no auth.
# Copy this file to: anpr-backend/app/pipeline_control.py
# Then add the 2 lines in PIPELINE_CONTROL_README.txt to app/main.py
# and restart the backend (double-click run.bat again).
#
# Start Camera in dashboard.html calls POST /api/pipeline/start,
# Stop Camera calls POST /api/pipeline/stop. The pipeline runs as
# a child process of the backend, so it stops when the backend stops.
# Only one pipeline runs at a time: start does nothing if one is
# already running.

import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_PIPELINE_FILE = Path(__file__).resolve().parent / "yolo_pipeline.py"
_BACKEND_ROOT = Path(__file__).resolve().parent.parent

_proc = None
_lock = threading.Lock()


def _is_running():
    return _proc is not None and _proc.poll() is None


@router.post("/start")
def start_pipeline():
    """Start app/yolo_pipeline.py, only if it is not already running."""
    global _proc
    with _lock:
        if _is_running():
            return {"running": True, "already_running": True, "pid": _proc.pid}
        if not _PIPELINE_FILE.exists():
            return {
                "running": False,
                "error": "yolo_pipeline.py not found at " + str(_PIPELINE_FILE),
            }
        _proc = subprocess.Popen(
            [sys.executable, str(_PIPELINE_FILE)],
            cwd=str(_BACKEND_ROOT),
        )
        return {"running": True, "pid": _proc.pid}


@router.post("/stop")
def stop_pipeline():
    """Stop the pipeline started by POST /api/pipeline/start."""
    global _proc
    with _lock:
        if not _is_running():
            _proc = None
            return {"running": False, "already_stopped": True}
        proc = _proc
        _proc = None
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    return {"running": False, "stopped": True}


@router.get("/status")
def pipeline_status():
    """Tell the dashboard whether the pipeline is running right now."""
    return {"running": _is_running(), "pid": _proc.pid if _is_running() else None}
