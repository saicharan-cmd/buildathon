@echo off
REM Buildathon - local only. Closes the Backend window started by run.bat.
REM Only the window titled Backend is closed. The pipeline runs as a
REM child of the backend, so it stops with it. Nothing else is touched.
taskkill /FI "WINDOWTITLE eq Backend*" /F
echo Done. If the Backend window is still open, close it by hand.
pause
