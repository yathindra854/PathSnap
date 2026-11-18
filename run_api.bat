@echo off
REM PathSnap API on Laptop 1. Run this from the PathSnap repository folder.
REM Optional: point at the evaluator trips folder, e.g.
REM   set PATHSNAP_TRIPS_DIR=C:\path\to\evaluator_trips
cd /d "%~dp0"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
