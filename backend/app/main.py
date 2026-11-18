"""PathSnap API (Laptop 1). Owner: D

Run from the repository root:
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.predict import pipeline, router as prediction_router

app = FastAPI(title="PathSnap API",
              description="GPS map matching and route reconstruction system",
              version="1.0.0")

# The UI runs on a different laptop, so allow cross-origin requests on the LAN.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(prediction_router)


@app.on_event("startup")
def load_model():
    pipeline()          # load the model once, before the first request


@app.get("/")
def root():
    return {"project": "PathSnap", "message": "GPS Map Matching and Route Reconstruction API",
            "status": "running", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "PathSnap API",
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/v1/status")
def system_status():
    p = pipeline()
    return {"system": "PathSnap", "inference_ready": True, "model_loaded": True,
            "decoder_connected": True, "prediction_endpoint": "pipeline",
            "model": p.model_name, "trips_dir": str(p.trips_dir), "trips_available": len(p.trip_dirs())}
