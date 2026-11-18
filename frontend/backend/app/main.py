from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.predict import router as prediction_router

app = FastAPI(
    title="PathSnap API",
    description="GPS map matching and route reconstruction system",
    version="0.1.0",
)

# Frontend development server access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)


@app.get("/")
def root():
    return {
        "project": "PathSnap",
        "message": "GPS Map Matching and Route Reconstruction API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/status")
def system_status():
    return {
        "system": "PathSnap",
        "stage": "integration development",
        "inference_ready": False,
        "model_loaded": False,
        "decoder_connected": False,
        "prediction_endpoint": "mock",
    }