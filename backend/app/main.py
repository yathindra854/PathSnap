from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from backend.app.routes.predict import router as prediction_router

app = FastAPI(
    title="PathSnap API",
    description="GPS map matching and route reconstruction system",
    version="0.1.0"
)

# Allow the React frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
        "version": "0.1.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap API",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/v1/status")
def system_status():
    return {
        "system": "PathSnap",
        "stage": "real inference integrated",
        "inference_ready": True,
        "model_loaded": True,
        "decoder_connected": True,
        "prediction_endpoint": "hgb_viterbi"
    }



