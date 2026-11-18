from fastapi import APIRouter, HTTPException
from ..schemas.predict import PredictionRequest, PredictionResponse
from ..services.inference import run_prediction

router = APIRouter(prefix="/api/v1", tags=["Prediction"])

@router.post("/predict", response_model=PredictionResponse)
def predict_route(request: PredictionRequest):
    try:
        return run_prediction(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
