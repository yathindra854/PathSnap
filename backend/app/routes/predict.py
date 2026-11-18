from fastapi import APIRouter
from ..schemas.predict import PredictionRequest, PredictionResponse
from ..services.inference import run_mock_prediction

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_route(request: PredictionRequest):
    return run_mock_prediction(request)