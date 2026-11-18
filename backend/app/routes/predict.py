"""API routes. Owner: D"""
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from ..schemas.predict import PredictionRequest, PredictionResponse, TripList
from ..services.pipeline import Pipeline

router = APIRouter(prefix="/api/v1", tags=["Prediction"])
log = logging.getLogger("uvicorn.error")
_pipeline = None


def pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


@router.get("/trips", response_model=TripList)
def list_trips(request: Request):
    trips = pipeline().list_trips()
    log.info("TRIPS requested by %s -> %d trips", request.client.host if request.client else "?", len(trips))
    return {"trips": trips, "count": len(trips)}


@router.post("/predict", response_model=PredictionResponse)
def predict_route(body: PredictionRequest, request: Request):
    log.info("PREDICT request from %s: trip_id=%s", request.client.host if request.client else "?", body.trip_id)
    t0 = time.perf_counter()
    try:
        r = pipeline().predict(body.trip_id.strip())
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown trip_id '{body.trip_id}'")
    r["start"] = {k: r["gps_points"][0][k] for k in ("lat", "lon")}
    r["end"] = {k: r["gps_points"][-1][k] for k in ("lat", "lon")}
    r["matched_points"] = [p for seg in r["route_segments"] for p in seg]
    r["points_processed"] = r["summary"]["gps_points"]
    r["message"] = f"Map-matched {r['summary']['gps_points']} GPS points in {time.perf_counter() - t0:.2f}s"
    return r
