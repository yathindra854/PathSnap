"""Request/response models for the PathSnap API. Owner: D

The UI sends only a trip ID. Responses never contain the reference route or labels;
`evaluation` holds metrics computed server-side.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=64, examples=["00000052"])


class TripInfo(BaseModel):
    trip_id: str
    gps_points: int


class TripList(BaseModel):
    trips: List[TripInfo]
    count: int


class GPSPoint(BaseModel):
    lat: float
    lon: float
    t: float


class LatLon(BaseModel):
    lat: float
    lon: float


class SnappedPoint(BaseModel):
    point_idx: int
    lat: float
    lon: float
    arc_id: int


class Summary(BaseModel):
    gps_points: int
    points_with_candidates: int
    mean_candidates_per_point: float
    predicted_route_arcs: int
    predicted_route_length_km: float
    route_continuity_hmm: float
    route_continuity_classifier_only: float


class Evaluation(BaseModel):
    available: bool
    reason: Optional[str] = None
    labelled_points: Optional[int] = None
    point_accuracy_hmm: Optional[float] = None
    point_accuracy_classifier_only: Optional[float] = None
    candidate_miss_rate: Optional[float] = None
    route_arc_precision: Optional[float] = None
    route_arc_recall: Optional[float] = None
    route_arc_f1: Optional[float] = None


class PredictionResponse(BaseModel):
    trip_id: str
    status: str = "success"
    mode: str = "pipeline"
    start: LatLon
    end: LatLon
    gps_points: List[GPSPoint]
    route_segments: List[List[LatLon]]
    matched_points: List[LatLon]          # flattened route, kept for the existing UI
    snapped_points: List[SnappedPoint]
    predicted_arcs: List[int]
    summary: Summary
    evaluation: Evaluation
    timing: dict
    model: dict
    points_processed: int
    message: str
