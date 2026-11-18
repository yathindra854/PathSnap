from pydantic import BaseModel, Field
from typing import List


class GPSPoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    timestamp: float


class PredictionRequest(BaseModel):
    trip_id: str
    points: List[GPSPoint] = Field(min_length=2)


class MatchedPoint(BaseModel):
    lat: float
    lon: float


class PredictionResponse(BaseModel):
    trip_id: str
    status: str
    mode: str
    matched_points: List[MatchedPoint]
    points_processed: int
    message: str