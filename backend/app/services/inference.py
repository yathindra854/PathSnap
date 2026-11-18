from ..schemas.predict import PredictionRequest, PredictionResponse, MatchedPoint


def run_mock_prediction(request: PredictionRequest) -> PredictionResponse:
    # TEMPORARY: Echoes the input GPS coordinates.
    # This is for testing API integration only, not actual map matching.

    matched = [
        MatchedPoint(lat=point.lat, lon=point.lon)
        for point in request.points
    ]

    return PredictionResponse(
        trip_id=request.trip_id,
        status="success",
        mode="mock",
        matched_points=matched,
        points_processed=len(request.points),
        message="Mock inference only. Real map matching is not connected yet."
    )
