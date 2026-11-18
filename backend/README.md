# PathSnap API (Laptop 1)

FastAPI backend that runs the full PathSnap pipeline for one trip at a time:
candidates (50 m) -> features -> tuned HistGradientBoosting -> HMM/Viterbi -> server-side scoring.
The UI (Laptop 2) only sends a trip ID. Reference routes and labels never leave this laptop.

## Setup (once)
From the repository root:
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
```
`scikit-learn==1.7.2` is pinned because the model file was saved with that version.

## Run
```bash
# default trips folder: data/raw/student_test, else ../Student folder/student_test
set PATHSNAP_TRIPS_DIR=C:\path\to\trips     # optional, e.g. the evaluator trips (macOS/Linux: export)
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
or double-click `run_api.bat` in the repository root.

`--host 0.0.0.0` makes the API reachable from Laptop 2 on the same network.
Find this laptop's address with `ipconfig` (IPv4 Address), then from Laptop 2 open
`http://<that-ip>:8000/docs`. If it does not load, allow Python through Windows Firewall
(Private networks) or open port 8000.

Each request is logged in this terminal, for example:
```
INFO:     PREDICT request from 192.168.1.20: trip_id=00000065
INFO:     PREDICT trip=00000065 points=768 cands/pt=14.1 | accuracy HMM=0.8977 classifier=0.8105 | continuity=0.980 | 0.24s
INFO:     192.168.1.20:51234 - "POST /api/v1/predict HTTP/1.1" 200 OK
```
These are the API logs the evaluators ask to see.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/api/v1/status` | model loaded, trips folder, number of trips |
| GET | `/api/v1/trips` | list of trip IDs with GPS point counts (no reference data) |
| POST | `/api/v1/predict` | body `{"trip_id": "00000065"}` -> predicted route and metrics |
| GET | `/docs` | interactive API documentation |

## `/api/v1/predict` response (main fields)
| Field | Content |
|---|---|
| `start`, `end` | `{lat, lon}` of the first and last GPS point (A and B) |
| `gps_points` | raw GPS observations `[{lat, lon, t}]` |
| `route_segments` | predicted route as polylines `[[{lat, lon}, ...], ...]`; usually one segment, a new one starts only at a GPS gap or a disconnected part of the network |
| `matched_points` | all route segments flattened into one list (kept for the earlier UI) |
| `snapped_points` | each GPS point moved onto its predicted road arc `[{point_idx, lat, lon, arc_id}]` |
| `predicted_arcs` | ordered arc IDs of the decoded route |
| `summary` | point counts, candidates per point, route length (km), route continuity with and without HMM |
| `evaluation` | computed on this laptop against the reference route: `point_accuracy_hmm`, `point_accuracy_classifier_only`, `candidate_miss_rate`, `route_arc_precision/recall/f1`; `available: false` if the trip has no reference route |
| `timing` | seconds per pipeline stage and total |
| `model` | classifier file, features, candidate radius, HMM settings |

Unknown trip IDs return HTTP 404.

## Student-test check (10 trips, this API)
Pooled point accuracy: 0.9136 with HMM, 0.8043 classifier-only (same as `scripts/05_decode_hmm.py`).
Every trip returns in under 1 s.
