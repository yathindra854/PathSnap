import numpy as np
import joblib
import networkx as nx
import shapely
from pathlib import Path
from pyproj import Transformer

import sys
sys.path.insert(0, str(Path('src').resolve()))
from pathsnap.data.loader import load_trip, project_trip
from pathsnap.features.candidates import build_candidates
from ..schemas.predict import PredictionRequest, PredictionResponse, MatchedPoint

MODEL_PATH = Path("artifacts/final_hgb.joblib")
DATA_DIR = Path("data/raw")
RADIUS = 50
TOP_K = 8
BETA = 50.0


def wrap_angle(z):
    return np.mod(z + np.pi, 2 * np.pi) - np.pi


def run_prediction(request: PredictionRequest) -> PredictionResponse:
    # Load road network for this trip ID (no ground-truth labels used)
    trip_dir = None

    for partition in ("student_test", "train"):
        candidate = DATA_DIR / partition / request.trip_id
        if candidate.exists():
            trip_dir = candidate
            break

    if trip_dir is None:
        raise ValueError(f"Trip road network not found: {request.trip_id}")

    trip = load_trip(str(trip_dir))

    # Replace track with submitted GPS observations
    trip.track = trip.track.iloc[:0].copy()
    trip.track["lon"] = [p.lon for p in request.points]
    trip.track["lat"] = [p.lat for p in request.points]
    trip.track["t"] = [p.timestamp for p in request.points]

    trip = project_trip(trip)

    xy = trip.track[["x", "y"]].to_numpy()
    ts = trip.track["t"].to_numpy()
    n = len(ts)

    if n < 2:
        raise ValueError("At least 2 GPS points are required")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]

    # Candidate generation (without labels)
    c = build_candidates(trip, RADIUS)

    if c.empty:
        raise ValueError("No road candidates found within 50 metres")

    # GPS movement features
    d = np.diff(xy, axis=0)
    step = np.linalg.norm(d, axis=1)
    dt = np.diff(ts)

    head = np.arctan2(d[:, 1], d[:, 0])
    head = np.append(head, head[-1] if n > 1 else 0.0)

    speed = np.append(step / np.maximum(dt, 1.0), 0.0)[:n]

    # Road arc features
    nodes = trip.nodes[["x", "y"]].to_numpy()
    fr = trip.arcs["from_node"].to_numpy()
    to = trip.arcs["to_node"].to_numpy()

    arc_head = np.arctan2(
        nodes[to, 1] - nodes[fr, 1],
        nodes[to, 0] - nodes[fr, 0]
    )
    arc_len = np.asarray(shapely.length(trip.arc_geoms))

    pt = c.point_idx.to_numpy()
    a = c.arc_id.to_numpy()
    g = c.groupby("point_idx")["perp_dist"]

    X = np.column_stack([
        c.perp_dist - g.transform("min"),
        c.proj_frac,
        np.degrees(np.abs(wrap_angle(head[pt] - arc_head[a]))),
        speed[pt],
        g.transform("size")
    ])

    c["prob"] = model.predict_proba(X)[:, 1]

    states = [
        c[c.point_idx == i].nlargest(TOP_K, "prob").reset_index(drop=True)
        for i in range(n)
    ]

    # Directed road graph
    G = nx.DiGraph()
    for arc_id, (u, v) in enumerate(zip(fr, to)):
        G.add_edge(int(u), int(v), weight=float(arc_len[arc_id]))

    dist_cache = {}

    def node_dist(source, target):
        source, target = int(source), int(target)
        if source not in dist_cache:
            dist_cache[source] = nx.single_source_dijkstra_path_length(
                G, source, weight="weight"
            )
        return dist_cache[source].get(target, np.inf)

    # Viterbi decoding
    scores = [
        np.log(np.clip(states[0].prob.to_numpy(), 1e-8, 1.0))
    ]
    backpointers = []

    for i in range(1, n):
        gps_dist = float(step[i - 1])
        prev_states = states[i - 1]
        curr_states = states[i]
        prev_scores = scores[-1]

        curr_scores = np.full(len(curr_states), -np.inf)
        back = np.zeros(len(curr_states), dtype=int)

        for j, curr in curr_states.iterrows():
            emission = np.log(max(float(curr.prob), 1e-8))
            best_score = -np.inf
            best_k = 0

            for k, prev in prev_states.iterrows():
                pa = int(prev.arc_id)
                ca = int(curr.arc_id)
                pf = float(prev.proj_frac)
                cf = float(curr.proj_frac)

                if pa == ca and cf >= pf:
                    route_dist = (cf - pf) * arc_len[pa]
                else:
                    middle = node_dist(to[pa], fr[ca])

                    if not np.isfinite(middle):
                        candidate_score = prev_scores[k] - 30.0 + emission
                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_k = k
                        continue

                    route_dist = (
                        (1 - pf) * arc_len[pa]
                        + middle
                        + cf * arc_len[ca]
                    )

                transition = -abs(route_dist - gps_dist) / BETA
                candidate_score = prev_scores[k] + transition + emission

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_k = k

            curr_scores[j] = best_score
            back[j] = best_k

        scores.append(curr_scores)
        backpointers.append(back)

    last = int(np.argmax(scores[-1]))
    path = [last]

    for back in reversed(backpointers):
        last = int(back[last])
        path.append(last)

    path.reverse()

    selected = [
        states[i].iloc[path[i]]
        for i in range(n)
    ]

    # Project matched positions onto selected road arcs and return GPS coordinates
    matched_xy = []

    for state in selected:
        arc_id = int(state.arc_id)
        fraction = float(state.proj_frac)

        geom = trip.arc_geoms[arc_id]
        point = shapely.line_interpolate_point(geom, fraction, normalized=True)

        matched_xy.append((point.x, point.y))

    inverse = Transformer.from_crs(
        f"EPSG:{trip.epsg}",
        "EPSG:4326",
        always_xy=True
    )

    matched_lon, matched_lat = inverse.transform(
        [p[0] for p in matched_xy],
        [p[1] for p in matched_xy]
    )

    matched_points = [
        MatchedPoint(lat=float(lat), lon=float(lon))
        for lat, lon in zip(matched_lat, matched_lon)
    ]

    return PredictionResponse(
        trip_id=request.trip_id,
        status="success",
        mode="hgb_viterbi",
        matched_points=matched_points,
        points_processed=n,
        message="Real HGB candidate scoring and Viterbi map matching completed."
    )

