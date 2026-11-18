import sys
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import shapely
from pathlib import Path

sys.path.insert(0, "src")

from pathsnap.data.loader import load_trip, project_trip
from pathsnap.features.candidates import build_candidates

MODEL_PATH = "artifacts/final_hgb.joblib"
TRIPS_DIR = Path("data/raw/student_test")
RESULTS_DIR = Path("results")

RADIUS_M = 50
TOP_K = 8
BETAS = [50.0, 25.0, 10.0]


def wrap_angle(z):
    return np.mod(z + np.pi, 2 * np.pi) - np.pi


def evaluate_trip(trip_path, model, beta):
    trip = project_trip(load_trip(str(trip_path)))
    candidates = build_candidates(trip, RADIUS_M)

    xy = trip.track[["x", "y"]].to_numpy()
    ts = trip.track["t"].to_numpy()
    n = len(ts)

    d = np.diff(xy, axis=0)
    step = np.linalg.norm(d, axis=1)
    dt = np.diff(ts)

    head = np.arctan2(d[:, 1], d[:, 0])
    head = np.append(head, head[-1] if n > 1 else 0.0)

    speed = np.append(step / np.maximum(dt, 1.0), 0.0)[:n]

    nodes = trip.nodes[["x", "y"]].to_numpy()
    fr = trip.arcs["from_node"].to_numpy()
    to = trip.arcs["to_node"].to_numpy()

    arc_head = np.arctan2(
        nodes[to, 1] - nodes[fr, 1],
        nodes[to, 0] - nodes[fr, 0]
    )

    arc_len = np.asarray(shapely.length(trip.arc_geoms))

    pt = candidates["point_idx"].to_numpy()
    a = candidates["arc_id"].to_numpy()

    g = candidates.groupby("point_idx")["perp_dist"]

    X = np.column_stack([
        candidates.perp_dist - g.transform("min"),
        candidates.proj_frac,
        np.degrees(
            np.abs(wrap_angle(head[pt] - arc_head[a]))
        ),
        speed[pt],
        g.transform("size"),
    ])

    candidates["prob"] = model.predict_proba(X)[:, 1]

    states = []
    for i in range(n):
        group = candidates[candidates.point_idx == i].nlargest(
            TOP_K, "prob"
        ).copy()
        states.append(group.reset_index(drop=True))

    # Greedy baseline
    greedy = (
        candidates.sort_values("prob", ascending=False)
        .drop_duplicates("point_idx")
        .sort_values("point_idx")
        .reset_index(drop=True)
    )

    greedy_ids = greedy.arc_id.to_numpy(dtype=int)

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

    def transition_score(prev, curr, gps_dist):
        prev_arc = int(prev.arc_id)
        curr_arc = int(curr.arc_id)

        prev_frac = float(prev.proj_frac)
        curr_frac = float(curr.proj_frac)

        if prev_arc == curr_arc and curr_frac >= prev_frac:
            route_dist = (
                (curr_frac - prev_frac) * arc_len[prev_arc]
            )
        else:
            middle = node_dist(to[prev_arc], fr[curr_arc])

            if not np.isfinite(middle):
                return -30.0

            route_dist = (
                (1 - prev_frac) * arc_len[prev_arc]
                + middle
                + curr_frac * arc_len[curr_arc]
            )

        mismatch = abs(route_dist - gps_dist)
        return -mismatch / beta

    # Viterbi
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
            options = []

            for k, prev in prev_states.iterrows():
                trans = transition_score(prev, curr, gps_dist)
                options.append(prev_scores[k] + trans + emission)

            best_k = int(np.argmax(options))
            curr_scores[j] = options[best_k]
            back[j] = best_k

        scores.append(curr_scores)
        backpointers.append(back)

    last = int(np.argmax(scores[-1]))
    path = [last]

    for back in reversed(backpointers):
        last = int(back[last])
        path.append(last)

    path.reverse()

    decoded = pd.DataFrame([
        states[i].iloc[path[i]] for i in range(n)
    ]).reset_index(drop=True)

    decoded_ids = decoded.arc_id.to_numpy(dtype=int)

    # Direct connectivity
    decoded_connected = (
        to[decoded_ids[:-1]] == fr[decoded_ids[1:]]
    )
    greedy_connected = (
        to[greedy_ids[:-1]] == fr[greedy_ids[1:]]
    )

    # Snap distances
    def snap_metrics(ids, selected):
        geoms = np.asarray(trip.arc_geoms, dtype=object)[ids]

        snapped = shapely.line_interpolate_point(
            geoms, selected.proj_frac.to_numpy(), normalized=True
        )

        snap_xy = np.column_stack([
            shapely.get_x(snapped),
            shapely.get_y(snapped)
        ])

        snap_dist = np.linalg.norm(xy - snap_xy, axis=1)
        route_steps = np.linalg.norm(
            np.diff(snap_xy, axis=0), axis=1
        )

        return snap_dist, route_steps

    # Network reachability/hops
    def route_analysis(ids):
        hops = []
        reachable = 0

        for i in range(len(ids) - 1):
            try:
                h = nx.shortest_path_length(
                    G,
                    source=int(to[ids[i]]),
                    target=int(fr[ids[i + 1]])
                )
                hops.append(h)
                reachable += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                hops.append(np.nan)

        hops = np.asarray(hops, dtype=float)
        valid = hops[np.isfinite(hops)]

        return {
            "network_reachability_pct": (
                reachable / max(len(ids) - 1, 1) * 100
            ),
            "mean_hops": float(np.mean(valid)) if len(valid) else np.nan,
            "max_hops": float(np.max(valid)) if len(valid) else np.nan,
            "hops_gt5": int(np.sum(valid > 5)),
            "hops_gt10": int(np.sum(valid > 10)),
        }

    rows = []

    for method, ids, selected, connected in [
        ("Greedy", greedy_ids, greedy, greedy_connected),
        ("Viterbi", decoded_ids, decoded, decoded_connected),
    ]:
        snap_dist, route_steps = snap_metrics(ids, selected)
        network = route_analysis(ids)

        row = {
            "trip_id": trip_path.name,
            "method": method,
            "beta": beta if method == "Viterbi" else np.nan,
            "n_points": n,
            "direct_connectivity_pct": (
                np.mean(connected) * 100 if len(connected) else np.nan
            ),
            "road_switches": int(np.sum(ids[1:] != ids[:-1])),
            "unique_arcs": int(len(np.unique(ids))),
            "mean_snap_distance_m": float(np.mean(snap_dist)),
            "median_snap_distance_m": float(np.median(snap_dist)),
            "max_snap_distance_m": float(np.max(snap_dist)),
            "mean_route_step_m": (
                float(np.mean(route_steps)) if len(route_steps) else np.nan
            ),
            "route_jumps_gt100m": int(np.sum(route_steps > 100)),
            **network,
        }

        rows.append(row)

    return rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    model_artifact = joblib.load(MODEL_PATH)
    model = model_artifact["model"]

    trip_dirs = sorted([
        p for p in TRIPS_DIR.iterdir() if p.is_dir()
    ])

    all_results = []

    print(f"Found {len(trip_dirs)} trips")
    print(f"Testing Betas: {BETAS}")
    print("-" * 60)

    for trip_path in trip_dirs:
        print(f"\nProcessing trip: {trip_path.name}")

        for beta in BETAS:
            try:
                rows = evaluate_trip(trip_path, model, beta)
                for row in rows:
                    all_results.append(row)

                print(f"  Beta {beta:g}: completed")

            except Exception as e:
                print(f"  Beta {beta:g}: FAILED - {e}")

    results = pd.DataFrame(all_results)

    if results.empty:
        print("No results generated!")
        return

    per_trip_path = RESULTS_DIR / "multi_trip_metrics.csv"
    results.to_csv(per_trip_path, index=False)

    viterbi = results[results.method == "Viterbi"].copy()

    summary = (
        viterbi.groupby("beta")
        .agg({
            "direct_connectivity_pct": "mean",
            "network_reachability_pct": "mean",
            "mean_hops": "mean",
            "max_hops": "mean",
            "hops_gt5": "mean",
            "hops_gt10": "mean",
            "road_switches": "mean",
            "unique_arcs": "mean",
            "mean_snap_distance_m": "mean",
            "median_snap_distance_m": "mean",
            "max_snap_distance_m": "mean",
            "mean_route_step_m": "mean",
            "route_jumps_gt100m": "mean",
        })
        .reset_index()
    )

    summary_path = RESULTS_DIR / "multi_trip_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("MULTI-TRIP EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Trips evaluated: {len(trip_dirs)}")
    print(f"Results: {per_trip_path}")
    print(f"Summary: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()