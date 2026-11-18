import sys
import joblib
import numpy as np
import pandas as pd
import shapely
from pathlib import Path

sys.path.insert(0, "src")

from pathsnap.data.loader import load_trip, project_trip
from pathsnap.data.labels import align_labels
from pathsnap.features.candidates import build_candidates

MODEL_PATH = "artifacts/final_hgb.joblib"
TRIPS_DIR = Path("data/raw/student_test")
OUT = Path("results")
RADIUS_M = 50

def wrap_angle(z):
    return np.mod(z + np.pi, 2 * np.pi) - np.pi

def evaluate(trip_path, model):
    trip = project_trip(load_trip(str(trip_path)))
    xy = trip.track[["x", "y"]].to_numpy()
    ts = trip.track["t"].to_numpy()
    n = len(ts)

    true_arc, label_dist = align_labels(
        xy, trip.route, trip.arc_geoms,
        window=5, max_dist=50.0, lookahead_m=150.0
    )

    c = build_candidates(trip, RADIUS_M, true_arc)

    d = np.diff(xy, axis=0)
    step = np.linalg.norm(d, axis=1)
    dt = np.diff(ts)
    head = np.arctan2(d[:, 1], d[:, 0])
    head = np.append(head, head[-1] if n > 1 else 0.0)
    speed = np.append(step / np.maximum(dt, 1.0), 0.0)[:n]

    nodes = trip.nodes[["x", "y"]].to_numpy()
    fr = trip.arcs["from_node"].to_numpy()
    to = trip.arcs["to_node"].to_numpy()
    arc_head = np.arctan2(nodes[to, 1] - nodes[fr, 1],
                          nodes[to, 0] - nodes[fr, 0])

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

    best = (c.sort_values("prob", ascending=False)
              .drop_duplicates("point_idx")
              .sort_values("point_idx"))

    pred = best.arc_id.to_numpy(dtype=int)
    labelled = true_arc != -1
    n_lab = int(labelled.sum())

    accuracy = float(np.mean(pred[labelled] == true_arc[labelled])) if n_lab else np.nan
    covered = c.loc[c.label == 1, "point_idx"].nunique()
    miss_rate = 1 - covered / n_lab if n_lab else np.nan

    return {
        "trip_id": trip_path.name,
        "n_points": n,
        "labelled_points": n_lab,
        "labelled_pct": 100 * n_lab / n if n else 0,
        "top1_accuracy_pct": 100 * accuracy,
        "candidate_miss_rate_pct": 100 * miss_rate,
        "mean_label_alignment_distance_m": float(np.mean(label_dist[labelled])) if n_lab else np.nan,
    }

def main():
    OUT.mkdir(exist_ok=True)
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]

    trips = sorted(p for p in TRIPS_DIR.iterdir() if p.is_dir())
    rows = []

    print(f"Evaluating {len(trips)} trips...", flush=True)

    for i, p in enumerate(trips, 1):
        try:
            r = evaluate(p, model)
            rows.append(r)
            print(f"[{i}/{len(trips)}] {p.name}: accuracy={r['top1_accuracy_pct']:.2f}% | miss={r['candidate_miss_rate_pct']:.2f}%", flush=True)
        except Exception as e:
            print(f"[{i}/{len(trips)}] {p.name} FAILED: {e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gt_fast_per_trip.csv", index=False)

    summary = df.mean(numeric_only=True).to_frame().T
    summary.insert(0, "trips_evaluated", len(df))
    summary.to_csv(OUT / "gt_fast_summary.csv", index=False)

    print("\n=== GT EVALUATION SUMMARY ===")
    print(df.to_string(index=False))
    print("\nMEAN:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(OUT / "gt_fast_per_trip.csv")
    print(OUT / "gt_fast_summary.csv")

if __name__ == "__main__":
    main()
