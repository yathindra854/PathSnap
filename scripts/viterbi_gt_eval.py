import sys
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import shapely
from pathlib import Path

sys.path.insert(0, "src")

from pathsnap.data.loader import load_trip, project_trip
from pathsnap.data.labels import align_labels
from pathsnap.features.candidates import build_candidates

MODEL_PATH = "artifacts/final_hgb.joblib"
TRIPS_DIR = Path("data/raw/student_test")
OUT = Path("results")
RADIUS = 50
TOP_K = 8
BETAS = [10.0, 25.0, 50.0]

def wrap_angle(z):
    return np.mod(z + np.pi, 2 * np.pi) - np.pi

def prepare_trip(path, model):
    trip = project_trip(load_trip(str(path)))
    xy = trip.track[["x", "y"]].to_numpy()
    ts = trip.track["t"].to_numpy()
    n = len(ts)

    true_arc, label_dist = align_labels(
        xy, trip.route, trip.arc_geoms,
        window=5, max_dist=50.0, lookahead_m=150.0
    )

    c = build_candidates(trip, RADIUS, true_arc)

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

    greedy = (
        c.sort_values("prob", ascending=False)
         .drop_duplicates("point_idx")
         .sort_values("point_idx")
    )
    greedy_ids = greedy.arc_id.to_numpy(dtype=int)

    # Graph and shared shortest-path cache for all beta values
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

    labelled = true_arc != -1
    n_lab = int(labelled.sum())
    candidate_hit = c.loc[c.label == 1, "point_idx"].nunique()
    miss_pct = 100 * (1 - candidate_hit / n_lab) if n_lab else np.nan

    def accuracy(ids):
        return 100 * np.mean(ids[labelled] == true_arc[labelled]) if n_lab else np.nan

    def decode(beta):
        scores = [np.log(np.clip(states[0].prob.to_numpy(), 1e-8, 1.0))]
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
                            trans = -30.0
                            candidate_score = prev_scores[k] + trans + emission
                            if candidate_score > best_score:
                                best_score = candidate_score
                                best_k = k
                            continue

                        route_dist = (
                            (1 - pf) * arc_len[pa]
                            + middle
                            + cf * arc_len[ca]
                        )

                    trans = -abs(route_dist - gps_dist) / beta
                    candidate_score = prev_scores[k] + trans + emission

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

        return np.array([
            states[i].iloc[path[i]].arc_id
            for i in range(n)
        ], dtype=int)

    rows = [{
        "trip_id": path.name,
        "method": "Greedy",
        "beta": np.nan,
        "n_points": n,
        "labelled_points": n_lab,
        "accuracy_pct": accuracy(greedy_ids),
        "candidate_miss_pct": miss_pct
    }]

    for beta in BETAS:
        decoded_ids = decode(beta)
        rows.append({
            "trip_id": path.name,
            "method": "Viterbi",
            "beta": beta,
            "n_points": n,
            "labelled_points": n_lab,
            "accuracy_pct": accuracy(decoded_ids),
            "candidate_miss_pct": miss_pct
        })

    return rows, len(dist_cache)

def main():
    OUT.mkdir(exist_ok=True)
    model = joblib.load(MODEL_PATH)["model"]
    trips = sorted(p for p in TRIPS_DIR.iterdir() if p.is_dir())

    results = []
    print(f"Viterbi GT evaluation: {len(trips)} trips", flush=True)

    for i, path in enumerate(trips, 1):
        try:
            rows, cache_sources = prepare_trip(path, model)
            results.extend(rows)

            greedy_acc = rows[0]["accuracy_pct"]
            print(f"\n[{i}/{len(trips)}] {path.name} | Greedy {greedy_acc:.2f}%", flush=True)
            for r in rows[1:]:
                print(f"  Beta {r['beta']:g}: {r['accuracy_pct']:.2f}%", flush=True)

        except Exception as e:
            print(f"\n[{i}/{len(trips)}] {path.name} FAILED: {e}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "viterbi_gt_per_trip.csv", index=False)

    summary = (
        df.groupby(["method", "beta"], dropna=False)
          .agg(
              trips=("trip_id", "nunique"),
              mean_accuracy_pct=("accuracy_pct", "mean"),
              mean_candidate_miss_pct=("candidate_miss_pct", "mean")
          )
          .reset_index()
    )

    summary.to_csv(OUT / "viterbi_gt_summary.csv", index=False)

    print("\n========== FINAL GT COMPARISON ==========")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(OUT / "viterbi_gt_per_trip.csv")
    print(OUT / "viterbi_gt_summary.csv")

if __name__ == "__main__":
    main()

