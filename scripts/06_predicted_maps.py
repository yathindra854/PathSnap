"""Pipeline step 06: Folium maps with the predicted route (report section 6). Owner: A

For each trip: GPS observations, road network, reference route, the final
prediction (classifier + HMM/Viterbi) and, as an optional layer, the
classifier-only prediction. Uses exactly the API's steps and settings
(backend/app/services/pipeline.py): 50 m candidates, shared features, the tuned
HGB model and the D-012/D-015 decoder. The reference route is used only to draw
the green layer and to print accuracy; it never enters the prediction.

Usage (from repo root):
  python scripts/06_predicted_maps.py                     # all student_test trips
  python scripts/06_predicted_maps.py 00000000 00000052   # selected trips

Writes reports/figures/maps/pred_<trip>.html and prints per-trip accuracy.
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data.labels import align_labels  # noqa: E402
from pathsnap.data.loader import list_trips, load_trip, project_trip  # noqa: E402
from pathsnap.features.candidates import build_candidates  # noqa: E402
from pathsnap.features.engineering import trip_features  # noqa: E402
from pathsnap.hmm.viterbi import ArcGraph, HMMParams, viterbi  # noqa: E402
from pathsnap.viz.folium_maps import make_map  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
RADIUS_M = 50.0
HMM = HMMParams(beta=4.0, disconnect_pen=10.0, uturn_pen=3.0, max_hops=6, gap_s=300.0)
OUT = ROOT / "reports/figures/maps"


def predict(trip, model, feats_used):
    """Classifier-only and HMM arc per GPS point (-1 = no candidate)."""
    reference = trip.route
    trip.route = None                                  # prediction never sees the route
    trip = project_trip(trip)
    f = trip_features(trip, build_candidates(trip, RADIUS_M))
    f["p"] = model.predict_proba(f[feats_used].to_numpy(float))[:, 1]
    f = f.sort_values(["point_idx", "p"], ascending=[True, False])

    t = trip.track["t"].to_numpy(float)
    idx, points = [], []
    for pi, g in f.groupby("point_idx", sort=True):
        idx.append(int(pi))
        points.append((t[pi], g.arc_id.to_numpy(), g.p.to_numpy()))
    graph = ArcGraph(trip.arcs["from_node"], trip.arcs["to_node"], HMM.max_hops)

    n = len(trip.track)
    clf, hmm = np.full(n, -1), np.full(n, -1)
    clf[idx] = [int(a[0]) for _, a, _ in points]
    hmm[idx] = viterbi(points, graph, HMM)
    trip.route = reference
    return trip, clf, hmm


def main():
    bundle = joblib.load(ROOT / "artifacts/final_hgb_tuned.joblib")
    model, feats_used = bundle["model"], bundle["features"]
    dirs = {d.name: d for d in list_trips(ROOT / "data/raw/student_test")}
    wanted = sys.argv[1:] or list(dirs)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for tid in wanted:
        trip, clf, hmm = predict(load_trip(dirs[tid]), model, feats_used)
        true_arc, _ = align_labels(trip.track[["x", "y"]].to_numpy(), trip.route,
                                   trip.arc_geoms, 5, 50.0, 150.0)
        lab = true_arc >= 0
        acc_clf = float((clf[lab] == true_arc[lab]).mean())
        acc_hmm = float((hmm[lab] == true_arc[lab]).mean())
        rows.append({"trip_id": tid, "labelled_points": int(lab.sum()),
                     "top1_classifier": round(acc_clf, 4), "top1_hmm": round(acc_hmm, 4),
                     "change": round(acc_hmm - acc_clf, 4)})

        m = make_map(trip, predicted_arcs=hmm, classifier_arcs=clf,
                     predicted_name="Predicted route (classifier + HMM)")
        m.save(OUT / f"pred_{tid}.html")
        print(f"{tid}: classifier {acc_clf:.3f} -> HMM {acc_hmm:.3f}  map -> pred_{tid}.html")

    df = pd.DataFrame(rows)
    out = ROOT / "reports/tables/hmm_per_trip_student_test.csv"
    df.to_csv(out, index=False)
    print(f"per-trip table -> {out}")


if __name__ == "__main__":
    main()
