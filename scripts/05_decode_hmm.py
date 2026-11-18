"""Pipeline step 05: HMM/Viterbi route reconstruction and comparison. Owner: C

Scores every candidate with the selected classifier (artifacts/final_hgb_tuned.joblib, step 04),
then compares classifier-only (best candidate per point) with classifier + HMM.

Usage:
  python scripts/05_decode_hmm.py student_test            # default decoder settings
  python scripts/05_decode_hmm.py student_test --grid     # small grid over decoder settings

Metrics (labelled points only for accuracy):
  top1            share of points whose predicted arc is the reference arc
  continuity      share of consecutive predictions on the same or a directly connected arc
  route_f1        F1 between the set of predicted arcs and the set of reference-route arcs
Writes experiments/results/hmm_<partition>.csv (one row per decoder setting).
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data.loader import list_trips, load_trip  # noqa: E402
from pathsnap.hmm.viterbi import ArcGraph, HMMParams, route_continuity, viterbi  # noqa: E402


def prepare(partition, model_file="final_hgb_tuned.joblib"):
    bundle = joblib.load(ROOT / "artifacts" / model_file)
    feats = pd.read_parquet(ROOT / f"data/processed/feats_{partition}.parquet")
    feats["p"] = bundle["model"].predict_proba(feats[bundle["features"]].to_numpy(float))[:, 1]
    trips = []
    for d in list_trips(ROOT / "data/raw" / partition):
        tid = d.name.zfill(8)
        f = feats[feats.trip_id == tid].sort_values(["point_idx", "p"], ascending=[True, False])
        if f.empty:
            continue
        trip = load_trip(d)
        t = trip.track["t"].to_numpy(float)
        pts, truth = [], []
        for pi, g in f.groupby("point_idx", sort=True):
            pts.append((t[pi], g.arc_id.to_numpy(), g.p.to_numpy()))
            lab = g.label.to_numpy()
            truth.append(None if (lab == -1).all() else (int(g.arc_id.to_numpy()[lab == 1][0]) if (lab == 1).any() else -2))
        trips.append(dict(trip_id=tid, arcs=trip.arcs, route=set(int(a) for a in trip.route), points=pts, truth=truth))
    return trips


def score(trips, prm, use_hmm):
    hit = lab = 0
    cont, f1s, secs = [], [], 0.0
    for tr in trips:
        g = ArcGraph(tr["arcs"]["from_node"], tr["arcs"]["to_node"], prm.max_hops)
        t0 = time.time()
        pred = viterbi(tr["points"], g, prm) if use_hmm else [int(a[0]) for _, a, _ in tr["points"]]
        secs += time.time() - t0
        for p, y in zip(pred, tr["truth"]):
            if y is not None:
                lab += 1
                hit += p == y
        times = [pt[0] for pt in tr["points"]]
        cont.append(route_continuity(pred, times, g, prm.gap_s))
        ps = set(pred)
        tp = len(ps & tr["route"])
        prec, rec = tp / len(ps), tp / len(tr["route"])
        f1s.append(2 * prec * rec / (prec + rec) if tp else 0.0)
    return dict(top1=hit / lab, continuity=float(np.mean(cont)), route_f1=float(np.mean(f1s)),
                decode_s=round(secs, 2), labelled_points=lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("partition", choices=["train", "student_test"])
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    trips = prepare(a.partition)
    rows = [dict(method="classifier_only", **score(trips, HMMParams(), use_hmm=False))]
    base = HMMParams()
    rows.append(dict(method="hmm_default", **vars(base), **score(trips, base, True)))
    print(rows[0]); print(rows[1], flush=True)
    if a.grid:
        for beta, disc, ut in itertools.product([0.5, 1.0, 2.0, 4.0], [5.0, 10.0], [0.0, 3.0]):
            prm = HMMParams(beta=beta, disconnect_pen=disc, uturn_pen=ut)
            rows.append(dict(method="hmm_grid", **vars(prm), **score(trips, prm, True)))
            print(rows[-1], flush=True)
    out = ROOT / f"experiments/results/hmm_{a.partition}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
