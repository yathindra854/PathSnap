"""Pipeline step 02: build the candidate table. Owner: A

Usage (from repo root):
  python scripts/02_build_candidates.py --partition train --radius 50
  python scripts/02_build_candidates.py --partition student_test --radius 50
  python scripts/02_build_candidates.py --partition train --radius 50 --limit 5   # quick v0

Writes:
  data/processed/candidates_<partition>.parquet
  reports/tables/candidate_stats_<partition>_r<radius>.csv  (per-trip miss rate etc.)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.data.labels import align_labels  # noqa: E402
from pathsnap.data.loader import list_trips, load_trip, project_trip  # noqa: E402
from pathsnap.features.candidates import build_candidates, miss_rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=["train", "student_test"], required=True)
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--window", type=int, default=5, help="label alignment look-ahead")
    ap.add_argument("--limit", type=int, default=None, help="only first N trips (v0)")
    ap.add_argument("--no-save", action="store_true", help="stats only, skip parquet")
    args = ap.parse_args()

    trips = list_trips(ROOT / "data/raw" / args.partition)[: args.limit]
    tables, stats = [], []
    for d in trips:
        t0 = time.time()
        trip = project_trip(load_trip(d))
        true_arc, align_dist = align_labels(trip.track[["x", "y"]].to_numpy(),
                                            trip.route, trip.arc_geoms, args.window,
                                            max_dist=args.radius)
        cands = build_candidates(trip, args.radius, true_arc)
        n = len(trip.track)
        stats.append({
            "trip_id": trip.trip_id, "epsg": trip.epsg, "n_points": n,
            "n_arcs": len(trip.arcs), "n_route_arcs": len(trip.route),
            "radius_m": args.radius, "unlabelled_frac": float((true_arc == -1).mean()),
            "miss_rate": miss_rate(cands, true_arc),
            "mean_candidates": len(cands) / n,
            "align_dist_median_m": float(pd.Series(align_dist).median()),
            "align_dist_p95_m": float(pd.Series(align_dist).quantile(0.95)),
            "seconds": round(time.time() - t0, 2),
        })
        print(f"{trip.trip_id}: {n} pts, {len(cands)} cands, "
              f"unlabelled={stats[-1]['unlabelled_frac']:.3f}, "
              f"miss={stats[-1]['miss_rate']:.3f}, {stats[-1]['seconds']}s")
        if not args.no_save:
            tables.append(cands)

    stats = pd.DataFrame(stats)
    tag = f"{args.partition}_r{int(args.radius)}"
    (ROOT / "reports/tables").mkdir(parents=True, exist_ok=True)
    stats.to_csv(ROOT / f"reports/tables/candidate_stats_{tag}.csv", index=False)

    # overall miss rate weighted by points
    lab = stats["n_points"] * (1 - stats["unlabelled_frac"])
    overall = (stats["miss_rate"].fillna(0) * lab).sum() / lab.sum()
    unl = 1 - lab.sum() / stats["n_points"].sum()
    print(f"\n{len(stats)} trips | unlabelled {unl:.4f} | overall miss rate {overall:.4f} | "
          f"mean candidates/point {stats['mean_candidates'].mean():.1f}")

    if not args.no_save:
        out = ROOT / f"data/processed/candidates_{args.partition}.parquet"
        pd.concat(tables, ignore_index=True).to_parquet(out, index=False)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
