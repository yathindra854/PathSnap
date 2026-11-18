"""Pipeline step 03a: candidate-level feature table. Owner: B

Reads candidates_<partition>.parquet (from step 02) and adds features computed
from the raw trajectory and road network. Labels: 1 = reference arc, 0 = other
candidate; unlabelled points (-1) are kept in the file but flagged and excluded
from training/scoring by the caller.

Trips dropped for label quality (decisions log D-002): 00000054, 00000033, 00000002
(train only). Usage: python scripts/03a_features.py <partition>
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data.loader import list_trips, load_trip, project_trip  # noqa: E402
from pathsnap.features.engineering import ALL_FEATURES, trip_features  # noqa: E402

DROP_TRAIN = {"00000054", "00000033", "00000002"}
FEATURES = ALL_FEATURES   # kept for older scripts that import FEATURES from here


def main(partition):
    pq = ROOT / f"data/processed/candidates_{partition}.parquet"
    cands = pd.read_parquet(pq) if pq.exists() else pd.read_csv(pq.with_suffix(".csv"), dtype={"trip_id": str})
    cands["trip_id"] = cands["trip_id"].astype(str).str.zfill(8)
    by = {k: v for k, v in cands.groupby("trip_id")}
    parts = []
    for d in list_trips(ROOT / "data/raw" / partition):
        tid = d.name.zfill(8)
        if partition == "train" and tid in DROP_TRAIN:
            continue
        if tid not in by:
            continue
        trip = project_trip(load_trip(d))
        parts.append(trip_features(trip, by[tid]))
        print(tid, len(parts[-1]), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["unlabelled_point"] = df["label"].eq(-1)
    out = ROOT / f"data/processed/feats_{partition}.parquet"
    df.to_parquet(out, index=False)
    print("saved", out, df.shape, "labelled points:", df.loc[~df.unlabelled_point, "point_idx"].groupby(df.trip_id).nunique().sum())


if __name__ == "__main__":
    main(sys.argv[1])
