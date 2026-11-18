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
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data.loader import list_trips, load_trip, project_trip  # noqa: E402

DROP_TRAIN = {"00000054", "00000033", "00000002"}
FEATURES = ["perp_dist", "perp_rel", "proj_frac", "rank", "n_cands",
            "arc_len", "heading_diff", "speed", "log_dt_next"]


def wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def trip_features(trip, c):
    xy = trip.track[["x", "y"]].to_numpy()
    t = trip.track["t"].to_numpy(float)
    n = len(xy)
    d = np.diff(xy, axis=0)
    step = np.hypot(d[:, 0], d[:, 1])
    dt = np.diff(t)
    head = np.arctan2(d[:, 1], d[:, 0])
    head = np.append(head, head[-1] if n > 1 else 0.0)            # heading to next point
    speed = np.append(step / np.maximum(dt, 1.0), 0.0)[:n]
    dt_next = np.append(dt, 0.0)[:n]

    nxy = trip.nodes[["x", "y"]].to_numpy()
    fr = trip.arcs["from_node"].to_numpy()
    to = trip.arcs["to_node"].to_numpy()
    arc_head = np.arctan2(nxy[to, 1] - nxy[fr, 1], nxy[to, 0] - nxy[fr, 0])
    arc_len = shapely.length(trip.arc_geoms)

    pt = c["point_idx"].to_numpy()
    a = c["arc_id"].to_numpy()
    out = pd.DataFrame({
        "trip_id": c["trip_id"].to_numpy(), "point_idx": pt, "arc_id": a,
        "label": c["label"].to_numpy(),
        "perp_dist": c["perp_dist"].to_numpy(),
        "proj_frac": c["proj_frac"].to_numpy(),
        "arc_len": arc_len[a],
        "heading_diff": np.degrees(np.abs(wrap(head[pt] - arc_head[a]))),
        "speed": speed[pt],
        "log_dt_next": np.log1p(dt_next[pt]),
    })
    g = out.groupby("point_idx")["perp_dist"]
    out["rank"] = g.rank(method="first").astype(int)
    out["n_cands"] = g.transform("size")
    out["perp_rel"] = out["perp_dist"] - g.transform("min")
    return out


def main(partition):
    cands = pd.read_parquet(ROOT / f"data/processed/candidates_{partition}.parquet")
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
