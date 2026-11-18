"""Per-(observation, candidate) feature engineering. Owner: B

Shared by training (scripts/03a_features.py) and serving (backend API), so the
model always sees features computed by exactly the same code.

Input : a projected Trip and its candidate table (from features/candidates.py).
Output: one row per (GPS point, candidate arc) with the columns in ALL_FEATURES.
        The final model uses MODEL_FEATURES (decisions log D-009).
No reference-route information is used here; a `label` column is only passed
through when the candidate table already has one (training).
"""
import numpy as np
import pandas as pd
import shapely

ALL_FEATURES = ["perp_dist", "perp_rel", "proj_frac", "rank", "n_cands",
                "arc_len", "heading_diff", "speed", "log_dt_next"]
MODEL_FEATURES = ["perp_rel", "proj_frac", "heading_diff", "speed", "n_cands"]


def _wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def trip_features(trip, c: pd.DataFrame) -> pd.DataFrame:
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
        "perp_dist": c["perp_dist"].to_numpy(),
        "proj_frac": c["proj_frac"].to_numpy(),
        "arc_len": arc_len[a],
        "heading_diff": np.degrees(np.abs(_wrap(head[pt] - arc_head[a]))),
        "speed": speed[pt],
        "log_dt_next": np.log1p(dt_next[pt]),
    })
    if "label" in c.columns:
        out.insert(3, "label", c["label"].to_numpy())
    g = out.groupby("point_idx")["perp_dist"]
    out["rank"] = g.rank(method="first").astype(int)
    out["n_cands"] = g.transform("size")
    out["perp_rel"] = out["perp_dist"] - g.transform("min")
    return out
