"""Candidate arc generation per GPS observation (spatial index). Owner: A

For each GPS point, every arc within `radius_m` metres is a candidate.
An STRtree over the arc LineStrings keeps this fast even on the largest
trips (~300k arcs).

Output: one row per (point, candidate arc) — the shared table B and C use.
Columns:
  trip_id, point_idx, t, arc_id,
  perp_dist  : distance from GPS point to the arc (m)
  proj_frac  : where the point projects along the arc (0 = start, 1 = end)
  label      : 1 if arc_id is the aligned reference arc for this point,
               0 otherwise, -1 for every candidate of an unlabelled point
(label is only present when the trip has a route; never at inference time)

Note: two-way roads appear as two directed arcs with the same geometry,
so both get the same perp_dist. Direction features (B) separate them.
"""
import numpy as np
import pandas as pd
import shapely

from pathsnap.data.loader import Trip


def build_candidates(trip: Trip, radius_m: float,
                     true_arc: np.ndarray | None = None) -> pd.DataFrame:
    """Candidate table for one projected trip."""
    points = shapely.points(trip.track[["x", "y"]].to_numpy())
    tree = shapely.STRtree(trip.arc_geoms)

    # pairs[0] = point index, pairs[1] = arc index, for all arcs within radius
    pairs = tree.query(points, predicate="dwithin", distance=radius_m)
    pt_idx, arc_idx = pairs[0], pairs[1]

    geoms = trip.arc_geoms[arc_idx]
    pts = points[pt_idx]
    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "point_idx": pt_idx,
        "t": trip.track["t"].to_numpy()[pt_idx],
        "arc_id": arc_idx,
        "perp_dist": shapely.distance(pts, geoms),
        "proj_frac": shapely.line_locate_point(geoms, pts, normalized=True),
    })
    if true_arc is not None:
        ta = true_arc[pt_idx]
        lab = (df["arc_id"].to_numpy() == ta).astype(np.int8)
        lab[ta == -1] = -1
        df["label"] = lab
    return df.sort_values(["point_idx", "perp_dist"], ignore_index=True)


def miss_rate(cands: pd.DataFrame, true_arc: np.ndarray) -> float:
    """Share of labelled GPS points whose reference arc is NOT among the candidates."""
    n_labelled = int((true_arc != -1).sum())
    if n_labelled == 0:
        return float("nan")
    hit = cands.loc[cands["label"] == 1, "point_idx"].nunique()
    return 1 - hit / n_labelled
