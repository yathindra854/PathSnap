"""Per-point label alignment. Owner: A (reviewed by C)

The .route file is an ordered list of arcs, not one label per GPS point.
We assign each GPS point to one route arc with a forward-only walk:

  - keep a pointer into the route (starts at 0)
  - for each point in time order, look at the route arcs ahead of the pointer:
    at least `window` arcs, and further if needed to cover `lookahead_m`
    metres of route (some trips have very short arcs, e.g. trip 00000093
    has 590 route arcs for 1211 points, so 5 arcs is not enough)
  - pick the closest one and move the pointer there (never backwards)

Forward-only stops a point from snapping to a far-later part of the route
that happens to pass nearby (loops, return trips). The window limits how
far one point can skip ahead.

Points farther than `max_dist` from every route arc they could match get
label -1 (unlabelled): the GPS went somewhere the reference route does not
cover (e.g. trip 00000002 keeps recording ~550 points after the route ends).
These points are kept for inference but excluded from training/scoring.

Known limitations (mention in report): the route itself may be imperfect,
a very long GPS gap can need a bigger window, and a point near a junction
can be assigned to either of the two adjoining arcs.
"""
import numpy as np
import shapely


def align_labels(track_xy: np.ndarray, route: np.ndarray, arc_geoms: np.ndarray,
                 window: int = 5, max_dist: float = 50.0,
                 lookahead_m: float = 150.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_arc_id, dist_m) for every GPS point (-1 = unlabelled).

    track_xy : (n_points, 2) projected coordinates in time order
    route    : reference arc_ids in travel order
    arc_geoms: LineString per arc_id (metres)
    """
    route_geoms = arc_geoms[route]
    # cumulative route length, so the look-ahead can be measured in metres
    cum = np.concatenate([[0.0], np.cumsum(shapely.length(route_geoms))])
    points = shapely.points(track_xy)
    n = len(points)
    true_arc = np.empty(n, dtype=np.int64)
    dist = np.empty(n)

    ptr = 0
    for i, p in enumerate(points):
        hi_m = np.searchsorted(cum, cum[ptr] + lookahead_m, side="right")
        hi = min(max(ptr + window + 1, hi_m), len(route))
        d = shapely.distance(p, route_geoms[ptr:hi])
        k = int(np.argmin(d))
        ptr += k
        true_arc[i] = route[ptr] if d[k] <= max_dist else -1
        dist[i] = d[k]
    return true_arc, dist