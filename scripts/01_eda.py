"""Step 1: dataset exploration for PathSnap.

Usage: python scripts/01_eda.py <data_root> <out_dir>
  data_root: folder that contains train/ and student_test/ (the 'Student folder')
  out_dir:   repo root; tables go to reports/tables, figures to reports/figures

Only the 80 train and 10 student-test trips are read. Evaluator data is never touched.
Distances use a local equirectangular projection (metres) per trip, which is
accurate at city scale; pyproj/UTM can replace it later.
"""
import sys, os, glob
import numpy as np
import pandas as pd

R = 6371000.0


def project(lon, lat, lon0, lat0):
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y


def components(n, edges):
    """Union-find over undirected edges. Returns component label per node."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def load_trip(d):
    t = os.path.basename(d)
    trk = pd.read_csv(f"{d}/{t}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(f"{d}/{t}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(f"{d}/{t}.arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(f"{d}/{t}.route", header=None, names=["arc"])["arc"].to_numpy()
    return t, trk, nodes, arcs, route


def summarise(part, d):
    t, trk, nodes, arcs, route = load_trip(d)
    n_nodes, n_arcs = len(nodes), len(arcs)
    lon0, lat0 = nodes.lon.mean(), nodes.lat.mean()

    # GPS quality
    dt = np.diff(trk.t.to_numpy())
    gx, gy = project(trk.lon.to_numpy(), trk.lat.to_numpy(), lon0, lat0)
    step = np.hypot(np.diff(gx), np.diff(gy))
    speed = step / np.where(dt > 0, dt, np.nan)
    missing = int(trk.isna().sum().sum())
    bad_coord = int(((trk.lon.abs() > 180) | (trk.lat.abs() > 90)).sum())

    # Network
    nx_, ny_ = project(nodes.lon.to_numpy(), nodes.lat.to_numpy(), lon0, lat0)
    arc_ok = bool(((arcs.u >= 0) & (arcs.u < n_nodes) & (arcs.v >= 0) & (arcs.v < n_nodes)).all())
    comp = components(n_nodes, arcs[["u", "v"]].to_numpy())
    comp_sizes = np.bincount(np.unique(comp, return_inverse=True)[1])
    deg = np.bincount(np.r_[arcs.u, arcs.v], minlength=n_nodes)
    arc_len = np.hypot(nx_[arcs.v] - nx_[arcs.u], ny_[arcs.v] - ny_[arcs.u])
    # reverse arcs present? (are arcs directed both ways)
    pairs = set(zip(arcs.u, arcs.v))
    both = np.mean([(v, u) in pairs for u, v in pairs])

    # Reference route
    route_ok = bool(((route >= 0) & (route < n_arcs)).all())
    r = arcs.iloc[route[route < n_arcs]]
    consecutive = float(np.mean(r.v.to_numpy()[:-1] == r.u.to_numpy()[1:])) if len(r) > 1 else np.nan
    route_len = float(arc_len[route[route < n_arcs]].sum())

    # GPS-to-network: distance from GPS point to nearest node (cheap proxy), in metres
    sample = np.linspace(0, len(trk) - 1, min(len(trk), 300)).astype(int)
    dn = np.array([np.min(np.hypot(nx_ - gx[i], ny_ - gy[i])) for i in sample])

    return dict(
        partition=part, trip=t, gps_points=len(trk), duration_s=float(trk.t.iloc[-1] - trk.t.iloc[0]),
        missing_vals=missing, bad_coords=bad_coord,
        dt_eq_1s_pct=100 * float(np.mean(dt == 1)), dt_max_s=float(dt.max()), n_gaps_gt5s=int((dt > 5).sum()),
        non_monotonic_t=int((dt <= 0).sum()),
        step_median_m=float(np.nanmedian(step)), step_max_m=float(np.nanmax(step)),
        speed_median_ms=float(np.nanmedian(speed)), speed_p99_ms=float(np.nanpercentile(speed, 99)),
        lon_min=trk.lon.min(), lon_max=trk.lon.max(), lat_min=trk.lat.min(), lat_max=trk.lat.max(),
        extent_x_km=float((gx.max() - gx.min()) / 1000), extent_y_km=float((gy.max() - gy.min()) / 1000),
        nodes=n_nodes, arcs=n_arcs, arc_ids_valid=arc_ok,
        components=len(comp_sizes), largest_comp_pct=100 * float(comp_sizes.max() / n_nodes),
        isolated_nodes=int((deg == 0).sum()), deg_mean=float(deg.mean()), deg_max=int(deg.max()),
        arc_len_median_m=float(np.median(arc_len)), arcs_bidirectional_pct=100 * float(both),
        route_arcs=len(route), route_ids_valid=route_ok, route_unique_arcs=int(len(np.unique(route))),
        route_consecutive_pct=100 * consecutive, route_len_km=route_len / 1000,
        gps_to_nearest_node_median_m=float(np.median(dn)), gps_to_nearest_node_p95_m=float(np.percentile(dn, 95)),
    )


def main(data_root, out):
    rows = []
    for part in ["train", "student_test"]:
        for d in sorted(glob.glob(f"{data_root}/{part}/*")):
            rows.append(summarise(part, d))
            print(part, rows[-1]["trip"], rows[-1]["gps_points"], rows[-1]["nodes"], flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(f"{out}/reports/tables", exist_ok=True)
    df.to_csv(f"{out}/reports/tables/dataset_inventory.csv", index=False)
    print("saved inventory", df.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
