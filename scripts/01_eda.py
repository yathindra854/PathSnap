"""Pipeline step 01: exploratory data analysis. Owner: A

Usage (from repo root):
  python scripts/01_eda.py                         # train + student_test, tables + plots
  python scripts/01_eda.py --maps 00000048 00000093  # also save Folium maps for these train trips

Writes:
  reports/tables/eda_trips_<partition>.csv   one row per trip (counts, gaps, speeds, connectivity)
  reports/tables/eda_missing_values.csv      missing / invalid values per partition and file
  reports/figures/eda_*.png                  plots for report section 2
  reports/figures/maps/map_<trip>.html       Folium maps (section 6)

Only train and student_test are used. Evaluator trips are not touched.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.data.loader import list_trips, load_trip, project_trip  # noqa: E402

TABLES = ROOT / "reports/tables"
FIGS = ROOT / "reports/figures"
PARTITIONS = ["train", "student_test"]


def trip_summary(trip) -> dict:
    tr = trip.track
    t = tr["t"].to_numpy()
    dt = np.diff(t)
    step = np.hypot(np.diff(tr["x"].to_numpy()), np.diff(tr["y"].to_numpy()))
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)          # m/s

    # road network as a directed graph
    g = nx.DiGraph()
    g.add_nodes_from(range(len(trip.nodes)))
    g.add_edges_from(trip.arcs[["from_node", "to_node"]].to_numpy())
    wcc = [len(c) for c in nx.weakly_connected_components(g)]
    isolated = sum(1 for n in g.nodes if g.degree(n) == 0)

    # does each route arc start where the previous one ended?
    r = trip.route
    f, to = trip.arcs["from_node"].to_numpy(), trip.arcs["to_node"].to_numpy()
    route_breaks = int((to[r[:-1]] != f[r[1:]]).sum()) if r is not None and len(r) > 1 else 0

    return {
        "trip_id": trip.trip_id,
        "epsg": trip.epsg,
        "centre_lon": round(tr["lon"].mean(), 4),
        "centre_lat": round(tr["lat"].mean(), 4),
        "n_points": len(tr),
        "duration_s": float(t[-1] - t[0]),
        "t_start": float(t[0]),
        "gaps_gt1s": int((dt > 1).sum()),
        "gaps_gt5s": int((dt > 5).sum()),
        "gaps_gt30s": int((dt > 30).sum()),
        "max_gap_s": float(dt.max()) if len(dt) else 0.0,
        "share_1s_steps": float((dt == 1).mean()) if len(dt) else np.nan,
        "non_increasing_t": int((dt <= 0).sum()),
        "speed_median_ms": float(np.nanmedian(speed)),
        "speed_p99_ms": float(np.nanpercentile(speed, 99)),
        "n_speed_gt50ms": int((speed > 50).sum()),        # > 180 km/h: likely GPS jump
        "extent_x_km": float(np.ptp(tr["x"]) / 1000),
        "extent_y_km": float(np.ptp(tr["y"]) / 1000),
        "path_length_km": float(np.nansum(step) / 1000),
        "n_nodes": len(trip.nodes),
        "n_arcs": len(trip.arcs),
        "isolated_nodes": isolated,
        "n_components": len(wcc),
        "largest_component_share": max(wcc) / len(trip.nodes),
        "n_route_arcs": 0 if r is None else len(r),
        "route_breaks": route_breaks,
    }


def missing_values(trip_dirs, partition) -> list[dict]:
    rows = []
    for ext, cols in [("track", 3), ("nodes", 2), ("arcs", 2), ("route", 1)]:
        missing_files = n_rows = n_nan = n_bad = 0
        for d in trip_dirs:
            p = d / f"{d.name}.{ext}"
            if not p.exists():
                missing_files += 1
                continue
            df = pd.read_csv(p, sep=r"\s+", header=None)
            n_rows += len(df)
            n_nan += int(df.isna().sum().sum())
            if df.shape[1] != cols:
                n_bad += len(df)
            elif ext in ("track", "nodes"):
                lon, lat = df[0], df[1]
                n_bad += int(((lon.abs() > 180) | (lat.abs() > 90) | ((lon == 0) & (lat == 0))).sum())
        rows.append({"partition": partition, "file": ext, "trips": len(trip_dirs),
                     "missing_files": missing_files, "rows": n_rows,
                     "nan_values": n_nan, "invalid_rows": n_bad})
    return rows


def plots(summ: pd.DataFrame, gaps: dict, speeds: dict):
    FIGS.mkdir(parents=True, exist_ok=True)

    # 1. points per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in summ.groupby("partition"):
        ax.hist(g["n_points"], bins=30, alpha=0.6, label=f"{p} ({len(g)} trips)")
    ax.set(xlabel="GPS points per trip", ylabel="Trips", title="Trip length")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "eda_points_per_trip.png", dpi=150); plt.close(fig)

    # 2. sampling interval distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, dt in gaps.items():
        ax.hist(np.clip(dt, 0, 30), bins=np.arange(0, 31) - 0.5, alpha=0.6, label=p, log=True)
    ax.set(xlabel="Seconds between consecutive points (clipped at 30)", ylabel="Count (log)",
           title="GPS sampling intervals")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "eda_sampling_intervals.png", dpi=150); plt.close(fig)

    # 3. speed distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, s in speeds.items():
        ax.hist(np.clip(s[~np.isnan(s)] * 3.6, 0, 200), bins=80, alpha=0.6, label=p, log=True)
    ax.set(xlabel="Speed between consecutive points (km/h, clipped at 200)", ylabel="Count (log)",
           title="Implied vehicle speed")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "eda_speed.png", dpi=150); plt.close(fig)

    # 4. where the trips are
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, g in summ.groupby("partition"):
        ax.scatter(g["centre_lon"], g["centre_lat"], s=18, alpha=0.7, label=p)
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Trip locations (centre of each trajectory)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(FIGS / "eda_trip_locations.png", dpi=150); plt.close(fig)

    # 5. network size vs track length
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in summ.groupby("partition"):
        ax.scatter(g["n_points"], g["n_arcs"], s=18, alpha=0.7, label=p)
    ax.set(xscale="log", yscale="log", xlabel="GPS points", ylabel="Road arcs",
           title="Road-network size vs trajectory length")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "eda_network_size.png", dpi=150); plt.close(fig)

    # 6. connectivity
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in summ.groupby("partition"):
        ax.hist(g["largest_component_share"], bins=20, alpha=0.6, label=p)
    ax.set(xlabel="Share of nodes in largest connected component", ylabel="Trips",
           title="Road-network connectivity")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "eda_connectivity.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="*", default=[], help="train trip ids to save Folium maps for")
    args = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)

    all_rows, miss_rows, gaps, speeds = [], [], {}, {}
    for part in PARTITIONS:
        dirs = list_trips(ROOT / "data/raw" / part)
        miss_rows += missing_values(dirs, part)
        rows, g_all, s_all = [], [], []
        for d in dirs:
            trip = project_trip(load_trip(d))
            rows.append(trip_summary(trip))
            tr = trip.track
            dt = np.diff(tr["t"].to_numpy())
            step = np.hypot(np.diff(tr["x"].to_numpy()), np.diff(tr["y"].to_numpy()))
            g_all.append(dt)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_all.append(np.where(dt > 0, step / dt, np.nan))
        df = pd.DataFrame(rows)
        df.to_csv(TABLES / f"eda_trips_{part}.csv", index=False)
        df["partition"] = part
        all_rows.append(df)
        gaps[part], speeds[part] = np.concatenate(g_all), np.concatenate(s_all)
        print(f"{part}: {len(df)} trips, {df['n_points'].sum()} points, "
              f"{df['epsg'].nunique()} UTM zones, "
              f"{(df['gaps_gt5s'] > 0).sum()} trips with gaps > 5 s, "
              f"{(df['route_breaks'] > 0).sum()} trips with route breaks")

    summ = pd.concat(all_rows, ignore_index=True)
    pd.DataFrame(miss_rows).to_csv(TABLES / "eda_missing_values.csv", index=False)
    plots(summ, gaps, speeds)
    print(f"tables -> {TABLES}\nfigures -> {FIGS}")

    if args.maps:
        from pathsnap.viz.folium_maps import make_map
        (FIGS / "maps").mkdir(parents=True, exist_ok=True)
        for tid in args.maps:
            m = make_map(load_trip(ROOT / "data/raw/train" / tid))
            out = FIGS / "maps" / f"map_{tid}.html"
            m.save(out)
            print(f"map -> {out}")


if __name__ == "__main__":
    main()