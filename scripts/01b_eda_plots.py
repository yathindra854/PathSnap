"""Step 1b: EDA figures from reports/tables/dataset_inventory.csv plus one trip overlay.

Usage: python scripts/01b_eda_plots.py <data_root> <repo_root> [trip_id] [partition]
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

eda = import_module("01a_eda_inventory")

def main(data_root, repo, trip="00000046", part="train"):
    fig_dir = f"{repo}/reports/figures"
    os.makedirs(fig_dir, exist_ok=True)
    d = pd.read_csv(f"{repo}/reports/tables/dataset_inventory.csv", dtype={"trip": str})
    tr, te = d[d.partition == "train"], d[d.partition == "student_test"]

    # 1. size distributions (partition-labelled)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, lab in zip(ax, ["gps_points", "nodes", "route_arcs"], ["GPS points / trip", "Road nodes / trip", "Reference route arcs / trip"]):
        a.hist(np.log10(tr[col]), bins=20, alpha=.7, label="train (80)")
        a.hist(np.log10(te[col]), bins=10, alpha=.7, label="student_test (10)")
        a.set_xlabel(f"log10({lab})"); a.set_ylabel("trips")
    ax[0].legend(); fig.suptitle("Trip size distributions"); fig.tight_layout()
    fig.savefig(f"{fig_dir}/eda_size_distributions.png", dpi=130); plt.close(fig)

    # 2. sampling quality
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(d.dt_eq_1s_pct, bins=25); ax[0].set_xlabel("% of steps exactly 1 s"); ax[0].set_ylabel("trips")
    ax[1].scatter(d.n_gaps_gt5s + 1, d.dt_max_s, c=(d.partition == "train").map({True: "C0", False: "C1"}), s=18)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("# gaps > 5 s (+1)"); ax[1].set_ylabel("largest gap (s)")
    fig.suptitle("GPS sampling regularity (blue = train, orange = student_test)"); fig.tight_layout()
    fig.savefig(f"{fig_dir}/eda_sampling_gaps.png", dpi=130); plt.close(fig)

    # 3. network connectivity
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(d.largest_comp_pct, bins=25); ax[0].set_xlabel("% nodes in largest component"); ax[0].set_ylabel("trips")
    ax[1].scatter(d.nodes, d.components, s=18); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("nodes"); ax[1].set_ylabel("connected components")
    fig.suptitle("Road-network connectivity (all 90 trips)"); fig.tight_layout()
    fig.savefig(f"{fig_dir}/eda_network_connectivity.png", dpi=130); plt.close(fig)

    # 4. GPS-to-network distance
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d.gps_to_nearest_node_p95_m.clip(upper=300), bins=30)
    ax.set_xlabel("p95 distance GPS point -> nearest node (m, clipped at 300)"); ax.set_ylabel("trips")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/eda_gps_to_network.png", dpi=130); plt.close(fig)

    # 5. one-trip overlay: network (grey), reference route (green), GPS (blue)
    t, trk, nodes, arcs, route = eda.load_trip(f"{data_root}/{part}/{trip}")
    fig, ax = plt.subplots(figsize=(8, 8))
    nx_, ny_ = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    for u, v in arcs[["u", "v"]].to_numpy():
        ax.plot([nx_[u], nx_[v]], [ny_[u], ny_[v]], c="0.8", lw=.4)
    for a in route:
        u, v = arcs.u[a], arcs.v[a]
        ax.plot([nx_[u], nx_[v]], [ny_[u], ny_[v]], c="g", lw=2)
    ax.scatter(trk.lon, trk.lat, s=4, c="b", zorder=3)
    ax.set_title(f"Trip {trip} ({part}): network (grey), reference route (green), GPS (blue)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); fig.tight_layout()
    fig.savefig(f"{fig_dir}/eda_trip_{trip}_overlay.png", dpi=130); plt.close(fig)
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
