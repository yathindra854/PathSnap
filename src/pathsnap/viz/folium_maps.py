"""Folium maps: GPS observations, road network, reference route, predicted route. Owner: A

Layers (toggle with the layer control): road network, GPS points + path,
reference route, optional predicted route.

CLI: python folium_maps.py <data_root> <partition> <trip_id> <out_html> [max_network_arcs]
"""
import sys, os
import numpy as np
import pandas as pd
import folium


def load_trip(data_root, partition, trip):
    d = f"{data_root}/{partition}/{trip}"
    trk = pd.read_csv(f"{d}/{trip}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(f"{d}/{trip}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(f"{d}/{trip}.arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(f"{d}/{trip}.route", header=None, names=["arc"])["arc"].to_numpy()
    return trk, nodes, arcs, route


def _arc_coords(nodes, arcs, arc_ids):
    lon, lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    u, v = arcs.u.to_numpy()[arc_ids], arcs.v.to_numpy()[arc_ids]
    return [[(lat[a], lon[a]), (lat[b], lon[b])] for a, b in zip(u, v)]


def build_map(trk, nodes, arcs, route, predicted_arcs=None, max_network_arcs=6000, title=""):
    centre = [trk.lat.mean(), trk.lon.mean()]
    m = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap")
    m.fit_bounds([[trk.lat.min(), trk.lon.min()], [trk.lat.max(), trk.lon.max()]])

    # Road network: only arcs near the trajectory bounding box (padded) to keep the HTML small
    pad = 0.01
    lon, lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    in_box = lambda i: (trk.lon.min() - pad < lon[i] < trk.lon.max() + pad) and (trk.lat.min() - pad < lat[i] < trk.lat.max() + pad)
    ids = np.array([k for k, (a, b) in enumerate(zip(arcs.u, arcs.v)) if in_box(a) and in_box(b)])
    if len(ids) > max_network_arcs:
        ids = np.random.default_rng(0).choice(ids, max_network_arcs, replace=False)
    net = folium.FeatureGroup(name=f"Road network ({len(ids)} arcs shown)", show=True)
    for seg in _arc_coords(nodes, arcs, ids):
        folium.PolyLine(seg, color="#9aa0a6", weight=1, opacity=.7).add_to(net)
    net.add_to(m)

    ref = folium.FeatureGroup(name=f"Reference route ({len(route)} arcs)", show=True)
    for seg in _arc_coords(nodes, arcs, route):
        folium.PolyLine(seg, color="#1e8e3e", weight=5, opacity=.9).add_to(ref)
    ref.add_to(m)

    if predicted_arcs is not None:
        pr = folium.FeatureGroup(name="Predicted route", show=True)
        for seg in _arc_coords(nodes, arcs, np.asarray(predicted_arcs)):
            folium.PolyLine(seg, color="#d93025", weight=3, opacity=.9, dash_array="6").add_to(pr)
        pr.add_to(m)

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(trk) // 1500)
    for i in range(0, len(trk), step):
        r = trk.iloc[i]
        folium.CircleMarker((r.lat, r.lon), radius=2, color="#1a73e8", fill=True, fill_opacity=.8,
                            tooltip=f"t={r.t:.0f}s  idx={i}").add_to(gps)
    folium.PolyLine(list(zip(trk.lat, trk.lon)), color="#1a73e8", weight=1, opacity=.4).add_to(gps)
    gps.add_to(m)
    folium.Marker((trk.lat.iloc[0], trk.lon.iloc[0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((trk.lat.iloc[-1], trk.lon.iloc[-1]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)

    legend = f"""<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;
    border:1px solid #888;font-size:12px"><b>{title}</b><br>
    <span style="color:#1a73e8">&#9679;</span> GPS observations<br>
    <span style="color:#1e8e3e">&#9644;</span> Reference route<br>
    <span style="color:#d93025">&#9644;</span> Predicted route (when supplied)<br>
    <span style="color:#9aa0a6">&#9644;</span> Road network</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


if __name__ == "__main__":
    root, part, trip, out = sys.argv[1:5]
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
    trk, nodes, arcs, route = load_trip(root, part, trip)
    m = build_map(trk, nodes, arcs, route, max_network_arcs=cap, title=f"Trip {trip} ({part})")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    m.save(out)
    print("saved", out)
