"""Folium maps: GPS, road network, reference route, predicted route. Owner: A

Usage:
    from pathsnap.data.loader import load_trip
    from pathsnap.viz.folium_maps import make_map
    m = make_map(load_trip("data/raw/train/00000048"))
    m.save("map.html")

Also used by the UI (D): pass `predicted_arcs` (one arc_id per GPS point,
or an ordered arc list) to draw the model's route on top.
Coordinates are drawn in lon/lat (EPSG:4326), which is what Folium expects.
"""
import folium
import numpy as np
from folium.plugins import Fullscreen

from pathsnap.data.loader import Trip

COLORS = {
    "network": "#9e9e9e",
    "gps": "#1f77b4",
    "reference": "#2ca02c",
    "predicted": "#d62728",
}


def _arc_lines(trip: Trip, arc_ids) -> list:
    """[[ (lat, lon), (lat, lon) ], ...] for the given arcs."""
    lon = trip.nodes["lon"].to_numpy()
    lat = trip.nodes["lat"].to_numpy()
    a = trip.arcs.iloc[np.asarray(arc_ids)]
    f, t = a["from_node"].to_numpy(), a["to_node"].to_numpy()
    return [[(lat[i], lon[i]), (lat[j], lon[j])] for i, j in zip(f, t)]


def _network_in_view(trip: Trip, pad_deg: float = 0.003) -> np.ndarray:
    """Arc ids with at least one end inside the track's bounding box (+ padding).
    Big trips have ~300k arcs; drawing all of them would make the HTML huge."""
    lo_lon, hi_lon = trip.track["lon"].min() - pad_deg, trip.track["lon"].max() + pad_deg
    lo_lat, hi_lat = trip.track["lat"].min() - pad_deg, trip.track["lat"].max() + pad_deg
    lon, lat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
    inside = (lon >= lo_lon) & (lon <= hi_lon) & (lat >= lo_lat) & (lat <= hi_lat)
    f, t = trip.arcs["from_node"].to_numpy(), trip.arcs["to_node"].to_numpy()
    return np.flatnonzero(inside[f] | inside[t])


def _legend(items: dict) -> folium.Element:
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;'
        f'background:{c};margin-right:6px;vertical-align:middle"></span>{name}</div>'
        for name, c in items.items())
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
            f'background:white;padding:8px 10px;border:1px solid #bbb;'
            f'border-radius:4px;font:12px sans-serif">{rows}</div>')
    return folium.Element(html)


def make_map(trip: Trip, predicted_arcs=None, max_gps_points: int = 3000,
             show_network: bool = True) -> folium.Map:
    """Interactive map of one trip with toggleable layers."""
    tr = trip.track
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CARTO light", max_zoom=20).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri streets", max_zoom=19).add_to(m)
    m.fit_bounds([[tr["lat"].min(), tr["lon"].min()], [tr["lat"].max(), tr["lon"].max()]])
    legend = {}

    if show_network:
        net = folium.FeatureGroup(name="Road network", show=True)
        folium.PolyLine(_arc_lines(trip, _network_in_view(trip)),
                        color=COLORS["network"], weight=1, opacity=0.5).add_to(net)
        net.add_to(m)
        legend["Road network"] = COLORS["network"]

    if trip.route is not None:
        ref = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_arc_lines(trip, trip.route), color=COLORS["reference"],
                        weight=5, opacity=0.8).add_to(ref)
        ref.add_to(m)
        legend["Reference route"] = COLORS["reference"]

    if predicted_arcs is not None:
        # collapse consecutive repeats: per-point predictions -> ordered arc list
        arcs = np.asarray(predicted_arcs)
        arcs = arcs[arcs >= 0]
        if len(arcs):
            arcs = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
        pred = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(_arc_lines(trip, arcs), color=COLORS["predicted"],
                        weight=3, opacity=0.9, dash_array="6 4").add_to(pred)
        pred.add_to(m)
        legend["Predicted route"] = COLORS["predicted"]

    # GPS points last so they sit on top; subsample very long trips
    step = max(1, len(tr) // max_gps_points)
    gps = folium.FeatureGroup(name=f"GPS points (every {step})" if step > 1 else "GPS points")
    for lat, lon, t in tr[["lat", "lon", "t"]].to_numpy()[::step]:
        folium.CircleMarker((lat, lon), radius=2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    legend["GPS points"] = COLORS["gps"]

    # start / end markers
    folium.Marker((tr["lat"].iloc[0], tr["lon"].iloc[0]), tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((tr["lat"].iloc[-1], tr["lon"].iloc[-1]), tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    m.get_root().html.add_child(_legend(legend))
    return m