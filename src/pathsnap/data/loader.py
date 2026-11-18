"""Load raw GPS trajectories (.track) and road network (.nodes, .arcs, .route). Owner: A

Raw files have no headers and are whitespace-separated:
  .track  lon lat t        (one row per GPS observation, t in seconds)
  .nodes  lon lat          (row index = node_id)
  .arcs   from_node to_node (row index = arc_id, directed)
  .route  arc_id           (reference arcs in travel order)

Trips are in different cities, so each trip gets its own UTM projection
(metres) instead of a single CRS for the whole dataset.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame          # lon, lat, t  (+ x, y after projection)
    nodes: pd.DataFrame          # lon, lat     (+ x, y after projection)
    arcs: pd.DataFrame           # from_node, to_node
    route: np.ndarray | None     # reference arc_ids in order (None if absent)
    epsg: int | None = None
    arc_geoms: np.ndarray | None = None  # shapely LineStrings in metres, index = arc_id


def _read(path: Path, names, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_trip(trip_dir: str | Path) -> Trip:
    """Read the four raw files of one trip folder."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"])
    nodes = _read(trip_dir / f"{tid}.nodes", ["lon", "lat"])
    arcs = _read(trip_dir / f"{tid}.arcs", ["from_node", "to_node"], dtype=np.int64)
    route_path = trip_dir / f"{tid}.route"
    route = None
    if route_path.exists():
        route = _read(route_path, ["arc_id"], dtype=np.int64)["arc_id"].to_numpy()
    return Trip(tid, track, nodes, arcs, route)


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). 326xx = north, 327xx = south."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def project_trip(trip: Trip) -> Trip:
    """Add x/y metres to track and nodes, and build one LineString per arc."""
    epsg = utm_epsg(trip.track["lon"].mean(), trip.track["lat"].mean())
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    trip.track["x"], trip.track["y"] = tf.transform(trip.track["lon"].values, trip.track["lat"].values)
    trip.nodes["x"], trip.nodes["y"] = tf.transform(trip.nodes["lon"].values, trip.nodes["lat"].values)

    nxy = trip.nodes[["x", "y"]].to_numpy()
    start = nxy[trip.arcs["from_node"].to_numpy()]
    end = nxy[trip.arcs["to_node"].to_numpy()]
    coords = np.stack([start, end], axis=1)          # (n_arcs, 2 points, 2 coords)
    trip.arc_geoms = shapely.linestrings(coords)
    trip.epsg = epsg
    return trip


def list_trips(partition_dir: str | Path) -> list[Path]:
    """Sorted trip folders inside a partition (e.g. data/raw/train)."""
    return sorted(p for p in Path(partition_dir).iterdir() if p.is_dir())
