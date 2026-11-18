"""PathSnap prediction pipeline used by the API (Laptop 1). Owner: D

Runs the same steps as the training scripts, in order:
  1. load the trip (GPS track, road nodes, road arcs) from the trips folder
  2. project to the trip's UTM zone (metres)
  3. candidate arcs within 50 m of every GPS point          (decisions log D-007)
  4. features, from the shared module used in training      (D-009)
  5. tuned HistGradientBoosting scores every candidate      (D-014)
  6. HMM/Viterbi decoding over the road network             (D-012, D-015)
  7. ONLY AFTER prediction: score against the reference route, server-side

Ground truth handling: the trip's .route file is removed from the Trip object
before step 2, so it can never reach candidates, features, the model or the
decoder. It is used only in step 7, and the response contains metrics only,
never the reference arcs or labels.

Configuration (environment variables):
  PATHSNAP_TRIPS_DIR  folder containing one sub-folder per trip
                      (default: data/raw/student_test, else ../Student folder/student_test)
  PATHSNAP_MODEL      model file (default: artifacts/final_hgb_tuned.joblib)
"""
import logging
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data.labels import align_labels  # noqa: E402
from pathsnap.data.loader import list_trips, load_trip, project_trip  # noqa: E402
from pathsnap.features.candidates import build_candidates  # noqa: E402
from pathsnap.features.engineering import trip_features  # noqa: E402
from pathsnap.hmm.viterbi import ArcGraph, HMMParams, route_continuity, viterbi  # noqa: E402

log = logging.getLogger("uvicorn.error")

RADIUS_M = 50.0                                                     # D-007
HMM = HMMParams(beta=4.0, disconnect_pen=10.0, uturn_pen=3.0,       # D-012 / D-015
                max_hops=6, gap_s=300.0)
TRIP_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _default_trips_dir() -> Path:
    env = os.environ.get("PATHSNAP_TRIPS_DIR")
    if env:
        return Path(env)
    for p in (ROOT / "data/raw/student_test", ROOT.parent / "Student folder" / "student_test"):
        if p.is_dir():
            return p
    return ROOT / "data/raw/student_test"


class Pipeline:
    def __init__(self):
        self.trips_dir = _default_trips_dir()
        model_path = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "artifacts/final_hgb_tuned.joblib"))
        bundle = joblib.load(model_path)
        self.model, self.features = bundle["model"], bundle["features"]
        self.model_name = model_path.name
        log.info("PathSnap pipeline ready | model=%s | features=%s | trips_dir=%s",
                 self.model_name, self.features, self.trips_dir)

    # ------------------------------------------------------------------ trips
    def trip_dirs(self) -> dict:
        if not self.trips_dir.is_dir():
            return {}
        return {d.name: d for d in list_trips(self.trips_dir)}

    def list_trips(self) -> list[dict]:
        out = []
        for tid, d in self.trip_dirs().items():
            track = d / f"{tid}.track"
            with open(track) as f:
                n = sum(1 for line in f if line.strip())
            out.append({"trip_id": tid, "gps_points": n})
        return out

    # -------------------------------------------------------------- predict
    def predict(self, trip_id: str) -> dict:
        if not TRIP_ID.match(trip_id):
            raise KeyError(trip_id)
        dirs = self.trip_dirs()
        if trip_id not in dirs:
            raise KeyError(trip_id)
        t0 = time.perf_counter()
        timing = {}

        trip = load_trip(dirs[trip_id])
        reference = trip.route          # held aside for scoring only
        trip.route = None               # the prediction path never sees it
        trip = project_trip(trip)
        timing["load_project_s"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        cands = build_candidates(trip, RADIUS_M)
        feats = trip_features(trip, cands)
        timing["candidates_features_s"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        feats["p"] = self.model.predict_proba(feats[self.features].to_numpy(float))[:, 1]
        timing["classifier_s"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        tt = trip.track["t"].to_numpy(float)
        feats = feats.sort_values(["point_idx", "p"], ascending=[True, False])
        point_idx, points = [], []
        for pi, g in feats.groupby("point_idx", sort=True):
            point_idx.append(int(pi))
            points.append((tt[pi], g.arc_id.to_numpy(), g.p.to_numpy()))
        graph = ArcGraph(trip.arcs["from_node"], trip.arcs["to_node"], HMM.max_hops)
        clf_pred = [int(a[0]) for _, a, _ in points]
        hmm_pred = viterbi(points, graph, HMM)
        timing["hmm_s"] = time.perf_counter() - t1

        to_ll = Transformer.from_crs(f"EPSG:{trip.epsg}", "EPSG:4326", always_xy=True)
        result = {
            "trip_id": trip_id,
            "gps_points": [{"lat": float(la), "lon": float(lo), "t": float(t)}
                           for lo, la, t in trip.track[["lon", "lat", "t"]].to_numpy()],
            "route_segments": self._route_segments(trip, hmm_pred, graph),
            "snapped_points": self._snapped(trip, point_idx, hmm_pred, to_ll),
            "predicted_arcs": self._collapse(hmm_pred),
            "summary": {
                "gps_points": int(len(trip.track)),
                "points_with_candidates": len(points),
                "mean_candidates_per_point": round(len(feats) / max(len(points), 1), 2),
                "predicted_route_arcs": len(self._collapse(hmm_pred)),
                "predicted_route_length_km": round(float(sum(
                    trip.arc_geoms[a].length for a in self._collapse(hmm_pred))) / 1000, 3),
                "route_continuity_hmm": round(route_continuity(hmm_pred, [p[0] for p in points], graph, HMM.gap_s), 4),
                "route_continuity_classifier_only": round(route_continuity(clf_pred, [p[0] for p in points], graph, HMM.gap_s), 4),
            },
        }
        t1 = time.perf_counter()
        segs, snap = result["route_segments"], result["snapped_points"]
        if segs and snap:      # trim the route ends to where the trip actually starts/ends
            segs[0][0] = {"lat": snap[0]["lat"], "lon": snap[0]["lon"]}
            segs[-1][-1] = {"lat": snap[-1]["lat"], "lon": snap[-1]["lon"]}
        result["evaluation"] = self._score(trip, reference, point_idx, points, clf_pred, hmm_pred)
        timing["scoring_s"] = time.perf_counter() - t1
        timing["total_s"] = time.perf_counter() - t0
        result["timing"] = {k: round(v, 3) for k, v in timing.items()}
        result["model"] = {"classifier": self.model_name, "features": self.features,
                           "candidate_radius_m": RADIUS_M, "hmm": vars(HMM)}
        ev = result["evaluation"]
        log.info("PREDICT trip=%s points=%d cands/pt=%.1f | accuracy HMM=%s classifier=%s | continuity=%.3f | %.2fs",
                 trip_id, len(trip.track), result["summary"]["mean_candidates_per_point"],
                 ev.get("point_accuracy_hmm"), ev.get("point_accuracy_classifier_only"),
                 result["summary"]["route_continuity_hmm"], timing["total_s"])
        return result

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _collapse(arcs):
        out = []
        for a in arcs:
            if not out or out[-1] != a:
                out.append(int(a))
        return out

    @staticmethod
    def _connect(graph, a, b):
        """Arcs strictly between a and b on a shortest directed path (<= max_hops), or None."""
        from collections import deque
        parent, q = {a: None}, deque([(a, 0)])
        while q:
            x, d = q.popleft()
            if x == b:
                path = []
                while parent[x] is not None and parent[x] != a:
                    x = parent[x]
                    path.append(x)
                return path[::-1]
            if d == graph.max_hops:
                continue
            for y in graph.out.get(int(graph.to[x]), ()):
                if y not in parent:
                    parent[y] = x
                    q.append((y, d + 1))
        return None

    def _route_segments(self, trip, arcs, graph=None):
        """Predicted route as polylines. Where the decoder moved between arcs that are
        not adjacent, the connecting arcs (shortest directed path, <= max_hops) are
        filled in for display. A new segment starts only where no such path exists
        (GPS gap or disconnected network piece)."""
        lon, lat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
        fr, to = trip.arcs["from_node"].to_numpy(), trip.arcs["to_node"].to_numpy()
        seq = []
        for a in self._collapse(arcs):
            if seq and graph is not None and fr[a] != to[seq[-1]]:
                fill = self._connect(graph, seq[-1], a)
                if fill:
                    seq.extend(int(x) for x in fill)
            seq.append(a)
        segs, cur, prev = [], [], None
        for a in seq:
            if prev is not None and fr[a] != to[prev]:
                segs.append(cur)
                cur = []
            if not cur:
                cur.append({"lat": float(lat[fr[a]]), "lon": float(lon[fr[a]])})
            cur.append({"lat": float(lat[to[a]]), "lon": float(lon[to[a]])})
            prev = a
        if cur:
            segs.append(cur)
        return segs

    @staticmethod
    def _snapped(trip, point_idx, arcs, to_ll):
        """Each GPS point projected onto its predicted arc (lat/lon)."""
        import shapely
        pts = shapely.points(trip.track[["x", "y"]].to_numpy()[point_idx])
        geoms = trip.arc_geoms[np.asarray(arcs)]
        snapped = shapely.line_interpolate_point(geoms, shapely.line_locate_point(geoms, pts))
        x, y = shapely.get_x(snapped), shapely.get_y(snapped)
        lo, la = to_ll.transform(x, y)
        return [{"point_idx": int(i), "lat": float(b), "lon": float(a), "arc_id": int(c)}
                for i, a, b, c in zip(point_idx, lo, la, arcs)]

    @staticmethod
    def _score(trip, reference, point_idx, points, clf_pred, hmm_pred):
        if reference is None or len(reference) == 0:
            return {"available": False, "reason": "no reference route on the server for this trip"}
        true_arc, _ = align_labels(trip.track[["x", "y"]].to_numpy(), reference, trip.arc_geoms,
                                   window=5, max_dist=50.0)
        truth = true_arc[point_idx]
        lab = truth != -1
        in_cands = np.array([t in set(a.tolist()) for t, (_, a, _) in zip(truth, points)])
        hmm, clf = np.asarray(hmm_pred), np.asarray(clf_pred)
        ref_set, pred_set = set(int(a) for a in reference), set(int(a) for a in hmm_pred)
        tp = len(ref_set & pred_set)
        prec, rec = tp / max(len(pred_set), 1), tp / max(len(ref_set), 1)
        return {
            "available": True,
            "labelled_points": int(lab.sum()),
            "point_accuracy_hmm": round(float((hmm[lab] == truth[lab]).mean()), 4),
            "point_accuracy_classifier_only": round(float((clf[lab] == truth[lab]).mean()), 4),
            "candidate_miss_rate": round(float(1 - in_cands[lab].mean()), 4),
            "route_arc_precision": round(prec, 4),
            "route_arc_recall": round(rec, 4),
            "route_arc_f1": round(2 * prec * rec / (prec + rec), 4) if tp else 0.0,
        }
