import { useState, useEffect, useRef } from "react";
import { MapContainer, TileLayer, Polyline, CircleMarker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

// Backend address. Laptop 2 sets it in frontend/.env.local, e.g.
// VITE_API_URL=http://192.168.1.23:8000   (IP of the laptop running the API)
const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

function FitBounds({ points }) {
  const map = useMap();
  useEffect(() => {
    if (points?.length > 1) {
      map.fitBounds(points.map((p) => [p.lat, p.lon]), { padding: [40, 40] });
    }
  }, [points, map]);
  return null;
}

function RouteMap({ result, layers }) {
  const gps = result?.gps_points ?? [];
  return (
    <MapContainer center={[30.3165, 78.0322]} zoom={12} scrollWheelZoom className="map">
      <TileLayer
        attribution='&copy; <a href="https://carto.com/attributions">CARTO</a> &copy; OpenStreetMap contributors'
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />
      <FitBounds points={gps} />

      {layers.gps && gps.length > 0 && (
        <>
          <Polyline
            positions={gps.map((p) => [p.lat, p.lon])}
            pathOptions={{ color: "#2563eb", weight: 2, opacity: 0.6, dashArray: "5 6" }}
          />
          {gps.filter((_, i) => i % Math.ceil(gps.length / 400) === 0).map((p, i) => (
            <CircleMarker
              key={`g${i}`}
              center={[p.lat, p.lon]}
              radius={3}
              pathOptions={{ color: "#2563eb", fillColor: "#2563eb", fillOpacity: 0.9, weight: 0 }}
            >
              <Popup>GPS point<br />{p.lat.toFixed(5)}, {p.lon.toFixed(5)}</Popup>
            </CircleMarker>
          ))}
        </>
      )}

      {layers.route &&
        result?.route_segments?.map((seg, i) => (
          <Polyline
            key={`s${i}`}
            positions={seg.map((p) => [p.lat, p.lon])}
            pathOptions={{ color: "#dc2626", weight: 5, opacity: 0.9 }}
          />
        ))}

      {layers.ends && result && (
        <>
          <CircleMarker center={[result.start.lat, result.start.lon]} radius={9}
            pathOptions={{ color: "#fff", fillColor: "#16a34a", fillOpacity: 1, weight: 3 }}>
            <Popup>Start</Popup>
          </CircleMarker>
          <CircleMarker center={[result.end.lat, result.end.lon]} radius={9}
            pathOptions={{ color: "#fff", fillColor: "#111827", fillOpacity: 1, weight: 3 }}>
            <Popup>End</Popup>
          </CircleMarker>
        </>
      )}
    </MapContainer>
  );
}

export default function App() {
  const [trips, setTrips] = useState([]);
  const [tripId, setTripId] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [online, setOnline] = useState(null);
  const [layers, setLayers] = useState({ gps: true, route: true, ends: true });
  const reqId = useRef(0);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_URL}/api/v1/trips`);
        if (!r.ok) throw new Error(`API error ${r.status}`);
        const data = await r.json();
        setTrips(data.trips);
        setOnline(true);
      } catch {
        setOnline(false);
        setError(`Cannot reach the API at ${API_URL}. Check the address in .env.local and that the backend is running.`);
      }
    })();
  }, []);

  async function run(id) {
    setTripId(id);
    if (!id) return;
    const mine = ++reqId.current;
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API_URL}/api/v1/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trip_id: id }),
      });
      if (!r.ok) throw new Error(r.status === 404 ? "Trip not found on the server" : `API error ${r.status}`);
      const data = await r.json();
      if (mine === reqId.current) { setResult(data); setOnline(true); }
    } catch (e) {
      if (mine === reqId.current) { setError(e.message || "Request failed"); setOnline(false); }
    } finally {
      if (mine === reqId.current) setLoading(false);
    }
  }

  const s = result?.summary;
  const ev = result?.evaluation;
  const toggle = (k) => setLayers((l) => ({ ...l, [k]: !l[k] }));

  return (
    <div className="app">
      <header className="bar">
        <h1>PathSnap</h1>
        <div className="picker">
          <label htmlFor="trip">Trip</label>
          <select id="trip" value={tripId} onChange={(e) => run(e.target.value)} disabled={!trips.length || loading}>
            <option value="">Select a trip…</option>
            {trips.map((t) => (
              <option key={t.trip_id} value={t.trip_id}>
                {t.trip_id} · {t.gps_points.toLocaleString()} points
              </option>
            ))}
          </select>
          {tripId && (
            <button onClick={() => run(tripId)} disabled={loading}>
              {loading ? "Running…" : "Run again"}
            </button>
          )}
        </div>
        <div className={`status ${online ? "ok" : online === false ? "bad" : ""}`}>
          <i /> {online === null ? "Connecting…" : online ? "API connected" : "API offline"}
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <div className="body">
        <div className="mapwrap">
          <RouteMap result={result} layers={layers} />
          {loading && <div className="overlay">Matching route…</div>}
          {!result && !loading && !error && <div className="overlay hint">Select a trip to reconstruct its route</div>}

          <div className="legend">
            {[
              ["gps", "GPS observations", "blue"],
              ["route", "Predicted route", "red"],
              ["ends", "Start / end", "green"],
            ].map(([k, label, c]) => (
              <label key={k}>
                <input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} />
                <i className={`sw ${c}`} />
                {label}
              </label>
            ))}
          </div>
        </div>

        <aside className="results">
          {!result ? (
            <p className="muted">Results appear here after you select a trip.</p>
          ) : (
            <>
              <div className="big">
                <span>Route length</span>
                <strong>{s.predicted_route_length_km.toFixed(2)} km</strong>
                <em>{s.predicted_route_arcs} road segments · {s.gps_points.toLocaleString()} GPS points</em>
              </div>

              <h3>Accuracy</h3>
              <dl>
                <dt>Point accuracy (with HMM)</dt><dd>{pct(ev?.point_accuracy_hmm)}</dd>
                <dt>Classifier only</dt><dd>{pct(ev?.point_accuracy_classifier_only)}</dd>
                <dt>Route arc F1</dt><dd>{pct(ev?.route_arc_f1)}</dd>
                <dt>Precision / recall</dt><dd>{pct(ev?.route_arc_precision)} / {pct(ev?.route_arc_recall)}</dd>
                <dt>Route continuity</dt><dd>{pct(s.route_continuity_hmm)}</dd>
              </dl>

              <h3>Run</h3>
              <dl>
                <dt>Processing time</dt><dd>{result.timing?.total_s?.toFixed(2)} s</dd>
                <dt>Model</dt><dd>{result.model?.classifier}</dd>
              </dl>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
