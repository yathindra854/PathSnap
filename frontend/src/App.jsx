import { useState, useEffect } from "react";
import {
  Activity,
  Map,
  Route,
  Play,
  Navigation,
  Clock3,
  Layers,
  CheckCircle2,
  RefreshCw,
} from "lucide-react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

// Backend address. Laptop 2 sets this in frontend/.env.local, e.g.
// VITE_API_URL=http://192.168.1.23:8000   (the IP of the laptop running the API)
const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function FitRoute({ points }) {
  const map = useMap();

  useEffect(() => {
    if (points?.length > 1) {
      const bounds = points.map((point) => [point.lat, point.lon]);
      map.fitBounds(bounds, { padding: [40, 40] });
    }
  }, [points, map]);

  return null;
}

function RouteMap({ rawPoints, segments, start, end }) {
  const defaultCenter = [30.3165, 78.0322];

  return (
    <MapContainer
      center={defaultCenter}
      zoom={13}
      scrollWheelZoom={true}
      className="leaflet-map"
    >
      <TileLayer
        attribution='&copy; <a href="https://carto.com/attributions">CARTO</a> &copy; OpenStreetMap contributors'
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />

      {rawPoints?.length > 0 && (
        <>
          <FitRoute points={rawPoints} />

          {rawPoints.map((point, index) => (
            <CircleMarker
              key={`raw-${index}`}
              center={[point.lat, point.lon]}
              radius={3}
              pathOptions={{
                color: "#38a8ff",
                fillColor: "#38a8ff",
                fillOpacity: 0.8,
                weight: 1,
              }}
            >
              <Popup>
                <strong>GPS point {index + 1}</strong>
                <br />
                {point.lat.toFixed(5)}, {point.lon.toFixed(5)}
              </Popup>
            </CircleMarker>
          ))}
        </>
      )}

      {segments?.map((seg, index) => (
        <Polyline
          key={`seg-${index}`}
          positions={seg.map((point) => [point.lat, point.lon])}
          pathOptions={{ color: "#b18aff", weight: 6, opacity: 0.95 }}
        />
      ))}

      {start && (
        <CircleMarker
          center={[start.lat, start.lon]}
          radius={9}
          pathOptions={{ color: "#ffffff", fillColor: "#22c55e", fillOpacity: 1, weight: 3 }}
        >
          <Popup><strong>Start (A)</strong></Popup>
        </CircleMarker>
      )}
      {end && (
        <CircleMarker
          center={[end.lat, end.lon]}
          radius={9}
          pathOptions={{ color: "#ffffff", fillColor: "#ef4444", fillOpacity: 1, weight: 3 }}
        >
          <Popup><strong>End (B)</strong></Popup>
        </CircleMarker>
      )}
    </MapContainer>
  );
}

function App() {
  const [trips, setTrips] = useState([]);
  const [tripId, setTripId] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [online, setOnline] = useState(null);

  useEffect(() => {
    async function loadTrips() {
      try {
        const response = await fetch(`${API_URL}/api/v1/trips`);
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        const data = await response.json();
        setTrips(data.trips);
        if (data.trips.length > 0) setTripId(data.trips[0].trip_id);
        setOnline(true);
      } catch {
        setOnline(false);
        setError(`Cannot reach the PathSnap API at ${API_URL}`);
      }
    }
    loadTrips();
  }, []);

  async function runPrediction() {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/v1/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trip_id: tripId }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      setResult(await response.json());
      setOnline(true);
    } catch (err) {
      setOnline(false);
      setError(err.message || "Unable to connect to PathSnap API");
    } finally {
      setLoading(false);
    }
  }

  const summary = result?.summary;
  const evaluation = result?.evaluation;
  const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Route size={22} />
          </div>
          <div>
            <h1>PathSnap</h1>
            <span>ROUTE INTELLIGENCE</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <div className="nav-item active">
          <Map size={18} />
          Dashboard
        </div>

        <div className="nav-item">
          <Navigation size={18} />
          Map Matching
        </div>

        <div className="nav-item">
          <Layers size={18} />
          Dataset Explorer
        </div>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" />
            System online
          </div>
          <span className="version">PathSnap v0.1.0</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="breadcrumb">WORKSPACE / DASHBOARD</span>
            <h2>Route Reconstruction</h2>
          </div>

          <div className="top-status">
            <span className="status-dot" />
            {online === null ? "Checking API..." : online ? `API connected (${API_URL})` : "API unreachable"}
          </div>
        </header>

        <section className="welcome">
          <div>
            <p className="eyebrow">GPS INTELLIGENCE PLATFORM</p>
            <h1>
              Turn GPS traces into <span>road routes.</span>
            </h1>
            <p className="welcome-sub">
              Analyze trajectory data and reconstruct routes on the road network.
            </p>
          </div>

          <div className="welcome-icon">
            <Route size={42} />
          </div>
        </section>

        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon purple">
              <Activity size={20} />
            </div>
            <span className="stat-label">SYSTEM STATUS</span>
            <h3>{online === null ? "Checking" : online ? "Operational" : "Offline"}</h3>
            <span className="stat-foot">{online ? "API responding" : "Check the API address"}</span>
          </div>

          <div className="stat-card">
            <div className="stat-icon blue">
              <Navigation size={20} />
            </div>
            <span className="stat-label">GPS POINTS</span>
            <h3>{result?.points_processed ?? "—"}</h3>
            <span className="stat-foot">Points processed</span>
          </div>

          <div className="stat-card">
            <div className="stat-icon green">
              <Route size={20} />
            </div>
            <span className="stat-label">INFERENCE MODE</span>
            <h3>{result?.mode ?? "Standby"}</h3>
            <span className="stat-foot">Prediction engine</span>
          </div>

          <div className="stat-card">
            <div className="stat-icon orange">
              <Clock3 size={20} />
            </div>
            <span className="stat-label">LATEST RUN</span>
            <h3>{result ? "Completed" : "No runs"}</h3>
            <span className="stat-foot">Current session</span>
          </div>
        </section>

        <section className="workspace-grid">
          <div className="panel run-panel">
            <div className="panel-header">
              <div>
                <span className="eyebrow">INFERENCE WORKSPACE</span>
                <h3>Run a prediction</h3>
              </div>

              <div className="panel-icon">
                <Play size={18} />
              </div>
            </div>

            <label className="field-label">Trip ID</label>

            <select
              className="trip-input"
              value={tripId}
              onChange={(e) => setTripId(e.target.value)}
              disabled={trips.length === 0}
            >
              {trips.map((t) => (
                <option key={t.trip_id} value={t.trip_id}>
                  Trip {t.trip_id} ({t.gps_points} points)
                </option>
              ))}
            </select>

            <button
              className="run-button"
              onClick={runPrediction}
              disabled={loading || !tripId}
            >
              {loading ? (
                <>
                  <RefreshCw size={17} className="spin" />
                  Running prediction...
                </>
              ) : (
                <>
                  <Play size={17} />
                  Reconstruct route
                </>
              )}
            </button>

            <p className="helper-text">
              Runs the trained classifier and HMM decoder on the selected trip.
            </p>

            {error && <div className="error-message">{error}</div>}

            {result && (
              <div className="result-success">
                <CheckCircle2 size={18} />
                <div>
                  <strong>Prediction completed</strong>
                  <span>{result.message}</span>
                </div>
              </div>
            )}
          </div>

          <div className="panel map-panel">
            <div className="panel-header">
              <div>
                <span className="eyebrow">GEOSPATIAL VIEW</span>
                <h3>Route visualization</h3>
              </div>

              <span className="map-badge">
                {result ? "RESULT" : "LIVE MAP"}
              </span>
            </div>

            <div className="map-container">
              <RouteMap
                rawPoints={result?.gps_points ?? []}
                segments={result?.route_segments ?? []}
                start={result?.start}
                end={result?.end}
              />
            </div>

            <div className="map-legend">
              <span>
                <i className="legend-dot raw" /> GPS trace
              </span>
              <span>
                <i className="legend-dot matched" /> Matched route
              </span>
            </div>
          </div>
        </section>

        <section className="panel output-panel">
          <div className="panel-header">
            <div>
              <span className="eyebrow">OUTPUT</span>
              <h3>Prediction details</h3>
            </div>
          </div>

          {!result ? (
            <div className="empty-output">
              Your prediction results will appear here after you run the engine.
            </div>
          ) : (
            <>
              <div className="stats-grid">
                <div className="stat-card">
                  <span className="stat-label">ROUTE LENGTH</span>
                  <h3>{summary.predicted_route_length_km.toFixed(2)} km</h3>
                  <span className="stat-foot">{summary.predicted_route_arcs} road arcs</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">POINT ACCURACY</span>
                  <h3>{pct(evaluation?.point_accuracy_hmm)}</h3>
                  <span className="stat-foot">
                    Classifier only: {pct(evaluation?.point_accuracy_classifier_only)}
                  </span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">ROUTE ARC F1</span>
                  <h3>{pct(evaluation?.route_arc_f1)}</h3>
                  <span className="stat-foot">
                    P {pct(evaluation?.route_arc_precision)} · R {pct(evaluation?.route_arc_recall)}
                  </span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">TIME</span>
                  <h3>{result.timing?.total_s != null ? `${result.timing.total_s.toFixed(2)} s` : "—"}</h3>
                  <span className="stat-foot">{result.model?.classifier ?? "HGB + Viterbi"}</span>
                </div>
              </div>
              <details>
                <summary>Raw response</summary>
                <pre className="json-output">{JSON.stringify({ ...result, gps_points: undefined, route_segments: undefined, matched_points: undefined, snapped_points: undefined }, null, 2)}</pre>
              </details>
            </>
          )}
        </section>

        <footer>
          PathSnap · GPS Map Matching &amp; Route Reconstruction
        </footer>
      </main>
    </div>
  );
}

export default App;