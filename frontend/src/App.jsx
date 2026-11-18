import { useState, useEffect } from "react";
import {
  Activity,
  Map,
  Route,
  Upload,
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

const samplePoints = [
  { lat: 10.5276, lon: 76.2144, timestamp: 1000 },
  { lat: 10.5280, lon: 76.2150, timestamp: 1005 },
  { lat: 10.5285, lon: 76.2158, timestamp: 1010 },
];

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

function RouteMap({ rawPoints, matchedPoints }) {
  const defaultCenter = [10.5276, 76.2144];

  return (
    <MapContainer
      center={defaultCenter}
      zoom={15}
      scrollWheelZoom={true}
      className="leaflet-map"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {rawPoints?.length > 0 && (
        <>
          <FitRoute points={rawPoints} />

          <Polyline
            positions={rawPoints.map((point) => [point.lat, point.lon])}
            pathOptions={{
              color: "#38a8ff",
              weight: 4,
              opacity: 0.9,
              dashArray: "8 8",
            }}
          />

          {rawPoints.map((point, index) => (
            <CircleMarker
              key={`raw-${index}`}
              center={[point.lat, point.lon]}
              radius={6}
              pathOptions={{
                color: "#38a8ff",
                fillColor: "#38a8ff",
                fillOpacity: 1,
              }}
            >
              <Popup>
                <strong>GPS Point {index + 1}</strong>
                <br />
                Lat: {point.lat}
                <br />
                Lon: {point.lon}
              </Popup>
            </CircleMarker>
          ))}
        </>
      )}

      {matchedPoints?.length > 0 && (
        <>
          <Polyline
            positions={matchedPoints.map((point) => [point.lat, point.lon])}
            pathOptions={{
              color: "#b18aff",
              weight: 6,
              opacity: 1,
            }}
          />

          {matchedPoints.map((point, index) => (
            <CircleMarker
              key={`matched-${index}`}
              center={[point.lat, point.lon]}
              radius={5}
              pathOptions={{
                color: "#d3b8ff",
                fillColor: "#b18aff",
                fillOpacity: 1,
              }}
            >
              <Popup>
                <strong>Matched Point {index + 1}</strong>
                <br />
                Lat: {point.lat}
                <br />
                Lon: {point.lon}
              </Popup>
            </CircleMarker>
          ))}
        </>
      )}
    </MapContainer>
  );
}

function App() {
  const [tripId, setTripId] = useState("demo_trip_001");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function runPrediction() {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/v1/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          trip_id: tripId,
          points: samplePoints,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to connect to PathSnap API");
    } finally {
      setLoading(false);
    }
  }

  const matchedPoints = result?.matched_points ?? [];

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
            API connected locally
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
            <h3>Operational</h3>
            <span className="stat-foot">API responding</span>
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

            <input
              className="trip-input"
              value={tripId}
              onChange={(e) => setTripId(e.target.value)}
              placeholder="Enter trip ID"
            />

            <div className="upload-box">
              <Upload size={24} />
              <strong>Dataset integration coming next</strong>
              <span>
                Trip selection and file upload will be connected here.
              </span>
            </div>

            <button
              className="run-button"
              onClick={runPrediction}
              disabled={loading || !tripId.trim()}
            >
              {loading ? (
                <>
                  <RefreshCw size={17} className="spin" />
                  Running prediction...
                </>
              ) : (
                <>
                  <Play size={17} />
                  Run mock prediction
                </>
              )}
            </button>

            <p className="helper-text">
              Currently using sample GPS points for API integration testing.
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
                rawPoints={result ? samplePoints : []}
                matchedPoints={matchedPoints}
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
            <pre className="json-output">{JSON.stringify(result, null, 2)}</pre>
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