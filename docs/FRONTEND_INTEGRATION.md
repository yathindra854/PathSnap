# Connecting the UI (Laptop 2) to the PathSnap API (Laptop 1)

The evaluation flow (Models and Training Guidelines §6):
UI lists the trips -> evaluator picks one -> UI sends only the trip ID -> API runs the
pipeline and scores it -> UI draws the route and shows the results.

## 1. API address
Do not hardcode `127.0.0.1`; on Laptop 2 that points at Laptop 2 itself.
Create `frontend/.env.local` on Laptop 2:
```
VITE_API_URL=http://<laptop-1-ip>:8000
```
and in the code:
```js
const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
```
Laptop 1's IP comes from `ipconfig` on Laptop 1. Test by opening `http://<laptop-1-ip>:8000/docs` in Laptop 2's browser.

## 2. Load the trip list (replace the free-text trip ID input with a dropdown)
```js
const res = await fetch(`${API}/api/v1/trips`);
const { trips } = await res.json();      // [{ trip_id: "00000065", gps_points: 768 }, ...]
```

## 3. Request a prediction (send only the trip ID; remove samplePoints)
```js
const res = await fetch(`${API}/api/v1/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ trip_id: selectedTripId }),
});
if (!res.ok) throw new Error(`API error: ${res.status}`);
const r = await res.json();
```

## 4. Draw on the Leaflet map
```jsx
{/* raw GPS: small blue circles */}
{r.gps_points.map((p, i) => <CircleMarker key={i} center={[p.lat, p.lon]} radius={2} />)}
{/* predicted route: one Polyline per segment */}
{r.route_segments.map((seg, i) =>
  <Polyline key={i} positions={seg.map(p => [p.lat, p.lon])} pathOptions={{ color: "#d93025", weight: 4 }} />)}
{/* A and B */}
<Marker position={[r.start.lat, r.start.lon]}><Popup>A (start)</Popup></Marker>
<Marker position={[r.end.lat, r.end.lon]}><Popup>B (end)</Popup></Marker>
```
Fit the map to `r.gps_points`. Long trips have up to about 4,000 GPS points; drawing every
other point is fine if the map gets slow.

## 5. Results panel
| Show | Field |
|---|---|
| Point accuracy (HMM) | `r.evaluation.point_accuracy_hmm` |
| Point accuracy (classifier only) | `r.evaluation.point_accuracy_classifier_only` |
| Route arc F1 | `r.evaluation.route_arc_f1` |
| Route continuity | `r.summary.route_continuity_hmm` |
| Route length | `r.summary.predicted_route_length_km` km |
| GPS points | `r.summary.gps_points` |
| Processing time | `r.timing.total_s` s |

If `r.evaluation.available` is false, show "No reference route for this trip".
The response never contains the reference route; do not try to draw it in the UI.

## 6. Map tiles
OpenStreetMap tiles can be blocked; if the background is grey or shows "Access blocked",
switch the TileLayer to CARTO:
```
https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png
```
