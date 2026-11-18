# Two-laptop setup

**Laptop 1 (backend):** runs the model and the API. **Laptop 2 (frontend):** runs the UI in a browser.
Both must be on the same Wi-Fi / hotspot.

## Laptop 1: backend

1. Pull the latest `main`.
2. Install once: `pip install -r backend/requirements.txt` (needs scikit-learn 1.7.2).
3. Make sure the test trips exist at `Student folder/student_test` next to the repo, or set
   `PATHSNAP_TRIPS_DIR` to that folder.
4. Find the laptop's IP: run `ipconfig`, use the **IPv4 Address** of the Wi-Fi adapter (e.g. `192.168.1.23`).
5. Allow the port through Windows Firewall (PowerShell **as Administrator**, once):
   `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`
6. Start the API on all interfaces: double-click `run_api.bat`
   (or `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`).
7. Check from Laptop 1: open `http://<IP>:8000/health`.

## Laptop 2: frontend

1. Pull the latest `main`, install Node.js 20+.
2. Check the API is reachable first: open `http://<LAPTOP-1-IP>:8000/health` in a browser.
   If it hangs, it is the firewall or a different network (some campus Wi-Fi blocks device-to-device traffic; use a phone hotspot).
3. In `frontend/`, create `.env.local` containing: `VITE_API_URL=http://<LAPTOP-1-IP>:8000`
4. `npm install`, then `npm run dev`, open the printed `http://localhost:5173`.
5. Pick a trip, press **Reconstruct route**. The map shows the GPS trace (blue), the matched route (purple), start (green) and end (red).

## Screenshots needed for the deliverables

- Laptop 2: the UI after a successful prediction.
- Laptop 1: the uvicorn terminal showing `PREDICT request from <Laptop 2 IP>: trip_id=...`, which proves the request arrived over the network.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI says "Cannot reach the PathSnap API" | Wrong IP in `.env.local` (restart `npm run dev` after editing), firewall rule missing, or API started without `--host 0.0.0.0` |
| `health` works on Laptop 1 but not Laptop 2 | Different networks, or Wi-Fi client isolation: use a phone hotspot |
| API error 404 | Trip ID not in the student_test folder |
| Model fails to load | scikit-learn is not 1.7.2 |
