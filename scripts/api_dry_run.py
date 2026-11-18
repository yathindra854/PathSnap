import json
import time
import csv
import urllib.request
import urllib.error
from pathlib import Path

base_url = "http://127.0.0.1:8002/api/v1/predict"
test_dir = Path("data/raw/student_test")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

trips = sorted(p for p in test_dir.iterdir() if p.is_dir())
rows = []

print(f"PathSnap API dry run: {len(trips)} trips\n")

for i, trip_dir in enumerate(trips, 1):
    trip_id = trip_dir.name
    track_file = trip_dir / f"{trip_id}.track"

    try:
        points = []
        with track_file.open() as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    lon, lat, timestamp = map(float, parts[:3])
                    points.append({
                        "lon": lon,
                        "lat": lat,
                        "timestamp": timestamp
                    })
                if len(points) == 10:
                    break

        payload = {
            "trip_id": trip_id,
            "points": points
        }

        req = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode())
        elapsed = time.perf_counter() - start

        success = (
            result.get("status") == "success"
            and len(result.get("matched_points", [])) == len(points)
        )

        rows.append({
            "trip_id": trip_id,
            "status": "PASS" if success else "FAIL",
            "input_points": len(points),
            "output_points": len(result.get("matched_points", [])),
            "mode": result.get("mode", ""),
            "time_seconds": round(elapsed, 3),
            "error": ""
        })

        print(
            f"[{i}/{len(trips)}] {trip_id}: "
            f"{'PASS' if success else 'FAIL'} | "
            f"{len(points)} points | {elapsed:.2f}s"
        )

    except Exception as e:
        rows.append({
            "trip_id": trip_id,
            "status": "FAIL",
            "input_points": 0,
            "output_points": 0,
            "mode": "",
            "time_seconds": 0,
            "error": str(e)
        })
        print(f"[{i}/{len(trips)}] {trip_id}: FAIL | {e}")

csv_path = results_dir / "api_dry_run_10_trips.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
    writer.writeheader()
    writer.writerows(rows)

passed = sum(r["status"] == "PASS" for r in rows)
failed = len(rows) - passed

print("\n========== DRY RUN SUMMARY ==========")
print(f"Total trips : {len(rows)}")
print(f"Passed     : {passed}")
print(f"Failed    : {failed}")
print(f"Saved     : {csv_path}")
print("==================================")
