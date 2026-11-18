# PathSnap Decisions Log

Record every decision that affects data, labels, models, evaluation or the report.
Status values: **Proposed** (not yet agreed), **Accepted** (agreed, in use), **Superseded** (replaced by a later entry, which is linked).

Rules:
- Set thresholds from **training-data statistics only**. Student-test results may be used for development comparison, but never to tune a threshold after seeing its effect. Evaluator-test data is never used.
- Each entry gets a new ID. Do not edit an accepted decision in place. Supersede it with a new entry.
- Put the rationale in the entry. These records feed the report's methodology section and the viva.

---

## D-001: Candidate search radius (adaptive)
- **Date:** 2026-11-18
- **Owner:** A (with B, C)
- **Status:** Superseded by D-007
- **Choice at the time:** B, adaptive radius. Start at 30 m, widen to 100 m only for points with no candidate.

## D-002: Problem trips 54, 33 and 2
- **Date:** 2026-11-18
- **Owner:** A (with B, C)
- **Status:** Accepted
- **Options considered:**
  - A. Drop all three trips
  - B. Filter points: keep only points within 50 m of the reference route
  - C. Keep everything and flag them
  - D. Hold them out as a stress test
- **Choice:** A. Drop trips 54, 33 and 2 from training and CV.
- **Rationale:** the team chose the simplest option with clean labels. This departs from the earlier recommendation to keep trip 2 with a point filter. Trip 2 is 1 of 80 train trips, so the cost is small. About 83% and 58% of points in trips 54 and 33 are more than 50 m from the route. About 12% of points in trip 2 are, so its labels are less reliable than the rest.
- **Constraint:** the dropped trips are reported in the methodology, with the reason and the distance statistics.
- **Evidence:** `reports/tables/gps_to_route_distance.csv`. Alignment check on the `person-a-data` branch: trips 54 and 33 have 89% and 75% of points unlabelled.
- **Open:** trips 93 (64% unlabelled) and 34 (24% unlabelled) are not on this list yet. Review them before the final training run.

## D-003: Large time gaps (for example trips 78 and 90)
- **Date:** 2026-11-18
- **Owner:** C (with A)
- **Status:** Accepted (method). Threshold to be set from train-data gap statistics.
- **Options considered:**
  - A. Split trips at large gaps, with no HMM transition across a gap
  - B. Drop the spans after gaps
  - C. Time-aware transitions (weaker connectivity over longer gaps)
  - D. Drop the trips
- **Choice:** A. Split trips at large gaps, with no HMM transition across a gap.
- **Provisional threshold:** about 300 s. Confirm from train-data gap statistics before use.
- **Rationale:** simple to implement and explain in the viva. It keeps the regular stretches of trips 78 and 90 instead of discarding them.
- **Evidence:** trip 78 has a largest gap of 52,810 s and 124 gaps over 5 s. Trip 90 has a largest gap of 153,064 s. Trip 21 has a largest gap of 2,161 s. (`reports/tables/dataset_inventory.csv`)

## D-004: Projection for distance calculations
- **Date:** 2026-11-18
- **Owner:** A
- **Status:** Accepted (EDA only). Superseded for candidate generation by the per-trip UTM setting on the `person-a-data` branch.
- **Choice:** local equirectangular projection per trip, centred on the trip's node centroid, for EDA distances.
- **Rationale:** accurate at city scale, with no extra dependencies. Plain degree distances are not appropriate (Dataset Exploration §5).

## D-005: Base map tiles for Folium
- **Date:** 2026-11-18
- **Owner:** A
- **Status:** Accepted (provisional). Under review.
- **Choice:** OpenStreetMap tiles.
- **Known issue:** OpenStreetMap's volunteer servers block tile requests from HTML files opened directly from disk, because those requests don't meet the tile usage policy. The route and GPS layers still render. Fix options: no basemap for the report maps, or serve the maps through a local web server for the demo.

## D-006: Repo workflow
- **Date:** 2026-11-18
- **Owner:** D
- **Status:** Accepted
- **Choice:** commit and push from a local terminal, not from the cloud sandbox. The sandbox can't push to GitHub.
- **Rationale:** the push was blocked with a 403 in this environment, and git lock files left behind need clearing on the local machine.

## D-007: Candidate search radius (fixed 50 m)
- **Date:** 2026-11-18
- **Owner:** A (with B, C)
- **Status:** Accepted. Supersedes D-001.
- **Choice:** fixed 50 m candidate radius, set as `candidates.radius_m` in `configs/config.yaml`.
- **Rationale:**
  - Already implemented on the `person-a-data` branch, so no extra build time before the 4 pm deadline.
  - On normal trips, GPS sits about 4–8 m from the reference route at the median, and about 15–25 m at the 95th percentile. A 50 m radius leaves a wide margin.
  - Mean candidates per point are about 12, with a maximum of about 41, which is manageable.
  - An adaptive radius would make candidate sets and feature scales vary between points. That complicates the classifier and HMM, and is harder to explain in the viva. It can be added later if the honest miss-rate check shows real misses.
- **Still to do:**
  - Replace the circular miss-rate check. The current check is always 0 because labels and candidates use the same 50 m limit. Measure misses with a smaller candidate radius than the label tolerance, or set the two separately.
  - Sweep 30, 50, 75 and 100 m on train only, and record the results.
- **Confirmation:** chosen by the team member directing the work. The rest of the team should confirm before training starts.

---

## Change history
| Date | ID | Change |
|---|---|---|
| 2026-11-18 | D-001 to D-006 | Initial log created |
| 2026-11-18 | D-001, D-002, D-003 | Team chose B, A and A. Status set to Accepted. Numeric thresholds still to confirm on train. |
| 2026-11-18 | D-001, D-004, D-007 | D-001 superseded by D-007 (fixed 50 m). D-004 superseded for candidate generation by per-trip UTM. D-005 marked provisional, with the tile issue recorded. |
