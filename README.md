# PathSnap: Distributed Map-Matching and Route Reconstruction

Team project for UE24CS352A Machine Learning hackathon.

PathSnap matches noisy GPS trajectories to a road network. A classifier scores
candidate road arcs for each GPS observation, and an HMM/Viterbi decoder
produces a connected route.

## Status
Skeleton only. See `docs/` (to be added) for the plan.

## Repository layout

| Path | Purpose | Owner |
|---|---|---|
| `src/pathsnap/data/` | Loading, cleaning, label alignment | A |
| `src/pathsnap/features/` | Candidate generation and feature engineering | A, B |
| `src/pathsnap/models/` | Model training, comparison, CV | B |
| `src/pathsnap/hmm/` | HMM and Viterbi route reconstruction | C |
| `src/pathsnap/evaluation/` | Metrics: top-1, macro-F1, log loss, route continuity | B, C |
| `src/pathsnap/viz/` | Folium maps and plots | A |
| `scripts/` | Entry points (EDA, train, tune, decode, export) | all |
| `api/` | FastAPI backend (model loading, inference endpoint) | D |
| `ui/` | Evaluator-facing interface | D |
| `configs/` | Experiment and pipeline configuration | all |
| `experiments/` | Logs, CV results, Optuna trial records | B, C |
| `artifacts/` | Exported model and inference assets | D |
| `reports/` | Figures, tables, and the technical report | all |
| `notebooks/` | Exploration notebooks | A |
| `tests/` | Unit tests | all |

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data
Place the course-provided `Student folder/` contents under `data/raw/`.
Raw data is git-ignored.

## Running
Step 02 — candidate table (label alignment + candidate arcs within R metres):
```bash
python scripts/02_build_candidates.py --partition train --radius 50 --limit 5   # quick v0
python scripts/02_build_candidates.py --partition train --radius 50
python scripts/02_build_candidates.py --partition student_test --radius 50
```
Outputs `data/processed/candidates_<partition>.parquet` (git-ignored) and
`reports/tables/candidate_stats_<partition>_r<R>.csv`.

Candidate table columns: `trip_id, point_idx, t, arc_id, perp_dist, proj_frac, label`
(`label`: 1 = reference arc, 0 = other candidate, -1 = point not covered by the reference route).

Other steps: TBD.

## Integrity
Evaluator-test trips and labels are never used for training, tuning, or
selection, and are never exposed to the prediction pipeline or UI.
