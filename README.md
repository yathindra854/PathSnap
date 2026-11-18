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
TBD. Commands will be added as each stage is implemented.

## Integrity
Evaluator-test trips and labels are never used for training, tuning, or
selection, and are never exposed to the prediction pipeline or UI.
