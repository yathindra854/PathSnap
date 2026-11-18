# PathSnap — Person B Modelling Work Package

This folder contains the Person B modelling notebooks. All source code is in `.ipynb` notebooks, as requested.

## Current status

- `01_dataset_exploration.ipynb`: executed against the supplied 80 training + 10 student-testing trips.
- `02_candidate_table_validation.ipynb`: validates Person A's candidate-table interface.
- `03_baseline_models_groupkfold.ipynb`: five-model grouped cross-validation.
- `04_bayesian_optimization.ipynb`: Optuna TPE Bayesian optimisation and retained search records.
- `05_student_test_and_export.ipynb`: permitted student-test evaluation and model export.

## Findings from the executed raw-data exploration

Notebook 01 was executed against the supplied archive (not a synthetic dataset). It found:

- Training partition: 80 trips, 208,417 GPS observations, 1,933,485 nodes, 3,876,904 arcs, and 31,336 reference-route arc entries.
- Student-testing partition: 10 trips, 19,160 GPS observations, 90,689 nodes, 185,041 arcs, and 3,477 reference-route arc entries.
- All four raw files were present for all 90 development trips.
- No missing or invalid GPS rows were detected in these two partitions.
- Non-1-second timestamp intervals were present (7,289 in training and 264 in student-testing). The largest observed timestamp gap was 153,064 seconds in training and 210 seconds in student-testing; inspect the trip-level CSV before deciding how to handle gaps.

These are descriptive dataset findings, not model-performance results. Evaluator-testing data was not inspected.

## Data placement

Place the supplied dataset at `data/raw/Student folder/` with `train/` and `student_test/` beneath it. Do not commit raw data. Notebook 01 resolves paths from the repository root.

Person A must provide these processed files before notebooks 02–05 can run:

- `data/processed/train_candidates.csv`
- `data/processed/student_test_candidates.csv`

Each CSV must contain `trip_id`, `observation_idx`, `candidate_arc_id`, `is_match`, plus numeric feature columns. Person A must document how the supplied `.route` reference labels are aligned to GPS observations. Do not assume one GPS row maps to one route arc.

## Run order

1. Open the repository root in VS Code. Select the project's Python environment as the notebook kernel.
2. Install dependencies: `python -m pip install pandas numpy matplotlib networkx scikit-learn optuna joblib jupyter`.
3. Run `01_dataset_exploration.ipynb`.
4. After Person A supplies the agreed candidate CSVs, run `02_candidate_table_validation.ipynb`. Resolve all validation issues with Person A.
5. Run `03_baseline_models_groupkfold.ipynb`.
6. Run `04_bayesian_optimization.ipynb`.
7. Run `05_student_test_and_export.ipynb`.

## Output locations

- `outputs/exploration/`: raw-data exploration CSVs.
- `configs/person_b_feature_manifest.json`: selected numeric feature list and data contract.
- `experiments/results/person_b/`: fold scores, model comparison, optimisation database/trials, student-test metrics and probabilities.
- `artifacts/person_b/`: exported classifier and metadata.

## Important integrity rules

- No evaluator-testing data is used by these notebooks.
- No model metrics are included as pre-filled claims; they are generated only when the real candidate data is supplied and the notebooks are run.
- Keep Optuna trial records and fold-level scores.
- The exported model is a development integration artifact; final evaluation must be performed through the evaluator's permitted interface.
- Review and understand every modelling choice before presenting or submitting it.
