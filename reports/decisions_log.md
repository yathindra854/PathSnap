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

## D-008: Model family for candidate classification
- **Date:** 2026-11-18
- **Owner:** B (with C)
- **Status:** Accepted (selected on CV only; student-test used as confirmation)
- **Candidates compared (5 families, GroupKFold by trip, 5 folds on 77 train trips):**

| Model | CV macro-F1 | CV log loss | CV top-1 | Student-test top-1 | Notes |
|---|---|---|---|---|---|
| Logistic Regression | 0.860 | 0.143 | 0.792 | 0.794 | baseline, full data |
| KNN (k=15) | 0.862 | 0.189 | 0.783 | 0.782 | fit on 60k subsample; slowest inference |
| Random Forest | 0.883 | 0.108 | 0.803 | 0.803 | fit on 300k subsample |
| HistGradientBoosting | 0.883 | 0.110 | 0.801 | 0.801 | full data, fastest to train among the strong models |
| Shallow MLP (64, 32) | 0.878 | 0.113 | 0.801 | 0.799 | fit on 150k subsample |

- **Choice:** HistGradientBoosting.
- **Rationale:** Random Forest and HGB are tied within noise on macro-F1 and top-1. Random Forest has slightly better log loss, but HGB uses all the training data without subsampling, trains in about 8 s, and has more room for Bayesian tuning in step 6. The MLP is a close third. Logistic Regression and KNN are clearly weaker.
- **Caveat for the report:** the comparison used the 9-feature set. The final model uses the reduced set (D-009).
- **Evidence:** `experiments/results/model_comparison.csv`.

## D-009: Feature set (reduced to 5)
- **Date:** 2026-11-18
- **Owner:** B
- **Status:** Accepted
- **Method:** leave-one-feature-out ablation with HGB under the same GroupKFold(5) protocol, then a combined run of the reduced set.
- **Results (CV macro-F1 / top-1 / log loss):**
  - Baseline, 9 features: 0.883 / 0.801 / 0.110
  - Drop `heading_diff`: 0.751 / 0.551 / 0.169. **Critical.**
  - Drop `proj_frac`: 0.876 / 0.795 / 0.118. Keep.
  - Drop `perp_rel`: 0.880 / 0.797 / 0.112. Keep.
  - Drop `speed`: 0.880 / 0.796 / 0.119. Keep.
  - Drop `perp_dist`: 0.882 / 0.801 / 0.110. Redundant with `perp_rel`.
  - Drop `rank`, `arc_len`, `log_dt_next`, `n_cands`: no meaningful change.
  - **Reduced set** (`perp_rel`, `proj_frac`, `heading_diff`, `speed`, `n_cands`): 0.881 / 0.803 / 0.109.
- **Choice:** the reduced 5-feature set.
- **Rationale:** the same accuracy with fewer features, and a simpler set to explain in the viva. `heading_diff` is the most important feature: true arcs have a median heading difference of about 23°, versus about 96° for other candidates. The log-loss change is within the range the team should check again after tuning.
- **Evidence:** `experiments/results/feature_ablation.csv`.

## D-010: Final model result (student-test, reported once)
- **Date:** 2026-11-18
- **Owner:** B
- **Status:** Accepted
- **Result:** HGB on the reduced set, trained on all 77 train trips, scored on the 10 student-test trips: macro-F1 0.887, log loss 0.100, top-1 0.803.
- **Artifact:** `artifacts/final_hgb.joblib`. Metrics: `experiments/results/final_model_metrics.csv`.
- **Caveat:** the student-test set was not used to choose the model or features. The evaluator-test set is still untouched.

## D-011: Person B pipeline vs Phase 1 model (keep Phase 1 model)
- **Date:** 2026-11-18
- **Owner:** B
- **Status:** Accepted
- **Compared:** the Phase 1 model (`artifacts/final_hgb.joblib`) against the Person B notebook pipeline (`artifacts/person_b/candidate_classifier.joblib`). Both are HistGradientBoosting.

| Student-test (10 trips) | Phase 1 model | Person B model |
|---|---|---|
| Features | 5 (`perp_rel`, `proj_frac`, `heading_diff`, `speed`, `n_cands`) | 2 (`perp_dist`, `proj_frac`) |
| Training data | all labelled rows, 77 trips | 10k balanced rows per CV fold; final fit on 80 trips |
| Top-1 accuracy (per GPS point) | **0.803** | 0.524 |
| Macro-F1 | **0.887** | 0.657 |
| Log loss | **0.100** | 0.174 |
| Bayesian tuning | not yet | yes, Optuna TPE, 12 trials |

- **Choice:** keep the Phase 1 model as the selected classifier.
- **Rationale:** it is better on every accuracy measure. The main cause is features, not the algorithm: the Person B model has no heading feature, and dropping `heading_diff` in our ablation (D-009) gave a similar collapse in top-1 (0.551). Person B also trained on the three trips dropped under D-002.
- **Kept from Person B:** the Bayesian optimisation setup (Optuna TPE, GroupKFold on train only, trial log) and the 5-model baseline. These satisfy the tuning requirement in method, but the tuned parameters were found for 2 features and must be re-searched on the 5-feature set before being reported as the tuned model.
- **Note for the report:** the two baseline tables are not comparable. Person B's CV scores are on sampled, balanced validation rows, and its model set includes a Decision Tree instead of an MLP.
- **Evidence:** `experiments/results/final_model_metrics.csv`, `experiments/results/person_b/student_test_metrics.json`.

## D-012: HMM/Viterbi decoder design and settings
- **Date:** 2026-11-18
- **Owner:** C
- **Status:** Accepted
- **Design:**
  - Hidden states: the candidate arcs of each GPS point (50 m radius, D-007).
  - Emission: log of the HGB classifier probability.
  - Transition: `-beta x hops`, where hops is the number of directed arc-to-arc moves from the previous arc to the next (0 = same arc), searched up to 6 hops. Reversing onto the opposite-direction arc costs an extra `uturn_pen`.
  - Unreachable or disconnected arcs get a finite penalty (`-beta x 7 - disconnect_pen`) instead of zero probability, so decoding never fails on fragmented networks.
  - GPS gaps over 300 s restart the chain (D-003).
- **Results, student-test (10 trips, 18,846 labelled points):**

| Method | Top-1 | Route continuity | Route arc F1 |
|---|---|---|---|
| Classifier only | 0.803 | 0.899 | 0.801 |
| + HMM, default (beta 1, uturn 0) | 0.879 | 0.975 | 0.901 |
| + HMM, best of grid (beta 4, disconnect 10, uturn 3) | 0.915 | 0.981 | 0.938 |

- **Choice:** beta 4, disconnect penalty 10, U-turn penalty 3.
- **How chosen:** a 16-setting grid on the student-test trips. The guideline permits optimising decoder parameters on training and student-test data. Because the same trips are used to report the result, the tuned figure is optimistic. The default-setting row is the untuned comparison.
- **Open:** the best beta is at the edge of the grid. Check higher values on training trips (with out-of-fold classifier probabilities) if time allows. The evaluator set was not used.
- **Evidence:** `experiments/results/hmm_student_test.csv`, `reports/figures/hmm_vs_classifier.png`.

## D-013: Full metric set for all 5 models (final 5-feature set)
- **Date:** 2026-11-18
- **Owner:** B
- **Status:** Accepted
- **Why:** the guideline asks for accuracy, precision, recall, F1, confusion matrix, log loss, top-1, route continuity, and training and inference time for every model. The D-008 comparison used the 9-feature set and lacked several of these. All 5 models were rerun on the final 5-feature set.
- **Protocol:** out-of-fold predictions from GroupKFold(5) by trip on 77 training trips; one refit on all training trips; one score on the 10 student-test trips.
- **Results:**

| Model | CV macro-F1 (±std) | CV top-1 | CV log loss | Student acc. | Student precision | Student recall | Student macro-F1 | Student log loss | Student top-1 | Route continuity | Train s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.843 (±0.029) | 0.766 | 0.152 | 0.952 | 0.860 | 0.841 | 0.850 | 0.147 | 0.769 | 0.890 | 0.8 |
| KNN | 0.860 (±0.025) | 0.778 | 0.192 | 0.958 | 0.886 | 0.850 | 0.867 | 0.181 | 0.782 | 0.888 | 0.04 |
| Random Forest | 0.880 (±0.026) | 0.803 | 0.109 | 0.964 | 0.900 | 0.874 | 0.887 | 0.099 | 0.803 | 0.900 | 21.3 |
| **HistGradientBoosting** | **0.881 (±0.026)** | 0.802 | 0.109 | 0.964 | 0.899 | 0.876 | **0.887** | 0.099 | **0.804** | 0.901 | 5.5 |
| Shallow MLP | 0.873 (±0.025) | 0.796 | 0.116 | 0.962 | 0.898 | 0.865 | 0.881 | 0.105 | 0.797 | 0.902 | 3.7 |

  Precision and recall are macro-averaged at threshold 0.5. Route continuity is classifier-only (best candidate per point).
- **Effect on D-008:** none. HistGradientBoosting is still tied with Random Forest at the top and trains about 4x faster.
- **MLP training curve:** training and validation log loss both fall steadily over 30 epochs with no rise in validation loss, so there is no overfitting. Validation loss is lower than training loss because the held-out trips are easier, not because of leakage.
- **Evidence:** `experiments/results/full_metrics/`, `reports/figures/models/01-09`.

## D-014: Bayesian hyperparameter optimisation (HistGradientBoosting)
- **Date:** 2026-11-18
- **Owner:** C
- **Status:** Accepted
- **Search:** Optuna TPE sampler (seed 42), 20 trials, 231 s. Trial 0 is the untuned default.
- **Space:** learning_rate log-uniform [0.02, 0.3]; max_iter [100, 500] step 50; max_leaf_nodes [15, 127] log; min_samples_leaf [10, 200] log; l2_regularization log-uniform [0.001, 10].
- **Objective:** mean macro-F1, GroupKFold(3) by trip, on a fixed 600k-row sample of the training rows. Student-test and evaluator data were not used in the search.
- **Best:** learning_rate 0.104, max_iter 150, max_leaf_nodes 17, min_samples_leaf 172, l2_regularization 7.29. These settings give smaller, more regularised trees.
- **Tuned vs untuned (both refit on all training rows; full GroupKFold(5) CV; same student-test trips):**

| | CV macro-F1 | CV log loss | CV top-1 | Student top-1 | Student macro-F1 | Student log loss | Fit s | Inference ms / 1k rows |
|---|---|---|---|---|---|---|---|---|
| Untuned | 0.8813 | 0.1087 | 0.8032 | 0.8038 | 0.8869 | 0.0991 | 6.1 | 1.9 |
| Tuned | 0.8814 | 0.1080 | 0.8036 | 0.8043 | 0.8868 | 0.0988 | 7.6 | 2.9 |

- **Conclusion:** the tuned model is marginally better on every measure except student macro-F1, which is equal. The gains are far smaller than the fold standard deviation (about 0.023), so the model is insensitive to these hyperparameters; feature quality (D-009) and decoding (D-012) matter much more. The tuned model is kept as the final classifier: `artifacts/final_hgb_tuned.joblib`.
- **Caveat:** the per-point top-1 values recorded inside the search (about 0.38) are not meaningful, because the row sample keeps only part of each point's candidate set. The search optimised macro-F1, which is unaffected. All top-1 figures above come from the full data.
- **Evidence:** `experiments/results/bayes/`, `reports/figures/models/10_bayes_history.png`, `11_tuned_vs_untuned.png`.

## D-015: HMM rerun with the tuned classifier
- **Date:** 2026-11-18
- **Owner:** C
- **Status:** Accepted. Updates the D-012 figures.
- **Student-test results with the tuned model:** classifier-only top-1 0.804, continuity 0.900, route F1 0.803. HMM default 0.880 / 0.975 / 0.901. HMM best grid (beta 4, disconnect 10, U-turn 3) 0.914 / 0.981 / 0.938. The same decoder settings are best as in D-012.
- **Evidence:** `experiments/results/hmm_student_test.csv`, `reports/figures/models/12_hmm_vs_classifier.png`.

## D-016: API design (Laptop 1)
- **Date:** 2026-11-18
- **Owner:** D
- **Status:** Accepted
- **Choice:** FastAPI app in `backend/app`, run with `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`. The UI sends only a trip ID (`POST /api/v1/predict`); the trip list comes from `GET /api/v1/trips`.
- **Pipeline:** the same code as training. Feature computation moved to `src/pathsnap/features/engineering.py` and is now shared by `scripts/03a_features.py` and the API; the student-test feature table is byte-identical before and after the move.
- **Ground truth:** the reference route is removed from the trip before projection, candidates, features, classifier and decoder. It is used only afterwards to compute metrics on Laptop 1. Responses contain metrics only, never the reference arcs or labels.
- **Display:** where the decoder moves between non-adjacent arcs, the connecting arcs (shortest directed path, at most 6 hops) are filled in for drawing only. Metrics use the decoded arcs. Route ends are trimmed to the first and last snapped GPS points.
- **Verification:** all 10 student-test trips through the API give pooled point accuracy 0.9136 (HMM) and 0.8043 (classifier only), identical to `scripts/05_decode_hmm.py`. Each trip takes 0.06 to 0.8 s. Unknown trip IDs return 404.
- **Logging:** each request prints the caller IP, trip ID, accuracy, continuity and time in the uvicorn terminal, for the evaluators.

---

## Change history
| Date | ID | Change |
|---|---|---|
| 2026-11-18 | D-001 to D-006 | Initial log created |
| 2026-11-18 | D-001, D-002, D-003 | Team chose B, A and A. Status set to Accepted. Numeric thresholds still to confirm on train. |
| 2026-11-18 | D-001, D-004, D-007 | D-001 superseded by D-007 (fixed 50 m). D-004 superseded for candidate generation by per-trip UTM. D-005 marked provisional, with the tile issue recorded. |
| 2026-11-18 | D-008, D-009, D-010 | Model family selected (HistGradientBoosting, CV only). Feature set reduced from 9 to 5. Final student-test result recorded. |
| 2026-11-18 | D-011 | Phase 1 model kept over Person B model (top-1 0.803 vs 0.524). Person B Optuna setup to be rerun on the 5-feature set. |
| 2026-11-18 | D-012 | HMM/Viterbi decoder built. Student-test top-1 0.803 -> 0.915 (decoder settings tuned on student-test, disclosed). |
| 2026-11-18 | D-013, D-014, D-015 | Full metrics for all 5 models on the 5-feature set. Optuna tuning done (marginal gain; tuned model kept). HMM rerun with tuned model: top-1 0.914. |
| 2026-11-18 | D-016 | API built on the real pipeline; shared feature module; verified on all 10 student-test trips. |
