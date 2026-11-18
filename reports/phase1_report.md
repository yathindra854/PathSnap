# PathSnap: Phase 1 Report

**Scope:** data exploration, label alignment, candidate generation, feature engineering, model comparison, feature selection, and the first final-model score.
**Status:** Phase 1 complete. Bayesian tuning, HMM/Viterbi decoding, the API and the UI are Phase 2.
**Integrity:** the 10 evaluator trips were never loaded, used or inspected. Student-test results were reported once, after model and feature choices were made on training CV.

---

## 1. Dataset understanding

- **Partitions:** 80 training trips, 10 student-test trips. Each trip has GPS observations (`.track`), a road network (`.nodes`, `.arcs`) and a reference route (`.route`).
- **Data quality:** no missing values, invalid coordinates, invalid arc IDs or out-of-order timestamps.
- **Sampling:** about 99% of steps are exactly 1 s in most trips. Some trips have large gaps, up to 153,064 s in trip 90.
- **Road network:** every network is fragmented into disconnected pieces. The largest piece holds 69–100% of nodes.
- **Projection:** each trip uses its own UTM zone, since trips come from different cities.
- **Reference routes:** 11 routes have breaks where consecutive arcs don't connect. They are treated as reference, not verified ground truth.

Full inventory: `reports/tables/dataset_inventory.csv`. Figures: `reports/figures/`.

## 2. Label alignment and candidates

- **Labels:** each GPS point is assigned to one reference arc with a forward-only walk along the route. Points more than 50 m from the route are unlabelled (-1) and excluded from training and scoring.
- **Candidates:** every arc within 50 m of a point. Mean 12 candidates per point, maximum about 41.
- **Unlabelled share:** 12.6% of training points, 1.6% of student-test points. Most of the training figure comes from trips 54 and 33.
- **Miss rate:** reported as 0%, but this is not evidence. Labels and candidates both use the same 50 m limit, so the check is circular. A proper check is still required (see §8).

## 3. Trips dropped from training (D-002)

| Trip | Reason |
|---|---|
| 54 | 89% of points unlabelled; median distance to route about 140 m |
| 33 | 75% of points unlabelled; median distance to route about 120–214 m |
| 2 | 12% of points unlabelled; 95th-percentile alignment distance about 240 m |

Training uses 77 trips after these exclusions.

## 4. Features

Nine features were engineered per (point, candidate) pair, and five were kept (see §6):

- `perp_rel`: distance from the GPS point to the arc, minus the closest candidate's distance
- `proj_frac`: where the point projects along the arc
- `heading_diff`: angle between the GPS movement direction and the arc direction
- `speed`: GPS speed at the point
- `n_cands`: number of candidates for the point

## 5. Model comparison

Five model families, 5-fold GroupKFold by trip on the 77 training trips.

| Model | CV macro-F1 (±std) | CV log loss | CV top-1 | Student macro-F1 | Student log loss | Student top-1 | Train time (s) | Inference (s/10k) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.860 (±0.027) | 0.143 | 0.792 | 0.867 | 0.135 | 0.794 | 2.0 | 0.0008 |
| KNN (k=15) | 0.862 (±0.023) | 0.189 | 0.783 | 0.868 | 0.183 | 0.782 | ~0.05 | 0.89 |
| Random Forest | 0.883 (±0.026) | 0.108 | 0.803 | 0.887 | 0.099 | 0.803 | 27.5 | 0.025 |
| **HistGradientBoosting** | **0.883 (±0.025)** | 0.110 | 0.801 | 0.885 | 0.100 | 0.801 | 7.5 | 0.028 |
| Shallow MLP (64, 32) | 0.878 (±0.025) | 0.113 | 0.801 | 0.879 | 0.104 | 0.799 | 4.3 | 0.0045 |

**Notes:**
- Random Forest, KNN and the MLP were fit on subsamples (300k, 60k and 150k rows). Logistic Regression and HGB used the full data.
- Random Forest and HGB are tied within fold noise.

**Selected family: HistGradientBoosting.** It uses all the training data, trains in about 8 seconds, scores well on both CV and student-test, and leaves room for Bayesian tuning in Phase 2. Random Forest has a slightly better log loss, so this is a judgement call (D-008).

## 6. Feature selection

Leave-one-feature-out ablation with HGB, same CV protocol (D-009).

| Dropped feature | CV top-1 | Effect |
|---|---|---|
| None (9 features) | 0.801 | baseline |
| `heading_diff` | 0.551 | **critical**, kept |
| `proj_frac` | 0.795 | kept |
| `perp_rel` | 0.797 | kept |
| `speed` | 0.796 | kept |
| `perp_dist` | 0.801 | redundant with `perp_rel` |
| `rank`, `arc_len`, `log_dt_next`, `n_cands` | ~0.80 | no meaningful change |

**Reduced set (5 features):** top-1 0.803, macro-F1 0.881, log loss 0.109. Matches the full set and is simpler to explain.

## 7. Final model result (D-010)

HGB on the 5-feature set, trained on all 77 training trips, scored once on the 10 student-test trips:

- Macro-F1: **0.887**
- Log loss: **0.100**
- Top-1 accuracy: **80.3%**

Artifact: `artifacts/final_hgb.joblib`.

## 8. Known limitations and open risks

1. **Miss rate is circular** (always 0%). Needs a check with a smaller candidate radius than the label tolerance.
2. **Radius:** the team chose a fixed 50 m candidate radius (D-007), and the sweep (30/50/75/100 m on train) is not yet run.
3. **Trips 93 and 34** have 64% and 24% of points unlabelled. They are not yet on the drop list (D-002 open item).
4. **Subsampled comparisons:** three of the five models were fit on subsamples, so the comparison may slightly understate them.
5. **Comparison features:** the model table used the 9-feature set. The final model uses the reduced set.
6. **Top-1 is per point, not per route.** A high top-1 does not guarantee a connected route, which is what the HMM decoder addresses in Phase 2.

## 9. Phase 2 plan

1. Bayesian hyperparameter search for HGB (Optuna, TPE, GroupKFold objective).
2. HMM/Viterbi decoding, with the gap split (D-003) and fallback for disconnected arcs.
3. Classifier-only vs classifier-plus-HMM comparison, with route continuity.
4. Resolve trips 93 and 34, and the radius sweep, before the final training run.
5. Export the model and HMM configuration; build the FastAPI endpoint and evaluator UI.

## 10. Decisions made in Phase 1

Recorded in `reports/decisions_log.md`:
- D-002: drop trips 54, 33 and 2.
- D-003: split trips at large time gaps (provisional threshold about 300 s).
- D-007: fixed 50 m candidate radius (supersedes the adaptive option, D-001).
- D-008: HistGradientBoosting selected for candidate classification.
- D-009: five-feature set.
- D-010: final student-test result, reported once.
