"""Pipeline step 04: Bayesian hyperparameter optimisation of the selected model. Owner: C

Model    : HistGradientBoosting on the final 5-feature set (D-008, D-009).
Sampler  : Optuna TPE (Bayesian), seed 42.
Objective: mean macro-F1 over GroupKFold(3) by trip on the training trips only.
           For speed each trial uses a fixed random 600k-row sample of the training
           rows (seed 42); folds still split by trip. Trial 0 is the untuned default
           configuration, so tuned vs untuned is compared under the same protocol.
Then     : the best and the default configurations are both refit on all training
           rows, scored with full GroupKFold(5) CV and once on the student-test trips.
Student-test and evaluator data are never used inside the search.

Usage: python scripts/04_tune_bayes.py [n_trials]
Writes experiments/results/bayes/ and artifacts/final_hgb_tuned.joblib
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["perp_rel", "proj_frac", "heading_diff", "speed", "n_cands"]
SEED = 42
OUT = ROOT / "experiments/results/bayes"
DEFAULT = dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0)
SPACE = {
    "learning_rate": "log-uniform [0.02, 0.3]",
    "max_iter": "int [100, 500], step 50",
    "max_leaf_nodes": "int [15, 127], log",
    "min_samples_leaf": "int [10, 200], log",
    "l2_regularization": "log-uniform [1e-3, 10] (0 = untuned default)",
}


def load(p):
    d = pd.read_parquet(ROOT / f"data/processed/feats_{p}.parquet")
    return d[~d.unlabelled_point].reset_index(drop=True)


def top1(df, p):
    d = df[["trip_id", "point_idx", "label"]].assign(p=p)
    return d.loc[d.groupby(["trip_id", "point_idx"])["p"].idxmax()]["label"].mean()


def cv_score(params, df, folds):
    X, y, g = df[FEATS].to_numpy(float), df.label.to_numpy(), df.trip_id
    f1s, lls, t1s = [], [], []
    for a, b in GroupKFold(folds).split(X, y, groups=g):
        m = HistGradientBoostingClassifier(**params, random_state=SEED).fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        f1s.append(f1_score(y[b], p > 0.5, average="macro"))
        lls.append(log_loss(y[b], np.clip(p, 1e-7, 1 - 1e-7)))
        t1s.append(top1(df.iloc[b], p))
    return dict(cv_f1_macro=np.mean(f1s), cv_f1_macro_std=np.std(f1s), cv_log_loss=np.mean(lls), cv_top1=np.mean(t1s))


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = load("train"), load("student_test")
    sub = tr.sample(n=min(600_000, len(tr)), random_state=SEED).reset_index(drop=True)

    def objective(trial):
        params = dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                      max_iter=trial.suggest_int("max_iter", 100, 500, step=50),
                      max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
                      min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
                      l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True))
        r = cv_score(params, sub, 3)
        for k, v in r.items():
            trial.set_user_attr(k, v)
        return r["cv_f1_macro"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({**DEFAULT, "l2_regularization": 1e-3})   # trial 0 = untuned baseline (l2 ~ 0)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    search_s = time.time() - t0
    study.trials_dataframe().to_csv(OUT / "optuna_trials.csv", index=False)
    best = study.best_params

    rows = []
    for tag, params in [("untuned", DEFAULT), ("tuned", best)]:
        cv = cv_score(params, tr, 5)
        t0 = time.time()
        m = HistGradientBoostingClassifier(**params, random_state=SEED).fit(tr[FEATS].to_numpy(float), tr.label.to_numpy())
        fit_s = time.time() - t0
        t0 = time.time()
        p = m.predict_proba(te[FEATS].to_numpy(float))[:, 1]
        inf = time.time() - t0
        y = te.label.to_numpy()
        rows.append(dict(model=tag, **{k: round(float(v), 5) for k, v in cv.items()},
                         st_top1=round(top1(te, p), 5), st_f1_macro=round(f1_score(y, p > 0.5, average="macro"), 5),
                         st_log_loss=round(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), 5),
                         fit_s=round(fit_s, 2), infer_ms_per_1k_rows=round(inf / len(te) * 1e6, 4),
                         params=json.dumps(params)))
        if tag == "tuned":
            joblib.dump({"model": m, "features": FEATS, "params": params}, ROOT / "artifacts/final_hgb_tuned.joblib")
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    json.dump(dict(sampler="Optuna TPESampler(seed=42)", objective="mean macro-F1, GroupKFold(3) by trip, 600k-row train sample",
                   search_space=SPACE, n_trials=n_trials, search_seconds=round(search_s, 1),
                   best_value=study.best_value, best_params=best, untuned_params=DEFAULT,
                   student_test_used_in_search=False, evaluator_test_used=False),
              open(OUT / "optuna_summary.json", "w"), indent=2)
    print("best", study.best_value, best, "search_s", round(search_s, 1))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
