"""Pipeline step 03e: full metric set for all 5 models on the final 5-feature set. Owner: B

Covers the metrics in Models and Training Guidelines §5-§6 for every model:
accuracy, precision, recall, F1 (macro and positive class), confusion matrix, log loss,
observation-level top-1, route continuity (classifier-only), training and inference time.

Protocol: GroupKFold(5) by trip on the 77 training trips (out-of-fold predictions),
then one refit on all training trips and one score on the 10 student-test trips.
Random Forest, KNN and MLP are fit on fixed random subsamples (seed 42) for speed.
Also records the MLP's epoch-wise training and validation log loss.

Usage: python scripts/03e_full_metrics.py
Writes experiments/results/full_metrics/*.csv|json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data.loader import list_trips, load_trip  # noqa: E402
from pathsnap.hmm.viterbi import ArcGraph, route_continuity  # noqa: E402

FEATS = ["perp_rel", "proj_frac", "heading_diff", "speed", "n_cands"]
SEED = 42
OUT = ROOT / "experiments/results/full_metrics"

MODELS = {
    "Logistic Regression": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), None,
                            "linear / probabilistic", "L2, C=1.0, max_iter=500, standardised"),
    "KNN": (lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=2)), 60_000,
            "instance-based", "k=15, uniform weights, standardised, 60k-row fit sample"),
    "Random Forest": (lambda: RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=20,
                                                     n_jobs=2, random_state=SEED), 300_000,
                      "ensemble (bagging)", "100 trees, depth 12, min leaf 20, 300k-row fit sample"),
    "HistGradientBoosting": (lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                                    random_state=SEED), None,
                             "boosting", "200 iterations, lr 0.1, defaults otherwise"),
    "Shallow MLP": (lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20,
                                                                          early_stopping=True, random_state=SEED)),
                    150_000, "neural network", "layers (64, 32), ReLU, Adam, 20 epochs, 150k-row fit sample"),
}


def load(p):
    d = pd.read_parquet(ROOT / f"data/processed/feats_{p}.parquet")
    return d[~d.unlabelled_point].reset_index(drop=True)


def fit(name, X, y):
    make, cap, *_ = MODELS[name]
    if cap and len(X) > cap:
        i = np.random.default_rng(SEED).choice(len(X), cap, replace=False)
        X, y = X[i], y[i]
    m = make()
    t0 = time.time()
    m.fit(X, y)
    return m, time.time() - t0


def top1_preds(df, p):
    d = df[["trip_id", "point_idx", "arc_id", "label"]].assign(p=p)
    return d.loc[d.groupby(["trip_id", "point_idx"])["p"].idxmax()].sort_values(["trip_id", "point_idx"])


def metrics(df, p):
    y = df["label"].to_numpy()
    yh = (p > 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, yh),
                precision_macro=precision_score(y, yh, average="macro", zero_division=0),
                recall_macro=recall_score(y, yh, average="macro", zero_division=0),
                f1_macro=f1_score(y, yh, average="macro"),
                precision_pos=precision_score(y, yh, zero_division=0),
                recall_pos=recall_score(y, yh, zero_division=0),
                f1_pos=f1_score(y, yh),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
                top1=top1_preds(df, p)["label"].mean())


def continuity(df, p, graphs, times):
    best = top1_preds(df, p)
    vals = [route_continuity(g.arc_id.tolist(), times[t][g.point_idx.to_numpy()], graphs[t], 300.0)
            for t, g in best.groupby("trip_id")]
    return float(np.mean(vals))


def mlp_curve(tr):
    """Epoch-wise train/validation log loss for the MLP (grouped holdout of 20% of trips)."""
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(tr, groups=tr.trip_id))
    rng = np.random.default_rng(SEED)
    a = rng.choice(a, min(150_000, len(a)), replace=False)
    b = rng.choice(b, min(60_000, len(b)), replace=False)
    sc = StandardScaler().fit(tr.iloc[a][FEATS])
    Xa, ya = sc.transform(tr.iloc[a][FEATS]), tr.iloc[a].label.to_numpy()
    Xb, yb = sc.transform(tr.iloc[b][FEATS]), tr.iloc[b].label.to_numpy()
    m = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=SEED)
    rows = []
    for ep in range(1, 31):
        m.partial_fit(Xa, ya, classes=[0, 1])
        rows.append(dict(epoch=ep, train_log_loss=log_loss(ya, m.predict_proba(Xa)[:, 1]),
                         val_log_loss=log_loss(yb, m.predict_proba(Xb)[:, 1])))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = load("train"), load("student_test")
    Xtr, ytr = tr[FEATS].to_numpy(float), tr.label.to_numpy()
    Xte = te[FEATS].to_numpy(float)
    graphs, times = {}, {}
    for d in list_trips(ROOT / "data/raw/student_test"):
        t = load_trip(d)
        graphs[d.name] = ArcGraph(t.arcs.from_node, t.arcs.to_node, 6)
        times[d.name] = t.track.t.to_numpy(float)

    rows, cms, folds = [], {}, []
    for name, (_, cap, fam, cfg) in MODELS.items():
        oof = np.zeros(len(tr))
        tts = []
        for k, (a, b) in enumerate(GroupKFold(5).split(Xtr, ytr, groups=tr.trip_id)):
            m, tt = fit(name, Xtr[a], ytr[a])
            oof[b] = m.predict_proba(Xtr[b])[:, 1]
            tts.append(tt)
            fm = metrics(tr.iloc[b], oof[b])
            folds.append(dict(model=name, fold=k, **fm))
        cv = metrics(tr, oof)
        m, tt = fit(name, Xtr, ytr)
        t0 = time.time()
        p = m.predict_proba(Xte)[:, 1]
        inf = time.time() - t0
        st = metrics(te, p)
        fd = pd.DataFrame([f for f in folds if f["model"] == name])
        row = dict(model=name, family=fam, config=cfg,
                   **{f"cv_{k}": v for k, v in cv.items()},
                   cv_f1_macro_std=fd.f1_macro.std(), cv_top1_std=fd.top1.std(),
                   **{f"st_{k}": v for k, v in st.items()},
                   st_route_continuity=continuity(te, p, graphs, times),
                   train_s=tt, cv_train_s_mean=float(np.mean(tts)),
                   infer_s_student_test=inf, infer_ms_per_1k_rows=inf / len(te) * 1e6)
        rows.append(row)
        cms[name] = confusion_matrix(te.label, (p > 0.5).astype(int)).tolist()
        print(name, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "model_metrics.csv", index=False)
        pd.DataFrame(folds).to_csv(OUT / "cv_folds.csv", index=False)
        json.dump(cms, open(OUT / "confusion_matrices_student_test.json", "w"), indent=1)
    mlp_curve(tr).to_csv(OUT / "mlp_loss_curve.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
