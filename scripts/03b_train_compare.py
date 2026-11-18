"""Pipeline step 03b: train and compare candidate classifiers. Owner: B

Five model families (Models and Training Guidelines §3):
  lr   Logistic Regression            linear / probabilistic
  knn  K-Nearest Neighbours (k=15)     instance-based
  rf   Random Forest (100 trees)       ensemble / tree-based
  hgb  HistGradientBoosting            boosting
  mlp  Shallow MLP (64, 32)            neural network

Protocol: GroupKFold(5) by trip on the training trips (CV). Student-test is
scored once per model, after the CV, with --final. Unlabelled points excluded.
Large models are fit on a fixed random subsample (seed 42) for speed; that cap
is recorded in the results file.

Usage: python scripts/03b_train_compare.py <model> [--final]
Appends to experiments/results/model_comparison.csv
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data.loader import load_trip, project_trip, list_trips  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402
FE = import_module("03a_features")
FEATURES, SEED = FE.FEATURES, 42

MODELS = {
    "lr":  (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), None),
    "knn": (lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=2)), 60_000),
    "rf":  (lambda: RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=20,
                                           n_jobs=2, random_state=SEED), 300_000),
    "hgb": (lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED), None),
    "mlp": (lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20,
                                                                  early_stopping=True, random_state=SEED)), 150_000),
}


def load(partition):
    df = pd.read_parquet(ROOT / f"data/processed/feats_{partition}.parquet")
    return df[~df.unlabelled_point].reset_index(drop=True)


def top1(df, p):
    d = df[["trip_id", "point_idx", "label"]].copy()
    d["p"] = p
    best = d.loc[d.groupby(["trip_id", "point_idx"])["p"].idxmax()]
    return best["label"].mean()


def metrics(df, p):
    y = df["label"].to_numpy()
    return dict(macro_f1=f1_score(y, p > 0.5, average="macro"),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
                top1=top1(df, p))


def fit(name, X, y, cap):
    make, _ = MODELS[name]
    if cap and len(X) > cap:
        idx = np.random.default_rng(SEED).choice(len(X), cap, replace=False)
        X, y = X[idx], y[idx]
    m = make()
    t0 = time.time()
    m.fit(X, y)
    return m, time.time() - t0


def cv(name, df):
    _, cap = MODELS[name]
    X = df[FEATURES].to_numpy(float)
    y = df["label"].to_numpy()
    rows = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=df["trip_id"])):
        m, tt = fit(name, X[tr], y[tr], cap)
        t0 = time.time()
        p = m.predict_proba(X[te])[:, 1]
        inf = (time.time() - t0) / len(te) * 10_000
        r = metrics(df.iloc[te], p)
        r.update(fold=k, train_s=tt, infer_s_per_10k=inf)
        rows.append(r)
        print(name, "fold", k, {a: round(b, 4) for a, b in r.items()}, flush=True)
    return pd.DataFrame(rows)


def final(name, tr_df):
    te_df = load("student_test")
    _, cap = MODELS[name]
    m, tt = fit(name, tr_df[FEATURES].to_numpy(float), tr_df["label"].to_numpy(), cap)
    t0 = time.time()
    p = m.predict_proba(te_df[FEATURES].to_numpy(float))[:, 1]
    inf = (time.time() - t0) / len(te_df) * 10_000
    r = metrics(te_df, p)
    r.update(train_s=tt, infer_s_per_10k=inf)
    return r, te_df, p


def save(out, rows):
    """Replace this model's rows in the shared results file (safe to re-run)."""
    path = out / "model_comparison.csv"
    res = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keys = {(r["model"], r["scope"]) for r in rows}
        old = old[[(m, sc) not in keys for m, sc in zip(old["model"], old["scope"])]]
        res = pd.concat([old, res], ignore_index=True)
    res.to_csv(path, index=False)
    print("saved", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--skip-cv", action="store_true", help="only the student-test row (CV already saved)")
    a = ap.parse_args()

    tr = load("train")
    out = ROOT / "experiments/results"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if not a.skip_cv:
        cvr = cv(a.model, tr)
        summary = dict(model=a.model, scope="cv_mean", **cvr.drop(columns="fold").mean().round(5).to_dict(),
                       cv_macro_f1_std=round(cvr.macro_f1.std(), 5), n_train_rows=len(tr))
        print("CV", summary, flush=True)
        rows = [summary]
        save(out, rows)
    if a.final:
        r, te_df, p = final(a.model, tr)
        rows.append(dict(model=a.model, scope="student_test", **{k: round(v, 5) for k, v in r.items()},
                         n_train_rows=len(tr)))
        print("STUDENT", rows[-1], flush=True)
        te_df.assign(p=p)[["trip_id", "point_idx", "arc_id", "label", "p"]].to_parquet(
            ROOT / f"experiments/results/student_test_scores_{a.model}.parquet", index=False)
    save(out, rows)


if __name__ == "__main__":
    main()
