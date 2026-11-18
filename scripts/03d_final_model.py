"""Pipeline step 03d: fit the selected model on all training trips and score student-test.

Selected model: HistGradientBoosting, reduced feature set (decisions log D-009).
Usage: python scripts/03d_final_model.py
Writes experiments/results/final_model_metrics.csv and models/final_hgb.joblib
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["perp_rel", "proj_frac", "heading_diff", "speed", "n_cands"]


def load(p):
    d = pd.read_parquet(ROOT / f"data/processed/feats_{p}.parquet")
    return d[~d.unlabelled_point].reset_index(drop=True)


def top1(d, p):
    x = d[["trip_id", "point_idx", "label"]].assign(p=p)
    return x.loc[x.groupby(["trip_id", "point_idx"])["p"].idxmax()]["label"].mean()


tr, te = load("train"), load("student_test")
t0 = time.time()
m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42).fit(
    tr[FEATS].to_numpy(float), tr["label"].to_numpy())
train_s = time.time() - t0
t0 = time.time()
p = m.predict_proba(te[FEATS].to_numpy(float))[:, 1]
infer_s = (time.time() - t0) / len(te) * 10_000
y = te["label"].to_numpy()
row = dict(model="hgb_reduced", split="student_test", macro_f1=round(f1_score(y, p > 0.5, average="macro"), 4),
           log_loss=round(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), 4), top1=round(top1(te, p), 4),
           train_s=round(train_s, 2), infer_s_per_10k=round(infer_s, 4), n_features=len(FEATS))
print(row)
out = ROOT / "experiments/results"; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row]).to_csv(out / "final_model_metrics.csv", index=False)
Path(ROOT / "artifacts").mkdir(exist_ok=True)
joblib.dump({"model": m, "features": FEATS}, ROOT / "artifacts/final_hgb.joblib")
print("saved model and metrics")
