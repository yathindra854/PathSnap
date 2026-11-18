"""Pipeline step 03c: feature selection by leave-one-feature-out ablation. Owner: B

Uses the boosting model (hgb) with the same GroupKFold(5) protocol on the
training trips. A feature whose removal lowers CV macro-F1 and top-1 is kept.
Usage: python scripts/03c_feature_ablation.py <feature | baseline>
Appends to experiments/results/feature_ablation.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402
FE = import_module("03a_features")
ALL = FE.FEATURES

tr = pd.read_parquet(ROOT / "data/processed/feats_train.parquet")
tr = tr[~tr.unlabelled_point].reset_index(drop=True)
y = tr["label"].to_numpy()
arg = sys.argv[1]
REDUCED = ["perp_rel", "proj_frac", "heading_diff", "speed", "n_cands"]
feats = ALL if arg == "baseline" else REDUCED if arg == "reduced" else [f for f in ALL if f != arg]
X = tr[feats].to_numpy(float)
res = []
for k, (a, b) in enumerate(GroupKFold(5).split(X, y, groups=tr["trip_id"])):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42).fit(X[a], y[a])
    p = m.predict_proba(X[b])[:, 1]
    d = tr.iloc[b][["trip_id", "point_idx", "label"]].assign(p=p)
    best = d.loc[d.groupby(["trip_id", "point_idx"])["p"].idxmax()]
    res.append(dict(macro_f1=f1_score(y[b], p > 0.5, average="macro"),
                    log_loss=log_loss(y[b], np.clip(p, 1e-7, 1 - 1e-7)),
                    top1=best["label"].mean()))
r = pd.DataFrame(res).mean().round(5).to_dict()
row = dict(dropped=arg, n_features=len(feats), **r)
print(row, flush=True)
path = ROOT / "experiments/results/feature_ablation.csv"
path.parent.mkdir(parents=True, exist_ok=True)
pd.concat([pd.read_csv(path)[lambda d: d.dropped != arg] if path.exists() else pd.DataFrame(),
           pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
