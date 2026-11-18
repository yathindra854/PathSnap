"""Pipeline step 07: report figures for model comparison, tuning and decoding.

Every figure is drawn from result files written by steps 03e, 04 and 05; nothing is
typed in by hand. Training curves are only drawn for the MLP, the one model with
epoch-wise training history.

Usage: python scripts/07_model_plots.py   -> reports/figures/models/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "experiments/results"
OUT = ROOT / "reports/figures/models"
CHOSEN = "HistGradientBoosting"
GREY, BLUE, DARK = "#9aa0a6", "#1a73e8", "#5f6368"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    print("saved", name)


def bars(ax, x, vals, w, label, color, fmt="{:.3f}"):
    b = ax.bar(x, vals, w, label=label, color=color)
    for xx, v in zip(x, vals):
        ax.text(xx, v, fmt.format(v), ha="center", va="bottom", fontsize=7)
    return b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(R / "full_metrics/model_metrics.csv")
    folds = pd.read_csv(R / "full_metrics/cv_folds.csv")
    names = m.model.tolist()
    short = [n.replace(" ", "\n") for n in names]
    cols = [BLUE if n == CHOSEN else GREY for n in names]
    x = np.arange(len(names))

    # 1. CV macro-F1 with fold std
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, m.cv_f1_macro, yerr=m.cv_f1_macro_std, capsize=5, color=cols)
    for i, v in enumerate(m.cv_f1_macro):
        ax.text(i, v + m.cv_f1_macro_std[i] + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, short); ax.set_ylim(0.78, 0.93)
    ax.set_ylabel("Macro-F1 (out-of-fold, ± fold std)")
    ax.set_title("Cross-validation macro-F1, GroupKFold(5) on 77 training trips (selected model in blue)")
    save(fig, "01_cv_macro_f1.png")

    # 2. Fold stability
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [folds[folds.model == n].f1_macro.to_numpy() for n in names]
    ax.boxplot(data, tick_labels=short, showmeans=True)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1), d, s=12, color=cols[i], zorder=3)
    ax.set_ylabel("Macro-F1 per fold"); ax.set_title("Stability across the 5 validation folds")
    save(fig, "02_cv_fold_stability.png")

    # 3. Student-test top-1 and macro-F1
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars(ax, x - w / 2, m.st_top1, w, "Top-1 accuracy (per GPS point)", DARK)
    bars(ax, x + w / 2, m.st_f1_macro, w, "Macro-F1 (per candidate)", BLUE)
    ax.set_xticks(x, short); ax.set_ylim(0.7, 0.95); ax.legend(loc="upper left")
    ax.set_title("Student-test (10 trips): top-1 accuracy and macro-F1")
    save(fig, "03_student_top1_macro_f1.png")

    # 4. Log loss CV vs student
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars(ax, x - w / 2, m.cv_log_loss, w, "CV (training trips)", DARK)
    bars(ax, x + w / 2, m.st_log_loss, w, "Student-test", BLUE)
    ax.set_xticks(x, short); ax.set_ylabel("Log loss (lower is better)"); ax.legend()
    ax.set_title("Probability quality: log loss")
    save(fig, "04_log_loss.png")

    # 5. Precision / recall / F1 (student-test, macro)
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars(ax, x - w, m.st_precision_macro, w, "Precision (macro)", "#8ab4f8")
    bars(ax, x, m.st_recall_macro, w, "Recall (macro)", DARK)
    bars(ax, x + w, m.st_f1_macro, w, "F1 (macro)", BLUE)
    ax.set_xticks(x, short); ax.set_ylim(0.75, 0.95); ax.legend(loc="upper left", ncol=3)
    ax.set_title("Student-test precision, recall and F1 (threshold 0.5)")
    save(fig, "05_precision_recall_f1.png")

    # 6. Confusion matrices
    cms = json.load(open(R / "full_metrics/confusion_matrices_student_test.json"))
    fig, axes = plt.subplots(1, len(names), figsize=(16, 3.6))
    for ax, n in zip(axes, names):
        c = np.array(cms[n])
        cn = c / c.sum(axis=1, keepdims=True)
        ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{c[i, j]:,}\n({cn[i, j]:.1%})", ha="center", va="center",
                        color="white" if cn[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks([0, 1], ["other arc", "true arc"]); ax.set_yticks([0, 1], ["other arc", "true arc"])
        ax.set_xlabel("predicted"); ax.set_title(n, fontsize=9, color=BLUE if n == CHOSEN else "black")
    axes[0].set_ylabel("actual")
    fig.suptitle("Confusion matrices, student-test candidates (row-normalised %)")
    save(fig, "06_confusion_matrices.png")

    # 7. MLP training curve (only model with epoch history)
    c = pd.read_csv(R / "full_metrics/mlp_loss_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(c.epoch, c.train_log_loss, marker="o", ms=3, label="training")
    ax.plot(c.epoch, c.val_log_loss, marker="o", ms=3, label="validation (held-out trips)")
    ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax.legend()
    ax.set_title("Shallow MLP: training vs validation loss per epoch")
    save(fig, "07_mlp_loss_curve.png")

    # 8. Route continuity per model (classifier only) + HMM
    h = pd.read_csv(R / "hmm_student_test.csv")
    best = h[h.method == "hmm_grid"].sort_values("top1").iloc[-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars(ax, x, m.st_route_continuity, 0.6, "classifier only", cols)
    ax.axhline(best.continuity, color="#d93025", ls="--", label=f"HGB + HMM ({best.continuity:.3f})")
    ax.set_xticks(x, short); ax.set_ylim(0.85, 1.0); ax.set_ylabel("Route continuity")
    ax.legend(loc="lower right"); ax.set_title("Route continuity on student-test: classifier-only vs HMM")
    save(fig, "08_route_continuity.png")

    # 9. Compute cost
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(x, m.train_s, color=cols); ax[0].set_xticks(x, short); ax[0].set_ylabel("seconds")
    ax[0].set_title("Training time (full training set or sample)")
    ax[1].bar(x, m.infer_ms_per_1k_rows, color=cols); ax[1].set_xticks(x, short)
    ax[1].set_ylabel("ms per 1,000 candidates"); ax[1].set_title("Inference time (student-test)")
    save(fig, "09_compute_cost.png")

    # 10-11. Bayesian optimisation
    bp = R / "bayes"
    if (bp / "optuna_trials.csv").exists():
        t = pd.read_csv(bp / "optuna_trials.csv")
        t = t[t.state == "COMPLETE"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.scatter(t.number, t.value, color=DARK, s=20, label="trial")
        ax.plot(t.number, t.value.cummax(), color=BLUE, label="best so far")
        ax.scatter([0], [t.value.iloc[0]], color="#d93025", s=40, zorder=3, label="trial 0 = untuned default")
        ax.set_xlabel("trial"); ax.set_ylabel("CV macro-F1 (GroupKFold 3, train sample)"); ax.legend()
        ax.set_title("Optuna TPE search history (HistGradientBoosting)")
        save(fig, "10_bayes_history.png")
    if (bp / "tuned_vs_untuned.csv").exists():
        tv = pd.read_csv(bp / "tuned_vs_untuned.csv").set_index("model")
        keys = [("cv_f1_macro", "CV macro-F1"), ("st_f1_macro", "Student macro-F1"), ("st_top1", "Student top-1")]
        xx = np.arange(len(keys)); w = 0.38
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
        bars(ax[0], xx - w / 2, [tv.loc["untuned", k] for k, _ in keys], w, "untuned", GREY, "{:.4f}")
        bars(ax[0], xx + w / 2, [tv.loc["tuned", k] for k, _ in keys], w, "tuned", BLUE, "{:.4f}")
        ax[0].set_xticks(xx, [l for _, l in keys]); ax[0].set_ylim(0.75, 0.92); ax[0].legend()
        ax[0].set_title("Tuned vs untuned (same data partitions)")
        k2 = [("cv_log_loss", "CV log loss"), ("st_log_loss", "Student log loss")]
        x2 = np.arange(2)
        bars(ax[1], x2 - w / 2, [tv.loc["untuned", k] for k, _ in k2], w, "untuned", GREY, "{:.4f}")
        bars(ax[1], x2 + w / 2, [tv.loc["tuned", k] for k, _ in k2], w, "tuned", BLUE, "{:.4f}")
        ax[1].set_xticks(x2, [l for _, l in k2]); ax[1].set_title("Log loss (lower is better)")
        save(fig, "11_tuned_vs_untuned.png")


if __name__ == "__main__":
    main()
