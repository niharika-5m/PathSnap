"""Over/underfitting evidence: in-sample (training-trip) scores of every model,
next to its group-CV and student-test scores from 03_compare_models.py.

A large gap between in-sample and CV scores means the model memorises training
trips (overfitting); low scores everywhere means it underfits.
Writes results/models/train_vs_cv.csv.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.experiment import evaluate, fit_predict, load_partition, train_rows  # noqa: E402
from pathsnap.models import MODEL_SPECS  # noqa: E402

RUNS = config.RESULTS_DIR / "models" / "runs"


def plot():
    """Train vs CV vs student-test Top-1 per model, and all confusion matrices in one grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out = config.RESULTS_DIR / "models" / "plots"
    df = pd.read_csv(config.RESULTS_DIR / "models" / "train_vs_cv.csv").sort_values("cv_top1", ascending=False)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, (col, lab) in enumerate([("train_top1", "training trips (in-sample)"), ("cv_top1", "5-fold group CV"),
                                    ("test_top1", "student-test trips")]):
        ax.bar(x + (k - 1) * 0.27, df[col], 0.27, label=lab)
    ax.set_xticks(x, df["model"], rotation=30, ha="right")
    ax.set_ylim(0.8, 0.95)
    ax.set_ylabel("observation Top-1 accuracy")
    ax.set_title("Over/underfitting check: in-sample vs cross-validated vs student-test Top-1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "train_vs_cv_top1.png", dpi=130)
    plt.close(fig)

    names = list(df["model"])
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.6))
    for ax, name in zip(axes.ravel(), names):
        cm = np.array(json.loads((RUNS / f"{name}.json").read_text())["test"]["confusion"])
        rates = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}\n({rates[i, j]:.1%})", ha="center", va="center", fontsize=8,
                    color="white" if rates[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1], ["pred wrong", "pred right"], fontsize=8)
        ax.set_yticks([0, 1], ["true wrong", "true right"], fontsize=8)
        ax.set_title(name, fontsize=10)
    fig.suptitle("Confusion matrices on student-test candidate rows (threshold 0.5; row-normalised share in brackets)")
    fig.tight_layout()
    fig.savefig(out / "confusion_all_models.png", dpi=130)
    plt.close(fig)


def main():
    if "--plot-only" in sys.argv:
        plot()
        return
    train = load_partition("train")
    fit_df = train_rows(train)
    rows = []
    for name in sys.argv[1:] or list(MODEL_SPECS):
        spec = MODEL_SPECS[name]
        _, fit_s, [(p, _)] = fit_predict(name, spec["params"], fit_df, [train], spec["max_rows"])
        m = evaluate(train, p)
        r = json.loads((RUNS / f"{name}.json").read_text())
        rows.append({"model": name,
                     "train_top1": m["top1_accuracy"], "cv_top1": r["cv_top1_mean"], "test_top1": r["test"]["top1_accuracy"],
                     "train_f1_macro": m["f1_macro"], "cv_f1_macro": r["cv_f1_macro_mean"], "test_f1_macro": r["test"]["f1_macro"],
                     "train_log_loss": m["log_loss"], "cv_log_loss": r["cv_log_loss_mean"], "test_log_loss": r["test"]["log_loss"],
                     "cv_top1_fold_std": r.get("cv_top1_std", pd.Series([f["top1_accuracy"] for f in r["cv"]]).std(ddof=0)),
                     "train_minus_cv_top1": m["top1_accuracy"] - r["cv_top1_mean"]})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "models" / "train_vs_cv.csv", index=False)
    print(df.round(4).to_string(index=False))
    plot()


if __name__ == "__main__":
    main()
