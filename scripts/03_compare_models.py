"""Train and compare candidate classifiers.

For every model:
  * 5-fold GroupKFold cross-validation on the 80 training trips (groups = trip,
    so no trip is split across folds),
  * refit on all training trips, evaluate on the 10 student-testing trips,
  * record configuration, timings and metrics.

Usage:  python scripts/03_compare_models.py [model ...] [--force]
Results: results/models/runs/<model>.json, results/models/comparison.csv, plots.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.experiment import load_partition, run_model  # noqa: E402
from pathsnap.models import MODEL_SPECS  # noqa: E402

OUT = config.RESULTS_DIR / "models"
RUNS = OUT / "runs"


def summarise():
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("base_model", r["model"]) != r["model"]:
            continue  # tuned variants are reported by 04_bayes_opt.py
        rows.append({
            "model": r["model"], "family": r["family"],
            "key_config": ", ".join(f"{k}={v}" for k, v in r["params"].items()
                                    if k not in ("n_jobs", "random_state", "verbose", "eval_metric")),
            "cv_f1_macro": r["cv_f1_macro_mean"], "cv_f1_macro_std": r["cv_f1_macro_std"],
            "cv_top1": r["cv_top1_mean"], "cv_log_loss": r["cv_log_loss_mean"],
            "test_top1": r["test"]["top1_accuracy"], "test_f1_macro": r["test"]["f1_macro"],
            "test_precision_macro": r["test"]["precision_macro"], "test_recall_macro": r["test"]["recall_macro"],
            "test_log_loss": r["test"]["log_loss"], "test_route_continuity": r["test"]["route_continuity"],
            "test_candidate_miss_rate": r["test"]["candidate_miss_rate"],
            "fit_seconds": r["fit_seconds_full"], "infer_ms_per_1k_rows": r["test_infer_ms_per_1k_rows"],
        })
    df = pd.DataFrame(rows).sort_values("cv_top1", ascending=False)
    df.to_csv(OUT / "comparison.csv", index=False)
    print(df.drop(columns=["key_config"]).to_string(index=False))
    plot(df)
    return df


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = OUT / "plots"
    figs.mkdir(parents=True, exist_ok=True)

    def bar(cols, labels, title, fname, err=None):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(len(df))
        w = 0.8 / len(cols)
        for i, (c, lab) in enumerate(zip(cols, labels)):
            ax.bar(x + i * w - 0.4 + w / 2, df[c], w, label=lab,
                   yerr=df[err] if (err and i == 0) else None, capsize=3)
        ax.set_xticks(x, df["model"], rotation=30, ha="right")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / fname, dpi=130)
        plt.close(fig)

    bar(["cv_f1_macro"], ["CV macro F1 (mean ± std over 5 folds)"], "Cross-validation macro F1 (80 training trips)",
        "cv_f1_macro.png", err="cv_f1_macro_std")
    bar(["test_top1", "test_f1_macro"], ["Top-1 accuracy", "Macro F1"], "Student-testing trips: Top-1 accuracy and macro F1",
        "student_test_top1_f1.png")
    bar(["cv_log_loss", "test_log_loss"], ["CV log loss", "Student-test log loss"], "Log loss (lower is better)",
        "log_loss.png")
    bar(["test_route_continuity"], ["Route continuity"], "Student-testing route continuity (classifier only)",
        "route_continuity.png")

    # Confusion matrices and training curves
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text())
        cm = np.array(r["test"]["confusion"])
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(r["model"], fontsize=9)
        fig.tight_layout()
        fig.savefig(figs / f"confusion_{r['model']}.png", dpi=130)
        plt.close(fig)
        h = r.get("history") or {}
        if h.get("train_loss"):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(h["train_loss"], label="training loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("log loss")
            if h.get("val_score"):
                ax2 = ax.twinx()
                ax2.plot(h["val_score"], color="tab:orange", label="validation accuracy")
                ax2.set_ylabel("validation accuracy")
                ax2.legend(loc="center right")
            ax.legend(loc="upper right")
            ax.set_title(f"{r['model']} training history")
            fig.tight_layout()
            fig.savefig(figs / f"history_{r['model']}.png", dpi=130)
            plt.close(fig)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    names = args or list(MODEL_SPECS)
    RUNS.mkdir(parents=True, exist_ok=True)
    train = load_partition("train")
    test = load_partition("student_test")
    print(f"train rows={len(train):,} trips={train['trip'].nunique()}  test rows={len(test):,}")
    for name in names:
        path = RUNS / f"{name}.json"
        if path.exists() and not force:
            print(f"skip {name} (exists)")
            continue
        t0 = time.time()
        r, _ = run_model(name, train, test)
        path.write_text(json.dumps(r, indent=1))
        print(f"== {name}: cv_top1={r['cv_top1_mean']:.4f} test_top1={r['test']['top1_accuracy']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    summarise()


if __name__ == "__main__":
    main()
