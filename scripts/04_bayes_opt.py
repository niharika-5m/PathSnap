"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model.

Objective: mean observation-level Top-1 accuracy over group-aware CV folds of
the 80 training trips (folds split by trip). Top-1 accuracy is what the HMM
ultimately relies on; log loss is recorded for every trial as well, because the
HMM also uses the probabilities as emission scores.

The student-testing trips are NOT used inside the search. After the search, the
best configuration is re-evaluated with exactly the same 5-fold CV and
student-test protocol as the untuned model so the two can be compared.

Usage: python scripts/04_bayes_opt.py [model] [--trials N] [--folds K]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.experiment import evaluate, fit_predict, load_partition, run_model, train_rows  # noqa: E402
from pathsnap.models import MODEL_SPECS, search_space as space  # noqa: E402

OUT = config.RESULTS_DIR / "bayes_opt"


def main():
    argv = sys.argv[1:]
    get = lambda flag, d: type(d)(argv[argv.index(flag) + 1]) if flag in argv else d  # noqa: E731
    n_trials = get("--trials", 30)
    n_folds = get("--folds", 3)
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or not argv[i - 1].startswith("--"))]
    if pos:
        name = pos[0]
    else:
        comp = pd.read_csv(config.RESULTS_DIR / "models" / "comparison.csv")
        name = comp.sort_values("cv_top1", ascending=False).iloc[0]["model"]
    print(f"Optimising {name}: {n_trials} trials, {n_folds}-fold GroupKFold on training trips")

    OUT.mkdir(parents=True, exist_ok=True)
    train = load_partition("train")
    test = load_partition("student_test")
    splits = list(GroupKFold(n_splits=n_folds).split(train, groups=train["trip"]))
    max_rows = MODEL_SPECS[name]["max_rows"]

    def objective(trial):
        params = space(name, trial)
        scores, losses, fits = [], [], []
        for k, (tr_i, va_i) in enumerate(splits):
            va = train.iloc[va_i]
            _, fit_s, [(p, _)] = fit_predict(name, params, train_rows(train.iloc[tr_i]), [va], max_rows)
            m = evaluate(va, p)
            scores.append(m["top1_accuracy"])
            losses.append(m["log_loss"])
            fits.append(fit_s)
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_log_loss", float(np.mean(losses)))
        trial.set_user_attr("fit_seconds", float(np.sum(fits)))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=8)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
                                study_name=f"pathsnap_{name}")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_s = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(OUT / f"trials_{name}.csv", index=False)
    print(trials[["number", "value", "state"]].to_string(index=False))
    print("best", study.best_value, study.best_params)

    best = space(name, optuna.trial.FixedTrial(study.best_params))
    tuned, model = run_model(name, train, test, params=best, tag=f"{name}_tuned")
    untuned_path = config.RESULTS_DIR / "models" / "runs" / f"{name}.json"
    untuned = json.loads(untuned_path.read_text())
    (config.RESULTS_DIR / "models" / "runs" / f"{name}_tuned.json").write_text(json.dumps(tuned, indent=1))

    def row(r, label):
        return {"variant": label, "cv_f1_macro": r["cv_f1_macro_mean"], "cv_top1": r["cv_top1_mean"],
                "cv_log_loss": r["cv_log_loss_mean"], "test_top1": r["test"]["top1_accuracy"],
                "test_f1_macro": r["test"]["f1_macro"], "test_log_loss": r["test"]["log_loss"],
                "test_route_continuity": r["test"]["route_continuity"],
                "infer_ms_per_1k_rows": r["test_infer_ms_per_1k_rows"], "fit_seconds": r["fit_seconds_full"]}
    cmp = pd.DataFrame([row(untuned, "untuned"), row(tuned, "tuned")])
    cmp.to_csv(OUT / f"tuned_vs_untuned_{name}.csv", index=False)
    print(cmp.to_string(index=False))

    summary = {"model": name, "n_trials": n_trials, "n_folds": n_folds, "objective": "mean CV observation Top-1 accuracy",
               "best_value": study.best_value, "best_params": study.best_params,
               "best_full_params": {k: str(v) if isinstance(v, tuple) else v for k, v in best.items()},
               "search_seconds": search_s,
               "n_pruned": int((trials["state"] == "PRUNED").sum())}
    (OUT / f"best_{name}.json").write_text(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    done = trials[trials["state"] == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(done["number"], done["value"], s=18, label="trial")
    ax.plot(done["number"], done["value"].cummax(), color="tab:red", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV Top-1 accuracy")
    ax.set_title(f"Bayesian optimisation history ({name}, TPE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"history_{name}.png", dpi=130)
    plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1])
        ax.set_title("Hyperparameter importance (fANOVA)")
        fig.tight_layout()
        fig.savefig(OUT / f"importance_{name}.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # importance needs >1 completed trial
        print("importance plot skipped:", e)


if __name__ == "__main__":
    main()
