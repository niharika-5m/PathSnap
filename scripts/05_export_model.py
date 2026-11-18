"""Fit the final classifier on all 80 training trips and export inference artifacts.

Picks the tuned configuration from results/bayes_opt/best_<model>.json when it
exists (otherwise the untuned params of the best model in the comparison).
Writes artifacts/model.joblib and artifacts/pipeline_config.json.
Usage: python scripts/05_export_model.py [model]
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.experiment import jsonable, load_partition, subsample, train_rows  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.models import MODEL_SPECS, build_model, search_space  # noqa: E402


def final_params(name):
    best = config.RESULTS_DIR / "bayes_opt" / f"best_{name}.json"
    if best.exists():
        params = search_space(name, optuna.trial.FixedTrial(json.loads(best.read_text())["best_params"]))
        return params, "bayesian-optimised"
    return MODEL_SPECS[name]["params"], "untuned"


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        tuned = sorted((config.RESULTS_DIR / "bayes_opt").glob("best_*.json"))
        if tuned:
            name = json.loads(tuned[0].read_text())["model"]
        else:
            comp = pd.read_csv(config.RESULTS_DIR / "models" / "comparison.csv")
            name = comp.sort_values("cv_top1", ascending=False).iloc[0]["model"]
    params, source = final_params(name)
    train = train_rows(load_partition("train"))
    fd = subsample(train, MODEL_SPECS[name]["max_rows"])
    model = build_model(name, params)
    t0 = time.time()
    model.fit(fd[FEATURES].to_numpy(np.float32), fd["y"].to_numpy())
    print(f"fitted {name} ({source}) on {len(fd):,} rows in {time.time() - t0:.1f}s")

    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.ARTIFACT_DIR / "model.joblib", compress=3)
    cfg_path = config.ARTIFACT_DIR / "pipeline_config.json"
    old = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg = {
        "model_name": name,
        "model_params": jsonable(params),
        "params_source": source,
        "trained_on": "train partition (80 trips), reliable labels only",
        "features": FEATURES,
        "candidates": {"radius": config.CANDIDATE_RADIUS_M, "max_k": config.MAX_CANDIDATES,
                       "min_k": config.MIN_CANDIDATES},
        "hmm": old.get("hmm", dict(config.HMM_DEFAULTS, top_m=8)),
        "label_max_dist_m": config.LABEL_MAX_DIST_M,
    }
    cfg_path.write_text(json.dumps(cfg, indent=1))
    size = (config.ARTIFACT_DIR / "model.joblib").stat().st_size / 1e6
    print(f"wrote artifacts/model.joblib ({size:.1f} MB) and pipeline_config.json")


if __name__ == "__main__":
    main()
