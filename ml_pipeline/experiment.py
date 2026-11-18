"""Shared experiment helpers: group-aware CV + student-test evaluation."""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config
from .features import FEATURES
from .metrics import observation_metrics, row_metrics
from .models import MODEL_SPECS, build_model

N_FOLDS = 5


def load_partition(name):
    return pd.read_parquet(config.CACHE_DIR / f"{name}.parquet")


def train_rows(df):
    """Rows usable for fitting: reliable labels only."""
    return df[df["label_dist"] <= config.LABEL_MAX_DIST_M]


def subsample(df, max_rows, seed=config.SEED):
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def fit_predict(name, params, fit_df, eval_dfs, max_rows):
    model = build_model(name, params)
    fd = subsample(fit_df, max_rows)
    t0 = time.time()
    model.fit(fd[FEATURES].to_numpy(np.float32), fd["y"].to_numpy())
    fit_s = time.time() - t0
    preds = []
    for ed in eval_dfs:
        t0 = time.time()
        p = model.predict_proba(ed[FEATURES].to_numpy(np.float32))[:, 1]
        preds.append((p, time.time() - t0))
    return model, fit_s, preds


def evaluate(df, p):
    d = df.assign(p=p)
    m = row_metrics(d["y"].to_numpy(), p)
    m.update(observation_metrics(d))
    return m


def jsonable(params):
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in params.items()}


def run_model(name, train, test, params=None, n_folds=N_FOLDS, tag=None):
    spec = MODEL_SPECS[name]
    params = spec["params"] if params is None else params
    folds = []
    for k, (tr_i, va_i) in enumerate(GroupKFold(n_splits=n_folds).split(train, groups=train["trip"])):
        tr, va = train.iloc[tr_i], train.iloc[va_i]
        _, fit_s, [(p, inf_s)] = fit_predict(name, params, train_rows(tr), [va], spec["max_rows"])
        m = evaluate(va, p)
        m.update(fold=k, fit_seconds=fit_s, infer_seconds=inf_s, n_val_rows=len(va))
        folds.append(m)
        print(f"  [{tag or name}] fold {k}: f1_macro={m['f1_macro']:.4f} top1={m['top1_accuracy']:.4f} "
              f"logloss={m['log_loss']:.4f} fit={fit_s:.1f}s", flush=True)

    model, fit_s, [(p, inf_s)] = fit_predict(name, params, train_rows(train), [test], spec["max_rows"])
    tm = evaluate(test, p)
    history = {}
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "loss_curve_"):
        history["train_loss"] = list(map(float, est.loss_curve_))
        history["val_score"] = list(map(float, getattr(est, "validation_scores_", []) or []))
    result = {
        "model": tag or name,
        "base_model": name,
        "family": spec["family"],
        "params": jsonable(params),
        "max_train_rows": spec["max_rows"],
        "cv": folds,
        "cv_f1_macro_mean": float(np.mean([f["f1_macro"] for f in folds])),
        "cv_f1_macro_std": float(np.std([f["f1_macro"] for f in folds])),
        "cv_top1_mean": float(np.mean([f["top1_accuracy"] for f in folds])),
        "cv_top1_std": float(np.std([f["top1_accuracy"] for f in folds])),
        "cv_log_loss_mean": float(np.mean([f["log_loss"] for f in folds])),
        "cv_continuity_mean": float(np.mean([f["route_continuity"] for f in folds])),
        "test": tm,
        "fit_seconds_full": fit_s,
        "test_infer_seconds": inf_s,
        "test_infer_ms_per_1k_rows": 1000 * inf_s / len(test) * 1000,
        "history": history,
    }
    return result, model
