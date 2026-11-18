"""Evaluation metrics: candidate-level classification and observation/route-level."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def row_metrics(y, p, threshold=0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y, yhat, average="macro"),
        "f1_positive": f1_score(y, yhat, pos_label=1),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def pick_top1(df: pd.DataFrame, score_col="p") -> pd.DataFrame:
    """Highest-scoring candidate per (trip, point)."""
    idx = df.groupby(["trip", "pidx"], sort=True)[score_col].idxmax()
    return df.loc[idx]


def continuity(arc, src, dst) -> float:
    """Share of consecutive predictions that stay on the same arc or move to a
    directly connected arc (to-node of previous == from-node of next)."""
    if len(arc) < 2:
        return 1.0
    ok = (arc[1:] == arc[:-1]) | (src[1:] == dst[:-1])
    return float(ok.mean())


def observation_metrics(df: pd.DataFrame, score_col="p") -> dict:
    """Observation-level Top-1 accuracy, candidate miss rate and route continuity.
    Only points whose reference label is reliable are scored."""
    top = pick_top1(df, score_col)
    has_pos = df.groupby(["trip", "pidx"])["y"].max()
    reliable = df.groupby(["trip", "pidx"])["label_dist"].first() <= 100.0
    top = top.set_index(["trip", "pidx"])
    correct = top["y"].astype(bool) & reliable.reindex(top.index).to_numpy()
    n_rel = int(reliable.sum())
    cont = []
    per_trip = []
    for trip, g in top.groupby(level=0):
        cont.append(continuity(g["arc"].to_numpy(), g["arc_src"].to_numpy(), g["arc_dst"].to_numpy()))
        r = reliable.loc[trip]
        per_trip.append(float(correct.loc[trip][r.to_numpy()].mean()) if r.any() else np.nan)
    return {
        "top1_accuracy": float(correct.sum() / max(n_rel, 1)),
        "candidate_miss_rate": float(1 - has_pos[reliable].mean()),
        "route_continuity": float(np.mean(cont)),
        "trip_top1_mean": float(np.nanmean(per_trip)),
        "trip_top1_min": float(np.nanmin(per_trip)),
    }
