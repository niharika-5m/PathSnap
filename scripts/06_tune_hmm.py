"""Tune the HMM decoder and compare classifier-only vs classifier + HMM/Viterbi.

To avoid optimistic probabilities on trips the classifier was trained on, the
training-trip probabilities used for decoder tuning are out-of-fold (5-fold
GroupKFold, same final model configuration). Decoder parameters are chosen on
those training trips; the student-testing trips (scored with the exported model,
which never saw them) are reported for every configuration as a development check.

Writes results/hmm/*.csv, plots, and the chosen parameters into
artifacts/pipeline_config.json.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config, io  # noqa: E402
from pathsnap.experiment import subsample, train_rows  # noqa: E402
from pathsnap.features import FEATURES, attach_labels, prepare_trip  # noqa: E402
from pathsnap.hmm import build_route, classifier_only, point_continuity, route_metrics, viterbi  # noqa: E402
from pathsnap.models import MODEL_SPECS  # noqa: E402

OUT = config.RESULTS_DIR / "hmm"
N_TUNE_TRIPS = 40   # training trips used for the decoder grid (speed)


def contexts(partition, trips, cand):
    out = {}
    for tid in trips:
        ctx = prepare_trip(partition, tid, **cand)
        ref, ref_dist = attach_labels(ctx, partition)
        out[tid] = (ctx, ref, ref_dist <= config.LABEL_MAX_DIST_M, io.load_route(partition, tid))
    return out


def score(ctxs, params):
    rows = []
    for tid, (ctx, ref, ok, route) in ctxs.items():
        h = viterbi(ctx.net, ctx.xy, ctx.t, ctx.frame, params)
        c = classifier_only(ctx.frame, len(ctx.xy))
        r = {"trip": tid, "points": len(ref),
             "clf_acc": float((c == ref)[ok].mean()), "hmm_acc": float((h == ref)[ok].mean()),
             "clf_continuity": point_continuity(ctx.net, c), "hmm_continuity": point_continuity(ctx.net, h)}
        rc = route_metrics(ctx.net, build_route(ctx.net, c), route)
        rh = route_metrics(ctx.net, build_route(ctx.net, h), route)
        r.update({f"clf_{k}": v for k, v in rc.items()})
        r.update({f"hmm_{k}": v for k, v in rh.items()})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = config.ARTIFACT_DIR / "pipeline_config.json"
    cfg = json.loads(cfg_path.read_text())
    name, cand = cfg["model_name"], cfg["candidates"]
    max_rows = MODEL_SPECS[name]["max_rows"]
    final = joblib.load(config.ARTIFACT_DIR / "model.joblib")

    # ---- out-of-fold probabilities for training trips
    rng = np.random.default_rng(config.SEED)
    all_train = io.list_trips("train")
    tune_trips = sorted(rng.choice(all_train, size=min(N_TUNE_TRIPS, len(all_train)), replace=False))
    print(f"building contexts for {len(tune_trips)} training trips + student_test ...", flush=True)
    tr_ctx = contexts("train", tune_trips, cand)
    st_ctx = contexts("student_test", io.list_trips("student_test"), cand)

    train_df = pd.read_parquet(config.CACHE_DIR / "train.parquet")
    trips = np.array(all_train)
    for k, (fit_i, hold_i) in enumerate(GroupKFold(n_splits=5).split(trips, groups=trips)):
        hold = set(trips[hold_i]) & set(tune_trips)
        if not hold:
            continue
        fd = subsample(train_rows(train_df[train_df["trip"].isin(trips[fit_i])]), max_rows)
        m = clone(final).fit(fd[FEATURES].to_numpy(np.float32), fd["y"].to_numpy())
        for tid in hold:
            f = tr_ctx[tid][0].frame
            f["p"] = m.predict_proba(f[FEATURES].to_numpy(np.float32))[:, 1]
        print(f"  OOF fold {k} done", flush=True)

    for ctx, *_ in st_ctx.values():
        ctx.frame["p"] = final.predict_proba(ctx.frame[FEATURES].to_numpy(np.float32))[:, 1]

    # ---- grid search on training trips (student_test reported alongside)
    base = dict(config.HMM_DEFAULTS)
    grid = list(itertools.product([2, 3, 5, 10, 20, 40, 80], [0.25, 0.5, 1.0, 2.0], [4, 8]))
    rows = []
    for beta, power, top_m in grid:
        p = dict(base, beta=beta, emission_power=power, top_m=top_m)
        t0 = time.time()
        tr = score(tr_ctx, p)
        st = score(st_ctx, p)
        rows.append({"beta": beta, "emission_power": power, "top_m": top_m,
                     "train_oof_hmm_acc": tr["hmm_acc"].mean(), "train_oof_route_f1": tr["hmm_route_f1"].mean(),
                     "student_test_hmm_acc": st["hmm_acc"].mean(), "student_test_route_f1": st["hmm_route_f1"].mean(),
                     "seconds": time.time() - t0})
        print(rows[-1], flush=True)
    g = pd.DataFrame(rows).sort_values("train_oof_hmm_acc", ascending=False)
    g.to_csv(OUT / "grid_beta_power_topm.csv", index=False)
    best = g.iloc[0]
    chosen = dict(base, beta=float(best["beta"]), emission_power=float(best["emission_power"]), top_m=int(best["top_m"]))

    gap_rows = []
    for gap in [10, 30, 60, 180, 1e9]:
        p = dict(chosen, gap_seconds=gap)
        tr = score(tr_ctx, p)
        gap_rows.append({"gap_seconds": gap, "train_oof_hmm_acc": tr["hmm_acc"].mean(),
                         "train_oof_route_f1": tr["hmm_route_f1"].mean()})
    gg = pd.DataFrame(gap_rows)
    gg.to_csv(OUT / "grid_gap_seconds.csv", index=False)
    # Tie-break: among thresholds within 0.0005 of the best accuracy, prefer the
    # smallest one, i.e. split the trip at real gaps instead of trusting a
    # transition across minutes of missing data.
    best_acc = gg["train_oof_hmm_acc"].max()
    chosen["gap_seconds"] = float(gg[gg["train_oof_hmm_acc"] >= best_acc - 0.0005]["gap_seconds"].min())
    print("chosen decoder params:", chosen)

    # ---- final comparison with the chosen parameters
    tr = score(tr_ctx, chosen).assign(partition="train (out-of-fold)")
    st = score(st_ctx, chosen).assign(partition="student_test")
    per_trip = pd.concat([tr, st], ignore_index=True)
    per_trip.to_csv(OUT / "classifier_vs_hmm_per_trip.csv", index=False)
    cols = ["clf_acc", "hmm_acc", "clf_continuity", "hmm_continuity", "clf_route_f1", "hmm_route_f1",
            "clf_route_mismatch_fraction", "hmm_route_mismatch_fraction"]
    summ = per_trip.groupby("partition")[cols].mean()
    summ.to_csv(OUT / "classifier_vs_hmm_summary.csv")
    print(summ.T.to_string())

    cfg["hmm"] = chosen
    cfg_path.write_text(json.dumps(cfg, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = st.sort_values("trip")
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, s["clf_acc"], 0.4, label="classifier only")
    ax.bar(x + 0.2, s["hmm_acc"], 0.4, label="classifier + HMM/Viterbi")
    ax.set_xticks(x, s["trip"], rotation=45)
    ax.set_ylim(max(0, s[["clf_acc", "hmm_acc"]].min().min() - 0.1), 1.0)
    ax.set_ylabel("point accuracy")
    ax.set_title("Student-testing trips: classifier only vs HMM decoding")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "clf_vs_hmm_student_test.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for pw in sorted(g["emission_power"].unique()):
        sub = g[(g["emission_power"] == pw) & (g["top_m"] == chosen["top_m"])].sort_values("beta")
        ax.plot(sub["beta"], sub["train_oof_hmm_acc"], marker="o", label=f"emission power {pw}")
    ax.set_xscale("log")
    ax.set_xlabel("beta (m)")
    ax.set_ylabel("train OOF point accuracy")
    ax.set_title("Decoder sensitivity to beta")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "beta_sensitivity.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
