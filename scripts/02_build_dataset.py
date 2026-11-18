"""Build the candidate/feature tables for the train and student_test partitions.

Output: cache/<partition>.parquet (one row per GPS point x candidate arc) and
results/dataset/candidate_summary_<partition>.csv (per-trip label/candidate stats).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config, io  # noqa: E402
from pathsnap.features import attach_labels, prepare_trip  # noqa: E402


def build(partition: str):
    frames, summary = [], []
    for tid in io.list_trips(partition):
        t0 = time.time()
        ctx = prepare_trip(partition, tid)
        ref, ref_dist = attach_labels(ctx, partition)
        f = ctx.frame
        f.insert(0, "trip", tid)
        n = len(ctx.xy)
        hit = f.groupby("pidx")["y"].max().reindex(range(n), fill_value=0).to_numpy()
        nearest_ok = f[f["dist_rank"] == 0].set_index("pidx")["y"].reindex(range(n), fill_value=0).to_numpy()
        summary.append({
            "trip": tid, "points": n, "arcs": ctx.net.n_arcs, "rows": len(f),
            "mean_candidates": len(f) / n,
            "candidate_miss_rate": 1 - hit.mean(),
            "nearest_arc_top1": nearest_ok.mean(),
            "label_dist_median": np.median(ref_dist),
            "label_dist_p95": np.percentile(ref_dist, 95),
            "unreliable_label_frac": np.mean(ref_dist > config.LABEL_MAX_DIST_M),
            "seconds": time.time() - t0,
        })
        frames.append(f)
        print(f"[{partition}] {tid}: {n} pts, {len(f)} rows, miss={1 - hit.mean():.4f}, {time.time() - t0:.1f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    config.CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(config.CACHE_DIR / f"{partition}.parquet", index=False)
    out = config.RESULTS_DIR / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(summary)
    s.to_csv(out / f"candidate_summary_{partition}.csv", index=False)
    print(s.describe().T.to_string())


if __name__ == "__main__":
    for part in sys.argv[1:] or ["train", "student_test"]:
        build(part)
