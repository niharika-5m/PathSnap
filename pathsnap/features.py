"""Candidate generation + feature engineering.

One row per (GPS observation, candidate arc). The classifier learns
P(candidate is the arc the vehicle was on | features). Those probabilities are
used directly as HMM emission scores.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, io
from .geo import angle_diff_deg, bearing_deg, point_segment
from .network import RoadNetwork, build_network

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "win_dist", "win_dist_rank", "win_dist_minus_min",
    "heading_diff", "heading_cos", "stationary",
    "speed", "log_arc_len", "t_along", "on_segment",
    "n_cand", "dt_prev", "dt_next", "jump_prev",
    "conn_prev", "conn_next", "has_reverse", "out_deg", "in_deg",
]


@dataclass
class TripContext:
    trip_id: str
    net: RoadNetwork
    xy: np.ndarray        # projected GPS points (n, 2)
    t: np.ndarray         # timestamps (n,)
    lon: np.ndarray
    lat: np.ndarray
    frame: pd.DataFrame   # candidate rows


def _rank_within(groups: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.lexsort((values, groups))
    rank = np.empty(len(values), dtype=np.int64)
    g_sorted = groups[order]
    start = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1]
    pos = np.arange(len(values)) - np.repeat(start, np.diff(np.r_[start, len(values)]))
    rank[order] = pos
    return rank


def _group_min(groups, values, n):
    out = np.full(n, np.inf)
    np.minimum.at(out, groups, values)
    return out[groups]


def build_features(net: RoadNetwork, xy: np.ndarray, t: np.ndarray,
                   radius=config.CANDIDATE_RADIUS_M, max_k=config.MAX_CANDIDATES,
                   min_k=config.MIN_CANDIDATES) -> pd.DataFrame:
    n = len(xy)
    cands = net.candidates(xy, radius=radius, max_k=max_k, min_k=min_k)
    counts = np.array([len(c[0]) for c in cands])
    pidx = np.repeat(np.arange(n), counts)
    arc = np.concatenate([c[0] for c in cands])
    dist = np.concatenate([c[1] for c in cands])
    rank = np.concatenate([np.arange(k) for k in counts])

    # --- per-point kinematics
    idx = np.arange(n)
    w = config.HEADING_WINDOW
    lo, hi = np.clip(idx - w, 0, n - 1), np.clip(idx + w, 0, n - 1)
    hv = xy[hi] - xy[lo]
    hlen = np.linalg.norm(hv, axis=1)
    gps_bearing = bearing_deg(hv[:, 0], hv[:, 1])
    span = np.maximum(t[hi] - t[lo], 1e-6)
    speed = hlen / span
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    jump_prev = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    stationary = hlen < 3.0

    # --- per-row geometry
    _, t_clip, t_raw = point_segment(xy[pidx], net.A[arc], net.B[arc])
    hd = angle_diff_deg(gps_bearing[pidx], net.bearing[arc])
    hd = np.where(stationary[pidx], 90.0, hd)

    W = config.DIST_WINDOW
    wd = np.zeros(len(arc))
    for o in range(-W, W + 1):
        j = np.clip(pidx + o, 0, n - 1)
        d_o, _, _ = point_segment(xy[j], net.A[arc], net.B[arc])
        wd += d_o
    wd /= (2 * W + 1)

    # --- sequential connectivity with neighbours' nearest candidates
    top = np.full((n, 3), -1, dtype=np.int64)
    for i, c in enumerate(cands):
        k = min(3, len(c[0]))
        top[i, :k] = c[0][:k]
    prev_top = np.vstack([np.full((1, 3), -1), top[:-1]])[pidx]
    next_top = np.vstack([top[1:], np.full((1, 3), -1)])[pidx]
    valid_p = prev_top >= 0
    valid_n = next_top >= 0
    pt = np.where(valid_p, prev_top, 0)
    nt = np.where(valid_n, next_top, 0)
    conn_prev = (valid_p & ((pt == arc[:, None]) | (net.dst[pt] == net.src[arc][:, None]))).any(axis=1)
    conn_next = (valid_n & ((nt == arc[:, None]) | (net.src[nt] == net.dst[arc][:, None]))).any(axis=1)

    dmin = _group_min(pidx, dist, n)
    wdmin = _group_min(pidx, wd, n)
    df = pd.DataFrame({
        "pidx": pidx,
        "arc": arc,
        "arc_src": net.src[arc],
        "arc_dst": net.dst[arc],
        "t_along": t_clip,
        "dist": dist,
        "dist_rank": rank,
        "dist_minus_min": dist - dmin,
        "dist_ratio": dist / (dmin + 1.0),
        "win_dist": wd,
        "win_dist_rank": _rank_within(pidx, wd),
        "win_dist_minus_min": wd - wdmin,
        "heading_diff": hd,
        "heading_cos": np.cos(np.radians(hd)),
        "stationary": stationary[pidx].astype(np.int8),
        "speed": np.minimum(speed[pidx], 80.0),
        "log_arc_len": np.log1p(net.length[arc]),
        "on_segment": ((t_raw >= 0) & (t_raw <= 1)).astype(np.int8),
        "n_cand": counts[pidx],
        "dt_prev": np.nan_to_num(np.minimum(dt_prev[pidx], 600.0), nan=1.0),
        "dt_next": np.nan_to_num(np.minimum(dt_next[pidx], 600.0), nan=1.0),
        "jump_prev": np.nan_to_num(np.minimum(jump_prev[pidx], 2000.0), nan=0.0),
        "conn_prev": conn_prev.astype(np.int8),
        "conn_next": conn_next.astype(np.int8),
        "has_reverse": net.has_reverse[arc].astype(np.int8),
        "out_deg": net.out_deg[net.dst[arc]],
        "in_deg": net.in_deg[net.src[arc]],
    })
    return df


def prepare_trip(partition: str, trip_id: str, **cand_kwargs) -> TripContext:
    """Prediction-time entry point: uses only the track and the road network."""
    tr = io.load_track(partition, trip_id)
    nodes, arcs = io.load_network(partition, trip_id)
    net = build_network(nodes, arcs, tr.lon, tr.lat)
    xy = net.proj.forward(tr.lon, tr.lat)
    frame = build_features(net, xy, tr.t, **cand_kwargs)
    return TripContext(trip_id, net, xy, tr.t, tr.lon, tr.lat, frame)


def attach_labels(ctx: TripContext, partition: str):
    """Training / evaluation only: align the reference route and label rows."""
    from .labels import align_route
    route = io.load_route(partition, ctx.trip_id)
    ref, ref_dist = align_route(ctx.xy, route, ctx.net)
    f = ctx.frame
    f["y"] = (f["arc"].to_numpy() == ref[f["pidx"].to_numpy()]).astype(np.int8)
    f["label_dist"] = ref_dist[f["pidx"].to_numpy()]
    return ref, ref_dist
