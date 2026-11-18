"""HMM map matching with Viterbi decoding.

Hidden state at observation i: one of its candidate arcs (together with the
point's projected position t along that arc).

Emission  log P(z_i | arc)        = power * log p_clf(arc | features_i)
          (the selected classifier's probability that this candidate is right)

Transition log P(arc_b | arc_a)   = -| route_dist(a -> b) - gc_dist(z_{i-1}, z_i) | / beta
          route_dist is the driving distance along the directed road network
          from the previous projected position to the current one. Moving the
          same distance along the road as the GPS moved is likely; long detours
          or jumps to unconnected roads are not (Newson & Krumm, 2009).
          If no path exists within a speed-based limit the transition gets a
          fixed low log-probability (`unreachable_logp`) instead of -inf, so the
          decoder can still recover from a broken network or bad label.

GPS gaps: if the time gap between two fixes exceeds `gap_seconds`, the trip is
split there and each piece is decoded independently (no transition is scored
across the gap because the vehicle could have gone anywhere).

Viterbi recurrence:
  V_0(s)   = E_0(s)
  V_i(s)   = E_i(s) + max_r [ V_{i-1}(r) + T(r -> s) ]
  path     = backtrack argmax pointers from argmax_s V_{n-1}(s)
"""
import numpy as np
import pandas as pd

from . import config


def _split_points(t, gap_seconds):
    cut = np.flatnonzero(np.diff(t) > gap_seconds) + 1
    return np.split(np.arange(len(t)), cut)


def viterbi(net, xy, t, frame: pd.DataFrame, params: dict | None = None, score_col="p"):
    """Decode one trip. `frame` holds candidate rows (pidx, arc, t_along, p).
    Returns an array with the chosen arc for every GPS point."""
    prm = {**config.HMM_DEFAULTS, **(params or {})}
    beta = float(prm["beta"])
    power = float(prm["emission_power"])
    top_m = int(prm.get("top_m", 8))
    unreach = float(prm["unreachable_logp"])
    vmax = float(prm["max_speed_mps"])

    f = frame.sort_values(["pidx", score_col], ascending=[True, False])
    f = f.groupby("pidx", sort=True).head(top_m)
    starts = np.searchsorted(f["pidx"].to_numpy(), np.arange(len(xy) + 1))
    arcs_all = f["arc"].to_numpy()
    tal_all = f["t_along"].to_numpy()
    em_all = power * np.log(np.clip(f[score_col].to_numpy(), 1e-6, 1.0))

    pred = np.full(len(xy), -1, dtype=np.int64)
    for seg in _split_points(t, prm["gap_seconds"]):
        V = None
        back = []
        prev = None
        for i in seg:
            s, e = starts[i], starts[i + 1]
            arcs, tal, em = arcs_all[s:e], tal_all[s:e], em_all[s:e]
            if V is None:
                V = em.copy()
                back.append(None)
                prev = (i, arcs, tal)
                continue
            pi, parcs, ptal = prev
            gc = float(np.linalg.norm(xy[i] - xy[pi]))
            dt = max(float(t[i] - t[pi]), 1.0)
            limit = min(vmax * dt, 3.0 * gc + 200.0) + 50.0
            T = np.full((len(parcs), len(arcs)), unreach)
            for r, (a, ta) in enumerate(zip(parcs, ptal)):
                rest_a = (1.0 - ta) * net.length[a]
                reach = None
                for c, (b, tb) in enumerate(zip(arcs, tal)):
                    if a == b:
                        rd = abs(tb - ta) * net.length[a]
                    else:
                        head_b = tb * net.length[b]
                        budget = limit - rest_a - head_b
                        if budget < 0:
                            continue
                        if reach is None:
                            reach = net.dists_from(int(net.dst[a]), limit)
                        d = reach.get(int(net.src[b]))
                        if d is None or d > budget:
                            continue
                        rd = rest_a + d + head_b
                    T[r, c] = -abs(rd - gc) / beta
            scores = V[:, None] + T
            bp = np.argmax(scores, axis=0)
            V = scores[bp, np.arange(len(arcs))] + em
            back.append(bp)
            prev = (i, arcs, tal)
        # backtrack
        state = int(np.argmax(V))
        for k in range(len(seg) - 1, -1, -1):
            i = seg[k]
            pred[i] = arcs_all[starts[i] + state]
            if back[k] is not None:
                state = int(back[k][state])
    return pred


def classifier_only(frame: pd.DataFrame, n_points: int, score_col="p"):
    top = frame.loc[frame.groupby("pidx")[score_col].idxmax()]
    pred = np.full(n_points, -1, dtype=np.int64)
    pred[top["pidx"].to_numpy()] = top["arc"].to_numpy()
    return pred


def build_route(net, point_arcs: np.ndarray, fill_limit=3000.0) -> list[int]:
    """Turn per-point arcs into a continuous arc sequence: drop repeats and fill
    holes between non-adjacent arcs with the shortest connecting path."""
    seq = [int(a) for k, a in enumerate(point_arcs) if a >= 0 and (k == 0 or a != point_arcs[k - 1])]
    route = []
    for a in seq:
        if route and route[-1] != a and not net.connected(route[-1], a):
            route.extend(net.shortest_arc_path(route[-1], a, limit=fill_limit))
        if not route or route[-1] != a:
            route.append(a)
    return route


def route_metrics(net, pred_route, ref_route) -> dict:
    """Length-weighted overlap between predicted and reference arc sets, and the
    route mismatch fraction (Newson & Krumm): (erroneously added + missed) / ref."""
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    L = net.length
    inter = sum(L[a] for a in P & R)
    lp, lr = sum(L[a] for a in P), sum(L[a] for a in R)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    return {
        "route_precision": prec,
        "route_recall": rec,
        "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "route_mismatch_fraction": ((lp - inter) + (lr - inter)) / lr if lr else np.nan,
        "pred_route_km": lp / 1000,
        "ref_route_km": lr / 1000,
    }


def point_continuity(net, point_arcs) -> float:
    a = point_arcs[point_arcs >= 0]
    if len(a) < 2:
        return 1.0
    ok = (a[1:] == a[:-1]) | (net.src[a[1:]] == net.dst[a[:-1]])
    return float(ok.mean())
