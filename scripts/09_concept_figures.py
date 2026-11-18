"""Explanatory figures drawn from real data (one student-test trip, final model).

  01_trip_overview.png      whole trip: GPS over its road network, zoom box
  02_nodes_and_arcs.png     what nodes and (directed) arcs are; a two-way street = 2 arcs
  03_gps_route_labels.png   GPS fixes, reference route, and the per-point label from alignment
  04_candidates.png         one GPS point: 60 m radius, candidate arcs, distances, XGBoost probs
  05_features_table.png     the feature values of that point's top candidates
  06_transition.png         HMM transition: straight-line GPS distance vs driving distance
  07_trellis.png            HMM trellis: states per point, transitions, Viterbi path

Usage: python scripts/09_concept_figures.py [trip_id]
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config, io  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.geo import point_segment  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.pipeline import PathSnap  # noqa: E402

OUT = config.RESULTS_DIR / "concepts"
PART = "student_test"
GREY, BLUE, PURPLE, GREEN, ORANGE, RED = "#9ca3af", "#2563eb", "#7c3aed", "#16a34a", "#f97316", "#dc2626"


def arrow(ax, a, b, color, lw=1.2, offset=0.0, z=2, alpha=1.0, head=7):
    """Directed arc a->b, shifted `offset` metres to its right so two-way pairs separate."""
    v = b - a
    n = np.array([v[1], -v[0]]) / (np.linalg.norm(v) + 1e-9)
    a2, b2 = a + n * offset, b + n * offset
    ax.add_patch(FancyArrowPatch(a2, b2, arrowstyle=f"-|>,head_length={head / 10},head_width={head / 20}",
                                 mutation_scale=10, color=color, lw=lw, zorder=z, alpha=alpha,
                                 shrinkA=0, shrinkB=0))


def window_arcs(net, c, half):
    m = (net.A + net.B) / 2
    return np.flatnonzero(np.all(np.abs(m - c) <= half * 1.2, axis=1))


def pick_point(ctx, ref, rel):
    f = ctx.frame
    best = f.loc[f.groupby("pidx")["p"].idxmax()].set_index("pidx")
    n = len(ctx.xy)
    for i in range(10, n - 10):
        if not rel[i] or best.loc[i, "arc"] != ref[i]:
            continue
        g = f[f.pidx == i]
        a = ref[i]
        twin = g[(g.arc_src == ctx.net.dst[a]) & (g.arc_dst == ctx.net.src[a])]
        if len(g) >= 6 and len(twin) and g.speed.iloc[0] > 6 and 0.2 < g[g.arc == a].t_along.iloc[0] < 0.8 \
                and ref[i - 3] != ref[i + 3]:
            return i
    return n // 2


def main():
    tid = sys.argv[1] if len(sys.argv) > 1 else "00000000"
    OUT.mkdir(parents=True, exist_ok=True)
    eng = PathSnap()
    pred = eng.predict(PART, tid)
    ctx = pred["ctx"]
    net, xy, f = ctx.net, ctx.xy, ctx.frame
    ref, ref_d = align_route(xy, io.load_route(PART, tid), net)
    rel = ref_d <= config.LABEL_MAX_DIST_M
    i0 = pick_point(ctx, ref, rel)
    c = xy[i0]
    route = io.load_route(PART, tid)
    print("focus point", i0, "trip", tid)

    # ---------------------------------------------------------------- 01 overview
    span = xy.max(0) - xy.min(0)
    fig, ax = plt.subplots(figsize=(12, max(4.5, 12 * span[1] / max(span[0], 1) + 1.2)))
    m = (net.A + net.B) / 2
    near = np.flatnonzero(np.all((m >= xy.min(0) - 400) & (m <= xy.max(0) + 400), axis=1))
    for a in near:
        ax.plot(*np.c_[net.A[a], net.B[a]], color=GREY, lw=0.4, alpha=0.6, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=14, color=BLUE, alpha=0.35, lw=0, zorder=2, label="GPS fixes (.track)")
    for k, a in enumerate(route):
        ax.plot(*np.c_[net.A[a], net.B[a]], color=PURPLE, lw=1.6, zorder=3,
                label="reference route (.route)" if k == 0 else None)
    ax.plot([], [], color=GREY, lw=1, label="road network (.nodes + .arcs)")
    ax.scatter(*c, s=260, facecolor="none", edgecolor=RED, lw=2.5, zorder=4)
    ax.annotate("area zoomed into\nin the next figures", c, xytext=(-140, -40), textcoords="offset points",
                color=RED, fontsize=9, arrowprops=dict(arrowstyle="->", color=RED))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"Trip {tid}: {len(xy):,} GPS fixes over its road network ({net.n_arcs:,} arcs) "
                 f"and its reference route ({len(route)} arcs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "01_trip_overview.png", dpi=140)
    plt.close(fig)

    half = 110
    arcs_w = window_arcs(net, c, half)

    def frame_ax(ax, title, cen=None, hw=None):
        cen = c if cen is None else cen
        hw = half if hw is None else hw
        ax.set_xlim(cen[0] - hw, cen[0] + hw)
        ax.set_ylim(cen[1] - hw, cen[1] + hw)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.plot([cen[0] - hw + 10, cen[0] - hw + 60], [cen[1] - hw + 10] * 2, color="black", lw=2)
        ax.text(cen[0] - hw + 35, cen[1] - hw + 14, "50 m", ha="center", fontsize=8)

    # ---------------------------------------------------------------- 02 nodes & arcs
    a_lab = ref[i0]
    c2 = (net.A[a_lab] + net.B[a_lab]) / 2           # centre on the labelled arc so both its nodes show
    h2a = max(half, 0.65 * net.length[a_lab])
    arcs_w2 = window_arcs(net, c2, h2a)
    nodes_w2 = np.unique(np.r_[net.src[arcs_w2], net.dst[arcs_w2]])
    fig, ax = plt.subplots(figsize=(8, 8))
    twin = [a for a in arcs_w2 if net.src[a] == net.dst[a_lab] and net.dst[a] == net.src[a_lab]]
    for a in arcs_w2:
        two = net.has_reverse[a]
        arrow(ax, net.A[a], net.B[a], GREY, lw=1.1, offset=2.2 if two else 0)
    arrow(ax, net.A[a_lab], net.B[a_lab], GREEN, lw=2.6, offset=2.2, z=4, head=9)
    if twin:
        arrow(ax, net.A[twin[0]], net.B[twin[0]], ORANGE, lw=2.6, offset=2.2, z=4, head=9)
    ax.scatter(net.node_xy[nodes_w2, 0], net.node_xy[nodes_w2, 1], s=14, color="black", zorder=5)
    s, d = net.src[a_lab], net.dst[a_lab]
    for nd, tag in [(s, "from"), (d, "to")]:
        lon, lat = net.nodes_lonlat[nd]
        ax.annotate(f"node {nd}\n({lon:.5f}, {lat:.5f})", net.node_xy[nd], xytext=(8, 8),
                    textcoords="offset points", fontsize=8, zorder=6,
                    bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.5))
    mid = (net.A[a_lab] + net.B[a_lab]) / 2
    txt = f"arc {a_lab}: node {s} → node {d}\n(row {a_lab} of .arcs = \"{s} {d}\")"
    if twin:
        txt += f"\narc {twin[0]}: node {d} → node {s}\n(the same street, other direction)"
    ax.annotate(txt, mid, xytext=(-200, -120), textcoords="offset points", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                bbox=dict(boxstyle="round", fc="#f8fafc", ec=GREY))
    frame_ax(ax, "Nodes (black dots, from .nodes) and directed arcs (arrows, from .arcs).\n"
                 "A two-way street is TWO arcs with the same geometry: green and orange.", c2, h2a)
    fig.tight_layout()
    fig.savefig(OUT / "02_nodes_and_arcs.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 03 GPS, route, labels
    fig, ax = plt.subplots(figsize=(8, 8))
    for a in arcs_w:
        arrow(ax, net.A[a], net.B[a], GREY, lw=0.8, offset=2.2 if net.has_reverse[a] else 0, alpha=0.7)
    rset = set(route.tolist())
    for a in arcs_w:
        if a in rset:
            arrow(ax, net.A[a], net.B[a], PURPLE, lw=3, offset=2.2 if net.has_reverse[a] else 0, z=3, head=8)
    ids = [i for i in range(max(0, i0 - 25), min(len(xy), i0 + 26))
           if np.all(np.abs(xy[i] - c) < half)]
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    uniq = list(dict.fromkeys(ref[ids]))
    for i in ids:
        _, t, _ = point_segment(xy[i][None], net.A[ref[i]][None], net.B[ref[i]][None])
        pr = net.A[ref[i]] + t[0] * (net.B[ref[i]] - net.A[ref[i]])
        col = cols[uniq.index(ref[i]) % 10]
        ax.plot([xy[i, 0], pr[0]], [xy[i, 1], pr[1]], ls=":", color=col, lw=1.2, zorder=4)
        ax.scatter(*xy[i], s=36, color=col, edgecolor="black", lw=0.5, zorder=5)
        if (i - i0) % 5 == 0:
            ax.annotate(f"#{i}\nt={ctx.t[i]:.0f}s", xy[i], xytext=(6, -14), textcoords="offset points", fontsize=7)
    frame_ax(ax, "GPS fixes (dots, from .track) and the reference route (purple, from .route).\n"
                 "Dot colour = the route arc each fix is assigned to by the alignment; dotted line = its projection")
    fig.tight_layout()
    fig.savefig(OUT / "03_gps_route_labels.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 04 candidates
    g = f[f.pidx == i0].sort_values("p", ascending=False)
    fig, ax = plt.subplots(figsize=(8.6, 8))
    for a in arcs_w:
        arrow(ax, net.A[a], net.B[a], GREY, lw=0.7, offset=2.2 if net.has_reverse[a] else 0, alpha=0.5)
    ax.add_patch(Circle(c, config.CANDIDATE_RADIUS_M, fill=False, ls="--", ec=BLUE, lw=1.2))
    ax.text(c[0], c[1] + config.CANDIDATE_RADIUS_M + 4, "60 m candidate radius", ha="center", color=BLUE, fontsize=8)
    cmap = plt.cm.viridis
    for k, r in enumerate(g.itertuples()):
        a = int(r.arc)
        arrow(ax, net.A[a], net.B[a], cmap(r.p), lw=3, offset=2.2 if net.has_reverse[a] else 0, z=3, head=8)
        pr = net.A[a] + r.t_along * (net.B[a] - net.A[a])
        if k < 5:
            ax.plot([c[0], pr[0]], [c[1], pr[1]], color="black", lw=0.6, ls="--", zorder=4)
            ax.annotate(f"arc {a}\nd={r.dist:.1f} m, Δθ={r.heading_diff:.0f}°\np={r.p:.3f}" + ("  ✓ correct" if a == ref[i0] else ""),
                        (pr + (net.A[a] + net.B[a]) / 2) / 2, xytext=(10 - 70 * (k % 2), 18 - 26 * k),
                        textcoords="offset points", fontsize=7.5, zorder=6,
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.4),
                        bbox=dict(boxstyle="round", fc="white", ec=GREEN if a == ref[i0] else GREY, lw=1.2 if a == ref[i0] else 0.5))
    lo, hi = max(0, i0 - 2), min(len(xy) - 1, i0 + 2)
    ax.annotate("", xy[hi], xy[lo], arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2), zorder=6)
    ax.text(*(xy[hi] + 4), "GPS heading\n(fix i−2 → i+2)", color=RED, fontsize=8, zorder=6)
    ax.scatter(*c, s=90, color=BLUE, edgecolor="black", zorder=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="XGBoost P(candidate is correct)")
    frame_ax(ax, f"One GPS fix (#{i0}) and its {len(g)} candidate arcs, coloured by XGBoost probability.\n"
                 "Dashed: perpendicular distance d. Δθ = angle between GPS heading and arc direction")
    fig.tight_layout()
    fig.savefig(OUT / "04_candidates.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 05 feature table
    show = ["arc", "dist", "dist_rank", "win_dist", "heading_diff", "speed", "t_along", "on_segment",
            "conn_prev", "conn_next", "has_reverse", "n_cand", "p"]
    tab = g.head(6)[show].copy()
    tab.insert(len(show), "correct", (tab["arc"] == ref[i0]).map({True: "yes", False: "no"}))
    cells = [[f"{v:.2f}" if isinstance(v, float) and k not in ("arc",) else str(int(v)) if k != "correct" and not isinstance(v, str) else v
              for k, v in row.items()] for _, row in tab.iterrows()]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=list(tab.columns), loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.5)
    for (r_, c_), cell in t.get_celld().items():
        if r_ == 0:
            cell.set_facecolor("#e5e7eb")
        elif cells[r_ - 1][-1] == "yes":
            cell.set_facecolor("#dcfce7")
    ax.set_title(f"Feature rows for fix #{i0}: one row per candidate (13 of the {len(FEATURES)} features shown), "
                 "sorted by XGBoost probability p", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "05_features_table.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 06 transition
    j = next(k for k in range(i0, min(len(xy) - 1, i0 + 40)) if pred["hmm_arcs"][k] != pred["hmm_arcs"][k + 1])
    a, b = int(pred["hmm_arcs"][j]), int(pred["hmm_arcs"][j + 1])
    ta = point_segment(xy[j][None], net.A[a][None], net.B[a][None])[1][0]
    tb = point_segment(xy[j + 1][None], net.A[b][None], net.B[b][None])[1][0]
    gc = float(np.linalg.norm(xy[j + 1] - xy[j]))
    beta = eng.cfg["hmm"]["beta"]
    unreach = eng.cfg["hmm"]["unreachable_logp"]

    def path_pts(a_, ta_, b_, tb_):
        pa_ = net.A[a_] + ta_ * (net.B[a_] - net.A[a_])
        pb_ = net.A[b_] + tb_ * (net.B[b_] - net.A[b_])
        if a_ == b_:
            return np.array([pa_, pb_])
        return np.array([pa_, net.B[a_]] + [net.B[e] for e in net.shortest_arc_path(a_, b_)] + [pb_])

    rd = net.route_distance(a, ta, b, tb, 1000.0)
    # a contrasting alternative: another candidate of fix j+1 that needs a detour (or is unreachable)
    limit = min(eng.cfg["hmm"]["max_speed_mps"] * max(ctx.t[j + 1] - ctx.t[j], 1.0), 3 * gc + 200) + 50
    alts = []
    for r in f[(f.pidx == j + 1) & (f.arc != b)].itertuples():
        d_ = net.route_distance(a, ta, int(r.arc), r.t_along, 1000.0)
        if np.isfinite(d_) and r.dist < 40:
            alts.append((d_ <= limit, abs(d_ - gc), int(r.arc), r.t_along, d_))
    within_l = sorted([z for z in alts if z[0]], key=lambda z: -z[1])
    beyond_l = sorted([z for z in alts if not z[0]], key=lambda z: z[1])
    panels = [("A. Two plausible moves (close-up)", within_l[0] if within_l else None),
              ("B. A detour the decoder rules out", beyond_l[0] if beyond_l else None)]
    panels = [pp for pp in panels if pp[1] is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 8))
    axes = np.atleast_1d(axes)
    P1 = path_pts(a, ta, b, tb)
    lp1 = -abs(rd - gc) / beta
    for ax, (ptitle, (within, _, b2, tb2, rd2)) in zip(axes, panels):
        P2 = path_pts(a, ta, b2, tb2)
        allp = np.vstack([P1, P2, xy[j:j + 2]])
        cc = (allp.min(0) + allp.max(0)) / 2
        h2 = max(35, (allp.max(0) - allp.min(0)).max() / 2 + 25)
        d_cc, _, _ = point_segment(cc[None], net.A, net.B)
        for e in np.flatnonzero(d_cc < h2 * 1.5):
            arrow(ax, net.A[e], net.B[e], GREY, lw=0.8, offset=1.5 if net.has_reverse[e] else 0, alpha=0.6)
        lp2 = -abs(rd2 - gc) / beta
        ax.plot(P1[:, 0], P1[:, 1], color=GREEN, lw=6, alpha=0.8, zorder=3,
                label=f"chosen: arc {a} → arc {b}, d_route = {rd:.1f} m → log T = {lp1:.2f}")
        ax.plot(P2[:, 0], P2[:, 1], color=ORANGE, lw=2.5, alpha=0.95, zorder=4, ls="-.",
                label=(f"alternative: arc {a} → arc {b2}, d_route = {rd2:.1f} m → log T = {lp2:.2f}" if within else
                       f"alternative: arc {a} → arc {b2} needs d_route = {rd2:.0f} m (turn into the side street, U-turn at the next junction, come back)\n"
                       f"> search limit {limit:.0f} m → treated as unreachable, log T = {unreach:.0f}"))
        ax.plot([xy[j, 0], xy[j + 1, 0]], [xy[j, 1], xy[j + 1, 1]], color=RED, ls="--", lw=2, zorder=5,
                label=f"straight-line GPS movement d_gc = {gc:.1f} m (1 s)")
        for p_, lab in [(xy[j], f"fix #{j}"), (xy[j + 1], f"fix #{j + 1}")]:
            ax.scatter(*p_, s=80, color=BLUE, edgecolor="black", zorder=6)
            ax.annotate(lab, p_, xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.set_xlim(cc[0] - h2, cc[0] + h2)
        ax.set_ylim(cc[1] - h2, cc[1] + h2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title(ptitle, fontsize=10)
    fig.suptitle(f"HMM transition score: log T = −|d_route − d_gc| / β  (β = {beta:.0f} m).  "
                 "Driving about as far as the GPS moved scores near 0; detours score lower", fontsize=11)
    fig.tight_layout()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "06_transition.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- 07 trellis
    cols_i = list(range(j - 2, j + 4))
    M = 4
    states = []
    for i in cols_i:
        s_ = f[f.pidx == i].sort_values("p", ascending=False).head(M)
        states.append(list(zip(s_.arc.astype(int), s_.t_along, s_.p)))
    fig, ax = plt.subplots(figsize=(13, 6))
    pos = {}
    for ci, (i, st) in enumerate(zip(cols_i, states)):
        for ri, (a_, _, p_) in enumerate(st):
            pos[(ci, ri)] = (ci * 2.2, -ri * 1.1)
    hmm = pred["hmm_arcs"]
    beta, gam = eng.cfg["hmm"]["beta"], eng.cfg["hmm"]["emission_power"]
    for ci in range(len(cols_i) - 1):
        i, i2 = cols_i[ci], cols_i[ci + 1]
        gcd = float(np.linalg.norm(xy[i2] - xy[i]))
        for ri, (a_, t_, _) in enumerate(states[ci]):
            for rj, (b_, tb_, _) in enumerate(states[ci + 1]):
                d_ = net.route_distance(a_, t_, b_, tb_, 3 * gcd + 250)
                lt = -abs(d_ - gcd) / beta if np.isfinite(d_) else eng.cfg["hmm"]["unreachable_logp"]
                on = hmm[i] == a_ and hmm[i2] == b_
                w = max(0.3, 3.0 * np.exp(lt / 3))
                (x1, y1), (x2, y2) = pos[(ci, ri)], pos[(ci + 1, rj)]
                ax.plot([x1 + 0.55, x2 - 0.55], [y1, y2], color=RED if on else GREY, lw=3.5 if on else w,
                        alpha=1 if on else 0.55, zorder=3 if on else 1)
                if on:
                    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, f"log T={lt:.2f}", color=RED, fontsize=7, ha="center")
    for ci, (i, st) in enumerate(zip(cols_i, states)):
        ax.text(ci * 2.2, 0.9, f"fix #{i}", ha="center", fontsize=10, weight="bold")
        for ri, (a_, _, p_) in enumerate(st):
            x, y = pos[(ci, ri)]
            on = hmm[i] == a_
            ok = ref[i] == a_
            ax.add_patch(Rectangle((x - 0.55, y - 0.32), 1.1, 0.64, fc="#fee2e2" if on else "white",
                                   ec=RED if on else GREY, lw=2 if on else 0.8, zorder=4))
            ax.text(x, y + 0.08, f"arc {a_}" + (" ✓" if ok else ""), ha="center", fontsize=8, zorder=5,
                    color=GREEN if ok else "black")
            ax.text(x, y - 0.17, f"p={p_:.2f}  logE={gam * np.log(max(p_, 1e-6)):.2f}", ha="center", fontsize=7, zorder=5)
    ax.set_xlim(-0.9, (len(cols_i) - 1) * 2.2 + 0.9)
    ax.set_ylim(-(M - 1) * 1.1 - 0.6, 1.2)
    ax.axis("off")
    ax.set_title("HMM trellis for 6 consecutive fixes: boxes = hidden states (top-4 candidate arcs with emission), lines = transitions\n"
                 "(thicker = more likely). Red = the Viterbi path; ✓ = reference arc", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "07_trellis.png", dpi=140)
    plt.close(fig)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
