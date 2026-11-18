"""Folium maps of the full pipeline output.

For each requested trip: raw GPS, nearby road network, reference route,
classifier-only route and HMM-predicted route, with a legend and layer control.

Usage: python scripts/07_maps.py [partition] [trip ...]
       (default: all student_test trips)
"""
import sys
from pathlib import Path

import folium
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config, io  # noqa: E402
from pathsnap.hmm import build_route  # noqa: E402
from pathsnap.pipeline import PathSnap, evaluate  # noqa: E402

OUT = config.RESULTS_DIR / "maps"

LEGEND = """
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; padding: 10px 12px;
     border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.3); font: 13px sans-serif; line-height: 1.7">
<b>Trip {trip}</b> &nbsp; HMM acc {acc:.1%} &middot; classifier acc {cacc:.1%} &middot; route F1 {f1:.1%}<br>
<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#2563eb"></span> GPS observation<br>
<span style="display:inline-block;width:22px;height:4px;background:#9ca3af"></span> Nearby road network<br>
<span style="display:inline-block;width:22px;height:4px;background:#7c3aed"></span> Reference route<br>
<span style="display:inline-block;width:22px;height:4px;background:#f97316"></span> Classifier-only route<br>
<span style="display:inline-block;width:22px;height:4px;background:#16a34a"></span> Predicted route (HMM)
</div>"""


def polyline_groups(net, route):
    lines, cur = [], []
    for k, a in enumerate(route):
        s, d = int(net.src[a]), int(net.dst[a])
        if not cur or int(net.dst[route[k - 1]]) != s:
            if cur:
                lines.append(cur)
            cur = [net.nodes_lonlat[s][::-1].tolist()]
        cur.append(net.nodes_lonlat[d][::-1].tolist())
    if cur:
        lines.append(cur)
    return lines


def make_map(engine, partition, tid):
    pred = engine.predict(partition, tid)
    scores = evaluate(partition, tid, pred)
    ctx = pred["ctx"]
    net = ctx.net
    m = folium.Map(location=[float(ctx.lat.mean()), float(ctx.lon.mean())], zoom_start=14, tiles="OpenStreetMap")

    roads = folium.FeatureGroup(name="Nearby road network")
    for a in np.unique(ctx.frame["arc"].to_numpy()):
        folium.PolyLine([net.nodes_lonlat[net.src[a]][::-1].tolist(), net.nodes_lonlat[net.dst[a]][::-1].tolist()],
                        color="#9ca3af", weight=2, opacity=0.6).add_to(roads)
    roads.add_to(m)

    layers = [("Reference route", io.load_route(partition, tid), "#7c3aed", 7, None, True),
              ("Classifier-only route", build_route(net, pred["classifier_arcs"]), "#f97316", 4, "6 6", False),
              ("Predicted route (HMM)", pred["route"], "#16a34a", 4, None, True)]
    for name, route, color, w, dash, show in layers:
        g = folium.FeatureGroup(name=name, show=show)
        for line in polyline_groups(net, list(route)):
            folium.PolyLine(line, color=color, weight=w, opacity=0.85, dash_array=dash).add_to(g)
        g.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS observations")
    for k in range(len(ctx.lat)):
        folium.CircleMarker([float(ctx.lat[k]), float(ctx.lon[k])], radius=2, color="#2563eb", fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"#{k} t={ctx.t[k]:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(ctx.lat.min()), float(ctx.lon.min())], [float(ctx.lat.max()), float(ctx.lon.max())]])
    m.get_root().html.add_child(folium.Element(LEGEND.format(
        trip=tid, acc=scores["hmm_point_accuracy"], cacc=scores["classifier_point_accuracy"], f1=scores["route_f1"])))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{partition}_{tid}.html"
    m.save(str(path))
    print(f"{path.name}: HMM {scores['hmm_point_accuracy']:.4f}  classifier {scores['classifier_point_accuracy']:.4f}  "
          f"route F1 {scores['route_f1']:.4f}")


def main():
    partition = sys.argv[1] if len(sys.argv) > 1 else "student_test"
    trips = sys.argv[2:] or io.list_trips(partition)
    engine = PathSnap()
    for tid in trips:
        make_map(engine, partition, tid)


if __name__ == "__main__":
    main()
