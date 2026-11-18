"""Dataset exploration for the train and student_test partitions.

Outputs (results/eda/):
  inventory.csv            one row per trip: files, sizes, counts, gaps, extent,
                           network connectivity, reference-route structure
  partition_summary.csv    totals per partition
  *.png                    plots
  map_trip_<id>.html       Folium map: GPS, road network, reference route
  data_quality_notes.md    automatically generated list of anomalies
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config, io  # noqa: E402
from pathsnap.geo import LocalProjection, haversine, point_segment  # noqa: E402

OUT = config.RESULTS_DIR / "eda"
PARTS = ["train", "student_test"]


def trip_stats(part, tid):
    d = io.trip_dir(part, tid)
    raw = np.loadtxt(d / f"{tid}.track", ndmin=2)
    nodes, arcs = io.load_network(part, tid)
    route = io.load_route(part, tid)
    lon, lat, t = raw[:, 0], raw[:, 1], raw[:, 2]
    dt = np.diff(t)
    step = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    n = len(nodes)
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    n_weak, weak = connected_components(g, directed=True, connection="weak")
    n_strong, strong = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(arcs.ravel(), minlength=n)
    route_breaks = int(np.sum(arcs[route[:-1], 1] != arcs[route[1:], 0]))
    proj = LocalProjection.for_points(lon, lat)
    xy = proj.forward(lon, lat)
    nxy = proj.forward(nodes[:, 0], nodes[:, 1])
    ra, rb = nxy[arcs[route, 0]][None], nxy[arcs[route, 1]][None]
    to_route = np.concatenate([point_segment(xy[k:k + 500, None, :], ra, rb)[0].min(axis=1)
                               for k in range(0, len(xy), 500)])
    w = (xy.max(0) - xy.min(0)) / 1000
    return {
        "partition": part, "trip": tid,
        "files": len([p for p in d.iterdir() if p.is_file()]),
        "track_kb": (d / f"{tid}.track").stat().st_size / 1024,
        "nodes_kb": (d / f"{tid}.nodes").stat().st_size / 1024,
        "arcs_kb": (d / f"{tid}.arcs").stat().st_size / 1024,
        "gps_points": len(raw),
        "non_finite_values": int((~np.isfinite(raw)).sum()),
        "out_of_range_coords": int(((np.abs(lon) > 180) | (np.abs(lat) > 90)).sum()),
        "non_increasing_timestamps": int((dt <= 0).sum()),
        "duration_min": (t[-1] - t[0]) / 60,
        "frac_dt_1s": float(np.mean(dt == 1)),
        "dt_max_s": float(dt.max()) if len(dt) else 0,
        "gaps_over_5s": int((dt > 5).sum()),
        "gaps_over_60s": int((dt > 60).sum()),
        "track_km": float(step.sum() / 1000),
        "max_step_m": float(step.max()) if len(step) else 0,
        "max_speed_kmh": float(np.max(step / np.maximum(dt, 1)) * 3.6) if len(step) else 0,
        "center_lon": float(lon.mean()), "center_lat": float(lat.mean()),
        "extent_ew_km": w[0], "extent_ns_km": w[1],
        "nodes": n, "arcs": len(arcs),
        "isolated_nodes": int((deg == 0).sum()),
        "weak_components": int(n_weak),
        "largest_weak_frac": float(np.bincount(weak).max() / n),
        "strong_components": int(n_strong),
        "largest_strong_frac": float(np.bincount(strong).max() / n),
        "route_arcs": len(route),
        "route_unique_arcs": len(np.unique(route)),
        "route_breaks": route_breaks,
        "gps_to_route_median_m": float(np.median(to_route)),
        "gps_to_route_p95_m": float(np.percentile(to_route, 95)),
    }


def folium_map(part, tid, path):
    import folium
    tr = io.load_track(part, tid)
    nodes, arcs = io.load_network(part, tid)
    route = io.load_route(part, tid)
    proj = LocalProjection.for_points(tr.lon, tr.lat)
    xy = proj.forward(tr.lon, tr.lat)
    nxy = proj.forward(nodes[:, 0], nodes[:, 1])
    from scipy.spatial import cKDTree
    mid = (nxy[arcs[:, 0]] + nxy[arcs[:, 1]]) / 2
    keep = cKDTree(xy).query(mid, distance_upper_bound=200)[0] < np.inf   # roads within 200 m of the track
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    net = folium.FeatureGroup(name=f"Road network within 200 m ({keep.sum():,} of {len(arcs):,} arcs)")
    for a, b in arcs[keep]:
        folium.PolyLine([nodes[a][::-1], nodes[b][::-1]], color="#9ca3af", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup(name="Reference route")
    folium.PolyLine([nodes[arcs[route[0], 0]][::-1]] + [nodes[arcs[r, 1]][::-1] for r in route],
                    color="#16a34a", weight=5, opacity=0.8).add_to(ref)
    ref.add_to(m)
    gps = folium.FeatureGroup(name="Raw GPS observations")
    for k in range(0, len(tr.lon)):
        folium.CircleMarker([tr.lat[k], tr.lon[k]], radius=2, color="#2563eb", fill=True,
                            fill_opacity=0.8, weight=0).add_to(gps)
    gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    m.save(str(path))


def plots(inv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = inv[inv["partition"] == "train"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].hist(inv["gps_points"], bins=30)
    axes[0].set_title("GPS points per trip")
    axes[1].hist(inv["track_km"], bins=30)
    axes[1].set_title("Trip length (km)")
    axes[2].hist(inv["duration_min"], bins=30)
    axes[2].set_title("Trip duration (min)")
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle("Trajectory statistics (train + student_test)")
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_stats.png", dpi=130)
    plt.close(fig)

    dts = []
    for _, r in tr.iterrows():
        t = np.loadtxt(io.trip_dir("train", r["trip"]) / f"{r['trip']}.track", ndmin=2)[:, 2]
        dts.append(np.diff(t))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(np.clip(dts, 0, 120), bins=np.arange(0, 122, 2), log=True)
    ax.set_xlabel("time between consecutive fixes (s, clipped at 120)")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Sampling intervals, train partition ({np.mean(dts == 1):.1%} exactly 1 s)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sampling_intervals.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for part, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        s = inv[inv["partition"] == part]
        ax.scatter(s["center_lon"], s["center_lat"], s=25, c=c, label=part)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Trip locations (trip centres)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "trip_locations.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(inv["nodes"], inv["arcs"], s=15)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("nodes")
    axes[0].set_ylabel("arcs")
    axes[0].set_title("Road-network size per trip")
    axes[1].hist(inv["largest_strong_frac"], bins=20)
    axes[1].set_xlabel("share of nodes in largest strongly connected component")
    axes[1].set_title("Road-network connectivity")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "network_size_connectivity.png", dpi=130)
    plt.close(fig)

    # one sample trajectory in metres (not degrees)
    tid = tr.sort_values("gps_points").iloc[len(tr) // 2]["trip"]
    t = io.load_track("train", tid)
    xy = LocalProjection.for_points(t.lon, t.lat).forward(t.lon, t.lat)
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=t.t - t.t[0], s=3, cmap="viridis")
    fig.colorbar(sc, label="seconds since start")
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"GPS trajectory, train trip {tid}")
    fig.tight_layout()
    fig.savefig(OUT / "sample_trajectory.png", dpi=130)
    plt.close(fig)


def notes(inv):
    lines = ["# Data-quality notes (generated by scripts/01_eda.py)", ""]
    for part in PARTS:
        s = inv[inv["partition"] == part]
        lines += [f"## {part}: {len(s)} trips, {s['gps_points'].sum():,} GPS points", "",
                  f"- Non-finite values: {s['non_finite_values'].sum()}; out-of-range coordinates: {s['out_of_range_coords'].sum()}; "
                  f"non-increasing timestamps: {s['non_increasing_timestamps'].sum()}",
                  f"- Share of 1 s steps (median over trips): {s['frac_dt_1s'].median():.1%}",
                  f"- Trips with a gap > 60 s: {(s['gaps_over_60s'] > 0).sum()} (largest gap {s['dt_max_s'].max():,.0f} s)",
                  f"- Trips whose reference route has breaks (consecutive arcs not connected): {(s['route_breaks'] > 0).sum()}",
                  f"- Trips with isolated nodes in the network: {(s['isolated_nodes'] > 0).sum()}",
                  f"- Median share of nodes in the largest strongly connected component: {s['largest_strong_frac'].median():.1%}",
                  f"- Max implied speed > 200 km/h (GPS spikes) in {(s['max_speed_kmh'] > 200).sum()} trips", ""]
        far = s[s["gps_to_route_median_m"] > 30]
        if len(far):
            lines.append("- Trips where GPS is far from the reference route (median > 30 m, possible label issues): "
                         + ", ".join(f"{r.trip} ({r.gps_to_route_median_m:.0f} m)" for r in far.itertuples()))
            lines.append("")
    (OUT / "data_quality_notes.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in PARTS:
        for tid in io.list_trips(part):
            rows.append(trip_stats(part, tid))
            print(part, tid, flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    summ = inv.groupby("partition").agg(trips=("trip", "count"), gps_points=("gps_points", "sum"),
                                        nodes=("nodes", "sum"), arcs=("arcs", "sum"),
                                        route_arcs=("route_arcs", "sum"), track_km=("track_km", "sum"),
                                        median_points=("gps_points", "median"),
                                        trips_with_gap_60s=("gaps_over_60s", lambda x: int((x > 0).sum())))
    summ.to_csv(OUT / "partition_summary.csv")
    print(summ.to_string())
    plots(inv)
    notes(inv)
    for tid in ["00000001"]:
        folium_map("train", tid, OUT / f"map_trip_{tid}.html")
    print("EDA written to", OUT)


if __name__ == "__main__":
    main()
