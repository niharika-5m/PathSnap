import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TRAIN_DIR = Path("train")

results = []

for trip_dir in sorted(TRAIN_DIR.iterdir()):
    if not trip_dir.is_dir():
        continue

    trip_id = trip_dir.name

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    if not all(f.exists() for f in
                [track_file, nodes_file, arcs_file, route_file]):
        continue

    track = pd.read_csv(track_file, sep=r"\s+", header=None)
    nodes = pd.read_csv(nodes_file, sep=r"\s+", header=None)
    arcs = pd.read_csv(arcs_file, sep=r"\s+", header=None)
    route = pd.read_csv(route_file, sep=r"\s+", header=None)

    results.append({
        "trip": trip_id,
        "gps_points": len(track),
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route_arcs": len(route)
    })

summary = pd.DataFrame(results)

print("\n========== DATASET EDA ==========\n")
print(summary.to_string(index=False))

print("\n========== TOTALS ==========")
print("Trips:", len(summary))
print("GPS Points:", summary["gps_points"].sum())
print("Road Nodes:", summary["nodes"].sum())
print("Road Arcs:", summary["arcs"].sum())
print("Route Arcs:", summary["route_arcs"].sum())

print("\n========== AVERAGES ==========")
print(summary[["gps_points", "nodes", "arcs", "route_arcs"]].mean())

summary.to_csv("eda_summary.csv", index=False)

# GPS points distribution
plt.figure(figsize=(10, 5))
plt.hist(summary["gps_points"], bins=20)
plt.title("GPS Points per Trip")
plt.xlabel("Number of GPS Points")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.savefig("gps_points_distribution.png")
plt.show()

# Plot one sample GPS trajectory
sample_trip = TRAIN_DIR / "00000001"
track_file = sample_trip / "00000001.track"

track = pd.read_csv(
    track_file,
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

plt.figure(figsize=(8, 8))
plt.plot(track["longitude"], track["latitude"], linewidth=0.7)
plt.scatter(track["longitude"], track["latitude"], s=2)
plt.title("GPS Trajectory - Trip 00000001")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")
plt.tight_layout()
plt.savefig("sample_trajectory.png")
plt.show()

print("\nEDA completed. Files saved.")