import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_DIR = Path("train")
OUTPUT_DIR = Path("cleaned")
OUTPUT_DIR.mkdir(exist_ok=True)

summary = []

for trip_dir in sorted(TRAIN_DIR.iterdir()):
    if not trip_dir.is_dir():
        continue

    track_file = trip_dir / f"{trip_dir.name}.track"

    if not track_file.exists():
        continue

    # Load raw GPS: longitude, latitude, timestamp
    df = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    original_count = len(df)

    # Convert to numeric and remove invalid values
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    # Remove invalid geographical coordinates
    df = df[
        df["longitude"].between(-180, 180) &
        df["latitude"].between(-90, 90)
    ]

    # Remove duplicate observations
    df = df.drop_duplicates()

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    cleaned_count = len(df)

    # Timestamp gap analysis
    gaps = df["timestamp"].diff().dropna()
    large_gaps = int((gaps > 5).sum())

    # Save cleaned trip
    df.to_csv(OUTPUT_DIR / f"{trip_dir.name}.csv", index=False)

    summary.append({
        "trip": trip_dir.name,
        "original_points": original_count,
        "cleaned_points": cleaned_count,
        "removed_points": original_count - cleaned_count,
        "large_gaps_over_5s": large_gaps
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUTPUT_DIR / "cleaning_summary.csv", index=False)

print("\n========== CLEANING SUMMARY ==========")
print(summary_df.to_string(index=False))

print("\nTOTAL TRIPS:", len(summary_df))
print("TOTAL ORIGINAL POINTS:", summary_df["original_points"].sum())
print("TOTAL CLEANED POINTS:", summary_df["cleaned_points"].sum())
print("TOTAL REMOVED:", summary_df["removed_points"].sum())
print("TRIPS WITH LARGE GAPS:", (summary_df["large_gaps_over_5s"] > 0).sum())

print("\nCleaning completed! Files saved in ./cleaned/")