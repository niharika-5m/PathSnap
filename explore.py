from pathlib import Path

ROOT = Path(".")
TRAIN = ROOT / "train"
TEST = ROOT / "student_test"

train_trips = sorted([p for p in TRAIN.iterdir() if p.is_dir()])
test_trips = sorted([p for p in TEST.iterdir() if p.is_dir()])

print("=" * 50)
print("PATHSNAP DATASET REPORT")
print("=" * 50)

print("Training trips:", len(train_trips))
print("Student test trips:", len(test_trips))

# Inspect first training trip
trip = train_trips[0]

print("\nInspecting trip:", trip.name)

for file in sorted(trip.iterdir()):
    print(f"\n{'-' * 40}")
    print("FILE:", file.name)

    try:
        lines = file.read_text(errors="replace").splitlines()
        print("Number of lines:", len(lines))
        print("First 5 lines:")
        for line in lines[:5]:
            print(line)
    except Exception as e:
        print("Error reading:", e)

print("\nDone.")