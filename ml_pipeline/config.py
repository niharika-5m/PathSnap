"""Central configuration: paths and default pipeline parameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Raw data lives outside git. Point PATHSNAP_DATA_DIR at the folder that holds
# train/, student_test/ and (on the API laptop) evaluator_test/.
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", ROOT / "data"))

ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"          # generated feature tables (gitignored)

SEED = 42

# Candidate generation
CANDIDATE_RADIUS_M = 60.0     # arcs within this distance of a GPS point
MAX_CANDIDATES = 12           # keep the closest K arcs
MIN_CANDIDATES = 3            # always keep at least this many (even if farther)
DENSIFY_STEP_M = 15.0         # spacing of sample points used by the spatial index

# Feature engineering
HEADING_WINDOW = 2            # points either side used to estimate GPS heading
DIST_WINDOW = 3               # points either side used for window distance

# Reference labels: points whose aligned reference arc is farther than this are
# treated as unreliable labels and excluded from training (still reported).
LABEL_MAX_DIST_M = 100.0

# HMM defaults (tuned later by scripts/05_hmm.py)
HMM_DEFAULTS = {
    "beta": 30.0,             # transition scale (m) on |route_dist - gc_dist|
    "emission_power": 1.0,    # exponent applied to classifier probability
    "gap_seconds": 60.0,      # time gap that splits the trip into independent pieces
    "max_speed_mps": 50.0,    # used to bound the shortest-path search
    "unreachable_logp": -30.0,  # log-prob for transitions with no path in range
}
