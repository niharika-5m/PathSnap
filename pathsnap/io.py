"""Reading raw trip files.

Each trip folder <id>/ contains:
  <id>.track  lon lat timestamp      (GPS observations)
  <id>.nodes  lon lat                (row index = node id)
  <id>.arcs   from_node to_node      (row index = arc id)
  <id>.route  arc_id                 (reference route, chronological)

The prediction pipeline only ever calls load_track / load_network. The route is
read separately by load_route, which is used for training labels and for the
server-side evaluation, never as an input to prediction.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


def trip_dir(partition: str, trip_id: str) -> Path:
    return config.DATA_DIR / partition / trip_id


def list_trips(partition: str) -> list[str]:
    root = config.DATA_DIR / partition
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@dataclass
class Track:
    trip_id: str
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    n_raw: int          # rows before cleaning
    n_dropped: int      # rows removed by cleaning


def load_track(partition: str, trip_id: str) -> Track:
    """Load and clean a GPS track: drop non-finite / out-of-range rows, sort by
    time and drop duplicate timestamps (keeping the first)."""
    raw = np.loadtxt(trip_dir(partition, trip_id) / f"{trip_id}.track", ndmin=2)
    n_raw = len(raw)
    ok = np.isfinite(raw).all(axis=1)
    ok &= (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
    raw = raw[ok]
    raw = raw[np.argsort(raw[:, 2], kind="stable")]
    keep = np.r_[True, np.diff(raw[:, 2]) > 0]
    raw = raw[keep]
    return Track(trip_id, raw[:, 0].copy(), raw[:, 1].copy(), raw[:, 2].copy(),
                 n_raw, n_raw - len(raw))


def load_network(partition: str, trip_id: str) -> tuple[np.ndarray, np.ndarray]:
    d = trip_dir(partition, trip_id)
    nodes = np.loadtxt(d / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    return nodes, arcs


def load_route(partition: str, trip_id: str) -> np.ndarray:
    return np.loadtxt(trip_dir(partition, trip_id) / f"{trip_id}.route", dtype=np.int64, ndmin=1)
