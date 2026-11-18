"""Geometry helpers.

Longitude/latitude are angles, so we never measure distance in degrees. Every
trip covers at most a few tens of kilometres, so we project it onto a local
equirectangular plane centred on the trip (x = east metres, y = north metres).
Over 50 km the scale error of this projection is well under 0.5 %, which is far
smaller than GPS noise (~5-30 m), and it lets us use fast planar geometry.
The haversine function is kept for checks and for reporting.
"""
import numpy as np

EARTH_R = 6371008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = np.radians(1.0) * EARTH_R * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * EARTH_R

    @classmethod
    def for_points(cls, lon, lat):
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def forward(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return np.stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky], axis=-1)

    def inverse(self, xy):
        xy = np.asarray(xy, dtype=float)
        return xy[..., 0] / self.kx + self.lon0, xy[..., 1] / self.ky + self.lat0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment(p, a, b):
    """Distance from points p to segments a->b (all (..., 2) arrays, broadcast).
    Returns (distance, t_clipped, t_raw) where t is the position along the
    segment as a fraction of its length."""
    ab = b - a
    ap = p - a
    L2 = np.einsum("...i,...i->...", ab, ab)
    t_raw = np.einsum("...i,...i->...", ap, ab) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(t_raw, 0.0, 1.0)
    proj = a + t[..., None] * ab
    d = np.linalg.norm(p - proj, axis=-1)
    return d, t, t_raw


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, 90 = east) of a planar vector."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
