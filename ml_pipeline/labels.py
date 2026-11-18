"""Per-observation reference labels.

The .route file is a chronological list of arcs, not one arc per GPS point, so
it has to be aligned to the track before it can be used as a label. We use a
monotone dynamic-programming alignment: every GPS point i is assigned a route
position j(i) with j(i) >= j(i-1) (the vehicle never goes back along the route,
but may pass several short arcs between two fixes), minimising the total
point-to-arc distance.

f[i, j] = d(i, route[j]) + min_{k <= j} f[i-1, k]
"""
import numpy as np

from .geo import point_segment


def align_route(pts_xy: np.ndarray, route: np.ndarray, net) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ref_arc per point, distance of point to that arc in metres)."""
    n, m = len(pts_xy), len(route)
    A = net.A[route]
    B = net.B[route]
    f = np.empty((n, m), dtype=np.float32)
    prev_best = np.zeros(m, dtype=np.float64)
    for i in range(n):
        d, _, _ = point_segment(pts_xy[i][None, :], A, B)
        row = d + prev_best
        f[i] = row
        prev_best = np.minimum.accumulate(row)
    # Backtrack
    pos = np.empty(n, dtype=np.int64)
    j = int(np.argmin(f[-1]))
    pos[-1] = j
    for i in range(n - 2, -1, -1):
        j = int(np.argmin(f[i, : j + 1]))
        pos[i] = j
    ref = route[pos]
    d, _, _ = point_segment(pts_xy, net.A[ref], net.B[ref])
    return ref, d
