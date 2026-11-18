"""Road network: arc geometry, spatial candidate search and shortest paths."""
import heapq

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .geo import LocalProjection, bearing_deg, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                 densify_step: float = config.DENSIFY_STEP_M):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.node_xy = proj.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.src = arcs[:, 0].astype(np.int64)
        self.dst = arcs[:, 1].astype(np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        self.A = self.node_xy[self.src]
        self.B = self.node_xy[self.dst]
        vec = self.B - self.A
        self.length = np.linalg.norm(vec, axis=1)
        self.bearing = bearing_deg(vec[:, 0], vec[:, 1])

        # Directed adjacency (CSR): arcs leaving each node.
        order = np.argsort(self.src, kind="stable")
        self.out_arcs = order
        self.out_ptr = np.searchsorted(self.src[order], np.arange(self.n_nodes + 1))
        self.out_deg = np.diff(self.out_ptr)
        self.in_deg = np.bincount(self.dst, minlength=self.n_nodes)

        # Is there an arc going the opposite way (two-way street)?
        key = self.src * self.n_nodes + self.dst
        rkey = self.dst * self.n_nodes + self.src
        self.has_reverse = np.isin(rkey, key)

        self._build_index(densify_step)
        self._sp_cache: dict[int, tuple[float, dict]] = {}

    # ------------------------------------------------------------------ spatial
    def _build_index(self, step):
        n_pts = np.maximum(1, np.ceil(self.length / step).astype(np.int64)) + 1
        arc_of = np.repeat(np.arange(self.n_arcs), n_pts)
        starts = np.cumsum(n_pts) - n_pts
        frac = (np.arange(len(arc_of)) - np.repeat(starts, n_pts)) / np.repeat(n_pts - 1, n_pts)
        pts = self.A[arc_of] + frac[:, None] * (self.B[arc_of] - self.A[arc_of])
        self._sample_arc = arc_of
        self._tree = cKDTree(pts)
        self._step = step

    def candidates(self, pts_xy: np.ndarray, radius=config.CANDIDATE_RADIUS_M,
                   max_k=config.MAX_CANDIDATES, min_k=config.MIN_CANDIDATES):
        """For each point return (arc_ids, distances, t) of candidate arcs sorted by
        distance: all arcs within `radius`, capped at max_k, but never fewer than
        min_k (falls back to the nearest arcs whatever their distance)."""
        reach = radius + self._step / 2 + 1.0
        balls = self._tree.query_ball_point(pts_xy, reach)
        out = []
        for i, idx in enumerate(balls):
            arcs = np.unique(self._sample_arc[idx]) if idx else np.empty(0, np.int64)
            if len(arcs) < min_k:
                k = 64
                while True:
                    _, nn = self._tree.query(pts_xy[i], k=min(k, self._tree.n))
                    arcs = np.unique(self._sample_arc[np.atleast_1d(nn)])
                    if len(arcs) >= min_k or k >= self._tree.n:
                        break
                    k *= 4
            d, t, _ = point_segment(pts_xy[i][None, :], self.A[arcs], self.B[arcs])
            order = np.argsort(d, kind="stable")
            keep = order[d[order] <= radius][:max_k]
            if len(keep) < min_k:
                keep = order[:min_k]
            out.append((arcs[keep], d[keep], t[keep]))
        return out

    # ------------------------------------------------------------ shortest path
    def dists_from(self, node: int, limit: float) -> dict:
        """Bounded Dijkstra from `node`; returns {node: distance} for nodes within
        `limit` metres. Results are cached because consecutive GPS points share
        most of their candidate arcs."""
        cached = self._sp_cache.get(node)
        if cached is not None and cached[0] >= limit:
            return cached[1]
        dist = {node: 0.0}
        heap = [(0.0, node)]
        done = set()
        out_ptr, out_arcs, dst, length = self.out_ptr, self.out_arcs, self.dst, self.length
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for k in range(out_ptr[u], out_ptr[u + 1]):
                a = out_arcs[k]
                v = int(dst[a])
                nd = d + length[a]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if len(self._sp_cache) > 20000:
            self._sp_cache.clear()
        self._sp_cache[node] = (limit, dist)
        return dist

    def route_distance(self, arc_a, t_a, arc_b, t_b, limit):
        """Driving distance from position t_a on arc_a to position t_b on arc_b
        (np.inf if not reachable within `limit`). Small backward movement on the
        same arc (GPS jitter) is allowed and counted as a positive distance."""
        if arc_a == arc_b:
            return abs(t_b - t_a) * self.length[arc_a]
        rest_a = (1.0 - t_a) * self.length[arc_a]
        head_b = t_b * self.length[arc_b]
        budget = limit - rest_a - head_b
        if budget < 0:
            return np.inf
        d = self.dists_from(int(self.dst[arc_a]), max(budget, 0.0)).get(int(self.src[arc_b]))
        if d is None or d > budget:
            return np.inf
        return rest_a + d + head_b

    def shortest_arc_path(self, arc_a, arc_b, limit=5000.0):
        """Arcs strictly between arc_a and arc_b on a shortest path (used to fill
        holes when drawing a continuous predicted route). [] if adjacent or none."""
        s, g = int(self.dst[arc_a]), int(self.src[arc_b])
        if s == g:
            return []
        dist = {s: 0.0}
        prev = {}
        heap = [(0.0, s)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u == g:
                break
            if u in done:
                continue
            done.add(u)
            for k in range(self.out_ptr[u], self.out_ptr[u + 1]):
                a = self.out_arcs[k]
                v = int(self.dst[a])
                nd = d + self.length[a]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = a
                    heapq.heappush(heap, (nd, v))
        if g not in prev:
            return []
        path = []
        u = g
        while u != s:
            a = prev[u]
            path.append(int(a))
            u = int(self.src[a])
        return path[::-1]

    def connected(self, arc_a, arc_b) -> bool:
        """True if arc_b equals arc_a or directly follows it."""
        return arc_a == arc_b or self.dst[arc_a] == self.src[arc_b]


def build_network(nodes, arcs, track_lon, track_lat) -> RoadNetwork:
    proj = LocalProjection.for_points(track_lon, track_lat)
    return RoadNetwork(nodes, arcs, proj)
