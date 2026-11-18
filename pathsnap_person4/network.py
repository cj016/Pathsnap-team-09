"""Read a trip's public inputs and construct its directed, metre-based road graph.

Crucial: this module NEVER reads .route files; those are references, not features.
Arc ids are 0-based row indices of .arcs, not node ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from pyproj import Transformer


def _table(path: Path, cols: int, dtype: type) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")
    try:
        data = np.loadtxt(path, dtype=dtype, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric table: {path}") from exc
    if data.shape[1] != cols or len(data) == 0:
        raise ValueError(f"Expected nonempty {cols}-column file: {path}")
    return data


@dataclass
class TripNetwork:
    trip_id: str
    track: np.ndarray      # longitude, latitude, time (N x 3)
    nodes: np.ndarray      # longitude, latitude (M x 2)
    arcs: np.ndarray       # from, to (E x 2)
    track_xy: np.ndarray   # metres in a per-trip UTM CRS
    node_xy: np.ndarray
    arc_length: np.ndarray
    graph: nx.DiGraph
    utm_epsg: int

    @classmethod
    def from_directory(cls, trip_dir: Path) -> "TripNetwork":
        trip_dir = Path(trip_dir)
        tid = trip_dir.name
        track = _table(trip_dir / f"{tid}.track", 3, float)
        nodes = _table(trip_dir / f"{tid}.nodes", 2, float)
        arcs = _table(trip_dir / f"{tid}.arcs", 2, int)
        if not np.isfinite(track).all() or not np.isfinite(nodes).all():
            raise ValueError("Coordinates and timestamps must all be finite")
        for name, table in [("GPS", track[:, :2]), ("nodes", nodes)]:
            if ((abs(table[:, 0]) > 180) | (abs(table[:, 1]) > 90)).any():
                raise ValueError(f"Implausible lon/lat in {name}")
        if (np.diff(track[:, 2]) <= 0).any():
            raise ValueError("GPS timestamps must be strictly increasing")
        if arcs.min() < 0 or arcs.max() >= len(nodes):
            raise ValueError("Arc endpoints refer to nonexistent nodes")

        mid_lon = float(np.median(track[:, 0]))
        mid_lat = float(np.median(track[:, 1]))
        if not -80 <= mid_lat <= 84:
            raise ValueError("Trip is outside the supported UTM latitude range")
        zone = max(1, min(60, int((mid_lon + 180) // 6) + 1))
        epsg = (32600 if mid_lat >= 0 else 32700) + zone
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        nx_, ny_ = transformer.transform(nodes[:, 0], nodes[:, 1])
        tx, ty = transformer.transform(track[:, 0], track[:, 1])
        node_xy = np.column_stack([nx_, ny_])
        track_xy = np.column_stack([tx, ty])
        vec = node_xy[arcs[:, 1]] - node_xy[arcs[:, 0]]
        lengths = np.linalg.norm(vec, axis=1)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(nodes)))
        for arc_id, (src, dst) in enumerate(arcs):
            # A DiGraph stores the shortest of any parallel edges. Individual
            # arc ids still remain distinct Viterbi hidden states.
            length = float(lengths[arc_id])
            if not graph.has_edge(int(src), int(dst)) or length < graph[int(src)][int(dst)]["weight"]:
                graph.add_edge(int(src), int(dst), weight=length, arc_id=arc_id)
        return cls(tid, track, nodes, arcs, track_xy, node_xy, lengths, graph, epsg)

    def position_fraction(self, point_idx: int, arc_id: int) -> float:
        a, b = map(int, self.arcs[arc_id])
        start, end = self.node_xy[a], self.node_xy[b]
        delta = end - start
        den = float(np.dot(delta, delta))
        return float(np.clip(np.dot(self.track_xy[point_idx] - start, delta) / den, 0.0, 1.0)) if den > 0 else 0.0

    def arc_lonlat(self, arc_id: int) -> list[list[float]]:
        a, b = map(int, self.arcs[arc_id])
        return [self.nodes[a].tolist(), self.nodes[b].tolist()]

    def distance_between(
        self, a: int, frac_a: float, b: int, frac_b: float,
        max_distance: float, source_distances: Optional[dict[int, dict[int, float]]] = None,
    ) -> float:
        """Directed distance between two positions on arcs, or infinity.

        A same-edge forward move is allowed; going backwards requires a network
        loop. Distances exceeding max_distance are rejected as implausible.
        """
        if a == b and frac_b >= frac_a:
            d = (frac_b - frac_a) * float(self.arc_length[a])
            return d if d <= max_distance else float("inf")
        _, to_a = map(int, self.arcs[a])
        from_b, _ = map(int, self.arcs[b])
        tail = (1.0 - frac_a) * float(self.arc_length[a])
        head = frac_b * float(self.arc_length[b])
        budget = max_distance - tail - head
        if budget < 0:
            return float("inf")
        if to_a == from_b:
            middle = 0.0
        else:
            if source_distances is None:
                source_distances = {}
            if to_a not in source_distances:
                # IMPORTANT: cache belongs to one transition/time step only,
                # so all calls use the same max_distance cutoff.
                source_distances[to_a] = nx.single_source_dijkstra_path_length(
                    self.graph, to_a, cutoff=max_distance, weight="weight"
                )
            middle = source_distances[to_a].get(from_b, float("inf"))
        result = tail + middle + head
        return float(result) if result <= max_distance else float("inf")

    def route_lines(self, predicted: list[int | None], breaks: set[int]) -> list[list[list[float]]]:
        """Polyline parts for the UI. Never draws an invented line across a break."""
        parts: list[list[list[float]]] = []
        path: list[list[float]] = []
        prev: int | None = None
        for idx, arc in enumerate(predicted):
            if arc is None or idx in breaks:
                if len(path) >= 2:
                    parts.append(path)
                path, prev = [], None
            if arc is None:
                continue
            arc = int(arc)
            if arc == prev:
                continue
            start, end = map(int, self.arcs[arc])
            if prev is not None:
                _, last_end = map(int, self.arcs[prev])
                try:
                    connector = nx.shortest_path(self.graph, last_end, start, weight="weight")
                    if connector:
                        coords = [self.nodes[k].tolist() for k in connector]
                        for pos in coords:
                            if not path or path[-1] != pos:
                                path.append(pos)
                except nx.NetworkXNoPath:
                    if len(path) >= 2:
                        parts.append(path)
                    path = []
            first = self.nodes[start].tolist()
            second = self.nodes[end].tolist()
            if not path or path[-1] != first:
                path.append(first)
            if path[-1] != second:
                path.append(second)
            prev = arc
        if len(path) >= 2:
            parts.append(path)
        return parts
