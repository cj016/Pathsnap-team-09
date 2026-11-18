"""PathSnap candidate features, built from Person 1's preprocessing objects.

One candidate row = one (GPS point, directed candidate arc) pair.
The source data is trip-local: src.data_prep loads .track/.nodes/.arcs,
projects coordinates to UTM metres and creates Shapely LineStrings.

Feature order shared with all classifiers:
  dist_to_arc, heading_difference, arc_length, dist_rank, speed,
  position_along_arc
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

FEATURE_COLUMNS = [
    "dist_to_arc",
    "heading_difference",
    "arc_length",
    "dist_rank",
    "speed",
    "position_along_arc",
]
BASE_COLUMNS = {
    "trip_id", "point_idx", "timestamp", "x", "y", "arc_id",
    "dist_to_arc", "dist_rank", "label", "label_uncertain",
}


def _bearing_deg(dx, dy):
    """Compass bearing clockwise from north, in degrees [0, 360)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def _angle_delta(a, b):
    """Smallest absolute circular difference in degrees."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def build_trip_features(candidate_rows: pd.DataFrame, trip: dict, lines=None) -> pd.DataFrame:
    """Add the four geometry/trajectory features using a loaded/projected trip.

    Parameters
    ----------
    candidate_rows:
        Output of src.data_prep.candidates(...), possibly filtered to the
        chosen radius. Candidate arc_id is the zero-based row index in trip
        'arcs', and point_idx is the row index in trip 'track'.
    trip:
        Return value from src.data_prep.project_trip(load_trip(...)).
        Must contain track (x,y,timestamp), nodes (x,y), arcs(from_node,to_node).
    lines:
        Optional Shapely array from src.data_prep.arc_lines(trip).

    Notes
    -----
    arc_length is the endpoint length. These arcs in Person 1's implementation
    are directed node-to-node segments, so this is their segment length.
    Heading difference compares GPS movement bearing to the directed arc
    bearing. At stationary GPS samples, GPS heading is carried forward from
    the previous valid movement within that trip; the first unavailable
    bearing remains NaN and must be imputed in a train-fitted pipeline.
    position_along_arc is the clamped scalar projection onto the directed
    segment: 0 at its start node and 1 at its end node.
    """
    required = BASE_COLUMNS
    missing = sorted(required - set(candidate_rows.columns))
    if missing:
        raise ValueError(f"Candidate rows missing columns: {missing}")
    for key in ("track", "nodes", "arcs"):
        if key not in trip:
            raise ValueError(f"trip must contain {key!r}; use src.data_prep.load_trip/project_trip")
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    # Work from ordered track rows, not from candidate-row order.
    gps = track[["x", "y", "timestamp"]].copy()
    dx = gps["x"].diff().to_numpy(dtype=float)
    dy = gps["y"].diff().to_numpy(dtype=float)
    movement_bearing = _bearing_deg(dx, dy)
    movement_bearing[0] = np.nan
    movement_bearing[np.hypot(dx, dy) < 0.01] = np.nan
    gps_heading = pd.Series(movement_bearing).ffill().to_numpy()

    # Person 1's .track loader reads timestamps as numeric seconds (their
    # validation/EDA reports median timestamp delta = 1 second). Do not pass
    # numeric seconds to pd.to_datetime: pandas would interpret them as ns.
    raw_timestamps = gps["timestamp"]
    if pd.api.types.is_numeric_dtype(raw_timestamps):
        dt = pd.to_numeric(raw_timestamps, errors="coerce").diff().to_numpy(dtype=float)
    else:
        timestamps = pd.to_datetime(raw_timestamps, errors="coerce", utc=True)
        dt = timestamps.diff().dt.total_seconds().to_numpy()
    step = np.hypot(dx, dy)
    speed = np.divide(step, dt, out=np.full(len(gps), np.nan), where=(dt > 0) & np.isfinite(dt))
    # Avoid absurd velocities over recording gaps; preserve valid speed values.
    speed[(dt > 30) | (speed > 70)] = np.nan

    arc_xy = nodes[["x", "y"]].to_numpy(dtype=float)
    arc_nodes = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    starts = arc_xy[arc_nodes[:, 0]]
    ends = arc_xy[arc_nodes[:, 1]]
    vectors = ends - starts
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    arc_bearing = _bearing_deg(vectors[:, 0], vectors[:, 1])

    c = candidate_rows.copy()
    pi = c["point_idx"].to_numpy(dtype=int)
    ai = c["arc_id"].to_numpy(dtype=int)
    if (pi < 0).any() or (pi >= len(gps)).any():
        raise ValueError("point_idx contains indices outside the trip track")
    if (ai < 0).any() or (ai >= len(arcs)).any():
        raise ValueError("arc_id contains indices outside the trip arcs")

    c["heading_difference"] = _angle_delta(gps_heading[pi], arc_bearing[ai])
    c["arc_length"] = lengths[ai]
    c["speed"] = speed[pi]

    # Scalar projection of GPS coordinate onto each directed arc's endpoint vector.
    p = gps[["x", "y"]].to_numpy(dtype=float)[pi]
    a = starts[ai]
    v = vectors[ai]
    denom = np.einsum("ij,ij->i", v, v)
    numer = np.einsum("ij,ij->i", p - a, v)
    c["position_along_arc"] = np.divide(
        numer, denom, out=np.full(len(c), np.nan), where=denom > 0
    ).clip(0.0, 1.0)

    # Keep distance/rank from Person 1's candidate builder; canonical numeric types.
    c["dist_to_arc"] = pd.to_numeric(c["dist_to_arc"], errors="coerce")
    c["dist_rank"] = pd.to_numeric(c["dist_rank"], errors="coerce")
    return c


def build_feature_table(candidates: pd.DataFrame, trip: dict) -> pd.DataFrame:
    """Convenience wrapper which returns all six agreed features plus metadata."""
    return build_trip_features(candidates, trip)


def get_feature_matrix(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Select the exact agreed feature order for all model implementations."""
    missing = [name for name in FEATURE_COLUMNS if name not in feature_table.columns]
    if missing:
        raise ValueError(f"Feature table lacks required features: {missing}")
    return feature_table.loc[:, FEATURE_COLUMNS].astype(float)
