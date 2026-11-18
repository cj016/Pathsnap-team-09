"""Build the shared candidate-arc feature matrix for PathSnap.

Input rows represent one (GPS point, candidate arc) pair.  The two existing
features are supplied by Person 1.  Geometry/trajectory-dependent features
are computed only when the necessary columns are present; otherwise the
function fails with an actionable message instead of fabricating values.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "dist_to_arc",
    "heading_difference",
    "arc_length",
    "dist_rank",
    "speed",
    "position_along_arc",
]

REQUIRED_BASE_COLUMNS = {
    "trip_id", "point_idx", "timestamp", "x", "y", "arc_id",
    "dist_to_arc", "dist_rank",
}


def _angle_difference_degrees(a: pd.Series, b: pd.Series) -> pd.Series:
    """Smallest absolute angular difference in degrees (range 0..180)."""
    return ((a - b + 180.0) % 360.0 - 180.0).abs()


def add_heading_difference(
    candidates: pd.DataFrame,
    *,
    gps_heading_col: str = "gps_heading",
    arc_heading_col: str = "arc_heading",
) -> pd.DataFrame:
    """Add absolute GPS-vs-arc bearing difference when bearings are available.

    Bearings use compass degrees clockwise from north. This function does not
    infer direction from arc IDs: both bearings must be explicit.
    """
    out = candidates.copy()
    if gps_heading_col not in out or arc_heading_col not in out:
        raise ValueError(
            "heading_difference needs gps_heading and arc_heading. "
            "Provide GPS bearings from consecutive track points and directed "
            "arc bearings from the .nodes/.arcs road geometry."
        )
    out["heading_difference"] = _angle_difference_degrees(
        pd.to_numeric(out[gps_heading_col], errors="coerce"),
        pd.to_numeric(out[arc_heading_col], errors="coerce"),
    )
    return out


def add_arc_length(
    candidates: pd.DataFrame,
    *,
    start_x_col: str = "arc_start_x",
    start_y_col: str = "arc_start_y",
    end_x_col: str = "arc_end_x",
    end_y_col: str = "arc_end_y",
) -> pd.DataFrame:
    """Add straight-line endpoint length in metres for each directed arc.

    This is a chord length based on projected UTM coordinates. If road arcs are
    polylines, replace this with the sum of segment lengths when full geometry
    becomes available.
    """
    out = candidates.copy()
    required = [start_x_col, start_y_col, end_x_col, end_y_col]
    missing = [c for c in required if c not in out]
    if missing:
        raise ValueError(
            "arc_length needs endpoint coordinates from .nodes/.arcs: "
            + ", ".join(missing)
        )
    dx = pd.to_numeric(out[end_x_col], errors="coerce") - pd.to_numeric(
        out[start_x_col], errors="coerce"
    )
    dy = pd.to_numeric(out[end_y_col], errors="coerce") - pd.to_numeric(
        out[start_y_col], errors="coerce"
    )
    out["arc_length"] = np.hypot(dx, dy)
    return out


def add_speed(
    candidates: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    x_col: str = "x",
    y_col: str = "y",
    group_cols: Iterable[str] = ("trip_id",),
) -> pd.DataFrame:
    """Add GPS speed in metres/second from consecutive points in each trip.

    Duplicate candidate rows for a point receive the same speed. Timestamps
    must be parseable; coordinates are UTM metres as supplied by Person 1.
    """
    out = candidates.copy()
    point_key = list(group_cols) + ["point_idx"]
    points = out[point_key + [timestamp_col, x_col, y_col]].drop_duplicates(point_key)
    points = points.sort_values(point_key).copy()
    points[timestamp_col] = pd.to_datetime(points[timestamp_col], errors="coerce", utc=True)
    group = points.groupby(list(group_cols), sort=False)
    dt = group[timestamp_col].diff().dt.total_seconds()
    dx = group[x_col].diff()
    dy = group[y_col].diff()
    points["speed"] = np.hypot(dx, dy) / dt.where(dt > 0)
    out = out.merge(points[point_key + ["speed"]], on=point_key, how="left", validate="many_to_one")
    return out


def add_position_along_arc(
    candidates: pd.DataFrame,
    *,
    start_x_col: str = "arc_start_x",
    start_y_col: str = "arc_start_y",
    end_x_col: str = "arc_end_x",
    end_y_col: str = "arc_end_y",
    point_x_col: str = "x",
    point_y_col: str = "y",
) -> pd.DataFrame:
    """Project the GPS point onto the arc endpoint chord, clipped to [0, 1]."""
    out = candidates.copy()
    required = [start_x_col, start_y_col, end_x_col, end_y_col, point_x_col, point_y_col]
    missing = [c for c in required if c not in out]
    if missing:
        raise ValueError(
            "position_along_arc needs GPS and arc endpoint coordinates: "
            + ", ".join(missing)
        )
    ax = out[start_x_col].astype(float)
    ay = out[start_y_col].astype(float)
    vx = out[end_x_col].astype(float) - ax
    vy = out[end_y_col].astype(float) - ay
    px = out[point_x_col].astype(float) - ax
    py = out[point_y_col].astype(float) - ay
    denom = vx * vx + vy * vy
    projection = (px * vx + py * vy) / denom.where(denom > 0)
    out["position_along_arc"] = projection.clip(0.0, 1.0)
    return out


def build_feature_table(
    candidates: pd.DataFrame,
    *,
    require_all_features: bool = True,
) -> pd.DataFrame:
    """Return candidates with the agreed six features in the agreed order.

    Expected optional geometry columns are gps_heading, arc_heading,
    arc_start_x/y, arc_end_x/y. Feature calculations are deterministic and
    preserve all original metadata columns.
    """
    missing = sorted(REQUIRED_BASE_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing required columns: {missing}")

    out = candidates.copy()
    out["dist_to_arc"] = pd.to_numeric(out["dist_to_arc"], errors="coerce")
    out["dist_rank"] = pd.to_numeric(out["dist_rank"], errors="coerce")

    # The following calls intentionally require explicit geometry columns.
    if "gps_heading" in out.columns and "arc_heading" in out.columns:
        out = add_heading_difference(out)
    if all(c in out.columns for c in ["arc_start_x", "arc_start_y", "arc_end_x", "arc_end_y"]):
        out = add_arc_length(out)
        out = add_position_along_arc(out)
    if "speed" not in out.columns:
        out = add_speed(out)

    missing_features = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing_features and require_all_features:
        raise ValueError(
            "Cannot build the complete model matrix yet. Missing feature(s): "
            + ", ".join(missing_features)
            + ". Join arc endpoint coordinates/bearings from .nodes/.arcs and "
              "derive GPS heading from the ordered .track points."
        )
    return out


def get_feature_matrix(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Select the exact common feature names and order for every model."""
    missing = [c for c in FEATURE_COLUMNS if c not in feature_table.columns]
    if missing:
        raise ValueError(f"Feature table lacks required columns: {missing}")
    return feature_table.loc[:, FEATURE_COLUMNS].astype(float)
