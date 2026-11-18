"""Build candidate features for an unlabelled trip."""

from features.feature_engineering import FEATURE_COLUMNS, build_trip_features
from src.data_prep import arc_lines, load_inputs, project_trip, spatial_candidates


def build_candidates(trip_directory, radius=75.0):
    trip = project_trip(load_inputs(trip_directory))
    lines = arc_lines(trip)
    rows = spatial_candidates(trip, lines, "inference", radius)
    return build_trip_features(rows, trip, lines)[
        ["point_idx", "arc_id", *FEATURE_COLUMNS]
    ]
