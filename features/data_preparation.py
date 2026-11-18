"""Leakage-safe preparation helpers for candidate classification."""
from __future__ import annotations

import pandas as pd

IDENTIFIER_COLUMNS = {
    "trip_id", "partition", "point_idx", "timestamp", "arc_id",
    "label", "label_uncertain", "reference_arc_id",
}


def training_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep only labelled, certain candidate points for model fitting.

    Candidate misses have no positive candidate row and are consequently not
    trainable from this candidate-level table. Keep them in point_inventory for
    evaluation instead.
    """
    required = {"label", "label_uncertain", "point_idx"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    out = candidates.loc[~candidates["label_uncertain"].astype(bool)].copy()
    # Exclude a whole GPS point if it has no positive candidate label.
    has_positive = out.groupby(["trip_id", "point_idx"])["label"].transform("max").eq(1)
    out = out.loc[has_positive]
    return out


def model_features(candidates: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Select only approved feature columns; metadata/labels cannot leak."""
    missing = [c for c in feature_columns if c not in candidates.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return candidates.loc[:, feature_columns].astype(float)


def prepare_xy(candidates: pd.DataFrame, feature_columns: list[str]):
    """Return X, y for training after applying the shared exclusions."""
    rows = training_rows(candidates)
    if rows.empty:
        raise ValueError("No certain candidate points with a positive match remain.")
    return model_features(rows, feature_columns), rows["label"].astype(int)
