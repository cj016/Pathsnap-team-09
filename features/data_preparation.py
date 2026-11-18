"""Leakage-aware data preparation for candidate-level map matching."""
from __future__ import annotations
import pandas as pd


def training_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """Exclude uncertain points and every point whose true arc is not a candidate."""
    required = {"trip_id", "point_idx", "label", "label_uncertain"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")
    rows = candidates.loc[~candidates["label_uncertain"].astype(bool)].copy()
    has_positive = rows.groupby(["trip_id", "point_idx"])["label"].transform("max").eq(1)
    return rows.loc[has_positive].copy()


def model_features(rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Select only the explicitly approved model inputs."""
    missing = [c for c in feature_columns if c not in rows.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return rows.loc[:, feature_columns].astype(float)


def prepare_xy(candidates: pd.DataFrame, feature_columns: list[str]):
    """Build X,y after removing uncertain points and candidate misses."""
    rows = training_rows(candidates)
    if rows.empty:
        raise ValueError("No certain candidate points with a positive candidate remain.")
    return model_features(rows, feature_columns), rows["label"].astype(int)


def point_inventory(labelled_points: pd.DataFrame) -> pd.DataFrame:
    """Build one evaluation row per GPS point from points_labelled.parquet.

    This includes GPS points with no candidate rows. Person 1's labelled points
    table has label (reference arc id), label_dist and label_uncertain.
    """
    required = {"trip_id", "timestamp", "label", "label_uncertain"}
    missing = required - set(labelled_points.columns)
    if missing:
        raise ValueError(f"Labelled point table missing columns: {sorted(missing)}")
    points = labelled_points.copy().reset_index(drop=True)
    if "point_idx" not in points:
        points["point_idx"] = points.groupby("trip_id", sort=False).cumcount()
    return points.rename(columns={"label": "reference_arc_id"})[
        ["trip_id", "point_idx", "reference_arc_id", "label_uncertain"]
    ]


def evaluate_candidate_coverage(candidates: pd.DataFrame, inventory: pd.DataFrame) -> dict:
    """Report candidate misses, including GPS points omitted from candidate CSV."""
    keys = ["trip_id", "point_idx"]
    present = candidates[keys].drop_duplicates().assign(has_candidate=True)
    scored = inventory[keys].drop_duplicates().merge(present, on=keys, how="left")
    misses = scored["has_candidate"].isna()
    n = len(scored)
    return {
        "n_points": int(n),
        "candidate_misses": int(misses.sum()),
        "candidate_miss_rate": float(misses.mean()) if n else float("nan"),
    }
