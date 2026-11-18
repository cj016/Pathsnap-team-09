"""Trip-grouped cross-validation helpers."""
from __future__ import annotations
from sklearn.model_selection import GroupKFold


def make_group_splits(rows, n_splits: int = 5):
    """Return train/validation indices grouped by trip_id (no trip leakage)."""
    if "trip_id" not in rows:
        raise ValueError("Rows must contain trip_id for trip-based CV.")
    n_groups = rows["trip_id"].nunique()
    if n_groups < n_splits:
        raise ValueError(f"Need at least {n_splits} distinct trips; found {n_groups}.")
    splitter = GroupKFold(n_splits=n_splits)
    groups = rows["trip_id"].to_numpy()
    y = rows["label"].to_numpy() if "label" in rows else None
    return list(splitter.split(rows, y, groups))


def assert_no_trip_leakage(rows, splits):
    """Verify no trip appears in both train and validation in any fold."""
    groups = rows["trip_id"].to_numpy()
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        overlap = set(groups[train_idx]) & set(groups[valid_idx])
        if overlap:
            raise AssertionError(f"Trip leakage in fold {fold}: {sorted(overlap)}")
    return True
