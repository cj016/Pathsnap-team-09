"""Trip-grouped cross-validation utilities."""
from __future__ import annotations
from sklearn.model_selection import GroupKFold


def make_group_splits(rows, n_splits: int = 5):
    """Return GroupKFold splits grouped by trip_id (no trip leakage)."""
    if "trip_id" not in rows:
        raise ValueError("Rows must contain trip_id for trip-based CV.")
    unique_trips = rows["trip_id"].nunique()
    if unique_trips < n_splits:
        raise ValueError(
            f"Need at least {n_splits} distinct trips for GroupKFold; got {unique_trips}."
        )
    splitter = GroupKFold(n_splits=n_splits)
    X = rows.index.to_numpy()
    y = rows["label"].to_numpy() if "label" in rows else None
    groups = rows["trip_id"].to_numpy()
    # Return indices relative to rows to keep the API straightforward.
    return list(splitter.split(X, y, groups))


def assert_no_trip_leakage(rows, splits):
    """Raise if any trip appears in both train and validation in a split."""
    groups = rows["trip_id"].to_numpy()
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        overlap = set(groups[train_idx]) & set(groups[valid_idx])
        if overlap:
            raise AssertionError(f"Trip leakage in fold {fold}: {sorted(overlap)}")
    return True
