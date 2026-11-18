"""Nearest-candidate baseline: always choose dist_rank == 1."""
from __future__ import annotations
import pandas as pd


def nearest_candidate_predictions(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"trip_id", "point_idx", "arc_id", "dist_rank"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")
    # Stable tie handling: preserve CSV order when two candidates share rank.
    return (
        candidates.sort_values(["trip_id", "point_idx", "dist_rank"], kind="stable")
        .drop_duplicates(["trip_id", "point_idx"])
        [["trip_id", "point_idx", "arc_id"]]
        .rename(columns={"arc_id": "predicted_arc_id"})
    )


def nearest_candidate_top1(candidates: pd.DataFrame, point_inventory: pd.DataFrame) -> dict:
    """Score nearest-candidate selection; zero-candidate points count as misses."""
    if "reference_arc_id" not in point_inventory:
        positives = (
            candidates.loc[candidates["label"].astype(int).eq(1),
                           ["trip_id", "point_idx", "arc_id"]]
            .drop_duplicates(["trip_id", "point_idx"])
            .rename(columns={"arc_id": "reference_arc_id"})
        )
        inventory = point_inventory.drop_duplicates(["trip_id", "point_idx"]).merge(
            positives, on=["trip_id", "point_idx"], how="left"
        )
    else:
        inventory = point_inventory.drop_duplicates(["trip_id", "point_idx"]).copy()

    predictions = nearest_candidate_predictions(candidates)
    scored = inventory.merge(predictions, on=["trip_id", "point_idx"], how="left")
    correct = scored["predicted_arc_id"].notna() & (
        scored["predicted_arc_id"] == scored["reference_arc_id"]
    )
    n = len(scored)
    misses = int(scored["predicted_arc_id"].isna().sum())
    return {
        "top1_accuracy": float(correct.sum() / n) if n else float("nan"),
        "correct_points": int(correct.sum()),
        "evaluated_points": int(n),
        "candidate_misses": misses,
        "candidate_miss_rate": float(misses / n) if n else float("nan"),
    }
