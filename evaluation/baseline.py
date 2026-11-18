"""Nearest-arc baseline, matching Person 1's distance-rank definition."""
from __future__ import annotations
import pandas as pd

from features.data_preparation import evaluate_candidate_coverage


def nearest_candidate_predictions(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"trip_id", "point_idx", "arc_id", "dist_rank"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")
    return (
        candidates.sort_values(["trip_id", "point_idx", "dist_rank"], kind="stable")
        .drop_duplicates(["trip_id", "point_idx"])
        [["trip_id", "point_idx", "arc_id"]]
        .rename(columns={"arc_id": "predicted_arc_id"})
    )


def nearest_candidate_top1(candidates: pd.DataFrame, inventory: pd.DataFrame,
                           *, include_uncertain: bool = False) -> dict:
    """Top-1 accuracy for rank=1; missing-candidate points count as incorrect."""
    required = {"trip_id", "point_idx", "reference_arc_id"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Point inventory missing columns: {sorted(missing)}")
    points = inventory.drop_duplicates(["trip_id", "point_idx"]).copy()
    if not include_uncertain:
        if "label_uncertain" not in points:
            raise ValueError("Certain-only baseline requires label_uncertain.")
        points = points.loc[~points["label_uncertain"].astype(bool)]
    pred = nearest_candidate_predictions(candidates)
    scored = points.merge(pred, on=["trip_id", "point_idx"], how="left")
    correct = scored["predicted_arc_id"].notna() & (
        scored["predicted_arc_id"] == scored["reference_arc_id"]
    )
    n = len(scored)
    misses = evaluate_candidate_coverage(candidates, points)["candidate_misses"]
    return {
        "top1_accuracy": float(correct.sum() / n) if n else float("nan"),
        "correct_points": int(correct.sum()),
        "evaluated_points": int(n),
        "candidate_misses": misses,
        "candidate_miss_rate": float(misses / n) if n else float("nan"),
    }
