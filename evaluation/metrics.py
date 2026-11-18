"""Candidate and GPS-point evaluation metrics for PathSnap."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


def _positive_probability(model, X):
    """Return P(label=1), locating class 1 safely on sklearn estimators/pipelines."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("The estimator must implement predict_proba for this evaluator.")
    probabilities = np.asarray(model.predict_proba(X))
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        classes = getattr(final_estimator, "classes_", None)
    if classes is None:
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1]
        raise ValueError("Could not determine which probability column represents label=1.")
    classes = list(classes)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def score_candidate_classifier(model, candidates: pd.DataFrame, feature_columns: list[str]):
    """Return candidate-level metrics and per-candidate probabilities."""
    X = candidates.loc[:, feature_columns].astype(float)
    y = candidates["label"].astype(int).to_numpy()
    t0 = time.perf_counter()
    pred = model.predict(X)
    probability = _positive_probability(model, X)
    elapsed = time.perf_counter() - t0

    metrics = {
        "candidate_accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, np.column_stack([1 - probability, probability]), labels=[0, 1])),
        "inference_seconds": float(elapsed),
        "candidate_rows": int(len(candidates)),
    }
    scored = candidates.copy()
    scored["predicted_label"] = pred
    scored["probability_label_1"] = probability
    return metrics, scored


def top1_accuracy(
    scored_candidates: pd.DataFrame,
    point_inventory: pd.DataFrame,
    *,
    include_uncertain: bool,
) -> dict:
    """Choose highest-scoring candidate per point; misses count as incorrect.

    point_inventory must contain one row per GPS point and reference_arc_id.
    It must include points with zero candidate rows. If candidate-level label
    is present, it may be used as the reference indicator for points with
    candidates, but reference_arc_id is preferred.
    """
    required_inventory = {"trip_id", "point_idx"}
    missing = required_inventory - set(point_inventory.columns)
    if missing:
        raise ValueError(f"point_inventory missing columns: {sorted(missing)}")
    inventory = point_inventory.drop_duplicates(["trip_id", "point_idx"]).copy()
    if not include_uncertain:
        if "label_uncertain" not in inventory:
            raise ValueError("Need label_uncertain in point_inventory for certain-only evaluation.")
        inventory = inventory.loc[~inventory["label_uncertain"].astype(bool)]

    best = (
        scored_candidates.sort_values(
            ["trip_id", "point_idx", "probability_label_1"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["trip_id", "point_idx"])
    )
    merged = inventory.merge(
        best[["trip_id", "point_idx", "arc_id"]],
        on=["trip_id", "point_idx"], how="left"
    )
    if "reference_arc_id" in merged:
        correct = merged["arc_id"].notna() & (merged["arc_id"] == merged["reference_arc_id"])
    elif "label" in scored_candidates:
        references = (
            scored_candidates.loc[scored_candidates["label"].astype(int).eq(1),
                                  ["trip_id", "point_idx", "arc_id"]]
            .drop_duplicates(["trip_id", "point_idx"])
            .rename(columns={"arc_id": "reference_arc_id"})
        )
        merged = merged.merge(references, on=["trip_id", "point_idx"], how="left")
        correct = merged["arc_id"].notna() & (merged["arc_id"] == merged["reference_arc_id"])
    else:
        raise ValueError("Need reference_arc_id in inventory or positive labels in candidates.")

    denominator = len(merged)
    return {
        "top1_accuracy": float(correct.sum() / denominator) if denominator else float("nan"),
        "correct_points": int(correct.sum()),
        "evaluated_points": int(denominator),
        "candidate_misses": int(merged["arc_id"].isna().sum()),
        "candidate_miss_rate": float(merged["arc_id"].isna().sum() / denominator) if denominator else float("nan"),
    }


def evaluate_model(
    model,
    candidates: pd.DataFrame,
    point_inventory: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Combined metrics. Requires one-row-per-point inventory for correct Top-1."""
    candidate_metrics, scored = score_candidate_classifier(model, candidates, feature_columns)
    certain = top1_accuracy(scored, point_inventory, include_uncertain=False)
    all_points = top1_accuracy(scored, point_inventory, include_uncertain=True)
    return {
        **candidate_metrics,
        "top1_certain": certain,
        "top1_with_uncertain": all_points,
        "scored_candidates": scored,
    }
