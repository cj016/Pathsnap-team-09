"""Candidate classification and per-GPS-point evaluation metrics."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


def _positive_probability(model, X):
    """Extract P(class=1) from an sklearn estimator or Pipeline."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Estimator must implement predict_proba.")
    p = np.asarray(model.predict_proba(X))
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(list(model.named_steps.values())[-1], "classes_", None)
    if classes is None:
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        raise ValueError("Cannot identify the positive-class probability column.")
    classes = list(classes)
    return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X), dtype=float)


def score_candidate_classifier(model, candidates: pd.DataFrame, feature_columns: list[str]):
    """Candidate-row metrics and scored candidate table."""
    X = candidates.loc[:, feature_columns].astype(float)
    y = candidates["label"].astype(int).to_numpy()
    t0 = time.perf_counter()
    pred = model.predict(X)
    p1 = _positive_probability(model, X)
    elapsed = time.perf_counter() - t0
    # Explicit two-class probabilities keep log_loss valid for sklearn pipelines.
    p2 = np.column_stack([1.0 - p1, p1]).clip(1e-15, 1 - 1e-15)
    metrics = {
        "candidate_accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p2, labels=[0, 1])),
        "inference_seconds": float(elapsed),
        "candidate_rows": int(len(candidates)),
    }
    scored = candidates.copy()
    scored["predicted_label"] = pred
    scored["probability_label_1"] = p1
    return metrics, scored


def top1_accuracy(scored_candidates: pd.DataFrame, inventory: pd.DataFrame,
                  *, include_uncertain: bool = False) -> dict:
    """Choose highest probability candidate per point; zero-candidate points wrong.

    inventory must include one row per GPS point, including points with no
    candidates. Its reference arc field is reference_arc_id (see
    features.data_preparation.point_inventory).
    """
    needed = {"trip_id", "point_idx", "reference_arc_id"}
    missing = needed - set(inventory.columns)
    if missing:
        raise ValueError(f"Point inventory missing columns: {sorted(missing)}")
    points = inventory.drop_duplicates(["trip_id", "point_idx"]).copy()
    if not include_uncertain:
        if "label_uncertain" not in points:
            raise ValueError("Certain-only Top-1 requires label_uncertain in inventory.")
        points = points.loc[~points["label_uncertain"].astype(bool)]
    best = (
        scored_candidates.sort_values(
            ["trip_id", "point_idx", "probability_label_1"],
            ascending=[True, True, False], kind="stable"
        )
        .drop_duplicates(["trip_id", "point_idx"])
        [["trip_id", "point_idx", "arc_id"]]
        .rename(columns={"arc_id": "predicted_arc_id"})
    )
    merged = points.merge(best, on=["trip_id", "point_idx"], how="left")
    correct = merged["predicted_arc_id"].notna() & (
        merged["predicted_arc_id"] == merged["reference_arc_id"]
    )
    n = len(merged)
    misses = int(merged["predicted_arc_id"].isna().sum())
    return {
        "top1_accuracy": float(correct.sum() / n) if n else float("nan"),
        "correct_points": int(correct.sum()),
        "evaluated_points": int(n),
        "candidate_misses": misses,
        "candidate_miss_rate": float(misses / n) if n else float("nan"),
    }


def evaluate_model(model, candidates: pd.DataFrame, inventory: pd.DataFrame,
                   feature_columns: list[str]) -> dict:
    """Return candidate metrics plus Top-1 with and without uncertain points."""
    candidate_metrics, scored = score_candidate_classifier(model, candidates, feature_columns)
    return {
        **candidate_metrics,
        "top1_certain": top1_accuracy(scored, inventory, include_uncertain=False),
        "top1_with_uncertain": top1_accuracy(scored, inventory, include_uncertain=True),
        "scored_candidates": scored,
    }
