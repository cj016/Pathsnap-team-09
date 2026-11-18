"""Server-only evaluation. Reference arc ids are never passed to inference/UI."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .decoder import DecoderConfig
from .network import TripNetwork


def load_point_labels(labels_file: Path, observation_count: int) -> list[int] | None:
    """Person 1 must supply exactly one row per GPS point: point_idx, arc_id.

    Raw .route is only a chronological *list of traversed arcs*, not per-point
    ground truth; NEVER zip it with GPS observations to fabricate labels.
    """
    if not labels_file.is_file():
        return None
    labels = pd.read_csv(labels_file)
    if not {"point_idx", "arc_id"}.issubset(labels.columns):
        raise ValueError(f"Bad label CSV {labels_file}: need point_idx,arc_id")
    expected = list(range(observation_count))
    if labels["point_idx"].tolist() != expected:
        raise ValueError(f"Labels in {labels_file} must contain ordered indices 0..N-1")
    if labels["arc_id"].isnull().any():
        raise ValueError("Point labels include null arc ids; decide a documented policy")
    return labels["arc_id"].astype(int).tolist()


def evaluate_predictions(prediction: dict, labels: list[int] | None) -> dict:
    if labels is None:
        return {"status": "unavailable", "reason": "No verified per-point reference labels on server"}
    n = prediction["observation_count"]
    if len(labels) != n:
        raise ValueError(f"Expected {n} reference labels, received {len(labels)}")
    truth = np.asarray(labels, dtype=int)
    results: dict = {"status": "available", "reference_observations": n}
    for method, key in (("classifier", "classifier_arc_id"), ("hmm", "hmm_arc_id")):
        raw = [row[key] for row in prediction["observations"]]
        # Arc ids cannot be negative; -1 explicitly means 'no candidate'.
        guess = np.asarray([int(x) if x is not None else -1 for x in raw], dtype=int)
        results[method] = {
            "top1_accuracy": float(accuracy_score(truth, guess)),
            "macro_f1": float(f1_score(truth, guess, average="macro", zero_division=0)),
            "exact_sequence_match": bool(np.array_equal(truth, guess)),
            "correct_transition_pair_fraction": float(np.mean(
                (guess[1:] == truth[1:]) & (guess[:-1] == truth[:-1])
            )) if n > 1 else None,
            "missing_prediction_count": int((guess == -1).sum()),
        }
    results["hmm_minus_classifier_top1"] = (
        results["hmm"]["top1_accuracy"] - results["classifier"]["top1_accuracy"]
    )
    return results


def evaluate_continuity(
    network: TripNetwork, predicted: list[int | None], cfg: DecoderConfig
) -> dict:
    """Directed reachable, plausible adjacent-pair rate. Big GPS gaps excluded.

    Denominator: adjacent observations with dt <= gap_seconds, including ones
    with missing predictions (they count as not continuous). This yields a
    comparable coverage-aware figure for classifier-only and HMM.
    """
    if len(predicted) != len(network.track):
        raise ValueError("Point-prediction vector length mismatch")
    eligible, feasible, missing = 0, 0, 0
    for idx in range(1, len(predicted)):
        dt = float(network.track[idx, 2] - network.track[idx-1, 2])
        if dt > cfg.gap_seconds:
            continue
        eligible += 1
        a, b = predicted[idx-1], predicted[idx]
        if a is None or b is None:
            missing += 1
            continue
        f_a = network.position_fraction(idx-1, a)
        f_b = network.position_fraction(idx, b)
        limit = cfg.max_speed_mps * dt + cfg.slack_m
        distance = network.distance_between(a, f_a, b, f_b, limit)
        if np.isfinite(distance):
            feasible += 1
    return {
        "rate": float(feasible / eligible) if eligible else None,
        "feasible_adjacent_pairs": feasible,
        "eligible_adjacent_pairs": eligible,
        "missing_candidate_pairs": missing,
        "infeasible_adjacent_pairs": eligible - feasible - missing,
        "definition": "directed reachable within max_speed*dt+slack; gaps excluded; missing counts as failure",
    }
