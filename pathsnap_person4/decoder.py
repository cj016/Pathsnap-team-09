"""Directed-road Hidden Markov Model / Viterbi decoder.

Each point's candidate arc is a hidden state. Emissions come from a fitted
classifier; transitions depend on directed shortest-path travel distance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .network import TripNetwork


@dataclass(frozen=True)
class DecoderConfig:
    beta_m: float = 35.0          # scale of route-vs-GPS displacement residual
    max_speed_mps: float = 55.0   # 198 km/h upper bound, adjust on development data
    slack_m: float = 50.0         # buffer for GPS error and endpoint approximation
    gap_seconds: float = 15.0     # restart after a long timestamp gap

    def __post_init__(self) -> None:
        if min(self.beta_m, self.max_speed_mps, self.slack_m, self.gap_seconds) <= 0:
            raise ValueError("Decoder parameters must be positive")


@dataclass
class DecodeResult:
    classifier_arc_ids: list[int | None]
    hmm_arc_ids: list[int | None]
    breaks: set[int]
    diagnostics: dict[str, int]


def decode(candidates: pd.DataFrame, network: TripNetwork, cfg: DecoderConfig = DecoderConfig()) -> DecodeResult:
    """Decode a candidate table with point_idx, arc_id and positive_prob.

    Not every point needs a candidate: those points return None and explicitly
    break the sequence. Never fill a missing observation with invented labels.
    """
    needed = {"point_idx", "arc_id", "positive_prob"}
    if not needed.issubset(candidates.columns):
        raise ValueError(f"Candidate table missing {sorted(needed - set(candidates.columns))}")
    n = len(network.track)
    if not candidates.empty:
        if candidates[list(needed)].isnull().any().any():
            raise ValueError("Candidate table contains missing indices, arc ids or probabilities")
        if not np.isfinite(candidates["positive_prob"].astype(float).to_numpy()).all():
            raise ValueError("Classifier emitted a nonfinite probability")
        if candidates["point_idx"].min() < 0 or candidates["point_idx"].max() >= n:
            raise ValueError("Candidate point_idx outside GPS observation range")
        if candidates["arc_id"].min() < 0 or candidates["arc_id"].max() >= len(network.arcs):
            raise ValueError("Candidate arc_id outside network arc range")
        if candidates.duplicated(["point_idx", "arc_id"]).any():
            raise ValueError("Duplicate (point_idx, arc_id) candidates are not allowed")
    by_point = {int(idx): df for idx, df in candidates.groupby("point_idx", sort=False)}
    independent: list[int | None] = [None] * n
    decoded: list[int | None] = [None] * n
    breaks: set[int] = set()
    events = {"no_candidates": 0, "gps_gaps": 0, "unreachable_restarts": 0}

    # Current contiguous segment: arrays of arc states / backpointer indices.
    segment_indices: list[int] = []
    state_history: list[np.ndarray] = []
    predecessor_history: list[np.ndarray] = []
    scores: np.ndarray | None = None
    fractions_prev: np.ndarray | None = None

    def finish_segment() -> None:
        nonlocal scores, fractions_prev
        if scores is None:
            return
        selected = int(np.argmax(scores))
        for t in range(len(segment_indices) - 1, -1, -1):
            decoded[segment_indices[t]] = int(state_history[t][selected])
            if t > 0:
                selected = int(predecessor_history[t][selected])
                if selected < 0:
                    raise RuntimeError("Broken Viterbi backpointer")
        segment_indices.clear()
        state_history.clear()
        predecessor_history.clear()
        scores, fractions_prev = None, None

    for idx in range(n):
        df = by_point.get(idx)
        if df is None or df.empty:
            finish_segment()
            breaks.add(idx)
            events["no_candidates"] += 1
            continue
        states = df["arc_id"].to_numpy(dtype=int)
        p = df["positive_prob"].to_numpy(dtype=float)
        if (p < 0).any() or (p > 1).any():
            raise ValueError("Candidate probabilities must lie in [0,1]")
        independent[idx] = int(states[np.argmax(p)])
        # Restrict classifier probabilities to the candidate set at this point.
        p = np.clip(p, 1e-12, 1.0)
        log_emission = np.log(p / p.sum())
        fractions = np.array([network.position_fraction(idx, int(a)) for a in states])

        if scores is not None:
            dt = float(network.track[idx, 2] - network.track[segment_indices[-1], 2])
            if dt > cfg.gap_seconds:
                finish_segment()
                breaks.add(idx)
                events["gps_gaps"] += 1

        if scores is None:
            segment_indices.append(idx)
            state_history.append(states)
            predecessor_history.append(np.full(len(states), -1, dtype=int))
            scores, fractions_prev = log_emission, fractions
            continue

        prev_idx = segment_indices[-1]
        dt = float(network.track[idx, 2] - network.track[prev_idx, 2])
        obs_distance = float(np.linalg.norm(network.track_xy[idx] - network.track_xy[prev_idx]))
        distance_limit = cfg.max_speed_mps * dt + cfg.slack_m
        prev_states = state_history[-1]
        assert fractions_prev is not None
        # Each previous-state row is normalised across feasible next states.
        log_trans = np.full((len(prev_states), len(states)), -np.inf)
        cached_sources: dict[int, dict[int, float]] = {}
        for i, a in enumerate(prev_states):
            for j, b in enumerate(states):
                d = network.distance_between(
                    int(a), float(fractions_prev[i]), int(b), float(fractions[j]),
                    distance_limit, cached_sources,
                )
                if math.isfinite(d):
                    log_trans[i, j] = -abs(d - obs_distance) / cfg.beta_m
            allowed = np.isfinite(log_trans[i])
            if allowed.any():
                row = log_trans[i, allowed]
                max_row = float(row.max())
                log_z = max_row + math.log(float(np.exp(row - max_row).sum()))
                log_trans[i, allowed] = row - log_z

        proposed = scores[:, None] + log_trans
        best_prev = np.argmax(proposed, axis=0)
        best = proposed[best_prev, np.arange(len(states))] + log_emission
        if not np.isfinite(best).any():
            finish_segment()
            breaks.add(idx)
            events["unreachable_restarts"] += 1
            segment_indices.append(idx)
            state_history.append(states)
            predecessor_history.append(np.full(len(states), -1, dtype=int))
            scores, fractions_prev = log_emission, fractions
        else:
            segment_indices.append(idx)
            state_history.append(states)
            predecessor_history.append(best_prev)
            scores, fractions_prev = best, fractions
    finish_segment()
    return DecodeResult(independent, decoded, breaks, events)
