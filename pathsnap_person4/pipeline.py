"""Classify candidate arcs and decode a directed route."""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import joblib
import numpy as np
import pandas as pd

from .decoder import DecoderConfig, decode
from .network import TripNetwork
from .metrics import evaluate_continuity


class NotConfigured(RuntimeError):
    """A required deployment setting or model file is missing."""


class Predictor:
    def __init__(self, artifacts_dir: Path, adapter: str):
        artifacts_dir = Path(artifacts_dir)

        config_path = artifacts_dir / "model_config.json"
        model_path = artifacts_dir / "model.joblib"

        if not (config_path.is_file() and model_path.is_file()):
            raise NotConfigured(
                f"Missing trained artifacts at {model_path} or {config_path}"
            )

        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        self.features = self.config.get("feature_columns", [])

        if (
            not isinstance(self.features, list)
            or not self.features
            or not all(isinstance(f, str) for f in self.features)
        ):
            raise ValueError(
                "model_config.json must specify a nonempty feature_columns list"
            )

        if "label" in self.features or "true_arc_id" in self.features:
            raise ValueError("Ground truth must never be included in model features")

        self.model = joblib.load(model_path)

        scaler_path = artifacts_dir / "scaler.joblib"

        separate_scaler = (
            self.config.get("preprocessing") != "pipeline" and scaler_path.is_file()
        )
        self.scaler = joblib.load(scaler_path) if separate_scaler else None

        self.decoder_config = DecoderConfig(**self.config.get("decoder", {}))

        self.positive_class = self.config.get("positive_class", 1)

        if ":" not in adapter:
            raise ValueError("PATHSNAP_ADAPTER must be module:function")

        module_name, callable_name = adapter.split(":", 1)

        try:
            self.build_candidates = getattr(import_module(module_name), callable_name)
        except (ImportError, AttributeError) as exc:
            raise NotConfigured(
                f"Could not import candidate adapter {adapter}"
            ) from exc

        if adapter == "inference_adapter:build_candidates":
            radius = float(self.config.get("candidate_radius_metres", 75))
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("Candidate radius must be positive and finite")
            self.build_candidates = partial(self.build_candidates, radius=radius)

        if not callable(self.build_candidates):
            raise TypeError("Inference adapter must be callable")

    def _score(self, table: pd.DataFrame) -> pd.DataFrame:
        required = {"point_idx", "arc_id", *self.features}

        absent = sorted(required - set(table.columns))

        if absent:
            raise ValueError(f"Missing candidate columns: {absent}")

        if table.empty:
            return table[["point_idx", "arc_id"]].assign(
                positive_prob=pd.Series(dtype=float)
            )

        if table.duplicated(["point_idx", "arc_id"]).any():
            raise ValueError("Duplicate candidate arcs for an observation")

        # Use the exact feature order from model_config.json.
        values = table.loc[:, self.features].astype(float)

        # The fitted estimator handles missing features.
        if np.isinf(values.to_numpy()).any():
            raise ValueError("Candidate features contain infinite values")

        if self.scaler is not None and hasattr(self.model, "steps"):
            raise ValueError(
                "A pipeline and a separate scaler cannot both preprocess features"
            )
        data = self.scaler.transform(values) if self.scaler is not None else values

        probs = np.asarray(self.model.predict_proba(data))

        if probs.ndim != 2 or probs.shape[0] != len(table):
            raise ValueError("model.predict_proba must return one row per candidate")

        classes = getattr(self.model, "classes_", None)

        if classes is None:
            raise ValueError(
                "Trained classifier must expose classes_ for positive class selection"
            )

        hits = np.flatnonzero(np.asarray(classes) == self.positive_class)

        if len(hits) != 1:
            raise ValueError(
                f"Positive class {self.positive_class!r} absent from {classes!r}"
            )

        positive = probs[:, int(hits[0])]

        if not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
            raise ValueError("Model returned invalid probabilities")

        scored = table[["point_idx", "arc_id"]].copy()

        scored["positive_prob"] = positive

        return scored

    def prepare_trip(self, trip_dir: Path) -> tuple[TripNetwork, pd.DataFrame]:
        """Load one trip and classify candidates once."""

        trip_dir = Path(trip_dir)

        network = TripNetwork.from_directory(trip_dir)

        # Create an isolated directory containing only
        # public input files, never ground-truth labels.
        with tempfile.TemporaryDirectory(prefix="pathsnap_unlabelled_") as tmp:
            isolated = Path(tmp) / trip_dir.name
            isolated.mkdir()

            for ext in ("track", "nodes", "arcs"):
                file_name = f"{trip_dir.name}.{ext}"

                shutil.copy2(trip_dir / file_name, isolated / file_name)

            candidate_table = self.build_candidates(isolated)

        if not isinstance(candidate_table, pd.DataFrame):
            raise TypeError("Inference adapter must return pandas.DataFrame")

        scored = self._score(candidate_table)

        return network, scored

    def predict(self, trip_dir: Path) -> dict:
        begin = time.perf_counter()

        network, scored = self.prepare_trip(trip_dir)

        classification_s = time.perf_counter() - begin

        start_decoding = time.perf_counter()

        result = decode(scored, network, self.decoder_config)

        decoding_s = time.perf_counter() - start_decoding

        continuity = {
            "classifier": evaluate_continuity(
                network, result.classifier_arc_ids, self.decoder_config
            ),
            "hmm": evaluate_continuity(
                network, result.hmm_arc_ids, self.decoder_config
            ),
        }

        n = len(network.track)

        rows = [
            {
                "point_idx": i,
                "longitude": float(network.track[i, 0]),
                "latitude": float(network.track[i, 1]),
                "timestamp": float(network.track[i, 2]),
                "classifier_arc_id": result.classifier_arc_ids[i],
                "hmm_arc_id": result.hmm_arc_ids[i],
            }
            for i in range(n)
        ]

        return {
            "trip_id": network.trip_id,
            "observation_count": n,
            "candidate_count": len(scored),
            "utm_epsg": network.utm_epsg,
            "observations": rows,
            "predicted_route_lines": network.route_lines(
                result.hmm_arc_ids, result.breaks
            ),
            "decoder_diagnostics": result.diagnostics,
            "route_continuity": continuity,
            "timing_seconds": {
                "features_and_classifier": round(classification_s, 6),
                "hmm_viterbi": round(decoding_s, 6),
                "total": round(time.perf_counter() - begin, 6),
            },
            "decoder_config": asdict(self.decoder_config),
        }


def predictor_from_environment() -> Predictor:
    return Predictor(
        artifacts_dir=Path(os.getenv("PATHSNAP_ARTIFACTS_DIR", "artifacts")),
        adapter=os.getenv("PATHSNAP_ADAPTER", "inference_adapter:build_candidates"),
    )
