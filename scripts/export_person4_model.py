"""Fit the team LightGBM on allowed training cache; export for Person 4 API."""

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from features.data_preparation import prepare_xy
from features.feature_engineering import FEATURE_COLUMNS
from model_experiments.models import make_boosting, make_mlp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path, help="e.g. processed/eval1-seed42")
    p.add_argument("output", type=Path, help="e.g. artifacts")
    p.add_argument("--model", choices=["lightgbm", "mlp"], default="lightgbm")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    files = sorted((a.cache / "train").glob("*.candidates.parquet"))
    if not files:
        raise SystemExit(
            "Missing training cache: run model_experiments.checkpoint prepare first"
        )
    training = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    X, y = prepare_xy(training, FEATURE_COLUMNS)
    estimator = make_boosting(a.seed) if a.model == "lightgbm" else make_mlp(a.seed)
    estimator.fit(X, y)
    a.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, a.output / "model.joblib")
    metadata = a.cache / "cache_config.json"
    radius = (
        json.loads(metadata.read_text())["candidate_radius_metres"]
        if metadata.is_file()
        else 75
    )
    config = {
        "model": a.model,
        "feature_columns": FEATURE_COLUMNS,
        "positive_class": 1,
        "seed": a.seed,
        "candidate_radius_metres": radius,
        "preprocessing": "pipeline",
    }
    (a.output / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"Exported {a.model} on {len(X)} candidate training rows / {len(files)} trips -> {a.output}"
    )


if __name__ == "__main__":
    main()
