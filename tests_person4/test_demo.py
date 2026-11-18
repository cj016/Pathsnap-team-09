import json

import joblib
import numpy as np
import pandas as pd
import pytest

from evaluation.baseline import nearest_candidate_top1
from evaluation.metrics import top1_accuracy
from features.data_preparation import training_rows
from features.feature_engineering import FEATURE_COLUMNS
from inference_adapter import build_candidates
from model_experiments.models import make_mlp
from pathsnap_person4.metrics import load_point_labels
from pathsnap_person4.pipeline import Predictor


def test_missing_reference_with_other_candidates():
    inventory = pd.DataFrame(
        {
            "trip_id": ["t"] * 3,
            "point_idx": [0, 1, 2],
            "reference_arc_id": [5, 6, 7],
            "label_uncertain": [False, True, False],
        }
    )
    rows = pd.DataFrame(
        {
            "trip_id": ["t", "t"],
            "point_idx": [0, 1],
            "arc_id": [5, 8],
            "dist_rank": [1, 1],
            "probability_label_1": [0.9, 0.6],
        }
    )
    for score in [nearest_candidate_top1, top1_accuracy]:
        assert score(rows, inventory, include_uncertain=True)["candidate_misses"] == 2
        assert score(rows, inventory)["candidate_misses"] == 1
        assert score(rows, inventory, include_uncertain=True)["correct_points"] == 1


def test_uncertain_point_excluded_as_a_group():
    rows = pd.DataFrame(
        {
            "trip_id": ["t"] * 4,
            "point_idx": [0, 0, 1, 1],
            "label": [1, 0, 1, 0],
            "label_uncertain": [False, True, False, False],
        }
    )
    assert training_rows(rows).point_idx.tolist() == [1, 1]


def test_no_candidates_and_no_reference(tiny_trip, fitted_artifacts):
    (tiny_trip / "12345678.route").unlink()
    (tiny_trip / "12345678.track").write_text("0.5 0.5 1\n0.5 0.5 2\n")
    table = build_candidates(tiny_trip)
    assert table.empty and list(table.columns) == [
        "point_idx",
        "arc_id",
        *FEATURE_COLUMNS,
    ]
    predictor = Predictor(fitted_artifacts, "inference_adapter:build_candidates")
    result = predictor.predict(tiny_trip)
    assert result["candidate_count"] == 0
    assert all(row["hmm_arc_id"] is None for row in result["observations"])
    assert result["predicted_route_lines"] == []


def test_mlp_uses_its_fitted_imputer(tmp_path):
    model = make_mlp().set_params(model__max_iter=2, model__batch_size=4)
    model.fit(pd.DataFrame({"speed": [1.0, 2.0, 3.0, 4.0]}), [0, 0, 1, 1])
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "model_config.json").write_text(
        json.dumps({"feature_columns": ["speed"]})
    )
    predictor = Predictor(tmp_path, "inference_adapter:build_candidates")
    row = pd.DataFrame({"point_idx": [0], "arc_id": [0], "speed": [np.nan]})
    assert 0 <= predictor._score(row).positive_prob.iloc[0] <= 1


@pytest.mark.parametrize("arc", ["-1", "3.2", ""])
def test_bad_reference_cannot_score_a_missing_prediction(tmp_path, arc):
    path = tmp_path / "trip.csv"
    path.write_text(f"point_idx,arc_id\n0,{arc}\n")
    with pytest.raises(ValueError):
        load_point_labels(path, 1)
