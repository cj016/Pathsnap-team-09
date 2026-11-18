"""A synthetic end-to-end contract test, NOT an evaluator prediction."""

import json
import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from pathsnap_person4.api import create_app, get_predictor


def test_api_runs_learned_model_and_does_not_expose_reference(monkeypatch, tiny_trip, tmp_path):
    # Test-only feature builder, using publicly available GPS/road files.
    # Reject any attempt to read a reference label or .route during inference.
    module_name = "toy_adapter_private_test"
    (tmp_path / f"{module_name}.py").write_text('''\
from pathlib import Path
import pandas as pd
import numpy as np

def build_candidates(trip_dir):
    trip_dir = Path(trip_dir)
    assert not any(trip_dir.glob("*.route")), "A label file leaked into inference!"
    tid = trip_dir.name
    points = np.loadtxt(trip_dir / (tid + ".track"), ndmin=2)
    arcs = np.loadtxt(trip_dir / (tid + ".arcs"), dtype=int, ndmin=2)
    rows = []
    for i in range(len(points)):
        for a in range(len(arcs)):
            rows.append({"point_idx":i, "arc_id":a, "distance":float(abs(i-a))})
    return pd.DataFrame(rows)
''', encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model = LogisticRegression(random_state=42).fit(
        pd.DataFrame({"distance": [0.0, 0.1, 1.0, 2.0, 3.0, 5.0]}),
        np.array([1, 1, 0, 0, 0, 0]),
    )
    joblib.dump(model, artifacts / "model.joblib")
    (artifacts / "model_config.json").write_text(json.dumps({
        "feature_columns": ["distance"], "positive_class": 1,
    }), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tiny_trip.parent))
    monkeypatch.setenv("PATHSNAP_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("PATHSNAP_ADAPTER", f"{module_name}:build_candidates")
    monkeypatch.setenv("PATHSNAP_REQUEST_LOG", str(tmp_path / "real_requests.jsonl"))
    monkeypatch.delenv("PATHSNAP_REFERENCE_LABELS_DIR", raising=False)
    get_predictor.cache_clear()
    response = TestClient(create_app()).get("/predict/12345678")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["observation_count"] == 3
    assert result["candidate_count"] == 9
    # The actual learned classifier and sequence model must both run.
    # Do not hardcode the synthetic estimator's exact output as a production route.
    assert all(isinstance(x["hmm_arc_id"], int) for x in result["observations"])
    assert result["route_continuity"]["hmm"]["rate"] == 1.0
    assert result["metrics"]["status"] == "unavailable"
    assert "ground_truth" not in response.text
    assert "reference_route" not in response.text
    assert len(result["predicted_route_lines"]) == 1
    get_predictor.cache_clear()
