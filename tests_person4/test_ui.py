from fastapi.testclient import TestClient

import ui.app as ui
from pathsnap_person4.api import create_app, get_predictor


def test_selection_calls_the_api_and_renders_map(
    tiny_trip, monkeypatch, tmp_path, fitted_artifacts
):
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tiny_trip.parent))
    monkeypatch.setenv("PATHSNAP_ARTIFACTS_DIR", str(fitted_artifacts))
    monkeypatch.setenv("PATHSNAP_REQUEST_LOG", str(tmp_path / "requests.jsonl"))
    monkeypatch.delenv("PATHSNAP_REFERENCE_LABELS_DIR", raising=False)
    get_predictor.cache_clear()
    backend = TestClient(create_app())
    monkeypatch.setattr(ui, "API", "")
    monkeypatch.setattr(ui.requests, "get", lambda path, timeout: backend.get(path))
    ui_client = ui.app.test_client()
    assert b"12345678" in ui_client.get("/").data
    page = ui_client.post("/", data={"trip": "12345678"}).get_data(as_text=True)
    assert "Predicted route for trip 12345678" in page
    assert "GPS points" in page and "Reference scores unavailable" in page
    assert 'role="alert"' not in page
    assert b'role="alert"' in ui_client.post("/", data={"trip": "99999999"}).data
    get_predictor.cache_clear()
