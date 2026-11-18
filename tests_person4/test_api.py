from fastapi.testclient import TestClient

from pathsnap_person4.api import create_app, get_predictor


def test_trips_real_files_but_no_fake_prediction(monkeypatch, tiny_trip, tmp_path):
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tiny_trip.parent))
    monkeypatch.setenv("PATHSNAP_ARTIFACTS_DIR", str(tmp_path / "missing_artifacts"))
    monkeypatch.setenv("PATHSNAP_REQUEST_LOG", str(tmp_path / "requests.jsonl"))
    get_predictor.cache_clear()
    client = TestClient(create_app())
    listing = client.get("/trips")
    assert listing.status_code == 200
    assert listing.json() == {"trip_ids": ["12345678"], "count": 1}
    assert client.get("/health").json()["status"] == "not_ready"
    response = client.get("/predict/12345678")
    assert response.status_code == 503
    assert "trained artifacts" in response.json()["detail"]
    assert client.get("/predict/hi").status_code == 400
    assert "not_configured" in (tmp_path / "requests.jsonl").read_text()
