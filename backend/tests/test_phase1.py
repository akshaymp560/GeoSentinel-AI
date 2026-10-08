import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.main import app
from backend.services.mock_change_detector import run_mock_change_detection


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def payload():
    return {"latitude": 10.1234, "longitude": 76.1234, "before_date": "2025-01-01", "after_date": "2025-02-01"}


def test_create_and_complete(client, tmp_path):
    response = client.post("/api/investigations", json=payload())
    assert response.status_code == 202
    data = response.json()
    assert data["investigation_id"].startswith("GS-")
    assert data["status"] == "processing"
    investigation_id = data["investigation_id"]
    result = client.get(f"/api/investigations/{investigation_id}")
    assert result.status_code == 200
    assert result.json()["status"] == "completed"
    assert result.json()["result"]["change_detection"]["success"] is True


@pytest.mark.parametrize("field,value", [("latitude", 91), ("longitude", 181)])
def test_invalid_coordinates(client, field, value):
    body = payload()
    body[field] = value
    assert client.post("/api/investigations", json=body).status_code == 422


def test_invalid_dates_and_missing_field(client):
    body = payload()
    body["before_date"], body["after_date"] = body["after_date"], body["before_date"]
    assert client.post("/api/investigations", json=body).status_code == 422
    body = payload()
    del body["latitude"]
    assert client.post("/api/investigations", json=body).status_code == 422


def test_status_and_unknown_investigation(client):
    created = client.post("/api/investigations", json=payload()).json()
    assert client.get(f"/api/investigations/{created['investigation_id']}/status").status_code == 200
    assert client.get("/api/investigations/GS-UNKNOWN/status").status_code == 404


def test_mock_contract(tmp_path):
    result = run_mock_change_detection("before.tif", "after.tif", tmp_path)
    assert set(result) == {"success", "mask_path", "changed_pixels", "total_pixels", "change_percentage"}
    assert result["success"] is True


def test_background_failure_is_persisted(client, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("mock detector failed")

    monkeypatch.setattr(main, "run_change_detection", fail)
    created = client.post("/api/investigations", json=payload()).json()
    result = client.get(f"/api/investigations/{created['investigation_id']}")
    assert result.status_code == 200
    assert result.json()["status"] == "failed"
    assert result.json()["error"] == "mock detector failed"
