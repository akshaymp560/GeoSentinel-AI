from pathlib import Path

import httpx
import pytest

from backend.agents.fire_api import retrieve_fire_evidence


FIRE_CSV = """latitude,longitude,acq_date,acq_time,confidence,frp,satellite,instrument,daynight,version
10.1234,76.1234,2025-01-02,1030,high,42.5,Suomi-NPP,VIIRS,D,2.0NRT
10.1240,76.1240,2025-01-04,1115,nominal,12.0,Suomi-NPP,VIIRS,D,2.0NRT
"""


class FakeResponse:
    def __init__(self, text="", error=None):
        self.text = text
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error


def successful_get(calls, text=FIRE_CSV):
    def get(url, *, timeout):
        calls.append((url, timeout))
        return FakeResponse(text=text)

    return get


def test_successful_response_contract_and_metadata(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "test-key")
    result = retrieve_fire_evidence(10.1234, 76.1234, "2025-01-01", "2025-01-05", http_get=successful_get([]))

    assert result["agent"] == "fire"
    assert result["status"] == "success"
    assert result["source"] == "NASA FIRMS"
    assert result["location"] == {"latitude": 10.1234, "longitude": 76.1234}
    assert result["period"] == {"before_date": "2025-01-01", "after_date": "2025-01-05"}
    assert result["metrics"]["detection_count"] == 2
    assert result["detections"][0]["confidence"] == "high"
    assert result["detections"][0]["frp"] == 42.5
    assert result["detections"][0]["satellite"] == "Suomi-NPP"
    assert result["detections"][0]["distance_km"] == 0.0
    assert result["evidence"]


def test_detection_count_and_metrics():
    result = retrieve_fire_evidence(10.1234, 76.1234, "2025-01-01", "2025-01-05", firms_map_key="key", http_get=successful_get([]))
    assert result["metrics"] == {
        "detection_count": 2,
        "detection_dates": ["2025-01-02", "2025-01-04"],
        "confidence_counts": {"high": 1, "nominal": 1},
        "nearest_detection_km": 0.0,
    }


def test_date_period_is_exact_and_long_period_is_chunked():
    calls = []
    result = retrieve_fire_evidence(10, 76, "2025-01-01", "2025-01-06", firms_map_key="key", http_get=successful_get(calls))
    assert result["status"] == "success"
    assert len(calls) == 2
    assert "/5/2025-01-01" in calls[0][0]
    assert "/1/2025-01-06" in calls[1][0]


def test_search_area_is_bounded_and_configurable():
    result = retrieve_fire_evidence(10, 76, "2025-01-01", "2025-01-01", radius_km=10, firms_map_key="key", http_get=successful_get([]))
    box = result["search_area"]["bounding_box"]
    assert result["search_area"]["radius_km"] == 10
    assert box["west"] < 76 < box["east"]
    assert box["south"] < 10 < box["north"]
    assert result["search_area"]["firms_area_parameter"].count(",") == 3


def test_spatial_filter_excludes_detection_outside_search_box():
    csv_text = FIRE_CSV + "10.5,76.5,2025-01-03,1200,high,20,Suomi-NPP,VIIRS,D,2\n"
    result = retrieve_fire_evidence(10.1234, 76.1234, "2025-01-01", "2025-01-05", radius_km=1, firms_map_key="key", http_get=successful_get([], csv_text))
    assert result["metrics"]["detection_count"] == 2


def test_zero_detection_is_success():
    result = retrieve_fire_evidence(10, 76, "2025-01-01", "2025-01-01", firms_map_key="key", http_get=successful_get([], "latitude,longitude,acq_date\n"))
    assert result["status"] == "success"
    assert result["metrics"]["detection_count"] == 0
    assert result["detections"] == []
    assert "No FIRMS detections" in result["evidence"][0]["description"]


@pytest.mark.parametrize("latitude,longitude", [(91, 76), (10, 181), (float("nan"), 76)])
def test_invalid_coordinates(latitude, longitude):
    result = retrieve_fire_evidence(latitude, longitude, "2025-01-01", "2025-01-01", firms_map_key="key")
    assert result["status"] == "failed"
    assert result["metrics"] is None
    assert result["detections"] == []


def test_invalid_date_range():
    result = retrieve_fire_evidence(10, 76, "2025-01-02", "2025-01-01", firms_map_key="key")
    assert result["status"] == "failed"
    assert "earlier than or equal" in result["error"]


def test_missing_map_key_is_configuration_failure(monkeypatch):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    result = retrieve_fire_evidence(10, 76, "2025-01-01", "2025-01-01")
    assert result["status"] == "failed"
    assert "FIRMS_MAP_KEY" in result["error"]


def test_http_failure():
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(403, request=request)
    result = retrieve_fire_evidence(
        10, 76, "2025-01-01", "2025-01-01", firms_map_key="key",
        http_get=lambda *args, **kwargs: FakeResponse(error=httpx.HTTPStatusError("forbidden", request=request, response=response)),
    )
    assert result["status"] == "failed"
    assert "HTTP 403" in result["error"]


def test_timeout_and_network_failure():
    timeout_result = retrieve_fire_evidence(
        10, 76, "2025-01-01", "2025-01-01", firms_map_key="key",
        http_get=lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("timeout")),
    )
    request = httpx.Request("GET", "https://example.test")
    network_result = retrieve_fire_evidence(
        10, 76, "2025-01-01", "2025-01-01", firms_map_key="key",
        http_get=lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline", request=request)),
    )
    assert timeout_result["status"] == "failed"
    assert "timed out" in timeout_result["error"]
    assert network_result["status"] == "failed"
    assert "request failed" in network_result["error"]


def test_malformed_and_missing_required_fields():
    for text in ["not,csv", "latitude,longitude\n10,76\n", "latitude,longitude,acq_date\nnot,76,2025-01-01\n"]:
        result = retrieve_fire_evidence(10, 76, "2025-01-01", "2025-01-01", firms_map_key="key", http_get=successful_get([], text))
        assert result["status"] == "failed"
        assert result["detections"] == []
        assert "Malformed" in result["error"]


def test_invalid_date_format_and_radius():
    invalid_date = retrieve_fire_evidence(10, 76, "01-01-2025", "2025-01-01", firms_map_key="key")
    invalid_radius = retrieve_fire_evidence(10, 76, "2025-01-01", "2025-01-01", radius_km=0, firms_map_key="key")
    assert invalid_date["status"] == "failed"
    assert "valid ISO date" in invalid_date["error"]
    assert invalid_radius["status"] == "failed"
    assert "radius_km" in invalid_radius["error"]
