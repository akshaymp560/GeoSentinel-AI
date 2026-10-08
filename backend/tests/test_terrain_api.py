import httpx
import pytest

from backend.agents.terrain_api import retrieve_terrain_evidence


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


def fake_get(calls, payload=None):
    def get(url, *, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse(payload if payload is not None else {"elevation": [123.45]})

    return get


def test_successful_elevation_response_contract():
    result = retrieve_terrain_evidence(10.1234, 76.1234, http_get=fake_get([]))
    assert result["agent"] == "terrain"
    assert result["status"] == "success"
    assert result["location"] == {"latitude": 10.1234, "longitude": 76.1234}
    assert result["metrics"] == {"elevation_m": 123.45}
    assert result["source"] == "Open-Meteo elevation DEM"
    assert result["evidence"][0]["unit"] == "m"


def test_correct_metric_extraction_and_request_parameters():
    calls = []
    result = retrieve_terrain_evidence(10, 76, timeout=8, http_get=fake_get(calls, {"elevation": [987.6543]}))
    assert result["metrics"]["elevation_m"] == 987.654
    _, params, timeout = calls[0]
    assert params == {"latitude": 10, "longitude": 76}
    assert timeout == 8


def test_search_area_is_explicit_and_configurable():
    result = retrieve_terrain_evidence(10, 76, radius_km=10, http_get=fake_get([]))
    area = result["search_area"]
    assert area["radius_km"] == 10
    assert area["measurement"] == "point elevation at the investigation coordinate"
    assert area["bounding_box"]["west"] < 76 < area["bounding_box"]["east"]
    assert area["bounding_box"]["south"] < 10 < area["bounding_box"]["north"]


@pytest.mark.parametrize("latitude,longitude", [(91, 76), (10, 181), (float("nan"), 76)])
def test_invalid_coordinates_return_structured_failure(latitude, longitude):
    result = retrieve_terrain_evidence(latitude, longitude, http_get=fake_get([]))
    assert result["status"] == "failed"
    assert result["metrics"] is None
    assert result["evidence"] == []


def test_invalid_radius_returns_failure():
    result = retrieve_terrain_evidence(10, 76, radius_km=0, http_get=fake_get([]))
    assert result["status"] == "failed"
    assert "radius_km" in result["error"]


def test_malformed_response_returns_failure():
    result = retrieve_terrain_evidence(10, 76, http_get=fake_get([], {"unexpected": []}))
    assert result["status"] == "failed"
    assert "missing an elevation" in result["error"]


def test_missing_elevation_value_returns_failure():
    result = retrieve_terrain_evidence(10, 76, http_get=fake_get([], {"elevation": []}))
    assert result["status"] == "failed"
    assert result["metrics"] is None
    assert result["evidence"] == []


def test_http_failure_returns_structured_failure():
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(503, request=request)
    result = retrieve_terrain_evidence(
        10,
        76,
        http_get=lambda *args, **kwargs: FakeResponse(
            error=httpx.HTTPStatusError("unavailable", request=request, response=response)
        ),
    )
    assert result["status"] == "failed"
    assert "HTTP 503" in result["error"]


def test_timeout_and_network_failure():
    timeout_result = retrieve_terrain_evidence(
        10,
        76,
        http_get=lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("timeout")),
    )
    request = httpx.Request("GET", "https://example.test")
    network_result = retrieve_terrain_evidence(
        10,
        76,
        http_get=lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline", request=request)),
    )
    assert timeout_result["status"] == "failed"
    assert "timed out" in timeout_result["error"]
    assert network_result["status"] == "failed"
    assert "request failed" in network_result["error"]


def test_non_numeric_elevation_returns_failure():
    result = retrieve_terrain_evidence(10, 76, http_get=fake_get([], {"elevation": [None]}))
    assert result["status"] == "failed"
    assert "invalid elevation" in result["error"]
