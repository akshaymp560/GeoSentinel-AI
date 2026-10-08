from datetime import date

import httpx
import pytest

from backend.agents.weather_api import retrieve_weather_evidence


def weather_payload():
    return {
        "daily": {
            "time": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "precipitation_sum": [0.0, 1.5, 2.0],
            "temperature_2m_mean": [20.0, 22.0, 21.0],
            "temperature_2m_max": [25.0, 26.0, 27.0],
            "wind_speed_10m_max": [10.0, 12.0, 11.0],
        }
    }


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


def successful_get(calls):
    def get(url, *, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse(weather_payload())

    return get


def test_successful_response_schema_and_source():
    result = retrieve_weather_evidence(10.1234, 76.1234, "2025-01-01", "2025-01-03", http_get=successful_get([]))

    assert result["status"] == "success"
    assert result["agent"] == "weather"
    assert result["source"] == "Open-Meteo"
    assert result["location"] == {"latitude": 10.1234, "longitude": 76.1234}
    assert result["period"] == {"before_date": "2025-01-01", "after_date": "2025-01-03"}
    assert isinstance(result["evidence"], list)
    assert result["metrics"] is not None


def test_metric_extraction_is_correct():
    result = retrieve_weather_evidence(10, 76, date(2025, 1, 1), date(2025, 1, 3), http_get=successful_get([]))

    assert result["metrics"] == {
        "total_precipitation_mm": 3.5,
        "mean_temperature_c": 21.0,
        "max_temperature_c": 27.0,
        "wet_days": 2,
        "observed_days": 3,
        "max_wind_speed_kmh": 12.0,
    }


def test_date_and_coordinate_parameters_are_sent_to_api():
    calls = []
    result = retrieve_weather_evidence(10.5, 76.5, "2025-02-01", "2025-02-05", http_get=successful_get(calls))

    assert result["status"] == "success"
    _, params, timeout = calls[0]
    assert params["latitude"] == 10.5
    assert params["longitude"] == 76.5
    assert params["start_date"] == "2025-02-01"
    assert params["end_date"] == "2025-02-05"
    assert "precipitation_sum" in params["daily"]
    assert timeout == 15.0


@pytest.mark.parametrize("latitude,longitude", [(91, 76), (10, 181), (float("nan"), 76)])
def test_invalid_coordinates_return_structured_failure(latitude, longitude):
    result = retrieve_weather_evidence(latitude, longitude, "2025-01-01", "2025-01-02")
    assert result["status"] == "failed"
    assert result["metrics"] is None
    assert result["evidence"] == []
    assert "latitude" in result["error"] or "longitude" in result["error"]


def test_invalid_date_format_returns_failure():
    result = retrieve_weather_evidence(10, 76, "01-01-2025", "2025-01-02")
    assert result["status"] == "failed"
    assert "valid ISO date" in result["error"]


def test_invalid_date_range_returns_failure():
    result = retrieve_weather_evidence(10, 76, "2025-01-03", "2025-01-02")
    assert result["status"] == "failed"
    assert "earlier than or equal" in result["error"]


def test_http_failure_returns_structured_failure():
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(503, request=request)
    result = retrieve_weather_evidence(
        10, 76, "2025-01-01", "2025-01-02",
        http_get=lambda *args, **kwargs: FakeResponse(
            error=httpx.HTTPStatusError("unavailable", request=request, response=response)
        ),
    )
    assert result["status"] == "failed"
    assert result["metrics"] is None
    assert result["evidence"] == []
    assert "HTTP 503" in result["error"]


def test_timeout_returns_structured_failure():
    result = retrieve_weather_evidence(
        10, 76, "2025-01-01", "2025-01-02",
        http_get=lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("timed out")),
    )
    assert result["status"] == "failed"
    assert "timed out" in result["error"]


def test_network_failure_returns_structured_failure():
    request = httpx.Request("GET", "https://example.test")
    result = retrieve_weather_evidence(
        10, 76, "2025-01-01", "2025-01-02",
        http_get=lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline", request=request)),
    )
    assert result["status"] == "failed"
    assert "request failed" in result["error"]


def test_malformed_response_and_missing_fields_fail_cleanly():
    for payload in [{}, {"daily": {"time": ["2025-01-01"]}}]:
        result = retrieve_weather_evidence(
            10, 76, "2025-01-01", "2025-01-01",
            http_get=lambda *args, **kwargs: FakeResponse(payload),
        )
        assert result["status"] == "failed"
        assert result["metrics"] is None
        assert result["evidence"] == []
        assert "Malformed" in result["error"]
