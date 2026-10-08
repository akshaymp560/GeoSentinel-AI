from datetime import date

import httpx
import pytest

from backend.agents.news_api import retrieve_news_evidence


ARTICLES = {
    "articles": [
        {
            "title": "River monitoring report",
            "domain": "example.org",
            "seendate": "20250102123000",
            "url": "https://example.org/river",
            "snippet": "A public report about the local area.",
        },
        {
            "title": "Older report",
            "domain": "old.example.org",
            "seendate": "20241231120000",
            "url": "https://old.example.org/report",
        },
    ]
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


def fake_get(calls, payload=ARTICLES):
    def get(url, *, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse(payload=payload)

    return get


def test_successful_response_contract():
    result = retrieve_news_evidence(10.1234, 76.1234, "2025-01-01", "2025-01-03", http_get=fake_get([]))
    assert result["agent"] == "news"
    assert result["status"] == "success"
    assert result["source"] == "GDELT DOC 2.1"
    assert result["location"] == {"latitude": 10.1234, "longitude": 76.1234}
    assert result["period"] == {"before_date": "2025-01-01", "after_date": "2025-01-03"}
    assert result["metrics"] == {"article_count": 1}
    assert result["evidence"]


def test_article_extraction():
    result = retrieve_news_evidence(10, 76, "2025-01-01", "2025-01-03", http_get=fake_get([]))
    assert result["articles"] == [
        {
            "title": "River monitoring report",
            "source": "example.org",
            "published_at": "20250102123000",
            "url": "https://example.org/river",
            "snippet": "A public report about the local area.",
        }
    ]


def test_date_filtering_and_exact_query_parameters():
    calls = []
    result = retrieve_news_evidence(10, 76, "2025-01-01", "2025-01-03", http_get=fake_get(calls))
    assert result["status"] == "success"
    _, params, timeout = calls[0]
    assert params["startdatetime"] == "20250101000000"
    assert params["enddatetime"] == "20250103235959"
    assert params["mode"] == "artlist"
    assert timeout == 15.0


def test_configurable_result_limit_and_location_context():
    calls = []
    result = retrieve_news_evidence(
        10, 76, "2025-01-01", "2025-01-03", radius_km=25, max_results=7,
        location_context="Kerala flood", http_get=fake_get(calls),
    )
    assert result["query_context"]["query"] == "Kerala flood"
    assert result["query_context"]["radius_km"] == 25
    assert result["query_context"]["max_results"] == 7
    assert result["query_context"]["coordinate_precision"].startswith("contextual")
    assert calls[0][1]["maxrecords"] == 7


def test_default_geographic_query_context():
    result = retrieve_news_evidence(10.5, 76.5, "2025-01-01", "2025-01-01", http_get=fake_get([]))
    assert result["query_context"]["query"] == "near:10.5,76.5,50km"


def test_date_objects_are_normalized_to_iso_period():
    result = retrieve_news_evidence(10, 76, date(2025, 1, 1), date(2025, 1, 1), http_get=fake_get([]))
    assert result["period"] == {"before_date": "2025-01-01", "after_date": "2025-01-01"}


def test_zero_result_is_success():
    result = retrieve_news_evidence(10, 76, "2025-01-01", "2025-01-01", http_get=fake_get([], {"articles": []}))
    assert result["status"] == "success"
    assert result["metrics"]["article_count"] == 0
    assert result["articles"] == []
    assert "No relevant public reports" in result["evidence"][0]["description"]


@pytest.mark.parametrize("latitude,longitude", [(91, 76), (10, 181), (float("nan"), 76)])
def test_invalid_coordinates(latitude, longitude):
    result = retrieve_news_evidence(latitude, longitude, "2025-01-01", "2025-01-01")
    assert result["status"] == "failed"
    assert result["metrics"] is None
    assert result["articles"] == []


def test_invalid_date_range_and_format():
    reversed_result = retrieve_news_evidence(10, 76, "2025-01-02", "2025-01-01")
    format_result = retrieve_news_evidence(10, 76, "01-01-2025", "2025-01-01")
    assert reversed_result["status"] == "failed"
    assert "earlier than or equal" in reversed_result["error"]
    assert format_result["status"] == "failed"
    assert "valid ISO date" in format_result["error"]


def test_invalid_spatial_and_result_parameters():
    radius_result = retrieve_news_evidence(10, 76, "2025-01-01", "2025-01-01", radius_km=0)
    limit_result = retrieve_news_evidence(10, 76, "2025-01-01", "2025-01-01", max_results=251)
    assert radius_result["status"] == "failed"
    assert "radius_km" in radius_result["error"]
    assert limit_result["status"] == "failed"
    assert "max_results" in limit_result["error"]


def test_http_failure():
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(503, request=request)
    result = retrieve_news_evidence(
        10, 76, "2025-01-01", "2025-01-01",
        http_get=lambda *args, **kwargs: FakeResponse(error=httpx.HTTPStatusError("down", request=request, response=response)),
    )
    assert result["status"] == "failed"
    assert "HTTP 503" in result["error"]


def test_timeout_and_network_failure():
    timeout_result = retrieve_news_evidence(
        10, 76, "2025-01-01", "2025-01-01",
        http_get=lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("timeout")),
    )
    request = httpx.Request("GET", "https://example.test")
    network_result = retrieve_news_evidence(
        10, 76, "2025-01-01", "2025-01-01",
        http_get=lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline", request=request)),
    )
    assert timeout_result["status"] == "failed"
    assert "timed out" in timeout_result["error"]
    assert network_result["status"] == "failed"
    assert "request failed" in network_result["error"]


def test_malformed_and_missing_fields():
    for payload in [{}, {"not_articles": []}, {"articles": [{"title": "Missing URL", "seendate": "20250101120000"}]}]:
        result = retrieve_news_evidence(10, 76, "2025-01-01", "2025-01-01", http_get=fake_get([], payload))
        assert result["status"] == "failed"
        assert result["articles"] == []
        assert "Malformed" in result["error"]


def test_invalid_publication_date_is_structured_failure():
    payload = {"articles": [{"title": "Bad date", "url": "https://example.org", "seendate": "unknown"}]}
    result = retrieve_news_evidence(10, 76, "2025-01-01", "2025-01-01", http_get=fake_get([], payload))
    assert result["status"] == "failed"
    assert "publication date" in result["error"]
