"""GDELT news evidence agent.

The agent uses GDELT DOC 2.1's bounded article-list search. The default query
uses GDELT's ``near`` geographic context around the coordinate; this is a
relevance hint, not proof that an article describes the exact point. Results
are factual reports only and are not causal conclusions.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time
from typing import Any, Callable

import httpx

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_RADIUS_KM = 50.0
DEFAULT_MAX_RESULTS = 25
MAX_RESULTS = 250


def _failure(
    error: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    before_date: str | None = None,
    after_date: str | None = None,
    query_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "agent": "news",
        "status": "failed",
        "error": error,
        "metrics": None,
        "articles": [],
        "evidence": [],
        "source": "GDELT DOC 2.1",
    }
    if latitude is not None and longitude is not None:
        result["location"] = {"latitude": latitude, "longitude": longitude}
    if before_date is not None and after_date is not None:
        result["period"] = {"before_date": before_date, "after_date": after_date}
    if query_context is not None:
        result["query_context"] = query_context
    return result


def _parse_date(value: date | str, field: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid ISO date") from exc


def _validate(
    latitude: float,
    longitude: float,
    before_date: date | str,
    after_date: date | str,
    radius_km: float,
    max_results: int,
) -> tuple[date, date] | str:
    if not isinstance(latitude, (int, float)) or not math.isfinite(latitude) or not -90 <= latitude <= 90:
        return "latitude must be between -90 and 90"
    if not isinstance(longitude, (int, float)) or not math.isfinite(longitude) or not -180 <= longitude <= 180:
        return "longitude must be between -180 and 180"
    if not isinstance(radius_km, (int, float)) or not math.isfinite(radius_km) or not 0 < radius_km <= 500:
        return "radius_km must be greater than 0 and at most 500"
    if not isinstance(max_results, int) or not 1 <= max_results <= MAX_RESULTS:
        return f"max_results must be between 1 and {MAX_RESULTS}"
    try:
        before = _parse_date(before_date, "before_date")
        after = _parse_date(after_date, "after_date")
    except ValueError as exc:
        return str(exc)
    if before > after:
        return "before_date must be earlier than or equal to after_date"
    return before, after


def _article_date(value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError("article is missing publication date")
    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.date()
        except ValueError:
            continue
    raise ValueError("article has an invalid publication date")


def _extract_articles(payload: Any, before: date, after: date) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
        raise ValueError("response is missing the articles list")
    articles = []
    for raw in payload["articles"]:
        if not isinstance(raw, dict):
            raise ValueError("response contains an invalid article")
        title = raw.get("title")
        url = raw.get("url")
        published_value = raw.get("seendate") or raw.get("published_at")
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str) or not url.strip():
            raise ValueError("article is missing a title or URL")
        published = _article_date(published_value)
        if not before <= published <= after:
            continue
        articles.append(
            {
                "title": title.strip(),
                "source": raw.get("domain") or raw.get("sourcecountry") or "Unknown source",
                "published_at": published_value,
                "url": url.strip(),
                "snippet": raw.get("snippet") or raw.get("description"),
            }
        )
    return articles


def retrieve_news_evidence(
    latitude: float,
    longitude: float,
    before_date: date | str,
    after_date: date | str,
    *,
    radius_km: float = DEFAULT_RADIUS_KM,
    max_results: int = DEFAULT_MAX_RESULTS,
    location_context: str | None = None,
    timeout: float = 15.0,
    http_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Search bounded GDELT news context for an inclusive date period."""
    before_text = before_date.isoformat() if isinstance(before_date, date) else str(before_date)
    after_text = after_date.isoformat() if isinstance(after_date, date) else str(after_date)
    validation = _validate(latitude, longitude, before_date, after_date, radius_km, max_results)
    query = location_context.strip() if isinstance(location_context, str) and location_context.strip() else f"near:{latitude},{longitude},{radius_km:g}km"
    context = {
        "strategy": "GDELT near geographic relevance hint",
        "query": query,
        "radius_km": radius_km,
        "max_results": max_results,
        "coordinate_precision": "contextual relevance only; not exact article geolocation",
    }
    if isinstance(validation, str):
        return _failure(validation, latitude=latitude if isinstance(latitude, (int, float)) else None, longitude=longitude if isinstance(longitude, (int, float)) else None, before_date=before_text, after_date=after_text, query_context=context)
    before, after = validation
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_results,
        "sort": "datedesc",
        "startdatetime": f"{before.isoformat().replace('-', '')}000000",
        "enddatetime": f"{after.isoformat().replace('-', '')}235959",
        "format": "json",
    }
    getter = http_get or httpx.get
    try:
        response = getter(GDELT_DOC_URL, params=params, timeout=timeout)
        response.raise_for_status()
        articles = _extract_articles(response.json(), before, after)
    except httpx.TimeoutException:
        return _failure("GDELT request timed out", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, query_context=context)
    except httpx.HTTPStatusError as exc:
        return _failure(f"GDELT returned HTTP {exc.response.status_code}", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, query_context=context)
    except httpx.RequestError as exc:
        return _failure(f"GDELT request failed: {exc}", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, query_context=context)
    except (ValueError, TypeError, KeyError) as exc:
        return _failure(f"Malformed GDELT response: {exc}", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, query_context=context)
    except Exception as exc:  # noqa: BLE001 - news is an optional evidence source
        return _failure(f"News retrieval failed: {exc}", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, query_context=context)

    evidence = [{
        "type": "news_reports",
        "description": f"{len(articles)} relevant public report(s) returned by the configured GDELT search context",
        "article_count": len(articles),
    }] if articles else [{
        "type": "news_reports",
        "description": "No relevant public reports were found within the configured search context and period",
        "article_count": 0,
    }]
    return {
        "agent": "news",
        "status": "success",
        "location": {"latitude": latitude, "longitude": longitude},
        "period": {"before_date": before.isoformat(), "after_date": after.isoformat()},
        "query_context": context,
        "metrics": {"article_count": len(articles)},
        "articles": articles,
        "evidence": evidence,
        "source": "GDELT DOC 2.1",
    }
