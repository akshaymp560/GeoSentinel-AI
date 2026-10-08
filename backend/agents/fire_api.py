"""NASA FIRMS fire-detection evidence agent.

This standalone agent reports remotely detected thermal anomalies and metadata;
it does not label them confirmed wildfires or infer a cause for satellite
change. FIRMS area requests are bounded to a configurable radius and are split
into exact five-day-or-shorter chunks because that is the API's area-query
limit.
"""

from __future__ import annotations

import csv
import io
import math
import os
from datetime import date, timedelta
from typing import Any, Callable

import httpx

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DEFAULT_SOURCE = "VIIRS_SNPP_SP"
DEFAULT_RADIUS_KM = 5.0
MAX_API_DAY_RANGE = 5
EARTH_RADIUS_KM = 6371.0088
REQUIRED_COLUMNS = {"latitude", "longitude", "acq_date"}


def _failure(
    error: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    before_date: str | None = None,
    after_date: str | None = None,
    search_area: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "agent": "fire",
        "status": "failed",
        "error": error,
        "metrics": None,
        "detections": [],
        "evidence": [],
        "source": "NASA FIRMS",
    }
    if latitude is not None and longitude is not None:
        result["location"] = {"latitude": latitude, "longitude": longitude}
    if before_date is not None and after_date is not None:
        result["period"] = {"before_date": before_date, "after_date": after_date}
    if search_area is not None:
        result["search_area"] = search_area
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
) -> tuple[date, date] | str:
    if not isinstance(latitude, (int, float)) or not math.isfinite(latitude) or not -90 <= latitude <= 90:
        return "latitude must be between -90 and 90"
    if not isinstance(longitude, (int, float)) or not math.isfinite(longitude) or not -180 <= longitude <= 180:
        return "longitude must be between -180 and 180"
    if not isinstance(radius_km, (int, float)) or not math.isfinite(radius_km) or not 0 < radius_km <= 500:
        return "radius_km must be greater than 0 and at most 500"
    try:
        before = _parse_date(before_date, "before_date")
        after = _parse_date(after_date, "after_date")
    except ValueError as exc:
        return str(exc)
    if before > after:
        return "before_date must be earlier than or equal to after_date"
    return before, after


def _search_area(latitude: float, longitude: float, radius_km: float) -> dict[str, Any]:
    lat_delta = radius_km / 111.32
    cosine = abs(math.cos(math.radians(latitude)))
    lon_delta = 180.0 if cosine < 1e-12 else min(180.0, radius_km / (111.32 * cosine))
    west = max(-180.0, longitude - lon_delta)
    east = min(180.0, longitude + lon_delta)
    south = max(-90.0, latitude - lat_delta)
    north = min(90.0, latitude + lat_delta)
    return {
        "radius_km": radius_km,
        "bounding_box": {"west": west, "south": south, "east": east, "north": north},
        "firms_area_parameter": f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}",
    }


def _date_chunks(start: date, end: date):
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=MAX_API_DAY_RANGE - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def _as_float(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _distance_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    lat_a, lat_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_lat = math.radians(latitude_b - latitude_a)
    delta_lon = math.radians(longitude_b - longitude_a)
    haversine = math.sin(delta_lat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(haversine)))


def _parse_csv(text: Any, latitude: float, longitude: float, start: date, end: date, area: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("FIRMS response was empty")
    reader = csv.DictReader(io.StringIO(text))
    fields = set(reader.fieldnames or [])
    if not REQUIRED_COLUMNS.issubset(fields):
        raise ValueError("FIRMS response is missing required detection fields")
    box = area["bounding_box"]
    detections = []
    for row in reader:
        detection_lat = _as_float(row.get("latitude"))
        detection_lon = _as_float(row.get("longitude"))
        acquired = row.get("acq_date")
        if detection_lat is None or detection_lon is None or not acquired:
            raise ValueError("FIRMS response contains an invalid detection row")
        try:
            acquired_date = date.fromisoformat(acquired)
        except ValueError as exc:
            raise ValueError("FIRMS response contains an invalid acquisition date") from exc
        if not (start <= acquired_date <= end):
            continue
        if not (box["west"] <= detection_lon <= box["east"] and box["south"] <= detection_lat <= box["north"]):
            continue
        detection = {
            "latitude": detection_lat,
            "longitude": detection_lon,
            "acq_date": acquired,
            "acq_time": row.get("acq_time"),
            "confidence": row.get("confidence"),
            "frp": _as_float(row.get("frp")),
            "distance_km": round(_distance_km(latitude, longitude, detection_lat, detection_lon), 3),
        }
        for field in ("satellite", "instrument", "daynight", "version"):
            if row.get(field) not in (None, ""):
                detection[field] = row[field]
        detections.append(detection)
    return detections


def _metrics(detections: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted({item["acq_date"] for item in detections})
    distances = [item["distance_km"] for item in detections]
    confidence_counts: dict[str, int] = {}
    for item in detections:
        confidence = item.get("confidence")
        if confidence not in (None, ""):
            key = str(confidence)
            confidence_counts[key] = confidence_counts.get(key, 0) + 1
    result = {
        "detection_count": len(detections),
        "detection_dates": dates,
        "confidence_counts": confidence_counts,
    }
    if distances:
        result["nearest_detection_km"] = min(distances)
    return result


def retrieve_fire_evidence(
    latitude: float,
    longitude: float,
    before_date: date | str,
    after_date: date | str,
    *,
    radius_km: float = DEFAULT_RADIUS_KM,
    firms_map_key: str | None = None,
    source: str = DEFAULT_SOURCE,
    timeout: float = 15.0,
    http_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Retrieve FIRMS detections in a bounded area and inclusive date period."""
    before_text = before_date.isoformat() if isinstance(before_date, date) else str(before_date)
    after_text = after_date.isoformat() if isinstance(after_date, date) else str(after_date)
    validation = _validate(latitude, longitude, before_date, after_date, radius_km)
    area = _search_area(latitude, longitude, radius_km) if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)) else None
    if isinstance(validation, str):
        return _failure(validation, latitude=latitude if isinstance(latitude, (int, float)) else None, longitude=longitude if isinstance(longitude, (int, float)) else None, before_date=before_text, after_date=after_text, search_area=area)
    if not firms_map_key and not os.getenv("FIRMS_MAP_KEY"):
        return _failure("FIRMS_MAP_KEY is required", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, search_area=area)
    key = firms_map_key or os.environ["FIRMS_MAP_KEY"]
    if not source or "/" in source:
        return _failure("source must be a valid FIRMS product name", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, search_area=area)
    getter = http_get or httpx.get
    detections: list[dict[str, Any]] = []
    try:
        for chunk_start, chunk_end in _date_chunks(validation[0], validation[1]):
            day_range = (chunk_end - chunk_start).days + 1
            url = f"{FIRMS_AREA_URL}/{key}/{source}/{area['firms_area_parameter']}/{day_range}/{chunk_start.isoformat()}"
            response = getter(url, timeout=timeout)
            response.raise_for_status()
            detections.extend(_parse_csv(response.text, latitude, longitude, chunk_start, chunk_end, area))
    except httpx.TimeoutException:
        return _failure("NASA FIRMS request timed out", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, search_area=area)
    except httpx.HTTPStatusError as exc:
        return _failure(f"NASA FIRMS returned HTTP {exc.response.status_code}", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, search_area=area)
    except httpx.RequestError as exc:
        return _failure(f"NASA FIRMS request failed: {exc}", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, search_area=area)
    except (ValueError, TypeError, KeyError) as exc:
        return _failure(f"Malformed NASA FIRMS response: {exc}", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, search_area=area)
    except Exception as exc:  # noqa: BLE001
        return _failure(f"NASA FIRMS retrieval failed: {exc}", latitude=latitude, longitude=longitude, before_date=before_text, after_date=after_text, search_area=area)

    metrics = _metrics(detections)
    if detections:
        evidence = [{"type": "fire_detections", "description": f"{len(detections)} FIRMS thermal anomaly detection(s) found in the searched area and period", "detection_count": len(detections)}]
    else:
        evidence = [{"type": "fire_detections", "description": "No FIRMS detections were found in the searched area and period", "detection_count": 0}]
    return {
        "agent": "fire",
        "status": "success",
        "location": {"latitude": latitude, "longitude": longitude},
        "period": {"before_date": validation[0].isoformat(), "after_date": validation[1].isoformat()},
        "search_area": area,
        "metrics": metrics,
        "detections": detections,
        "evidence": evidence,
        "source": "NASA FIRMS",
    }
