"""Weather evidence retrieval using the public Open-Meteo archive API.

This agent reports observed weather metrics only. It does not infer causes or
make environmental hypotheses. It is intentionally standalone and is not
connected to the frozen Phase 1 investigation pipeline.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Callable

import httpx

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARIABLES = (
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "wind_speed_10m_max",
)


def _failure(
    error: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    before_date: str | None = None,
    after_date: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "agent": "weather",
        "status": "failed",
        "error": error,
        "metrics": None,
        "evidence": [],
        "source": "Open-Meteo",
    }
    if latitude is not None and longitude is not None:
        result["location"] = {"latitude": latitude, "longitude": longitude}
    if before_date is not None and after_date is not None:
        result["period"] = {"before_date": before_date, "after_date": after_date}
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
) -> tuple[date, date] | str:
    if not isinstance(latitude, (int, float)) or not math.isfinite(latitude) or not -90 <= latitude <= 90:
        return "latitude must be between -90 and 90"
    if not isinstance(longitude, (int, float)) or not math.isfinite(longitude) or not -180 <= longitude <= 180:
        return "longitude must be between -180 and 180"
    try:
        before = _parse_date(before_date, "before_date")
        after = _parse_date(after_date, "after_date")
    except ValueError as exc:
        return str(exc)
    if before > after:
        return "before_date must be earlier than or equal to after_date"
    return before, after


def _numeric_values(values: Any, name: str, expected_length: int) -> list[float]:
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(f"weather field '{name}' is missing or has an invalid length")
    result = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"weather field '{name}' contains a non-numeric value")
        result.append(float(value))
    return result


def _extract_metrics(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("daily"), dict):
        raise ValueError("response is missing the daily weather object")
    daily = payload["daily"]
    times = daily.get("time")
    if not isinstance(times, list) or not times:
        raise ValueError("response is missing daily time values")
    length = len(times)
    precipitation = _numeric_values(daily.get("precipitation_sum"), "precipitation_sum", length)
    temperatures = _numeric_values(daily.get("temperature_2m_mean"), "temperature_2m_mean", length)
    maximum_temperatures = _numeric_values(
        daily.get("temperature_2m_max"), "temperature_2m_max", length
    )
    winds = _numeric_values(daily.get("wind_speed_10m_max"), "wind_speed_10m_max", length)
    if not precipitation or not temperatures or not maximum_temperatures:
        raise ValueError("response does not contain enough valid weather values")

    metrics = {
        "total_precipitation_mm": round(sum(precipitation), 3),
        "mean_temperature_c": round(sum(temperatures) / len(temperatures), 3),
        "max_temperature_c": round(max(maximum_temperatures), 3),
        "wet_days": sum(value > 0 for value in precipitation),
        "observed_days": length,
    }
    if winds:
        metrics["max_wind_speed_kmh"] = round(max(winds), 3)
    evidence = [
        {
            "metric": "total_precipitation_mm",
            "value": metrics["total_precipitation_mm"],
            "unit": "mm",
            "description": "Total reported daily precipitation over the requested period",
        },
        {
            "metric": "wet_days",
            "value": metrics["wet_days"],
            "unit": "days",
            "description": "Days with reported precipitation greater than 0 mm",
        },
        {
            "metric": "mean_temperature_c",
            "value": metrics["mean_temperature_c"],
            "unit": "°C",
            "description": "Mean of reported daily mean temperatures",
        },
        {
            "metric": "max_temperature_c",
            "value": metrics["max_temperature_c"],
            "unit": "°C",
            "description": "Maximum reported daily maximum temperature",
        },
    ]
    if "max_wind_speed_kmh" in metrics:
        evidence.append(
            {
                "metric": "max_wind_speed_kmh",
                "value": metrics["max_wind_speed_kmh"],
                "unit": "km/h",
                "description": "Maximum reported daily wind speed",
            }
        )
    return metrics, evidence


def retrieve_weather_evidence(
    latitude: float,
    longitude: float,
    before_date: date | str,
    after_date: date | str,
    *,
    timeout: float = 15.0,
    http_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Retrieve factual weather evidence for an inclusive date period."""
    before_text = before_date.isoformat() if isinstance(before_date, date) else str(before_date)
    after_text = after_date.isoformat() if isinstance(after_date, date) else str(after_date)
    validation = _validate(latitude, longitude, before_date, after_date)
    if isinstance(validation, str):
        return _failure(
            validation,
            latitude=latitude if isinstance(latitude, (int, float)) else None,
            longitude=longitude if isinstance(longitude, (int, float)) else None,
            before_date=before_text,
            after_date=after_text,
        )
    before, after = validation
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": before.isoformat(),
        "end_date": after.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "UTC",
    }
    getter = http_get or httpx.get
    try:
        response = getter(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        metrics, evidence = _extract_metrics(payload)
    except httpx.TimeoutException:
        return _failure(
            "Open-Meteo request timed out",
            latitude=latitude,
            longitude=longitude,
            before_date=before.isoformat(),
            after_date=after.isoformat(),
        )
    except httpx.HTTPStatusError as exc:
        return _failure(
            f"Open-Meteo returned HTTP {exc.response.status_code}",
            latitude=latitude,
            longitude=longitude,
            before_date=before.isoformat(),
            after_date=after.isoformat(),
        )
    except httpx.RequestError as exc:
        return _failure(
            f"Open-Meteo request failed: {exc}",
            latitude=latitude,
            longitude=longitude,
            before_date=before.isoformat(),
            after_date=after.isoformat(),
        )
    except (ValueError, TypeError, KeyError) as exc:
        return _failure(
            f"Malformed Open-Meteo response: {exc}",
            latitude=latitude,
            longitude=longitude,
            before_date=before.isoformat(),
            after_date=after.isoformat(),
        )
    except Exception as exc:  # noqa: BLE001 - one evidence source must fail safely
        return _failure(
            f"Weather retrieval failed: {exc}",
            latitude=latitude,
            longitude=longitude,
            before_date=before.isoformat(),
            after_date=after.isoformat(),
        )

    return {
        "agent": "weather",
        "status": "success",
        "location": {"latitude": latitude, "longitude": longitude},
        "period": {"before_date": before.isoformat(), "after_date": after.isoformat()},
        "metrics": metrics,
        "evidence": evidence,
        "source": "Open-Meteo",
    }
