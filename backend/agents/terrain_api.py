"""Terrain/elevation evidence agent using Open-Meteo's public DEM endpoint.

The endpoint provides point elevation, so this agent reports only
``elevation_m`` at the investigation coordinate. It does not invent slope or
area statistics. The configurable radius is returned as an explicit search
area context for later terrain-aware analysis; it is not silently used to
claim that the point elevation represents the whole area.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import httpx

OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
DEFAULT_RADIUS_KM = 5.0


def _search_area(latitude: float, longitude: float, radius_km: float) -> dict[str, Any]:
    lat_delta = radius_km / 111.32
    cosine = abs(math.cos(math.radians(latitude)))
    lon_delta = 180.0 if cosine < 1e-12 else min(180.0, radius_km / (111.32 * cosine))
    return {
        "radius_km": radius_km,
        "bounding_box": {
            "west": max(-180.0, longitude - lon_delta),
            "south": max(-90.0, latitude - lat_delta),
            "east": min(180.0, longitude + lon_delta),
            "north": min(90.0, latitude + lat_delta),
        },
        "measurement": "point elevation at the investigation coordinate",
    }


def _failure(
    error: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    search_area: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "agent": "terrain",
        "status": "failed",
        "error": error,
        "metrics": None,
        "evidence": [],
        "source": "Open-Meteo elevation DEM",
    }
    if latitude is not None and longitude is not None:
        result["location"] = {"latitude": latitude, "longitude": longitude}
    if search_area is not None:
        result["search_area"] = search_area
    return result


def _validate(latitude: float, longitude: float, radius_km: float) -> str | None:
    if not isinstance(latitude, (int, float)) or not math.isfinite(latitude) or not -90 <= latitude <= 90:
        return "latitude must be between -90 and 90"
    if not isinstance(longitude, (int, float)) or not math.isfinite(longitude) or not -180 <= longitude <= 180:
        return "longitude must be between -180 and 180"
    if not isinstance(radius_km, (int, float)) or not math.isfinite(radius_km) or not 0 < radius_km <= 500:
        return "radius_km must be greater than 0 and at most 500"
    return None


def _extract_elevation(payload: Any) -> float:
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    values = payload.get("elevation")
    if not isinstance(values, list) or not values:
        raise ValueError("response is missing an elevation value")
    value = values[0]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("response contains an invalid elevation value")
    return float(value)


def retrieve_terrain_evidence(
    latitude: float,
    longitude: float,
    *,
    radius_km: float = DEFAULT_RADIUS_KM,
    timeout: float = 15.0,
    http_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Retrieve point elevation evidence for a bounded terrain context."""
    area = (
        _search_area(latitude, longitude, radius_km)
        if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float))
        else None
    )
    error = _validate(latitude, longitude, radius_km)
    if error:
        return _failure(
            error,
            latitude=latitude if isinstance(latitude, (int, float)) else None,
            longitude=longitude if isinstance(longitude, (int, float)) else None,
            search_area=area,
        )

    getter = http_get or httpx.get
    try:
        response = getter(
            OPEN_METEO_ELEVATION_URL,
            params={"latitude": latitude, "longitude": longitude},
            timeout=timeout,
        )
        response.raise_for_status()
        elevation = _extract_elevation(response.json())
    except httpx.TimeoutException:
        return _failure("Elevation request timed out", latitude=latitude, longitude=longitude, search_area=area)
    except httpx.HTTPStatusError as exc:
        return _failure(f"Elevation service returned HTTP {exc.response.status_code}", latitude=latitude, longitude=longitude, search_area=area)
    except httpx.RequestError as exc:
        return _failure(f"Elevation request failed: {exc}", latitude=latitude, longitude=longitude, search_area=area)
    except (ValueError, TypeError, KeyError) as exc:
        return _failure(f"Malformed elevation response: {exc}", latitude=latitude, longitude=longitude, search_area=area)
    except Exception as exc:  # noqa: BLE001 - one evidence source must fail safely
        return _failure(f"Terrain retrieval failed: {exc}", latitude=latitude, longitude=longitude, search_area=area)

    return {
        "agent": "terrain",
        "status": "success",
        "location": {"latitude": latitude, "longitude": longitude},
        "search_area": area,
        "metrics": {"elevation_m": round(elevation, 3)},
        "evidence": [
            {
                "type": "elevation",
                "value": round(elevation, 3),
                "unit": "m",
                "description": "Point elevation reported for the investigation coordinate",
            }
        ],
        "source": "Open-Meteo elevation DEM",
    }
