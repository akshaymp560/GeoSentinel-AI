"""ESA WorldCover land-cover evidence agent.

WorldCover v200's ``Map`` band is sampled over a bounded AOI using Earth
Engine. Percentages are calculated from returned class pixel counts. The agent
reports contextual land-cover evidence only; it does not infer deforestation,
development, or any other change cause.
"""

from __future__ import annotations

import math
import os
from typing import Any

try:  # Keep importable for unit tests and configuration-error handling.
    import ee
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]

WORLD_COVER_COLLECTION = "ESA/WorldCover/v200"
WORLD_COVER_BAND = "Map"
DEFAULT_RADIUS_KM = 5.0
WORLD_COVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


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
        "measurement": "WorldCover pixel-class distribution within the bounding box",
    }


def _failure(
    error: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    search_area: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "agent": "landcover",
        "status": "failed",
        "error": error,
        "metrics": None,
        "evidence": [],
        "source": "ESA WorldCover",
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


def _class_distribution(histogram: Any) -> list[dict[str, Any]]:
    if not isinstance(histogram, dict) or not histogram:
        raise ValueError("WorldCover response contains no class data")
    counts: dict[int, float] = {}
    for raw_code, raw_count in histogram.items():
        try:
            code = int(raw_code)
        except (TypeError, ValueError) as exc:
            raise ValueError("WorldCover response contains an invalid class code") from exc
        if code not in WORLD_COVER_CLASSES:
            raise ValueError(f"WorldCover response contains unsupported class code: {code}")
        if not isinstance(raw_count, (int, float)) or not math.isfinite(raw_count) or raw_count < 0:
            raise ValueError("WorldCover response contains an invalid class pixel count")
        counts[code] = counts.get(code, 0) + float(raw_count)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("WorldCover response contains no valid pixels")
    return [
        {
            "class_code": code,
            "class_name": WORLD_COVER_CLASSES[code],
            "percentage": round(count / total * 100, 3),
        }
        for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def retrieve_landcover_evidence(
    latitude: float,
    longitude: float,
    *,
    radius_km: float = DEFAULT_RADIUS_KM,
    gee_project: str | None = None,
    ee_module: Any | None = None,
) -> dict[str, Any]:
    """Retrieve WorldCover class distribution for a bounded coordinate AOI."""
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

    earth_engine = ee_module or ee
    project = gee_project or os.getenv("GEE_PROJECT")
    if earth_engine is None:
        return _failure("earthengine-api is not installed", latitude=latitude, longitude=longitude, search_area=area)
    if not project:
        return _failure("GEE_PROJECT is required", latitude=latitude, longitude=longitude, search_area=area)

    try:
        earth_engine.Initialize(project=project)
        box = area["bounding_box"]
        aoi = earth_engine.Geometry.Rectangle(
            [box["west"], box["south"], box["east"], box["north"]]
        )
        image = earth_engine.ImageCollection(WORLD_COVER_COLLECTION).first().select(WORLD_COVER_BAND)
        histogram = image.reduceRegion(
            reducer=earth_engine.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=10,
            maxPixels=1_000_000,
        ).getInfo()
        class_data = histogram.get(WORLD_COVER_BAND) if isinstance(histogram, dict) else None
        classes = _class_distribution(class_data)
    except Exception as exc:  # noqa: BLE001 - GEE failures become agent failures
        return _failure(f"WorldCover retrieval failed: {exc}", latitude=latitude, longitude=longitude, search_area=area)

    primary = classes[0]
    return {
        "agent": "landcover",
        "status": "success",
        "location": {"latitude": latitude, "longitude": longitude},
        "search_area": area,
        "land_cover": {
            "primary_class": primary["class_name"],
            "class_code": primary["class_code"],
            "coverage_percentage": primary["percentage"],
        },
        "classes": classes,
        "evidence": [
            {
                "type": "land_cover_distribution",
                "description": f"{primary['class_name']} is the dominant ESA WorldCover class in the queried area",
                "class_code": primary["class_code"],
                "coverage_percentage": primary["percentage"],
            }
        ],
        "source": "ESA WorldCover",
    }
