"""Sentinel-2 retrieval service for Checkpoint 2.1.

This module is intentionally not called by the Phase 1 API yet. It provides a
small, testable boundary for selecting and downloading one least-cloudy
Sentinel-2 image for each investigation period.

Output GeoTIFF band order is always: B2, B3, B4, B8, B11, B12. The download
request uses a single 10 m scale and the B2 projection so the native 20 m B11
and B12 bands are delivered on the same grid as the 10 m bands.

The Sentinel-2 Scene Classification Layer (SCL) is exported separately as a
quality sidecar on the same 10 m grid. The six-band production output remains
unchanged; the sidecar is consumed by satellite_preprocessing.py.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

try:  # Keep unit tests/imports useful before the optional dependency is installed.
    import ee
except ImportError:  # pragma: no cover - exercised through the init failure path
    ee = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
OUTPUT_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
DEFAULT_CLOUD_LIMIT = 10
DEFAULT_PERIOD_DAYS = 30
DEFAULT_AOI_HALF_SIZE_DEGREES = 0.05
SAFE_INVESTIGATION_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _failure(error: str, stage: str) -> dict[str, Any]:
    return {"success": False, "error": error, "stage": stage}


def _parse_date(value: date | str, field_name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid ISO date") from exc


def _validate_inputs(
    investigation_id: str,
    latitude: float,
    longitude: float,
    before_date: date | str,
    after_date: date | str,
) -> tuple[date, date] | dict[str, Any]:
    if not investigation_id or not SAFE_INVESTIGATION_ID.fullmatch(investigation_id):
        return _failure("investigation_id must contain only letters, numbers, '_' or '-'", "validation")
    if not isinstance(latitude, (int, float)) or not math.isfinite(latitude) or not -90 <= latitude <= 90:
        return _failure("latitude must be between -90 and 90", "validation")
    if not isinstance(longitude, (int, float)) or not math.isfinite(longitude) or not -180 <= longitude <= 180:
        return _failure("longitude must be between -180 and 180", "validation")
    try:
        before = _parse_date(before_date, "before_date")
        after = _parse_date(after_date, "after_date")
    except ValueError as exc:
        return _failure(str(exc), "validation")
    if before >= after:
        return _failure("before_date must be earlier than after_date", "validation")
    return before, after


def _initialise_ee(project: str | None) -> dict[str, Any] | None:
    if ee is None:
        return _failure("earthengine-api is not installed", "gee_initialization")
    if not project:
        return _failure("GEE_PROJECT is required", "gee_initialization")
    try:
        # Authentication is deliberately not automated. Users must configure
        # Earth Engine credentials using the official ee CLI/workflow.
        ee.Initialize(project=project)
    except Exception as exc:  # noqa: BLE001 - SDK/network errors are returned to callers
        logger.exception("Google Earth Engine initialization failed")
        return _failure(f"GEE initialization failed: {exc}", "gee_initialization")
    return None


def _period_bounds(anchor: date, period_days: int) -> tuple[str, str]:
    start = anchor - timedelta(days=period_days)
    end = anchor + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _select_image(aoi: Any, anchor: date, period_days: int, cloud_limit: int) -> tuple[Any, str, Any] | dict[str, Any]:
    try:
        start, end = _period_bounds(anchor, period_days)
        collection = (
            ee.ImageCollection(COLLECTION_ID)
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_limit))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        if collection.size().getInfo() == 0:
            return _failure("No suitable Sentinel-2 image found for the requested period", "query")
        image = ee.Image(collection.first())
        acquired = image.date().format("YYYY-MM-dd").getInfo()
        cloud_cover = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        return image, acquired, cloud_cover
    except Exception as exc:  # noqa: BLE001 - SDK failures must not reach FastAPI
        logger.exception("Sentinel-2 collection query failed")
        return _failure(f"Sentinel-2 query failed: {exc}", "query")


def _download_image(image: Any, aoi: Any, destination: Path, name: str, crs: str) -> None:
    # scale + CRS are applied to the complete six-band image. B11/B12 are
    # resampled by Earth Engine onto the same 10 m grid as the 10 m bands.
    prepared = image.select(list(OUTPUT_BANDS)).resample("bilinear").reproject(crs=crs, scale=10)
    url = prepared.getDownloadURL(
        {
            "name": name,
            "bands": list(OUTPUT_BANDS),
            "region": aoi.coordinates().getInfo(),
            "scale": 10,
            "crs": crs,
            "filePerBand": False,
            "format": "GEO_TIFF",
        }
    )
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - URL is supplied by EE SDK
        payload = response.read()
    if payload[:2] == b"PK":
        with zipfile.ZipFile(__import__("io").BytesIO(payload)) as archive:
            tif_members = [member for member in archive.namelist() if member.lower().endswith((".tif", ".tiff"))]
            if not tif_members:
                raise ValueError("Earth Engine download archive did not contain a GeoTIFF")
            destination.write_bytes(archive.read(tif_members[0]))
    else:
        destination.write_bytes(payload)
    if not destination.exists() or destination.stat().st_size == 0:
        raise ValueError("Earth Engine returned an empty GeoTIFF")


def _download_quality_image(image: Any, aoi: Any, destination: Path, name: str, crs: str) -> None:
    """Export Sentinel-2 SCL as a nearest-neighbour 10 m quality sidecar."""
    # SCL is categorical; Earth Engine's default resampling is nearest-neighbour.
    prepared = image.select("SCL").reproject(crs=crs, scale=10)
    url = prepared.getDownloadURL(
        {
            "name": name,
            "bands": ["SCL"],
            "region": aoi.coordinates().getInfo(),
            "scale": 10,
            "crs": crs,
            "filePerBand": False,
            "format": "GEO_TIFF",
        }
    )
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - URL is supplied by EE SDK
        payload = response.read()
    if payload[:2] == b"PK":
        with zipfile.ZipFile(__import__("io").BytesIO(payload)) as archive:
            tif_members = [member for member in archive.namelist() if member.lower().endswith((".tif", ".tiff"))]
            if not tif_members:
                raise ValueError("Earth Engine SCL download archive did not contain a GeoTIFF")
            destination.write_bytes(archive.read(tif_members[0]))
    else:
        destination.write_bytes(payload)
    if not destination.exists() or destination.stat().st_size == 0:
        raise ValueError("Earth Engine returned an empty SCL GeoTIFF")


def retrieve_sentinel2_images(
    investigation_id: str,
    latitude: float,
    longitude: float,
    before_date: date | str,
    after_date: date | str,
    *,
    aoi_half_size_degrees: float = DEFAULT_AOI_HALF_SIZE_DEGREES,
    period_days: int = DEFAULT_PERIOD_DAYS,
    cloud_limit: int = DEFAULT_CLOUD_LIMIT,
    gee_project: str | None = None,
    artifact_root: str | Path | None = None,
    downloader: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Retrieve aligned six-band Sentinel-2 GeoTIFFs for an investigation.

    Each supplied date anchors a configurable lookback window (30 days by
    default), allowing a period to contain multiple candidate acquisitions.
    ``artifact_root`` is injectable for tests; production defaults to
    ``backend/artifacts/investigations/{investigation_id}``.
    """
    validation = _validate_inputs(investigation_id, latitude, longitude, before_date, after_date)
    if isinstance(validation, dict):
        return validation
    before, after = validation
    if not isinstance(aoi_half_size_degrees, (int, float)) or not 0 < aoi_half_size_degrees <= 1:
        return _failure("aoi_half_size_degrees must be greater than 0 and at most 1", "validation")
    if not isinstance(period_days, int) or period_days < 1:
        return _failure("period_days must be a positive integer", "validation")
    if not isinstance(cloud_limit, int) or not 0 <= cloud_limit <= 100:
        return _failure("cloud_limit must be between 0 and 100", "validation")

    init_failure = _initialise_ee(gee_project or os.getenv("GEE_PROJECT"))
    if init_failure:
        return init_failure

    try:
        aoi = ee.Geometry.Rectangle(
            [
                longitude - aoi_half_size_degrees,
                latitude - aoi_half_size_degrees,
                longitude + aoi_half_size_degrees,
                latitude + aoi_half_size_degrees,
            ]
        )
        before_result = _select_image(aoi, before, period_days, cloud_limit)
        if isinstance(before_result, dict):
            before_result["stage"] = "before_query"
            return before_result
        after_result = _select_image(aoi, after, period_days, cloud_limit)
        if isinstance(after_result, dict):
            after_result["stage"] = "after_query"
            return after_result
        before_image, before_acquired, before_cloud = before_result
        after_image, after_acquired, after_cloud = after_result
        crs = before_image.select("B2").projection().crs().getInfo()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sentinel-2 setup failed")
        return _failure(f"Sentinel-2 setup failed: {exc}", "query")

    root = Path(artifact_root) if artifact_root else Path(__file__).resolve().parents[1] / "artifacts" / "investigations"
    output_dir = root / investigation_id
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        download = downloader or _download_image
        download(before_image, aoi, output_dir / "before.tif", "before", crs)
        download(after_image, aoi, output_dir / "after.tif", "after", crs)
        if downloader:
            download(before_image, aoi, output_dir / "before_scl.tif", "before_scl", crs)
            download(after_image, aoi, output_dir / "after_scl.tif", "after_scl", crs)
        else:
            _download_quality_image(before_image, aoi, output_dir / "before_scl.tif", "before_scl", crs)
            _download_quality_image(after_image, aoi, output_dir / "after_scl.tif", "after_scl", crs)
        metadata = {
            "collection": COLLECTION_ID,
            "bands": list(OUTPUT_BANDS),
            "grid": {"crs": crs, "scale_meters": 10},
            "quality_source": "Sentinel-2 Scene Classification Layer (SCL)",
            "cloud_mask_available": True,
            "cloud_shadow_mask_available": True,
            "before_quality_path": str(output_dir / "before_scl.tif"),
            "after_quality_path": str(output_dir / "after_scl.tif"),
            "before_date_acquired": before_acquired,
            "after_date_acquired": after_acquired,
            "cloud_cover_before": before_cloud,
            "cloud_cover_after": after_cloud,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - filesystem/download failures are structured
        logger.exception("Sentinel-2 export failed")
        return _failure(f"Sentinel-2 export failed: {exc}", "export")

    return {
        "success": True,
        "before_image_path": str(output_dir / "before.tif"),
        "after_image_path": str(output_dir / "after.tif"),
        "metadata": metadata,
    }
