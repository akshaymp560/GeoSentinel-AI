"""Deterministic spectral indices for Checkpoint 2.2.

The input contract is the six-band Sentinel-2 GeoTIFF produced by
``gee_retriever``: B2, B3, B4, B8, B11, B12, in that exact order. This module
does not apply a ChangeFormer mask; later integration may use the returned
rasters/statistics together with that mask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

EXPECTED_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
INDEX_NAMES = ("NDVI", "NDWI", "NDBI")


def _failure(error: str, stage: str) -> dict[str, Any]:
    return {"success": False, "error": error, "stage": stage}


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Calculate a ratio, using NaN only for invalid/no-data pixels."""
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(numerator, denominator, out=result, where=valid)
    result[~np.isfinite(result)] = np.nan
    return result


def _calculate_indices(bands: np.ndarray) -> dict[str, np.ndarray]:
    # Fixed positions from the Checkpoint 2.1 six-band contract.
    blue, green, red, nir, swir1, swir2 = bands
    del blue, green, swir2  # Reserved for later indices; retained in the input contract.
    return {
        "NDVI": _safe_ratio(nir - red, nir + red),
        "NDWI": _safe_ratio(nir - swir1, nir + swir1),
        "NDBI": _safe_ratio(swir1 - nir, swir1 + nir),
    }


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"minimum": None, "maximum": None, "mean": None, "valid_pixel_count": 0}
    return {
        "minimum": float(np.min(valid)),
        "maximum": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "valid_pixel_count": int(valid.size),
    }


def _read_six_band_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]] | dict[str, Any]:
    if not path.exists():
        return _failure(f"Raster file does not exist: {path}", "read")
    if not path.is_file():
        return _failure(f"Raster path is not a file: {path}", "read")
    try:
        with rasterio.open(path) as source:
            if source.count != len(EXPECTED_BANDS):
                return _failure(
                    f"Expected exactly 6 bands in {path}, found {source.count}",
                    "validation",
                )
            if source.width <= 0 or source.height <= 0:
                return _failure(f"Raster has invalid dimensions: {path}", "validation")
            bands = source.read(masked=True).astype(np.float32).filled(np.nan)
            profile = source.profile.copy()
            profile["width"] = source.width
            profile["height"] = source.height
            profile["transform"] = source.transform
            profile["crs"] = source.crs
            return bands, profile
    except Exception as exc:  # noqa: BLE001 - callers receive a structured result
        logger.exception("Could not read spectral raster %s", path)
        return _failure(f"Could not read raster: {exc}", "read")


def _read_change_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]] | dict[str, Any]:
    if not path.exists() or not path.is_file():
        return _failure(f"Change mask file does not exist: {path}", "mask_read")
    try:
        with rasterio.open(path) as source:
            if source.count != 1:
                return _failure(f"Expected a single-band change mask, found {source.count}", "mask_validation")
            if source.width <= 0 or source.height <= 0:
                return _failure("Change mask has invalid dimensions", "mask_validation")
            mask = source.read(1, masked=True).astype(np.float32)
            profile = source.profile.copy()
            profile.update(width=source.width, height=source.height, transform=source.transform, crs=source.crs)
            return mask, profile
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not read change mask %s", path)
        return _failure(f"Could not read change mask: {exc}", "mask_read")


def _validate_grid(
    before_profile: dict[str, Any], after_profile: dict[str, Any], mask_profile: dict[str, Any] | None = None
) -> str | None:
    if before_profile["width"] != after_profile["width"] or before_profile["height"] != after_profile["height"]:
        return "Before and after rasters must have identical dimensions"
    if before_profile.get("crs") != after_profile.get("crs"):
        return "Before and after rasters must have identical CRS"
    if before_profile.get("transform") != after_profile.get("transform"):
        return "Before and after rasters must have identical transforms"
    if mask_profile is not None:
        if mask_profile["width"] != before_profile["width"] or mask_profile["height"] != before_profile["height"]:
            return "Change mask dimensions must match the input rasters"
        if mask_profile.get("crs") != before_profile.get("crs"):
            return "Change mask CRS must match the input rasters"
        if mask_profile.get("transform") != before_profile.get("transform"):
            return "Change mask transform must match the input rasters"
    return None


def _changed_region_statistics(
    before_indices: dict[str, np.ndarray],
    after_indices: dict[str, np.ndarray],
    before_bands: np.ndarray,
    after_bands: np.ndarray,
    mask: np.ma.MaskedArray,
) -> dict[str, Any]:
    changed = (~np.ma.getmaskarray(mask)) & np.isfinite(mask.filled(0)) & (mask.filled(0) > 0)
    before_values = np.asarray(before_bands, dtype=np.float32)
    after_values = np.asarray(after_bands, dtype=np.float32)
    valid_pixels = (
        (~np.ma.getmaskarray(before_bands).any(axis=0))
        & (~np.ma.getmaskarray(after_bands).any(axis=0))
        & np.isfinite(before_values).all(axis=0)
        & np.isfinite(after_values).all(axis=0)
    )
    changed_valid = changed & valid_pixels
    total_valid = int(np.count_nonzero(valid_pixels))
    changed_pixels = int(np.count_nonzero(changed_valid))
    percentage = round(changed_pixels / total_valid * 100, 2) if total_valid else 0.0
    indices: dict[str, Any] = {}
    for name in INDEX_NAMES:
        before_values = before_indices[name]
        after_values = after_indices[name]
        valid = changed_valid & np.isfinite(before_values) & np.isfinite(after_values)
        before_mean = float(np.mean(before_values[valid])) if np.any(valid) else None
        after_mean = float(np.mean(after_values[valid])) if np.any(valid) else None
        indices[name] = {
            "before_mean": before_mean,
            "after_mean": after_mean,
            "delta": (after_mean - before_mean) if before_mean is not None and after_mean is not None else None,
            "valid_changed_pixels": int(np.count_nonzero(valid)),
        }
    return {
        "changed_pixels": changed_pixels,
        "total_valid_pixels": total_valid,
        "change_percentage": percentage,
        "indices": indices,
    }


def _write_index_rasters(
    indices: dict[str, np.ndarray], profile: dict[str, Any], output_dir: Path
) -> dict[str, str] | dict[str, Any]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, str] = {}
        output_profile = profile.copy()
        output_profile.update(count=1, dtype="float32", nodata=np.nan)
        for name, values in indices.items():
            path = output_dir / f"{name.lower()}.tif"
            with rasterio.open(path, "w", **output_profile) as destination:
                destination.write(values.astype(np.float32), 1)
            output_paths[name] = str(path)
        return output_paths
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not write spectral rasters to %s", output_dir)
        return _failure(f"Could not write spectral rasters: {exc}", "write")


def analyze_spectral_image(
    raster_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    write_rasters: bool = False,
) -> dict[str, Any]:
    """Analyze one six-band Sentinel-2 raster and optionally write index rasters."""
    path = Path(raster_path)
    loaded = _read_six_band_raster(path)
    if isinstance(loaded, dict):
        return loaded
    bands, profile = loaded
    try:
        indices = _calculate_indices(bands)
        result: dict[str, Any] = {
            "success": True,
            "raster_path": str(path),
            "shape": [int(bands.shape[1]), int(bands.shape[2])],
            "band_order": list(EXPECTED_BANDS),
            "indices": {name: {"summary": _summary(values)} for name, values in indices.items()},
        }
        if write_rasters:
            if output_dir is None:
                return _failure("output_dir is required when write_rasters is true", "write")
            output_paths = _write_index_rasters(indices, profile, Path(output_dir))
            if isinstance(output_paths, dict) and output_paths.get("success") is False:
                return output_paths
            result["index_raster_paths"] = output_paths
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Spectral calculation failed for %s", path)
        return _failure(f"Spectral calculation failed: {exc}", "calculation")


def analyze_before_after(
    before_path: str | Path,
    after_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    write_rasters: bool = False,
    mask_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze before/after rasters and optionally calculate changed-region statistics.

    A nonzero mask pixel is treated as changed. The mask is deliberately an
    explicit input so a future real ChangeFormer output can replace the mock
    without changing this spectral API.
    """
    before_result = analyze_spectral_image(
        before_path,
        output_dir=Path(output_dir) / "before" if output_dir else None,
        write_rasters=write_rasters,
    )
    if not before_result.get("success"):
        before_result["stage"] = f"before_{before_result.get('stage', 'analysis')}"
        return before_result
    after_result = analyze_spectral_image(
        after_path,
        output_dir=Path(output_dir) / "after" if output_dir else None,
        write_rasters=write_rasters,
    )
    if not after_result.get("success"):
        after_result["stage"] = f"after_{after_result.get('stage', 'analysis')}"
        return after_result

    before_loaded = _read_six_band_raster(Path(before_path))
    after_loaded = _read_six_band_raster(Path(after_path))
    if isinstance(before_loaded, dict) or isinstance(after_loaded, dict):
        return _failure("Could not reload before/after rasters for change statistics", "calculation")
    before_bands, before_profile = before_loaded
    after_bands, after_profile = after_loaded
    mask_data = None
    mask_profile = None
    if mask_path is not None:
        loaded_mask = _read_change_mask(Path(mask_path))
        if isinstance(loaded_mask, dict):
            return loaded_mask
        mask_data, mask_profile = loaded_mask
    grid_error = _validate_grid(before_profile, after_profile, mask_profile)
    if grid_error:
        return _failure(grid_error, "validation")
    before_indices = _calculate_indices(before_bands)
    after_indices = _calculate_indices(after_bands)
    change = {
        name: _summary(after_indices[name] - before_indices[name]) for name in INDEX_NAMES
    }
    result = {
        "success": True,
        "before": before_result,
        "after": after_result,
        "change": change,
    }
    if mask_data is not None:
        result["changed_region"] = _changed_region_statistics(
            before_indices, after_indices, before_bands, after_bands, mask_data
        )
    return result
