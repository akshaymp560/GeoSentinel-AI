"""Sentinel-2 SCL-based cloud and cloud-shadow preprocessing.

The retriever exports the six production bands plus an SCL quality sidecar.
This module consumes those artifacts, masks invalid/cloud/shadow SCL classes,
and writes aligned six-band float32 GeoTIFFs for later ChangeFormer input
preparation. It does not perform model-specific normalization or resizing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

EXPECTED_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
SCL_BAND_COUNT = 1
INVALID_SCL_CLASSES = frozenset({0, 1})
CLOUD_SHADOW_SCL_CLASSES = frozenset({3})
CLOUD_SCL_CLASSES = frozenset({8, 9, 10, 11})


def _failure(error: str, stage: str) -> dict[str, Any]:
    return {"success": False, "error": error, "stage": stage}


def _read_pair(image_path: Path, quality_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | dict[str, Any]:
    if not image_path.is_file():
        return _failure(f"Image file does not exist: {image_path}", "read")
    if not quality_path.is_file():
        return _failure(f"Quality sidecar does not exist: {quality_path}", "read")
    try:
        with rasterio.open(image_path) as image, rasterio.open(quality_path) as quality:
            if image.count != len(EXPECTED_BANDS):
                return _failure(f"Expected exactly 6 image bands, found {image.count}", "validation")
            if quality.count != SCL_BAND_COUNT:
                return _failure(f"Expected exactly 1 SCL quality band, found {quality.count}", "validation")
            if image.width <= 0 or image.height <= 0 or quality.width <= 0 or quality.height <= 0:
                return _failure("Image or quality raster has invalid dimensions", "validation")
            if (image.width, image.height) != (quality.width, quality.height):
                return _failure("Image and SCL quality raster dimensions do not match", "validation")
            if image.crs != quality.crs or image.transform != quality.transform:
                return _failure("Image and SCL quality raster grids do not align", "validation")
            bands = image.read(masked=True).astype(np.float32)
            scl = quality.read(1, masked=True)
            profile = image.profile.copy()
            profile.update(width=image.width, height=image.height, count=len(EXPECTED_BANDS), dtype="float32")
            return bands, scl, profile
    except Exception as exc:  # noqa: BLE001 - preprocessing must fail structurally
        logger.exception("Could not read Sentinel-2 preprocessing inputs")
        return _failure(f"Could not read preprocessing input: {exc}", "read")


def _mask_image(bands: np.ndarray, scl: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    scl_values = scl.filled(0).astype(np.int16)
    quality_invalid = np.ma.getmaskarray(scl)
    source_invalid = np.ma.getmaskarray(bands).any(axis=0)
    cloud_shadow = np.isin(scl_values, list(CLOUD_SHADOW_SCL_CLASSES))
    cloud = np.isin(scl_values, list(CLOUD_SCL_CLASSES))
    invalid = quality_invalid | source_invalid | np.isin(scl_values, list(INVALID_SCL_CLASSES)) | cloud_shadow | cloud
    output = bands.filled(np.nan).astype(np.float32)
    output[:, invalid] = np.nan
    valid_fraction = float(np.count_nonzero(~invalid) / invalid.size) if invalid.size else 0.0
    return output, {
        "cloud_mask_applied": True,
        "cloud_shadow_handling": True,
        "quality_source": "Sentinel-2 Scene Classification Layer (SCL)",
        "masked_cloud_pixels": int(np.count_nonzero(cloud & ~quality_invalid)),
        "masked_cloud_shadow_pixels": int(np.count_nonzero(cloud_shadow & ~quality_invalid)),
        "masked_invalid_pixels": int(np.count_nonzero(invalid)),
        "valid_fraction": round(valid_fraction, 6),
    }


def _write_image(path: Path, bands: np.ndarray, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(count=len(EXPECTED_BANDS), dtype="float32", nodata=np.nan)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(bands)


def preprocess_sentinel2_images(
    before_image_path: str | Path,
    after_image_path: str | Path,
    before_quality_path: str | Path,
    after_quality_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Apply SCL cloud/shadow masking to aligned before/after six-band images."""
    before_loaded = _read_pair(Path(before_image_path), Path(before_quality_path))
    if isinstance(before_loaded, dict):
        before_loaded["stage"] = f"before_{before_loaded['stage']}"
        return before_loaded
    after_loaded = _read_pair(Path(after_image_path), Path(after_quality_path))
    if isinstance(after_loaded, dict):
        after_loaded["stage"] = f"after_{after_loaded['stage']}"
        return after_loaded
    before_bands, before_scl, before_profile = before_loaded
    after_bands, after_scl, after_profile = after_loaded
    if before_bands.shape != after_bands.shape:
        return _failure("Before and after image dimensions do not match", "validation")
    if before_profile["crs"] != after_profile["crs"] or before_profile["transform"] != after_profile["transform"]:
        return _failure("Before and after image grids do not align", "validation")
    try:
        before_masked, before_metadata = _mask_image(before_bands, before_scl)
        after_masked, after_metadata = _mask_image(after_bands, after_scl)
        output = Path(output_dir)
        before_output = output / "before.tif"
        after_output = output / "after.tif"
        _write_image(before_output, before_masked, before_profile)
        _write_image(after_output, after_masked, after_profile)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sentinel-2 preprocessing failed")
        return _failure(f"Satellite preprocessing failed: {exc}", "write")

    metadata = {
        "cloud_mask_applied": True,
        "cloud_shadow_handling": True,
        "quality_source": "Sentinel-2 Scene Classification Layer (SCL)",
        "bands": list(EXPECTED_BANDS),
        "crs": str(before_profile["crs"]),
        "transform": tuple(before_profile["transform"]),
        "dimensions": {"width": before_profile["width"], "height": before_profile["height"]},
        "scale_meters": 10,
        "before_valid_fraction": before_metadata["valid_fraction"],
        "after_valid_fraction": after_metadata["valid_fraction"],
        "before_masking": before_metadata,
        "after_masking": after_metadata,
    }
    return {
        "success": True,
        "before_image_path": str(before_output),
        "after_image_path": str(after_output),
        "metadata": metadata,
        "error": None,
    }
