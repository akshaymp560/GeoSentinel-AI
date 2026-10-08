from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.services.satellite_preprocessing import preprocess_sentinel2_images


TRANSFORM = from_origin(500000, 1200000, 10, 10)


def write_inputs(root: Path, *, transform=TRANSFORM, include_quality=True):
    image_paths = (root / "before.tif", root / "after.tif")
    quality_paths = (root / "before_scl.tif", root / "after_scl.tif")
    bands = np.stack([np.full((3, 3), value, dtype=np.float32) for value in range(1, 7)])
    scl_before = np.full((3, 3), 4, dtype=np.uint8)
    scl_after = np.full((3, 3), 4, dtype=np.uint8)
    scl_before[0, 0] = 8  # cloud
    scl_before[0, 1] = 3  # cloud shadow
    scl_after[1, 1] = 9  # high-probability cloud
    scl_after[1, 2] = 10  # cirrus
    root.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        with rasterio.open(path, "w", driver="GTiff", height=3, width=3, count=6, dtype="float32", crs="EPSG:32643", transform=transform, nodata=-9999) as dst:
            dst.write(bands)
    if include_quality:
        for path, values in zip(quality_paths, (scl_before, scl_after)):
            with rasterio.open(path, "w", driver="GTiff", height=3, width=3, count=1, dtype="uint8", crs="EPSG:32643", transform=transform, nodata=0) as dst:
                dst.write(values, 1)
    return image_paths, quality_paths


def test_six_bands_grid_and_quality_masks_are_preserved(tmp_path):
    images, quality = write_inputs(tmp_path / "inputs")
    result = preprocess_sentinel2_images(*images, *quality, tmp_path / "outputs")

    assert result["success"] is True
    assert result["metadata"]["bands"] == ["B2", "B3", "B4", "B8", "B11", "B12"]
    assert result["metadata"]["quality_source"].endswith("(SCL)")
    assert result["metadata"]["cloud_mask_applied"] is True
    assert result["metadata"]["cloud_shadow_handling"] is True
    with rasterio.open(result["before_image_path"]) as before, rasterio.open(result["after_image_path"]) as after:
        assert before.count == after.count == 6
        assert (before.height, before.width) == (after.height, after.width) == (3, 3)
        assert before.crs == after.crs
        assert before.transform == after.transform == TRANSFORM
        assert np.isnan(before.read(1)[0, 0])  # cloud
        assert np.isnan(before.read(1)[0, 1])  # cloud shadow
        assert np.isnan(after.read(1)[1, 1])  # high cloud
        assert np.isnan(after.read(1)[1, 2])  # cirrus
        assert not np.isnan(before.read(1)[2, 2])


def test_nodata_is_preserved_as_invalid(tmp_path):
    images, quality = write_inputs(tmp_path / "inputs")
    with rasterio.open(images[0], "r+") as source:
        values = source.read()
        values[:, 2, 2] = -9999
        source.write(values)
    result = preprocess_sentinel2_images(*images, *quality, tmp_path / "outputs")
    assert result["success"] is True
    with rasterio.open(result["before_image_path"]) as before:
        assert np.isnan(before.read(1)[2, 2])
    assert result["metadata"]["before_masking"]["masked_invalid_pixels"] >= 3


def test_missing_or_misaligned_quality_returns_structured_failure(tmp_path):
    images, quality = write_inputs(tmp_path / "inputs", include_quality=False)
    missing = preprocess_sentinel2_images(*images, *quality, tmp_path / "outputs")
    assert missing["success"] is False
    assert missing["stage"] == "before_read"

    images, quality = write_inputs(tmp_path / "misaligned")
    with rasterio.open(quality[0], "r+") as source:
        source.transform = from_origin(500001, 1200000, 10, 10)
    misaligned = preprocess_sentinel2_images(*images, *quality, tmp_path / "outputs2")
    assert misaligned["success"] is False
    assert misaligned["stage"] == "before_validation"


def test_before_after_alignment_is_required(tmp_path):
    images, quality = write_inputs(tmp_path / "inputs")
    with rasterio.open(images[1], "r+") as source:
        source.transform = from_origin(500001, 1200000, 10, 10)
    result = preprocess_sentinel2_images(*images, *quality, tmp_path / "outputs")
    assert result["success"] is False
    assert result["stage"] == "after_validation"


def test_invalid_band_count_is_structured(tmp_path):
    images, quality = write_inputs(tmp_path / "inputs")
    invalid = tmp_path / "inputs" / "before.tif"
    with rasterio.open(invalid, "w", driver="GTiff", height=3, width=3, count=5, dtype="float32", crs="EPSG:32643", transform=TRANSFORM) as dst:
        dst.write(np.zeros((5, 3, 3), dtype=np.float32))
    result = preprocess_sentinel2_images(*images, *quality, tmp_path / "outputs")
    assert result["success"] is False
    assert result["stage"] == "before_validation"
