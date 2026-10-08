from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.services.spectral_analysis import analyze_before_after, analyze_spectral_image


def write_synthetic_raster(path: Path, bands: np.ndarray, *, crs="EPSG:32643") -> None:
    height, width = bands.shape[1:]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands.shape[0],
        dtype="float32",
        crs=crs,
        transform=from_origin(500000, 1200000, 10, 10),
        nodata=-9999,
    ) as destination:
        destination.write(bands.astype(np.float32))


def write_mask(path: Path, values: np.ndarray, *, crs="EPSG:32643", transform=None) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform or from_origin(500000, 1200000, 10, 10),
        nodata=0,
    ) as destination:
        destination.write(values.astype(np.uint8), 1)


def six_band_values(red=1, nir=3, swir=1, shape=(2, 2)) -> np.ndarray:
    bands = np.zeros((6, *shape), dtype=np.float32)
    bands[0] = 99  # B2: should not affect the implemented indices.
    bands[1] = 88  # B3: should not affect the implemented indices.
    bands[2] = red
    bands[3] = nir
    bands[4] = swir
    bands[5] = 77  # B12: should not affect the implemented indices.
    return bands


def test_band_mapping_and_index_calculations(tmp_path):
    path = tmp_path / "input.tif"
    write_synthetic_raster(path, six_band_values())
    result = analyze_spectral_image(path)

    assert result["success"] is True
    assert result["band_order"] == ["B2", "B3", "B4", "B8", "B11", "B12"]
    assert result["shape"] == [2, 2]
    assert result["indices"]["NDVI"]["summary"]["mean"] == 0.5
    assert result["indices"]["NDWI"]["summary"]["mean"] == 0.5
    assert result["indices"]["NDBI"]["summary"]["mean"] == -0.5
    assert result["indices"]["NDVI"]["summary"]["valid_pixel_count"] == 4


def test_zero_denominators_and_invalid_values_are_safe(tmp_path):
    bands = six_band_values()
    bands[2, 0, 0] = -3  # B8 + B4 == 0 for NDVI.
    bands[3, 0, 0] = 3
    bands[4, 0, 1] = -3  # B8 + B11 == 0 for NDWI/NDBI.
    bands[3, 1, 0] = np.nan
    path = tmp_path / "invalid-values.tif"
    write_synthetic_raster(path, bands)

    result = analyze_spectral_image(path)
    assert result["success"] is True
    for index in ("NDVI", "NDWI", "NDBI"):
        summary = result["indices"][index]["summary"]
        assert summary["valid_pixel_count"] <= 3
        assert summary["minimum"] is not None
        assert summary["maximum"] is not None


def test_invalid_missing_and_wrong_band_inputs(tmp_path):
    missing = analyze_spectral_image(tmp_path / "missing.tif")
    assert missing["success"] is False
    assert missing["stage"] == "read"

    wrong_path = tmp_path / "wrong-count.tif"
    write_synthetic_raster(wrong_path, np.zeros((5, 2, 2), dtype=np.float32))
    wrong = analyze_spectral_image(wrong_path)
    assert wrong["success"] is False
    assert wrong["stage"] == "validation"
    assert "Expected exactly 6 bands" in wrong["error"]


def test_writes_index_rasters_with_shape_and_spatial_metadata(tmp_path):
    source = tmp_path / "source.tif"
    output = tmp_path / "indices"
    write_synthetic_raster(source, six_band_values(), crs="EPSG:32643")
    result = analyze_spectral_image(source, output_dir=output, write_rasters=True)

    assert result["success"] is True
    assert set(result["index_raster_paths"]) == {"NDVI", "NDWI", "NDBI"}
    with rasterio.open(result["index_raster_paths"]["NDVI"]) as written:
        assert written.count == 1
        assert (written.height, written.width) == (2, 2)
        assert written.crs.to_string() == "EPSG:32643"
        assert written.transform == from_origin(500000, 1200000, 10, 10)


def test_before_after_processing_and_change_statistics(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    write_synthetic_raster(before, six_band_values(red=1, nir=3, swir=1))
    write_synthetic_raster(after, six_band_values(red=1, nir=5, swir=1))

    result = analyze_before_after(before, after)
    assert result["success"] is True
    assert result["before"]["success"] is True
    assert result["after"]["success"] is True
    assert result["change"]["NDVI"]["valid_pixel_count"] == 4
    assert result["change"]["NDVI"]["mean"] == pytest.approx(4 / 6 - 0.5)


def test_before_after_dimension_mismatch_is_structured(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    write_synthetic_raster(before, six_band_values(shape=(2, 2)))
    write_synthetic_raster(after, six_band_values(shape=(3, 2)))

    result = analyze_before_after(before, after)
    assert result["success"] is False
    assert result["stage"] == "validation"


def test_changed_region_statistics_use_only_masked_pixels(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    mask = tmp_path / "change-mask.tif"
    before_values = six_band_values(red=1, nir=3, swir=1)
    after_values = six_band_values(red=1, nir=5, swir=1)
    write_synthetic_raster(before, before_values)
    write_synthetic_raster(after, after_values)
    write_mask(mask, np.array([[1, 0], [0, 1]], dtype=np.uint8))

    result = analyze_before_after(before, after, mask_path=mask)
    changed = result["changed_region"]
    assert result["success"] is True
    assert changed["changed_pixels"] == 2
    assert changed["total_valid_pixels"] == 4
    assert changed["change_percentage"] == 50.0
    assert changed["indices"]["NDVI"]["before_mean"] == 0.5
    assert changed["indices"]["NDVI"]["after_mean"] == pytest.approx(4 / 6)
    assert changed["indices"]["NDVI"]["delta"] == pytest.approx(4 / 6 - 0.5)
    assert changed["indices"]["NDWI"]["before_mean"] == 0.5
    assert changed["indices"]["NDBI"]["before_mean"] == -0.5


def test_zero_and_all_changed_masks_are_safe(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    zero_mask = tmp_path / "zero-mask.tif"
    all_mask = tmp_path / "all-mask.tif"
    write_synthetic_raster(before, six_band_values())
    write_synthetic_raster(after, six_band_values(nir=5))
    write_mask(zero_mask, np.zeros((2, 2), dtype=np.uint8))
    write_mask(all_mask, np.ones((2, 2), dtype=np.uint8))

    zero = analyze_before_after(before, after, mask_path=zero_mask)["changed_region"]
    all_changed = analyze_before_after(before, after, mask_path=all_mask)["changed_region"]
    assert zero["changed_pixels"] == 0
    assert zero["change_percentage"] == 0.0
    assert zero["indices"]["NDVI"]["before_mean"] is None
    assert all_changed["changed_pixels"] == 4
    assert all_changed["change_percentage"] == 100.0


def test_mask_nodata_and_invalid_pixels_are_excluded(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    mask = tmp_path / "mask.tif"
    before_values = six_band_values()
    after_values = six_band_values(nir=5)
    before_values[:, 0, 0] = np.nan
    write_synthetic_raster(before, before_values)
    write_synthetic_raster(after, after_values)
    write_mask(mask, np.ones((2, 2), dtype=np.uint8))

    result = analyze_before_after(before, after, mask_path=mask)
    changed = result["changed_region"]
    assert changed["changed_pixels"] == 3
    assert changed["total_valid_pixels"] == 3
    assert changed["change_percentage"] == 100.0
    assert changed["indices"]["NDVI"]["valid_changed_pixels"] == 3


def test_mask_dimensions_crs_and_transform_are_validated(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    write_synthetic_raster(before, six_band_values())
    write_synthetic_raster(after, six_band_values())

    wrong_dimensions = tmp_path / "wrong-dimensions.tif"
    write_mask(wrong_dimensions, np.ones((3, 2), dtype=np.uint8))
    dimension_result = analyze_before_after(before, after, mask_path=wrong_dimensions)
    assert dimension_result["success"] is False
    assert "dimensions" in dimension_result["error"]

    wrong_crs = tmp_path / "wrong-crs.tif"
    write_mask(wrong_crs, np.ones((2, 2), dtype=np.uint8), crs="EPSG:4326")
    crs_result = analyze_before_after(before, after, mask_path=wrong_crs)
    assert crs_result["success"] is False
    assert "CRS" in crs_result["error"]

    wrong_transform = tmp_path / "wrong-transform.tif"
    write_mask(wrong_transform, np.ones((2, 2), dtype=np.uint8), transform=from_origin(500001, 1200000, 10, 10))
    transform_result = analyze_before_after(before, after, mask_path=wrong_transform)
    assert transform_result["success"] is False
    assert "transform" in transform_result["error"]


def test_before_after_crs_and_transform_are_validated(tmp_path):
    before = tmp_path / "before.tif"
    after_crs = tmp_path / "after-crs.tif"
    after_transform = tmp_path / "after-transform.tif"
    write_synthetic_raster(before, six_band_values())
    write_synthetic_raster(after_crs, six_band_values(), crs="EPSG:4326")
    write_synthetic_raster(after_transform, six_band_values())
    with rasterio.open(after_transform, "r+") as destination:
        destination.transform = from_origin(500001, 1200000, 10, 10)

    crs_result = analyze_before_after(before, after_crs)
    transform_result = analyze_before_after(before, after_transform)
    assert crs_result["success"] is False
    assert "CRS" in crs_result["error"]
    assert transform_result["success"] is False
    assert "transforms" in transform_result["error"]
