"""Asynchronous investigation pipeline coordinator for Phase 3.4.

The orchestrator coordinates existing services and agents.  It contains no
remote-sensing calculations, ChangeFormer implementation, or Qwen model code.
Production providers are selected with ``GEOSENTINEL_MOCK_MODE=false``;
deterministic adapters keep the normal local test/application path offline.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ..agents.fire_api import retrieve_fire_evidence as _retrieve_fire_evidence
from ..agents.landcover_api import retrieve_landcover_evidence as _retrieve_landcover_evidence
from ..agents.news_api import retrieve_news_evidence as _retrieve_news_evidence
from ..agents.terrain_api import retrieve_terrain_evidence as _retrieve_terrain_evidence
from ..agents.weather_api import retrieve_weather_evidence as _retrieve_weather_evidence
from .change_interface import run_change_detection as _run_change_detection
from .consistency_engine import evaluate_consistency as _evaluate_consistency
from .fusion import fuse_evidence as _fuse_evidence
from .gee_retriever import retrieve_sentinel2_images as _retrieve_sentinel2_images
from .qwen_interface import run_qwen_reasoning as _run_qwen_reasoning
from .satellite_preprocessing import preprocess_sentinel2_images as _preprocess_sentinel2_images
from .spectral_analysis import analyze_before_after as _analyze_before_after

logger = logging.getLogger(__name__)

# These module-level boundaries are intentionally patchable for unit tests and
# replaceable by the future isolated model/service adapters.
retrieve_sentinel2_images = _retrieve_sentinel2_images
preprocess_sentinel2_images = _preprocess_sentinel2_images
run_change_detection = _run_change_detection
analyze_before_after = _analyze_before_after
retrieve_weather_evidence = _retrieve_weather_evidence
retrieve_fire_evidence = _retrieve_fire_evidence
retrieve_terrain_evidence = _retrieve_terrain_evidence
retrieve_landcover_evidence = _retrieve_landcover_evidence
retrieve_news_evidence = _retrieve_news_evidence
fuse_evidence = _fuse_evidence
run_qwen_reasoning = _run_qwen_reasoning
evaluate_consistency = _evaluate_consistency


class PipelineFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(message)


@dataclass
class PipelineState:
    investigation_id: str
    latitude: float
    longitude: float
    before_date: date
    after_date: date
    artifact_dir: Path


def _safe_error(error: Any) -> str:
    message = str(error)
    message = re.sub(r"(?i)(api[_-]?key|token|password|secret)=([^&\s]+)", r"\1=[REDACTED]", message)
    return message[:1000]


def _require_success(value: Any, stage: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineFailure(stage, "component returned a non-structured result")
    if value.get("success") is False:
        raise PipelineFailure(stage, _safe_error(value.get("error", "component reported failure")))
    return value


def _write_mock_raster(path: Path, values: np.ndarray, *, scl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = values.shape[-2:]
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1 if scl else 6,
        "dtype": "uint8" if scl else "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(76.0, 10.0, 0.001, 0.001),
        "nodata": 0 if scl else np.nan,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(values.astype(profile["dtype"]))


def _mock_satellite(state: PipelineState) -> dict[str, Any]:
    satellite_dir = state.artifact_dir / "satellite"
    shape = (6, 512, 512)
    before = np.zeros(shape, dtype=np.float32)
    after = np.zeros(shape, dtype=np.float32)
    before[:] = np.array([0.10, 0.12, 0.20, 0.40, 0.15, 0.10], dtype=np.float32)[:, None, None]
    after[:] = np.array([0.10, 0.12, 0.25, 0.32, 0.20, 0.15], dtype=np.float32)[:, None, None]
    before_path = satellite_dir / "before.tif"
    after_path = satellite_dir / "after.tif"
    before_scl = satellite_dir / "before_scl.tif"
    after_scl = satellite_dir / "after_scl.tif"
    _write_mock_raster(before_path, before)
    _write_mock_raster(after_path, after)
    _write_mock_raster(before_scl, np.full((1, 512, 512), 4, dtype=np.uint8), scl=True)
    _write_mock_raster(after_scl, np.full((1, 512, 512), 4, dtype=np.uint8), scl=True)
    return {
        "success": True,
        "before_image_path": str(before_path),
        "after_image_path": str(after_path),
        "before_scl_path": str(before_scl),
        "after_scl_path": str(after_scl),
        "metadata": {"bands": ["B2", "B3", "B4", "B8", "B11", "B12"], "mock": True},
    }


def _mock_environmental(state: PipelineState) -> dict[str, dict[str, Any]]:
    location = {"latitude": state.latitude, "longitude": state.longitude}
    period = {"before_date": state.before_date.isoformat(), "after_date": state.after_date.isoformat()}
    return {
        "weather": {"agent": "weather", "status": "success", "location": location, "period": period, "metrics": {"total_precipitation_mm": 0, "observed_days": 0}, "evidence": [], "source": "mock"},
        "fire": {"agent": "fire", "status": "success", "location": location, "period": period, "metrics": {"detection_count": 0}, "detections": [], "evidence": [], "source": "mock"},
        "terrain": {"agent": "terrain", "status": "success", "location": location, "metrics": {"elevation_m": 0}, "evidence": [], "source": "mock"},
        "landcover": {"agent": "landcover", "status": "success", "location": location, "land_cover": {"primary_class": "Tree cover", "class_code": 10, "coverage_percentage": 100}, "evidence": [], "source": "mock"},
        "news": {"agent": "news", "status": "success", "location": location, "period": period, "metrics": {"article_count": 0}, "articles": [], "evidence": [], "source": "mock"},
    }


def _aligned_change_mask(mask_path: str | Path, reference_path: str | Path, output_path: Path) -> str:
    """Adapt mask formats that lack geospatial metadata at the boundary.

    The current mock detector writes a compact PGM mask, while spectral
    analysis intentionally validates CRS and transform.  Real model wrappers
    may already return an aligned GeoTIFF; those paths pass through unchanged.
    """
    mask = Path(mask_path)
    with rasterio.open(mask) as source, rasterio.open(reference_path) as reference:
        source_profile = source.profile
        aligned = (
            source_profile.get("crs") is not None
            and source_profile.get("transform") == reference.transform
            and source.width == reference.width
            and source.height == reference.height
        )
        if aligned:
            return str(mask)
        if source.width != reference.width or source.height != reference.height:
            raise PipelineFailure("spectral_analysis", "Change mask dimensions must match the preprocessed imagery")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = reference.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0)
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write((source.read(1) > 0).astype(np.uint8), 1)
    return str(output_path)


def _collect_environmental(state: PipelineState, *, mock_mode: bool) -> dict[str, dict[str, Any]]:
    if mock_mode:
        return _mock_environmental(state)
    calls: dict[str, Callable[..., dict[str, Any]]] = {
        "weather": retrieve_weather_evidence,
        "fire": retrieve_fire_evidence,
        "terrain": retrieve_terrain_evidence,
        "landcover": retrieve_landcover_evidence,
        "news": retrieve_news_evidence,
    }
    results: dict[str, dict[str, Any]] = {}
    for name, function in calls.items():
        try:
            kwargs: dict[str, Any] = {"latitude": state.latitude, "longitude": state.longitude}
            if name in {"weather", "fire", "news"}:
                kwargs.update(before_date=state.before_date, after_date=state.after_date)
            if name == "landcover":
                kwargs["gee_project"] = os.getenv("GEE_PROJECT")
            result = function(**kwargs)
            results[name] = result if isinstance(result, dict) else {"agent": name, "status": "failed", "error": "invalid agent result", "metrics": None, "evidence": []}
        except Exception as exc:  # noqa: BLE001 - an optional source must not stop the pipeline
            logger.exception("Environmental agent %s failed", name)
            results[name] = {"agent": name, "status": "failed", "error": _safe_error(exc), "metrics": None, "evidence": []}
    return results


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, (date, Path)):
        return value.isoformat() if isinstance(value, date) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return str(value)


def _persist_failure(record: Any, db: Any, failure: PipelineFailure) -> dict[str, Any]:
    record.status = "failed"
    record.error = _safe_error(failure.message)
    record.completed_at = __import__("datetime").datetime.now(__import__("datetime").UTC)
    db.commit()
    return {"status": "failed", "stage": failure.stage, "error": record.error}


def run_investigation(
    record: Any,
    db: Any,
    *,
    change_detector: Callable[..., dict[str, Any]] | None = None,
    mock_mode: bool | None = None,
) -> dict[str, Any]:
    """Run one investigation and persist its terminal state."""
    state = PipelineState(record.investigation_id, record.latitude, record.longitude, record.before_date, record.after_date, Path(__file__).resolve().parents[1] / "artifacts" / "investigations" / record.investigation_id)
    use_mocks = mock_mode if mock_mode is not None else os.getenv("GEOSENTINEL_MOCK_MODE", "true").lower() in {"1", "true", "yes", "on"}
    detector = change_detector or run_change_detection
    try:
        state.artifact_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] Investigation started", state.investigation_id)
        logger.info("[%s] GEE retrieval started", state.investigation_id)
        satellite = _mock_satellite(state) if use_mocks else _require_success(retrieve_sentinel2_images(state.investigation_id, state.latitude, state.longitude, state.before_date, state.after_date, artifact_root=Path(__file__).resolve().parents[1] / "artifacts" / "investigations"), "gee_retrieval")
        logger.info("[%s] GEE retrieval completed", state.investigation_id)

        logger.info("[%s] Preprocessing started", state.investigation_id)
        preprocessing = _require_success(preprocess_sentinel2_images(satellite["before_image_path"], satellite["after_image_path"], satellite.get("before_scl_path") or satellite["metadata"].get("before_quality_path"), satellite.get("after_scl_path") or satellite["metadata"].get("after_quality_path"), state.artifact_dir / "preprocessed"), "preprocessing")
        logger.info("[%s] Preprocessing completed", state.investigation_id)

        logger.info("[%s] Change detection started", state.investigation_id)
        change = _require_success(detector(preprocessing["before_image_path"], preprocessing["after_image_path"], state.artifact_dir / "change"), "change_detection")
        logger.info("[%s] Change detection completed", state.investigation_id)

        logger.info("[%s] Spectral analysis started", state.investigation_id)
        mask_path = _aligned_change_mask(change["mask_path"], preprocessing["before_image_path"], state.artifact_dir / "change" / "change_mask_aligned.tif")
        spectral = _require_success(analyze_before_after(preprocessing["before_image_path"], preprocessing["after_image_path"], output_dir=state.artifact_dir / "spectral", mask_path=mask_path), "spectral_analysis")
        logger.info("[%s] Spectral analysis completed", state.investigation_id)

        environmental = _collect_environmental(state, mock_mode=use_mocks)
        logger.info("[%s] Environmental evidence collected", state.investigation_id)
        fusion = fuse_evidence(change, spectral, environmental, investigation_period={"before_date": state.before_date, "after_date": state.after_date})
        if not isinstance(fusion, dict):
            raise PipelineFailure("fusion", "fusion returned a non-structured result")
        logger.info("[%s] Fusion completed", state.investigation_id)
        qwen = run_qwen_reasoning(fusion)
        logger.info("[%s] Qwen reasoning completed", state.investigation_id)
        consistency = evaluate_consistency(fusion, qwen)
        logger.info("[%s] Consistency validation completed", state.investigation_id)

        final = _json_safe({"investigation_id": state.investigation_id, "status": "completed", "location": {"latitude": state.latitude, "longitude": state.longitude}, "period": {"before": state.before_date, "after": state.after_date}, "satellite": satellite, "preprocessing": preprocessing, "change_detection": change, "spectral_analysis": spectral, "environmental_evidence": environmental, "fusion": fusion, "ai_reasoning": qwen, "consistency": consistency, "uncertainty": consistency.get("uncertainty", {}), "warnings": []})
        json.dumps(final)
        record.result_json = final
        record.status = "completed"
        record.error = None
        record.completed_at = __import__("datetime").datetime.now(__import__("datetime").UTC)
        db.commit()
        logger.info("[%s] Investigation completed", state.investigation_id)
        return final
    except PipelineFailure as exc:
        logger.error("[%s] Stage failed: %s", state.investigation_id, exc.stage)
        return _persist_failure(record, db, exc)
    except Exception as exc:  # noqa: BLE001 - background jobs must persist failure state
        logger.exception("[%s] Investigation failed", state.investigation_id)
        return _persist_failure(record, db, PipelineFailure("pipeline", _safe_error(exc)))
