"""Deterministic evidence fusion for GeoSentinel-AI Phase 3.2.

This module consumes already-collected observations. It does not call external
APIs, import ChangeFormer, or invoke an LLM. HCS values are compatibility
scores, not probabilities or calibrated confidence values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


HYPOTHESES = (
    "wildfire_burn_scar",
    "deforestation_clearing",
    "urban_construction",
    "inundation_flood",
    "landslide_terrain_disturbance",
)
MODALITIES = ("change_geometry", "spectral", "weather", "fire", "terrain", "landcover", "news")


@dataclass(frozen=True)
class FusionConfig:
    """Transparent, adjustable signal thresholds and contribution weights."""

    vegetation_loss_delta: float = -0.10
    built_up_delta: float = 0.10
    water_delta: float = 0.10
    wet_precipitation_mm: float = 20.0
    steep_slope_degrees: float = 30.0
    small_gap: float = 0.10


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _item(source: str, observation: str, value: Any = None, status: str = "neutral", **extra: Any) -> dict[str, Any]:
    result = {"source": source, "observation": observation, "value": value, "status": status}
    result.update(extra)
    return result


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _period_relevance(value: Any, period: tuple[date, date] | None) -> str:
    if period is None:
        return "unknown"
    if isinstance(value, dict):
        start = _parse_date(value.get("before_date") or value.get("start_date"))
        end = _parse_date(value.get("after_date") or value.get("end_date"))
        if start is not None and end is not None:
            return "relevant" if not (end < period[0] or start > period[1]) else "outside_period"
    parsed = _parse_date(value)
    if parsed is not None:
        return "relevant" if period[0] <= parsed <= period[1] else "outside_period"
    return "unknown"


def _spatial_relevance(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        if value <= 1:
            return "inside_or_at_target"
        if value <= 50:
            return "nearby"
        return "distant"
    return "unknown"


def _change_data(change_detection: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    if not isinstance(change_detection, dict) or change_detection.get("success") is False:
        return {}, False
    source = change_detection.get("change_detection", change_detection)
    if not isinstance(source, dict):
        return {}, False
    return source, bool(source.get("success", True))


def _spectral_deltas(spectral: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(spectral, dict) or spectral.get("success") is False:
        return {}
    changed = spectral.get("changed_region", {})
    indices = changed.get("indices", {}) if isinstance(changed, dict) else {}
    result = {}
    for name, key in (("NDVI", "ndvi_delta"), ("NDWI", "ndwi_delta"), ("NDBI", "ndbi_delta")):
        value = indices.get(name, {}).get("delta") if isinstance(indices.get(name), dict) else None
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[key] = float(value)
    if not result and isinstance(spectral.get("change"), dict):
        # Compatibility with the existing whole-image output shape.
        for name, key in (("NDVI", "ndvi_delta"), ("NDWI", "ndwi_delta"), ("NDBI", "ndbi_delta")):
            value = spectral["change"].get(name, {}).get("mean")
            if isinstance(value, (int, float)) and math.isfinite(value):
                result[key] = float(value)
    return result


def _environmental_item(source_name: str, result: Any, period: tuple[date, date] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return _item(source_name, "evidence unavailable", status="missing")
    status = result.get("status")
    if status != "success":
        return _item(source_name, "evidence unavailable", status="missing", temporal_relevance="unknown", spatial_relevance="unknown")
    temporal = _period_relevance(result.get("period"), period)
    spatial = _spatial_relevance(result.get("distance_km"))
    return _item(source_name, "source completed successfully", status="available", temporal_relevance=temporal, spatial_relevance=spatial, original=result)


def _classifications(
    spectral: dict[str, float], environmental: dict[str, dict[str, Any]], config: FusionConfig
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[str]]:
    supporting = {hypothesis: [] for hypothesis in HYPOTHESES}
    contradicting = {hypothesis: [] for hypothesis in HYPOTHESES}
    missing = {hypothesis: [] for hypothesis in HYPOTHESES}
    flags: list[str] = []

    def usable(item: dict[str, Any]) -> bool:
        return item["status"] == "available" and item.get("temporal_relevance") != "outside_period"

    ndvi = spectral.get("ndvi_delta")
    ndwi = spectral.get("ndwi_delta")
    ndbi = spectral.get("ndbi_delta")
    if ndvi is None:
        for hypothesis in HYPOTHESES:
            missing[hypothesis].append(_item("spectral", "NDVI delta unavailable", status="missing"))
    elif ndvi <= config.vegetation_loss_delta:
        for hypothesis in ("wildfire_burn_scar", "deforestation_clearing", "landslide_terrain_disturbance"):
            supporting[hypothesis].append(_item("spectral", "NDVI decrease indicates vegetation loss", ndvi, "supporting"))
    if ndwi is None:
        missing["inundation_flood"].append(_item("spectral", "NDWI delta unavailable", status="missing"))
    elif ndwi >= config.water_delta:
        supporting["inundation_flood"].append(_item("spectral", "NDWI increase is compatible with increased surface water", ndwi, "supporting"))
    if ndbi is None:
        missing["urban_construction"].append(_item("spectral", "NDBI delta unavailable", status="missing"))
    elif ndbi >= config.built_up_delta:
        supporting["urban_construction"].append(_item("spectral", "NDBI increase is compatible with built-up change", ndbi, "supporting"))

    fire = environmental["fire"]
    if fire["status"] == "missing":
        missing["wildfire_burn_scar"].append(_item("fire", "FIRMS evidence unavailable", status="missing"))
    elif usable(fire):
        fire_result = fire["original"]
        count = fire_result.get("metrics", {}).get("detection_count", 0)
        if count:
            supporting["wildfire_burn_scar"].append(_item("fire", "FIRMS detections were observed", count, "supporting"))
        else:
            contradicting["wildfire_burn_scar"].append(_item("fire", "No FIRMS detections were observed in the searched context", count, "contradicting"))
            supporting["deforestation_clearing"].append(_item("fire", "Available fire search found no detections", count, "supporting"))

    weather = environmental["weather"]
    if weather["status"] == "missing":
        missing["inundation_flood"].append(_item("weather", "weather evidence unavailable", status="missing"))
    elif usable(weather):
        precipitation = weather["original"].get("metrics", {}).get("total_precipitation_mm")
        if isinstance(precipitation, (int, float)) and precipitation >= config.wet_precipitation_mm:
            supporting["inundation_flood"].append(_item("weather", "reported precipitation is relevant to flood context", precipitation, "supporting"))

    landcover = environmental["landcover"]
    if landcover["status"] == "missing":
        for hypothesis in HYPOTHESES:
            missing[hypothesis].append(_item("landcover", "land-cover context unavailable", status="missing"))
    elif usable(landcover):
        primary = landcover["original"].get("land_cover", {}).get("primary_class", "").lower()
        if any(word in primary for word in ("tree", "shrub", "grass", "cropland", "vegetation")):
            for hypothesis in ("wildfire_burn_scar", "deforestation_clearing"):
                supporting[hypothesis].append(_item("landcover", "pre-change cover is vegetated", primary, "supporting"))
        if "built" in primary:
            supporting["urban_construction"].append(_item("landcover", "land-cover context is built-up", primary, "supporting"))
        if "water" in primary or "wetland" in primary:
            supporting["inundation_flood"].append(_item("landcover", "land-cover context includes water/wetland", primary, "supporting"))

    terrain = environmental["terrain"]
    if terrain["status"] == "missing":
        missing["landslide_terrain_disturbance"].append(_item("terrain", "terrain evidence unavailable", status="missing"))
    elif usable(terrain):
        terrain_metrics = terrain["original"].get("metrics", {})
        slope = terrain_metrics.get("slope_degrees")
        if isinstance(slope, (int, float)) and slope >= config.steep_slope_degrees:
            supporting["landslide_terrain_disturbance"].append(_item("terrain", "reported slope is compatible with terrain disturbance context", slope, "supporting"))
            if ndbi is not None and ndbi >= config.built_up_delta:
                flags.append("High NDBI change occurs in steep terrain; urban interpretation requires caution.")
                contradicting["urban_construction"].append(_item("terrain", "steep terrain creates an urban-interpretation caution", slope, "contradicting"))

    news = environmental["news"]
    if news["status"] == "missing":
        missing["wildfire_burn_scar"].append(_item("news", "news evidence unavailable", status="missing"))
    elif usable(news):
        article_count = news["original"].get("metrics", {}).get("article_count", 0)
        if article_count:
            supporting["wildfire_burn_scar"].append(_item("news", "relevant public reports were found", article_count, "supporting"))
    return supporting, contradicting, missing, flags


def fuse_evidence(
    change_detection: dict[str, Any] | None,
    spectral_analysis: dict[str, Any] | None,
    environmental_evidence: dict[str, dict[str, Any]] | None,
    *,
    investigation_period: dict[str, Any] | None = None,
    config: FusionConfig | None = None,
) -> dict[str, Any]:
    """Fuse existing observations into an explainable hypothesis profile."""
    config = config or FusionConfig()
    period = None
    if investigation_period:
        start = _parse_date(investigation_period.get("before_date") or investigation_period.get("start_date"))
        end = _parse_date(investigation_period.get("after_date") or investigation_period.get("end_date"))
        if start and end:
            period = (start, end)
    environmental_evidence = environmental_evidence or {}
    environmental = {name: _environmental_item(name, environmental_evidence.get(name), period) for name in ("weather", "fire", "terrain", "landcover", "news")}
    spectral = _spectral_deltas(spectral_analysis)
    change, change_available = _change_data(change_detection)
    supporting, contradicting, missing, physical_flags = _classifications(spectral, environmental, config)
    temporal_flags = [
        f"{name} evidence is outside the investigation period."
        for name, item in environmental.items()
        if item.get("temporal_relevance") == "outside_period"
    ]
    spatial_flags = [
        f"{name} evidence has {item['spatial_relevance']} spatial relevance."
        for name, item in environmental.items()
        if item.get("spatial_relevance") in {"nearby", "distant"}
    ]

    weights = {
        "spectral": 0.15,
        "fire": 0.35,
        "weather": 0.20,
        "landcover": 0.20,
        "terrain": 0.25,
        "news": 0.05,
    }
    scored = []
    for hypothesis in HYPOTHESES:
        signals = []
        for evidence in supporting[hypothesis]:
            source = evidence["source"]
            signals.append({"source": source, "classification": "supporting", "weight": weights.get(source, 0.05), "observation": evidence["observation"]})
        for evidence in contradicting[hypothesis]:
            source = evidence["source"]
            signals.append({"source": source, "classification": "contradicting", "weight": -weights.get(source, 0.05), "observation": evidence["observation"]})
        score = _clamp(0.5 + sum(signal["weight"] for signal in signals))
        scored.append({"hypothesis": hypothesis, "hcs": score, "supporting_evidence": supporting[hypothesis], "contradicting_evidence": contradicting[hypothesis], "missing_evidence": missing[hypothesis], "physical_flags": physical_flags.copy(), "temporal_flags": temporal_flags.copy(), "spatial_flags": spatial_flags.copy(), "signals": signals})
    scored.sort(key=lambda entry: (-entry["hcs"], entry["hypothesis"]))
    for rank, entry in enumerate(scored, 1):
        entry["rank"] = rank
    available = ["change_geometry"] if change_available else []
    if isinstance(spectral_analysis, dict) and spectral_analysis.get("success") is not False:
        available.append("spectral")
    available.extend(name for name, item in environmental.items() if item["status"] == "available")
    available = list(dict.fromkeys(available))
    missing_modalities = [modality for modality in MODALITIES if modality not in available]
    top, second = scored[0], scored[1]
    gap = round(top["hcs"] - second["hcs"], 4)
    interpretation = "Competing hypotheses are weakly separated." if gap < config.small_gap else "Top hypothesis is substantially more compatible with available evidence."
    normalized = [_item("spectral", f"{key} delta", value, "neutral", temporal_relevance="unknown", spatial_relevance="unknown") for key, value in spectral.items()]
    normalized.extend(environmental.values())
    return {
        "fusion_version": "1.0",
        "change_geometry": change,
        "spectral_indicators": spectral,
        "normalized_evidence": normalized,
        "hypothesis_scores": scored,
        "cross_modal_analysis": {"supporting_evidence": [item for values in supporting.values() for item in values], "contradicting_evidence": [item for values in contradicting.values() for item in values], "missing_evidence": [item for values in missing.values() for item in values], "physical_sanity_flags": physical_flags, "temporal_flags": temporal_flags, "spatial_flags": spatial_flags},
        "observational_bounds": {"evidence_completeness_score": round(len(available) / len(MODALITIES), 4), "available_modalities": available, "missing_modalities": missing_modalities},
        "hypothesis_separation": {"top_hypothesis": top["hypothesis"], "top_score": top["hcs"], "second_hypothesis": second["hypothesis"], "second_score": second["hcs"], "hypothesis_gap": gap, "interpretation": interpretation},
    }


run_evidence_fusion = fuse_evidence
