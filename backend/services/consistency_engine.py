"""Deterministic consistency and uncertainty checks for Phase 3.3.

The engine accepts the structured output of ``fusion.py`` and optional claims
from a future Qwen wrapper.  It performs no network calls and has no Qwen
dependency.  Compatibility scores remain observations; they are not treated
as calibrated probabilities.
"""

from __future__ import annotations

import math
from typing import Any


SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNSUPPORTED = "UNSUPPORTED"

HYPOTHESES = {
    "wildfire_burn_scar",
    "deforestation_clearing",
    "urban_construction",
    "inundation_flood",
    "landslide_terrain_disturbance",
}


def _claims_input(claims: dict[str, Any] | list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(claims, dict):
        return (claims if isinstance(claims, list) else []), None
    primary = claims.get("primary_hypothesis")
    supplied = claims.get("claims", [])
    result = list(supplied) if isinstance(supplied, list) else []
    if isinstance(primary, str):
        result.insert(0, {"claim_id": "primary_hypothesis", "claim_type": "cause", "subject": "change", "value": primary})
    return result, primary if isinstance(primary, str) else None


def _hypothesis_scores(fusion: dict[str, Any]) -> list[dict[str, Any]]:
    scores = fusion.get("hypothesis_scores", []) if isinstance(fusion, dict) else []
    return [item for item in scores if isinstance(item, dict) and isinstance(item.get("hypothesis"), str)]


def _hypothesis_entry(fusion: dict[str, Any], hypothesis: str) -> dict[str, Any] | None:
    return next((item for item in _hypothesis_scores(fusion) if item["hypothesis"] == hypothesis), None)


def _available(fusion: dict[str, Any], source: str) -> bool:
    bounds = fusion.get("observational_bounds", {}) if isinstance(fusion, dict) else {}
    return source in bounds.get("available_modalities", [])


def _evidence_items(fusion: dict[str, Any], key: str) -> list[dict[str, Any]]:
    cross = fusion.get("cross_modal_analysis", {}) if isinstance(fusion, dict) else {}
    items = cross.get(key, []) if isinstance(cross, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _refs(items: list[dict[str, Any]]) -> list[Any]:
    refs: list[Any] = []
    for item in items:
        for key in ("evidence_id", "observation_id", "source_id", "id"):
            if item.get(key) is not None and item[key] not in refs:
                refs.append(item[key])
    return refs


def _text(claim: dict[str, Any]) -> str:
    values = [claim.get("claim_type"), claim.get("subject"), claim.get("value"), claim.get("text")]
    return " ".join(str(value).lower() for value in values if value is not None)


def _fire_state(fusion: dict[str, Any]) -> str:
    if not _available(fusion, "fire"):
        return "unknown"
    items = _evidence_items(fusion, "supporting_evidence") + _evidence_items(fusion, "contradicting_evidence")
    fire_items = [item for item in items if item.get("source") == "fire"]
    if any(isinstance(item.get("value"), (int, float)) and item["value"] > 0 for item in fire_items):
        return "detected"
    if any(item.get("source") == "fire" for item in fire_items):
        return "none_detected"
    return "unknown"


def _hypothesis_claim(fusion: dict[str, Any], hypothesis: str) -> tuple[str, str, list[dict[str, Any]]]:
    entry = _hypothesis_entry(fusion, hypothesis)
    scores = _hypothesis_scores(fusion)
    supporting = entry.get("supporting_evidence", []) if entry else []
    contradicting = entry.get("contradicting_evidence", []) if entry else []
    if not entry or not scores:
        return UNSUPPORTED, "No fusion hypothesis profile is available.", supporting + contradicting
    if not _available(fusion, "spectral") and not _available(fusion, "change_geometry"):
        return UNSUPPORTED, "No usable change evidence is available to verify the hypothesis.", supporting + contradicting
    top = scores[0].get("hypothesis")
    gap = fusion.get("hypothesis_separation", {}).get("hypothesis_gap", 0)
    if top == hypothesis:
        if supporting and not contradicting:
            return SUPPORTED, "The claim matches the top fusion hypothesis and available evidence supports it.", supporting
        if supporting:
            return SUPPORTED, "The claim matches the top fusion hypothesis, with mixed supporting signals.", supporting + contradicting
        return UNSUPPORTED, "The claim is ranked first, but direct supporting evidence is unavailable.", supporting + contradicting
    if contradicting and (isinstance(gap, (int, float)) and gap >= 0.10):
        return CONTRADICTED, f"Fusion ranks {top} above the claimed hypothesis with a material compatibility gap.", supporting + contradicting
    if top and isinstance(gap, (int, float)) and gap >= 0.10:
        return CONTRADICTED, f"Fusion ranks {top} above the claimed hypothesis with a material compatibility gap.", supporting + contradicting
    return UNSUPPORTED, "Available fusion evidence does not separate this hypothesis sufficiently.", supporting + contradicting


def _evaluate_claim(claim: dict[str, Any], fusion: dict[str, Any]) -> tuple[str, str, list[Any], dict[str, Any] | None]:
    text = _text(claim)
    refs: list[dict[str, Any]] = []
    fire_claim = (
        str(claim.get("claim_type", "")).lower() in {"fire", "fire_activity", "firms"}
        or any(token in text for token in ("no_fire", "no fire", "no detected", "none detected", "without fire"))
        or str(claim.get("value", "")).lower() in {"detected_fire", "fire_detected", "no_fire_detected"}
    )
    if fire_claim:
        state = _fire_state(fusion)
        no_fire = any(token in text for token in ("no_fire", "no fire", "no detected", "none detected", "without fire"))
        if state == "unknown":
            status, reason = UNSUPPORTED, "Fire evidence is unavailable; absence of detections cannot be established."
        elif no_fire and state == "none_detected":
            status, reason = SUPPORTED, "FIRMS evidence is available and reports zero detections in the searched context."
        elif no_fire and state == "detected":
            status, reason = CONTRADICTED, "Available FIRMS evidence contains one or more detections."
        elif state == "detected":
            status, reason = SUPPORTED, "Available FIRMS evidence contains one or more detections."
        else:
            status, reason = UNSUPPORTED, "Available FIRMS evidence does not directly verify this fire claim."
        refs = [item for item in _evidence_items(fusion, "supporting_evidence") + _evidence_items(fusion, "contradicting_evidence") if item.get("source") == "fire"]
        return status, reason, _refs(refs), None

    hypothesis = next((name for name in HYPOTHESES if name in text), None)
    if hypothesis:
        status, reason, evidence = _hypothesis_claim(fusion, hypothesis)
        physical = [flag for flag in (fusion.get("cross_modal_analysis", {}).get("physical_sanity_flags", [])) if isinstance(flag, str)]
        if hypothesis == "urban_construction" and physical and status == SUPPORTED:
            status, reason = CONTRADICTED, "Urban interpretation conflicts with a physical consistency warning."
            evidence = evidence + [{"evidence_id": flag} for flag in physical]
        return status, reason, _refs(evidence), _hypothesis_entry(fusion, hypothesis)

    return UNSUPPORTED, "The claim type or value is not covered by deterministic consistency rules.", [], None


def _uncertainty(fusion: dict[str, Any], results: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    bounds = fusion.get("observational_bounds", {}) if isinstance(fusion, dict) else {}
    completeness = bounds.get("evidence_completeness_score", 0.0)
    separation = fusion.get("hypothesis_separation", {}) if isinstance(fusion, dict) else {}
    gap = separation.get("hypothesis_gap", 0.0)
    missing = bounds.get("missing_modalities", [])
    reasons: list[str] = []
    if not isinstance(completeness, (int, float)) or completeness <= 0:
        return {"level": "INSUFFICIENT_EVIDENCE", "reasons": ["No usable evidence modalities are available."]}
    if completeness < 0.5:
        reasons.append("Evidence completeness is low and critical modalities are missing.")
    elif missing:
        reasons.append("Some evidence modalities are unavailable.")
    if not isinstance(gap, (int, float)) or gap < 0.10:
        reasons.append("Top two hypotheses have a small compatibility gap.")
    if any(result["status"] == CONTRADICTED for result in results):
        reasons.append("One or more claims conflict with available evidence.")
    if any(conflict["type"] == "physical_conflict" for conflict in conflicts):
        reasons.append("A claim conflicts with a physical consistency warning.")
    if len([result for result in results if result["status"] == CONTRADICTED]) >= 2:
        level = "HIGH"
    elif completeness < 0.5 or (not isinstance(gap, (int, float)) or gap < 0.10):
        level = "HIGH"
    elif reasons:
        level = "MODERATE"
    else:
        level = "LOW"
    return {"level": level, "reasons": reasons}


def evaluate_consistency(fusion_result: dict[str, Any] | None, qwen_claims: dict[str, Any] | list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Evaluate structured claims against deterministic fusion evidence."""
    fusion = fusion_result if isinstance(fusion_result, dict) else {}
    claims, _ = _claims_input(qwen_claims)
    claim_results: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for index, claim in enumerate(claims, 1):
        if not isinstance(claim, dict):
            continue
        status, reason, evidence_refs, hypothesis_entry = _evaluate_claim(claim, fusion)
        claim_id = claim.get("claim_id", f"claim_{index}")
        result = {"claim_id": claim_id, "status": status, "reason": reason, "evidence_refs": evidence_refs}
        claim_results.append(result)
        if hypothesis_entry and status == CONTRADICTED:
            separation = fusion.get("hypothesis_separation", {})
            conflicts.append({"type": "hypothesis_conflict", "claim": claim.get("value"), "fusion_top_hypothesis": separation.get("top_hypothesis"), "severity": "HIGH" if separation.get("hypothesis_gap", 0) >= 0.10 else "MODERATE", "reason": reason})
        if "urban_construction" in _text(claim) and any("urban interpretation" in str(flag).lower() for flag in fusion.get("cross_modal_analysis", {}).get("physical_sanity_flags", [])) and status == CONTRADICTED:
            conflicts.append({"type": "physical_conflict", "claim": claim.get("value"), "severity": "HIGH", "reason": "Urban interpretation conflicts with a physical consistency warning."})
        if status == UNSUPPORTED and ("fire" in _text(claim) or "firms" in _text(claim)) and not _available(fusion, "fire"):
            conflicts.append({"type": "missing_evidence", "claim": claim.get("value"), "severity": "MODERATE", "reason": "Fire evidence is unavailable."})

    summary = {"claims_evaluated": len(claim_results), "supported": sum(r["status"] == SUPPORTED for r in claim_results), "contradicted": sum(r["status"] == CONTRADICTED for r in claim_results), "unsupported": sum(r["status"] == UNSUPPORTED for r in claim_results)}
    if not claim_results and (not fusion or not fusion.get("observational_bounds", {}).get("available_modalities")):
        overall = "INSUFFICIENT_EVIDENCE"
    elif summary["contradicted"] and summary["supported"]:
        overall = "MIXED"
    elif summary["contradicted"]:
        overall = "INCONSISTENT"
    elif summary["unsupported"] == summary["claims_evaluated"] and summary["claims_evaluated"]:
        overall = "INSUFFICIENT_EVIDENCE"
    else:
        overall = "CONSISTENT"
    return {"consistency_version": "1.0", "claim_results": claim_results, "summary": summary, "conflicts": conflicts, "uncertainty": _uncertainty(fusion, claim_results, conflicts), "overall_consistency": overall}


run_consistency_check = evaluate_consistency
