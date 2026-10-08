from backend.services.consistency_engine import evaluate_consistency
from backend.services.fusion import fuse_evidence


def fusion(*, fire_status="success", fire_count=2, ndvi=-0.3, ndwi=0.0, ndbi=0.0, slope=None, available_change=True):
    env = {
        "weather": {"status": "success", "metrics": {"total_precipitation_mm": 5}},
        "fire": {"status": fire_status, "metrics": {"detection_count": fire_count}},
        "terrain": {"status": "success", "metrics": {"slope_degrees": slope or 10}},
        "landcover": {"status": "success", "land_cover": {"primary_class": "Tree cover"}},
        "news": {"status": "success", "metrics": {"article_count": 0}},
    }
    return fuse_evidence(
        {"success": available_change, "changed_pixels": 5} if available_change else None,
        {"success": True, "changed_region": {"indices": {"NDVI": {"delta": ndvi}, "NDWI": {"delta": ndwi}, "NDBI": {"delta": ndbi}}}} if available_change else None,
        env,
    )


def claim(value, claim_id="c1", claim_type="cause"):
    return {"claims": [{"claim_id": claim_id, "claim_type": claim_type, "subject": "change", "value": value}]}


def result(result):
    return result["claim_results"][0]


def test_supported_wildfire_claim():
    assert result(evaluate_consistency(fusion(), claim("wildfire_burn_scar"))) ["status"] == "SUPPORTED"


def test_contradicted_wildfire_claim():
    assert result(evaluate_consistency(fusion(fire_count=0, ndvi=0.0), claim("wildfire_burn_scar"))) ["status"] == "CONTRADICTED"


def test_unavailable_fire_is_not_no_fire():
    assert result(evaluate_consistency(fusion(fire_status="failed"), claim("no_fire_detected", claim_type="fire_activity")))["status"] == "UNSUPPORTED"


def test_available_zero_fire_differs_from_unavailable():
    assert result(evaluate_consistency(fusion(fire_count=0), claim("no_fire_detected", claim_type="fire_activity")))["status"] == "SUPPORTED"
    assert result(evaluate_consistency(fusion(fire_status="failed"), claim("no_fire_detected", claim_type="fire_activity")))["status"] == "UNSUPPORTED"


def test_supported_deforestation_claim():
    assert result(evaluate_consistency(fusion(fire_count=0), claim("deforestation_clearing")))["status"] == "SUPPORTED"


def test_physical_warning_contradicts_urban_claim():
    checked = evaluate_consistency(fusion(ndbi=0.4, slope=40), claim("urban_construction"))
    assert result(checked)["status"] == "CONTRADICTED"
    assert any(item["type"] == "physical_conflict" for item in checked["conflicts"])


def test_hypothesis_conflict_is_recorded():
    checked = evaluate_consistency(fusion(fire_count=0, ndvi=0.0), {"primary_hypothesis": "wildfire_burn_scar"})
    assert any(item["type"] == "hypothesis_conflict" for item in checked["conflicts"])


def test_large_gap_with_complete_evidence_has_low_uncertainty():
    checked = evaluate_consistency(fusion(), claim("wildfire_burn_scar"))
    assert checked["uncertainty"]["level"] in {"LOW", "MODERATE"}


def test_small_gap_increases_uncertainty():
    evidence = fusion(ndvi=0.0, ndbi=0.0, fire_count=0)
    evidence["hypothesis_separation"]["hypothesis_gap"] = 0.05
    checked = evaluate_consistency(evidence, claim("deforestation_clearing"))
    assert checked["uncertainty"]["level"] in {"HIGH", "MODERATE"}
    assert checked["uncertainty"]["reasons"]


def test_missing_modalities_increase_uncertainty():
    checked = evaluate_consistency(fusion(fire_status="failed"), claim("wildfire_burn_scar"))
    assert checked["uncertainty"]["level"] in {"HIGH", "MODERATE"}


def test_no_evidence_is_insufficient():
    checked = evaluate_consistency(fuse_evidence(None, None, {}), claim("wildfire_burn_scar"))
    assert checked["uncertainty"]["level"] == "INSUFFICIENT_EVIDENCE"


def test_multiple_contradictions_are_inconsistent():
    evidence = fusion(fire_count=0, ndvi=0.0)
    flood = next(item for item in evidence["hypothesis_scores"] if item["hypothesis"] == "inundation_flood")
    flood["contradicting_evidence"] = [{"source": "weather", "observation": "precipitation evidence contradicts flooding", "value": 0, "status": "contradicting"}]
    checked = evaluate_consistency(evidence, {"claims": [{"claim_id": "a", "value": "wildfire_burn_scar"}, {"claim_id": "b", "value": "inundation_flood"}]})
    assert checked["summary"]["contradicted"] >= 2
    assert checked["overall_consistency"] == "INCONSISTENT"


def test_claim_evidence_references_are_preserved():
    checked = fusion()
    checked["hypothesis_scores"][0]["supporting_evidence"][0]["evidence_id"] = "spectral-1"
    result_item = result(evaluate_consistency(checked, claim(checked["hypothesis_scores"][0]["hypothesis"])))
    assert "spectral-1" in result_item["evidence_refs"]


def test_no_qwen_claims_returns_safe_empty_evaluation():
    checked = evaluate_consistency(fusion(), None)
    assert checked["claim_results"] == []
    assert checked["summary"]["claims_evaluated"] == 0


def test_result_contains_structured_consistency_contract():
    checked = evaluate_consistency(fusion(), claim("wildfire_burn_scar"))
    assert checked["consistency_version"] == "1.0"
    assert set(checked) >= {"claim_results", "summary", "conflicts", "uncertainty", "overall_consistency"}
    assert set(checked["summary"]) == {"claims_evaluated", "supported", "contradicted", "unsupported"}
