from backend.services.fusion import HYPOTHESES, fuse_evidence


def spectral(ndvi=-0.3, ndwi=0.0, ndbi=0.2):
    return {
        "success": True,
        "changed_region": {
            "indices": {
                "NDVI": {"delta": ndvi},
                "NDWI": {"delta": ndwi},
                "NDBI": {"delta": ndbi},
            }
        },
    }


def environmental(*, fire_status="success", fire_count=2, slope=None, weather_period=None, distance=None):
    terrain_metrics = {"elevation_m": 100}
    if slope is not None:
        terrain_metrics["slope_degrees"] = slope
    fire = {"status": fire_status, "metrics": {"detection_count": fire_count}}
    if distance is not None:
        fire["distance_km"] = distance
    return {
        "weather": {"status": "success", "metrics": {"total_precipitation_mm": 50}, "period": weather_period or {"before_date": "2025-01-01", "after_date": "2025-01-03"}},
        "fire": fire,
        "terrain": {"status": "success", "metrics": terrain_metrics},
        "landcover": {"status": "success", "land_cover": {"primary_class": "Tree cover", "class_code": 10, "coverage_percentage": 80}},
        "news": {"status": "success", "metrics": {"article_count": 1}, "period": {"before_date": "2025-01-01", "after_date": "2025-01-03"}},
    }


def run(spectral_result=None, evidence=None, **kwargs):
    return fuse_evidence(
        {"success": True, "changed_pixels": 10, "total_pixels": 100, "change_percentage": 10},
        spectral_result or spectral(),
        evidence or environmental(),
        investigation_period={"before_date": "2025-01-01", "after_date": "2025-01-03"},
        **kwargs,
    )


def test_supporting_evidence_increases_compatibility():
    supported = run(evidence=environmental(fire_count=3))
    no_fire = run(evidence=environmental(fire_count=0))
    wildfire_supported = next(item for item in supported["hypothesis_scores"] if item["hypothesis"] == "wildfire_burn_scar")
    wildfire_without = next(item for item in no_fire["hypothesis_scores"] if item["hypothesis"] == "wildfire_burn_scar")
    assert wildfire_supported["hcs"] > wildfire_without["hcs"]


def test_contradicting_evidence_reduces_compatibility():
    with_fire = run(evidence=environmental(fire_count=2))
    without_fire = run(evidence=environmental(fire_count=0))
    assert next(x["hcs"] for x in with_fire["hypothesis_scores"] if x["hypothesis"] == "deforestation_clearing") < next(x["hcs"] for x in without_fire["hypothesis_scores"] if x["hypothesis"] == "deforestation_clearing")


def test_missing_fire_is_not_contradicting_fire():
    result = run(evidence=environmental(fire_status="failed"))
    wildfire = next(item for item in result["hypothesis_scores"] if item["hypothesis"] == "wildfire_burn_scar")
    assert any(item["status"] == "missing" for item in wildfire["missing_evidence"])
    assert not any(item["source"] == "fire" for item in wildfire["contradicting_evidence"])
    assert "fire" in result["observational_bounds"]["missing_modalities"]


def test_hypothesis_ranking_and_scores_are_bounded():
    result = run()
    scores = result["hypothesis_scores"]
    assert [item["rank"] for item in scores] == [1, 2, 3, 4, 5]
    assert all(0.0 <= item["hcs"] <= 1.0 for item in scores)
    assert {item["hypothesis"] for item in scores} == set(HYPOTHESES)


def test_completeness_changes_for_unavailable_modalities():
    complete = run()
    partial_evidence = environmental(fire_status="failed")
    partial_evidence["news"] = {"status": "failed"}
    partial = run(evidence=partial_evidence)
    assert complete["observational_bounds"]["evidence_completeness_score"] > partial["observational_bounds"]["evidence_completeness_score"]
    assert set(partial["observational_bounds"]["missing_modalities"]) >= {"fire", "news"}


def test_temporally_outside_evidence_is_not_direct_support():
    evidence = environmental(weather_period={"before_date": "2024-01-01", "after_date": "2024-01-03"})
    result = run(evidence=evidence)
    flood = next(item for item in result["hypothesis_scores"] if item["hypothesis"] == "inundation_flood")
    assert not any(item["source"] == "weather" for item in flood["supporting_evidence"])
    assert result["cross_modal_analysis"]["temporal_flags"]


def test_spatial_relevance_is_exposed_without_inventing_distance():
    result = run(evidence=environmental(distance=10))
    assert result["normalized_evidence"]
    fire_item = next(item for item in result["normalized_evidence"] if item["source"] == "fire")
    assert fire_item["spatial_relevance"] == "nearby"
    assert any("fire evidence has nearby" in flag for flag in result["cross_modal_analysis"]["spatial_flags"])


def test_steep_terrain_high_ndbi_adds_physical_caution():
    result = run(spectral_result=spectral(ndbi=0.4), evidence=environmental(slope=35))
    assert any("High NDBI change occurs in steep terrain" in flag for flag in result["cross_modal_analysis"]["physical_sanity_flags"])
    urban = next(item for item in result["hypothesis_scores"] if item["hypothesis"] == "urban_construction")
    assert urban["contradicting_evidence"]


def test_hypothesis_gap_is_calculated_and_interpreted():
    result = run()
    separation = result["hypothesis_separation"]
    assert separation["top_hypothesis"]
    assert separation["second_hypothesis"]
    assert separation["hypothesis_gap"] == round(separation["top_score"] - separation["second_score"], 4)
    assert separation["interpretation"]


def test_minimal_empty_evidence_does_not_crash():
    result = fuse_evidence(None, None, {})
    assert result["fusion_version"] == "1.0"
    assert len(result["hypothesis_scores"]) == 5
    assert result["observational_bounds"]["evidence_completeness_score"] == 0.0


def test_original_change_and_spectral_observations_are_preserved():
    result = run()
    assert result["change_geometry"]["changed_pixels"] == 10
    assert result["spectral_indicators"]["ndvi_delta"] == -0.3
    assert any(item["source"] == "spectral" for item in result["normalized_evidence"])


def test_signal_contributions_explain_the_hcs():
    result = run()
    for hypothesis in result["hypothesis_scores"]:
        assert "signals" in hypothesis
        assert all(signal["classification"] in {"supporting", "contradicting"} for signal in hypothesis["signals"])
        assert all(isinstance(signal["weight"], float) for signal in hypothesis["signals"])


def test_unavailable_environmental_input_is_not_negative_observation():
    evidence = environmental(fire_status="failed")
    result = run(evidence=evidence)
    fire_missing = [item for item in result["normalized_evidence"] if item["source"] == "fire"]
    assert fire_missing[0]["status"] == "missing"
    assert fire_missing[0]["observation"] == "evidence unavailable"
