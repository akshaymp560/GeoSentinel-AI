import json
from datetime import date
from pathlib import Path

import pytest

from backend.services import orchestrator


class FakeRecord:
    investigation_id = "GS-ORCH1234"
    latitude = 10.0
    longitude = 76.0
    before_date = date(2025, 1, 1)
    after_date = date(2025, 2, 1)
    status = "processing"
    result_json = None
    error = None
    completed_at = None


class FakeDB:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def component_success(name):
    return {"agent": name, "status": "success", "metrics": {}, "evidence": [], "source": "test"}


def patch_pipeline(monkeypatch, tmp_path, *, unavailable_agent=None):
    calls = []
    monkeypatch.setattr(
        orchestrator,
        "retrieve_sentinel2_images",
        lambda *args, **kwargs: calls.append(("gee", args, kwargs)) or {"success": True, "before_image_path": "before.tif", "after_image_path": "after.tif", "metadata": {"before_quality_path": "before_scl.tif", "after_quality_path": "after_scl.tif"}},
    )
    monkeypatch.setattr(
        orchestrator,
        "preprocess_sentinel2_images",
        lambda *args, **kwargs: calls.append(("preprocess", args, kwargs)) or {"success": True, "before_image_path": "pre_before.tif", "after_image_path": "pre_after.tif"},
    )
    monkeypatch.setattr(orchestrator, "_aligned_change_mask", lambda *args: calls.append(("align_mask", args, {})) or str(tmp_path / "mask.tif"))
    monkeypatch.setattr(
        orchestrator,
        "analyze_before_after",
        lambda *args, **kwargs: calls.append(("spectral", args, kwargs)) or {"success": True, "changed_region": {"indices": {}}, "change": {}},
    )
    monkeypatch.setattr(
        orchestrator,
        "run_change_detection",
        lambda *args, **kwargs: calls.append(("change", args, kwargs)) or {"success": True, "mask_path": "raw-mask.pgm", "changed_pixels": 2},
    )
    for name in ("weather", "fire", "terrain", "landcover", "news"):
        function_name = f"retrieve_{name}_evidence"
        monkeypatch.setattr(orchestrator, function_name, lambda *args, _name=name, **kwargs: calls.append((_name, args, kwargs)) or ({"agent": _name, "status": "failed", "error": "unavailable", "metrics": None, "evidence": []} if _name == unavailable_agent else component_success(_name)))
    monkeypatch.setattr(orchestrator, "fuse_evidence", lambda *args, **kwargs: calls.append(("fusion", args, kwargs)) or {"hypothesis_scores": [], "observational_bounds": {"available_modalities": ["spectral"], "evidence_completeness_score": 0.2}})
    monkeypatch.setattr(orchestrator, "run_qwen_reasoning", lambda *args, **kwargs: calls.append(("qwen", args, kwargs)) or {"claims": []})
    monkeypatch.setattr(orchestrator, "evaluate_consistency", lambda *args, **kwargs: calls.append(("consistency", args, kwargs)) or {"uncertainty": {"level": "MODERATE", "reasons": []}, "claim_results": []})
    return calls


def test_successful_pipeline_orders_stages_and_persists_json(monkeypatch, tmp_path):
    calls = patch_pipeline(monkeypatch, tmp_path)
    record, db = FakeRecord(), FakeDB()
    result = orchestrator.run_investigation(record, db, mock_mode=False)
    names = [item[0] for item in calls]
    assert record.status == "completed"
    assert db.commits == 1
    assert json.dumps(result)
    assert names.index("gee") < names.index("preprocess") < names.index("change") < names.index("spectral") < names.index("fusion") < names.index("qwen") < names.index("consistency")
    assert record.result_json["status"] == "completed"


def test_change_mask_reaches_spectral_analysis(monkeypatch, tmp_path):
    calls = patch_pipeline(monkeypatch, tmp_path)
    orchestrator.run_investigation(FakeRecord(), FakeDB(), mock_mode=False)
    aligned = next(item for item in calls if item[0] == "align_mask")
    spectral = next(item for item in calls if item[0] == "spectral")
    assert aligned[1][0] == "raw-mask.pgm"
    assert spectral[2]["mask_path"] == str(tmp_path / "mask.tif")


def test_all_environmental_agents_are_invoked(monkeypatch, tmp_path):
    calls = patch_pipeline(monkeypatch, tmp_path)
    orchestrator.run_investigation(FakeRecord(), FakeDB(), mock_mode=False)
    assert {name for name, *_ in calls} >= {"weather", "fire", "terrain", "landcover", "news"}


def test_qwen_claims_reach_consistency(monkeypatch, tmp_path):
    calls = patch_pipeline(monkeypatch, tmp_path)
    orchestrator.run_investigation(FakeRecord(), FakeDB(), mock_mode=False)
    qwen = next(item for item in calls if item[0] == "qwen")
    consistency = next(item for item in calls if item[0] == "consistency")
    assert consistency[1][1] == {"claims": []}
    assert qwen[0] == "qwen"


def test_unavailable_agent_is_preserved_without_failing_pipeline(monkeypatch, tmp_path):
    calls = patch_pipeline(monkeypatch, tmp_path, unavailable_agent="fire")
    result = orchestrator.run_investigation(FakeRecord(), FakeDB(), mock_mode=False)
    fusion_call = next(item for item in calls if item[0] == "fusion")
    environmental = fusion_call[1][2]
    assert result["status"] == "completed"
    assert environmental["fire"]["status"] == "failed"


@pytest.mark.parametrize("stage", ["gee", "preprocess", "change", "spectral", "fusion"])
def test_critical_stage_failure_persists_failed(monkeypatch, tmp_path, stage):
    calls = patch_pipeline(monkeypatch, tmp_path)
    if stage == "gee":
        monkeypatch.setattr(orchestrator, "retrieve_sentinel2_images", lambda *args, **kwargs: {"success": False, "error": "retrieval failed"})
    elif stage == "preprocess":
        monkeypatch.setattr(orchestrator, "preprocess_sentinel2_images", lambda *args, **kwargs: {"success": False, "error": "preprocessing failed"})
    elif stage == "change":
        monkeypatch.setattr(orchestrator, "run_change_detection", lambda *args, **kwargs: {"success": False, "error": "detector failed"})
    elif stage == "spectral":
        monkeypatch.setattr(orchestrator, "analyze_before_after", lambda *args, **kwargs: {"success": False, "error": "spectral failed"})
    else:
        monkeypatch.setattr(orchestrator, "fuse_evidence", lambda *args, **kwargs: None)
    record, db = FakeRecord(), FakeDB()
    result = orchestrator.run_investigation(record, db, mock_mode=False)
    assert record.status == "failed"
    assert result["status"] == "failed"
    assert db.commits == 1


def test_secret_like_errors_are_redacted(monkeypatch, tmp_path):
    patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(orchestrator, "retrieve_sentinel2_images", lambda *args, **kwargs: {"success": False, "error": "api_key=super-secret-value"})
    record, db = FakeRecord(), FakeDB()
    orchestrator.run_investigation(record, db, mock_mode=False)
    assert "super-secret-value" not in record.error
    assert "REDACTED" in record.error


def test_default_mock_mode_completes_without_external_services(tmp_path):
    record, db = FakeRecord(), FakeDB()
    result = orchestrator.run_investigation(record, db, mock_mode=True)
    assert result["status"] == "completed"
    assert record.status == "completed"
    assert Path(result["satellite"]["before_image_path"]).exists()

