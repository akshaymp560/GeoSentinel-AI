import logging
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .schemas.api_models import (
    InvestigationCreateRequest,
    InvestigationCreateResponse,
    InvestigationResultResponse,
    InvestigationStatusResponse,
)
from .services.change_interface import run_change_detection
from .services.orchestrator import run_investigation
from .storage.database import Base, engine, get_db
from .storage.models import Investigation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="GeoSentinel-AI API", version="1.0.0")
Base.metadata.create_all(bind=engine)


def _new_investigation_id() -> str:
    return f"GS-{uuid.uuid4().hex[:8].upper()}"


def _run_pipeline(investigation_id: str) -> None:
    db = next(get_db())
    try:
        investigation = db.scalar(select(Investigation).where(Investigation.investigation_id == investigation_id))
        if investigation is None:
            logger.error("Investigation %s disappeared before pipeline start", investigation_id)
            return
        # Pass the module-level detector so existing callers/tests can replace
        # the stable ChangeFormer boundary without changing the orchestrator.
        run_investigation(investigation, db, change_detector=run_change_detection)
    except Exception:  # noqa: BLE001 - the orchestrator normally persists failures
        db.rollback()
        logger.exception("Investigation %s failed before orchestration could persist its state", investigation_id)
    finally:
        db.close()


@app.post("/api/investigations", response_model=InvestigationCreateResponse, status_code=status.HTTP_202_ACCEPTED)
def create_investigation(
    request: InvestigationCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    investigation_id = _new_investigation_id()
    record = Investigation(
        investigation_id=investigation_id,
        latitude=request.latitude,
        longitude=request.longitude,
        before_date=request.before_date,
        after_date=request.after_date,
        status="processing",
    )
    db.add(record)
    db.commit()
    background_tasks.add_task(_run_pipeline, investigation_id)
    return {"investigation_id": investigation_id, "status": "processing"}


def _get_investigation(investigation_id: str, db: Session) -> Investigation:
    record = db.scalar(select(Investigation).where(Investigation.investigation_id == investigation_id))
    if record is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return record


@app.get("/api/investigations/{investigation_id}/status", response_model=InvestigationStatusResponse)
def investigation_status(investigation_id: str, db: Session = Depends(get_db)):
    record = _get_investigation(investigation_id, db)
    return {"investigation_id": record.investigation_id, "status": record.status}


@app.get("/api/investigations/{investigation_id}", response_model=InvestigationResultResponse)
def investigation_result(investigation_id: str, db: Session = Depends(get_db)):
    record = _get_investigation(investigation_id, db)
    return {
        "investigation_id": record.investigation_id,
        "status": record.status,
        "result": record.result_json,
        "error": record.error,
        "created_at": record.created_at,
        "completed_at": record.completed_at,
    }
