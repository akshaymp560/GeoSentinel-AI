from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Status = Literal["processing", "completed", "failed"]


class InvestigationCreateRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    before_date: date
    after_date: date

    @model_validator(mode="after")
    def dates_are_ordered(self):
        if self.before_date >= self.after_date:
            raise ValueError("before_date must be earlier than after_date")
        return self


class InvestigationCreateResponse(BaseModel):
    investigation_id: str
    status: Status


class InvestigationStatusResponse(BaseModel):
    investigation_id: str
    status: Status


class InvestigationResultResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    investigation_id: str
    status: Status
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
