from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DriftReportCreate(BaseModel):
    reference_materialization_id: int
    current_materialization_id: int
    name: str | None = Field(default=None, min_length=2, max_length=100)
    feature_columns: list[str] | None = None


class DriftColumnMetric(BaseModel):
    column: str
    dtype: str
    metric_type: str
    drift_score: float
    drift_level: str
    details: dict[str, Any]


class DriftReportResponse(BaseModel):
    id: int
    name: str

    reference_materialization_id: int
    current_materialization_id: int

    compared_columns: list[str]
    metrics: list[DriftColumnMetric]

    overall_drift_score: float
    drift_level: str

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DriftReportListItem(BaseModel):
    id: int
    name: str
    reference_materialization_id: int
    current_materialization_id: int
    overall_drift_score: float
    drift_level: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
