from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    row_index: int
    prediction: Any
    probabilities: dict[str, float] | None = None


class PredictionResponse(BaseModel):
    model_id: int
    model_name: str
    algorithm: str
    problem_type: str
    rows_predicted: int
    predictions: list[PredictionResult]
    created_at: datetime


class BatchPredictionRequest(BaseModel):
    materialization_id: int | None = None
    limit: int | None = Field(default=None, ge=1, le=10000)


class BatchPredictionResponse(BaseModel):
    model_id: int
    model_name: str
    source: str
    rows_predicted: int
    predictions: list[PredictionResult]
    created_at: datetime
