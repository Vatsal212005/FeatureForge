from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


DeduplicationStrategy = Literal["last", "first", "mean"]


class OnlineStoreMaterializeRequest(BaseModel):
    materialization_id: int
    entity_column: str | None = None
    deduplication_strategy: DeduplicationStrategy = "last"


class OnlineStoreMaterializeResponse(BaseModel):
    materialization_id: int
    entity_column: str
    deduplication_strategy: str
    entities_stored: int
    feature_columns: list[str]
    created_at: datetime


class OnlineFeatureLookupResponse(BaseModel):
    materialization_id: int
    entity_column: str
    entity_value: str
    features: dict[str, Any]
    updated_at: datetime


class BatchOnlineFeatureLookupRequest(BaseModel):
    entity_column: str
    entity_values: list[str | int | float] = Field(..., min_length=1)


class BatchOnlineFeatureLookupResponse(BaseModel):
    materialization_id: int
    entity_column: str
    found: int
    missing: int
    results: list[OnlineFeatureLookupResponse]


class OnlineStoreStatsResponse(BaseModel):
    materialization_id: int
    entity_count: int
    entity_columns: list[str]
    sample_feature_keys: list[str]


class OnlinePredictionRequest(BaseModel):
    materialization_id: int | None = None
    entity_column: str
    entity_values: list[str | int | float] = Field(..., min_length=1)


class OnlinePredictionResult(BaseModel):
    entity_value: str
    prediction: Any
    probabilities: dict[str, float] | None = None
    features_used: dict[str, Any]


class OnlinePredictionResponse(BaseModel):
    model_id: int
    model_name: str
    materialization_id: int
    entity_column: str
    rows_predicted: int
    predictions: list[OnlinePredictionResult]
    created_at: datetime
