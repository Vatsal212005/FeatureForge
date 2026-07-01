from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FeatureKind = Literal["column", "aggregate"]
Transformation = Literal["identity", "log1p", "zscore", "minmax", "abs", "square"]
AggregationFunction = Literal["count", "mean", "sum", "min", "max", "nunique"]


class FeatureCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    dataset_id: int
    description: str | None = None

    entity_column: str
    source_column: str | None = None
    timestamp_column: str | None = None

    feature_kind: FeatureKind = "column"
    transformation: Transformation = "identity"

    aggregation_function: AggregationFunction | None = None
    window_days: int | None = Field(default=None, ge=1, le=3650)

    output_dtype: str = "float"


class FeatureResponse(BaseModel):
    id: int
    name: str
    version: int
    description: str | None

    dataset_id: int
    entity_column: str
    source_column: str | None
    timestamp_column: str | None

    feature_kind: str
    transformation: str
    aggregation_function: str | None
    window_days: int | None

    output_dtype: str
    status: str
    definition_hash: str

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FeatureListItem(BaseModel):
    id: int
    name: str
    version: int
    dataset_id: int
    feature_kind: str
    transformation: str
    aggregation_function: str | None
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FeatureValidationResponse(BaseModel):
    valid: bool
    message: str
    available_columns: list[str]
