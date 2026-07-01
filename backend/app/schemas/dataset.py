from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    nullable: bool
    missing_count: int
    missing_percentage: float
    unique_count: int
    sample_values: list[Any]


class DatasetResponse(BaseModel):
    id: int
    name: str
    version: int
    original_filename: str
    stored_filename: str
    file_hash: str
    rows: int
    columns: int
    column_schema: list[ColumnSchema]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetListItem(BaseModel):
    id: int
    name: str
    version: int
    original_filename: str
    rows: int
    columns: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetPreviewResponse(BaseModel):
    dataset_id: int
    name: str
    version: int
    rows_returned: int
    preview: list[dict[str, Any]]
