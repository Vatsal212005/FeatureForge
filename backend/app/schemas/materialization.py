from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MaterializationCreate(BaseModel):
    dataset_id: int
    name: str | None = Field(default=None, min_length=2, max_length=100)
    feature_ids: list[int] | None = None
    label_column: str | None = None


class MaterializationResponse(BaseModel):
    id: int
    name: str
    dataset_id: int

    feature_ids: list[int]
    feature_names: list[str]

    label_column: str | None

    rows: int
    columns: int

    stored_filename: str
    stored_path: str

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MaterializationListItem(BaseModel):
    id: int
    name: str
    dataset_id: int
    rows: int
    columns: int
    stored_filename: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MaterializationPreviewResponse(BaseModel):
    materialization_id: int
    name: str
    rows_returned: int
    preview: list[dict[str, Any]]
