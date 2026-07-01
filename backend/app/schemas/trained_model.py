from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Algorithm = Literal["random_forest", "logistic_regression", "xgboost"]
ProblemType = Literal["classification", "regression"]


class TrainModelRequest(BaseModel):
    materialization_id: int
    name: str | None = Field(default=None, min_length=2, max_length=100)

    label_column: str | None = None
    algorithm: Algorithm = "random_forest"
    problem_type: ProblemType = "classification"

    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    random_state: int = 42


class TrainModelResponse(BaseModel):
    id: int
    name: str
    algorithm: str

    materialization_id: int
    dataset_id: int

    label_column: str
    problem_type: str

    feature_columns: list[str]
    metrics: dict[str, Any]

    artifact_filename: str
    artifact_path: str

    train_rows: int
    test_rows: int

    test_size: float
    random_state: int

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TrainedModelListItem(BaseModel):
    id: int
    name: str
    algorithm: str
    problem_type: str
    materialization_id: int
    dataset_id: int
    label_column: str
    artifact_filename: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
