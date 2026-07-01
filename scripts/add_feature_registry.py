from pathlib import Path

ROOT = Path.cwd()

files = {
    "backend/app/models/feature_definition.py": '''from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

from app.db.database import Base


class FeatureDefinition(Base):
    __tablename__ = "feature_definitions"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)
    description = Column(Text, nullable=True)

    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)

    entity_column = Column(String, nullable=False)
    source_column = Column(String, nullable=True)
    timestamp_column = Column(String, nullable=True)

    feature_kind = Column(String, nullable=False, default="column")
    transformation = Column(String, nullable=False, default="identity")

    aggregation_function = Column(String, nullable=True)
    window_days = Column(Integer, nullable=True)

    output_dtype = Column(String, nullable=False, default="float")
    status = Column(String, nullable=False, default="active")

    definition_hash = Column(String, nullable=False, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_feature_name_version"),
    )
''',

    "backend/app/schemas/feature_definition.py": '''from datetime import datetime
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
''',

    "backend/app/services/feature_service.py": '''import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.feature_definition import FeatureDefinition
from app.schemas.feature_definition import FeatureCreate

SUPPORTED_TRANSFORMATIONS = {"identity", "log1p", "zscore", "minmax", "abs", "square"}
SUPPORTED_AGGREGATIONS = {"count", "mean", "sum", "min", "max", "nunique"}

NUMERIC_DTYPE_MARKERS = ("int", "float", "double", "number")


def _safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_\\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "feature"


def _definition_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _next_feature_version(db: Session, feature_name: str) -> int:
    latest_feature = (
        db.query(FeatureDefinition)
        .filter(FeatureDefinition.name == feature_name)
        .order_by(FeatureDefinition.version.desc())
        .first()
    )

    if latest_feature is None:
        return 1

    return latest_feature.version + 1


def _load_dataset_schema(dataset: Dataset) -> list[dict[str, Any]]:
    try:
        return json.loads(dataset.column_schema_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Dataset schema metadata is corrupted.",
        ) from exc


def _column_map(dataset: Dataset) -> dict[str, dict[str, Any]]:
    schema = _load_dataset_schema(dataset)
    return {column["name"]: column for column in schema}


def _require_column(columns: dict[str, dict[str, Any]], column_name: str, label: str):
    if column_name not in columns:
        raise HTTPException(
            status_code=400,
            detail=f"{label} '{column_name}' does not exist in the selected dataset.",
        )


def _is_numeric_dtype(dtype: str) -> bool:
    return any(marker in dtype.lower() for marker in NUMERIC_DTYPE_MARKERS)


def validate_feature_definition(dataset: Dataset, feature: FeatureCreate) -> list[str]:
    columns = _column_map(dataset)
    available_columns = list(columns.keys())

    _require_column(columns, feature.entity_column, "Entity column")

    if feature.source_column is not None:
        _require_column(columns, feature.source_column, "Source column")

    if feature.timestamp_column is not None:
        _require_column(columns, feature.timestamp_column, "Timestamp column")

    if feature.transformation not in SUPPORTED_TRANSFORMATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported transformation: {feature.transformation}",
        )

    if feature.feature_kind == "column":
        if feature.source_column is None:
            raise HTTPException(
                status_code=400,
                detail="Column features require source_column.",
            )

        if feature.aggregation_function is not None:
            raise HTTPException(
                status_code=400,
                detail="Column features should not have aggregation_function.",
            )

        if feature.window_days is not None:
            raise HTTPException(
                status_code=400,
                detail="Column features should not have window_days.",
            )

    if feature.feature_kind == "aggregate":
        if feature.aggregation_function is None:
            raise HTTPException(
                status_code=400,
                detail="Aggregate features require aggregation_function.",
            )

        if feature.aggregation_function not in SUPPORTED_AGGREGATIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported aggregation function: {feature.aggregation_function}",
            )

        if feature.source_column is None:
            raise HTTPException(
                status_code=400,
                detail="Aggregate features require source_column.",
            )

        if feature.window_days is not None and feature.timestamp_column is None:
            raise HTTPException(
                status_code=400,
                detail="Windowed aggregate features require timestamp_column.",
            )

    numeric_transformations = {"log1p", "zscore", "minmax", "abs", "square"}

    if feature.transformation in numeric_transformations and feature.source_column is not None:
        dtype = columns[feature.source_column]["dtype"]

        if not _is_numeric_dtype(dtype):
            raise HTTPException(
                status_code=400,
                detail=f"Transformation '{feature.transformation}' requires a numeric source column. "
                f"Column '{feature.source_column}' has dtype '{dtype}'.",
            )

    return available_columns


def create_feature_definition(db: Session, feature: FeatureCreate) -> FeatureDefinition:
    dataset = db.query(Dataset).filter(Dataset.id == feature.dataset_id).first()

    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    validate_feature_definition(dataset, feature)

    clean_name = _safe_name(feature.name)
    version = _next_feature_version(db, clean_name)

    definition_payload = {
        "name": clean_name,
        "version": version,
        "dataset_id": feature.dataset_id,
        "entity_column": feature.entity_column,
        "source_column": feature.source_column,
        "timestamp_column": feature.timestamp_column,
        "feature_kind": feature.feature_kind,
        "transformation": feature.transformation,
        "aggregation_function": feature.aggregation_function,
        "window_days": feature.window_days,
        "output_dtype": feature.output_dtype,
    }

    feature_definition = FeatureDefinition(
        name=clean_name,
        version=version,
        description=feature.description,
        dataset_id=feature.dataset_id,
        entity_column=feature.entity_column,
        source_column=feature.source_column,
        timestamp_column=feature.timestamp_column,
        feature_kind=feature.feature_kind,
        transformation=feature.transformation,
        aggregation_function=feature.aggregation_function,
        window_days=feature.window_days,
        output_dtype=feature.output_dtype,
        definition_hash=_definition_hash(definition_payload),
    )

    db.add(feature_definition)
    db.commit()
    db.refresh(feature_definition)

    return feature_definition


def preview_feature_definition(
    db: Session,
    feature_id: int,
    limit: int = 10,
) -> dict[str, Any]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Preview limit must be between 1 and 100.")

    feature = (
        db.query(FeatureDefinition)
        .filter(FeatureDefinition.id == feature_id)
        .first()
    )

    if feature is None:
        raise HTTPException(status_code=404, detail="Feature definition not found.")

    dataset = db.query(Dataset).filter(Dataset.id == feature.dataset_id).first()

    if dataset is None:
        raise HTTPException(status_code=404, detail="Linked dataset not found.")

    path = Path(dataset.stored_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored dataset file not found.")

    df = pd.read_csv(path)

    if feature.feature_kind == "column":
        preview_df = df[[feature.entity_column, feature.source_column]].head(limit).copy()

        output_column = f"{feature.name}_v{feature.version}"

        if feature.transformation == "identity":
            preview_df[output_column] = preview_df[feature.source_column]
        elif feature.transformation == "log1p":
            preview_df[output_column] = preview_df[feature.source_column].clip(lower=0).map(lambda x: pd.NA if pd.isna(x) else __import__("math").log1p(x))
        elif feature.transformation == "abs":
            preview_df[output_column] = preview_df[feature.source_column].abs()
        elif feature.transformation == "square":
            preview_df[output_column] = preview_df[feature.source_column] ** 2
        elif feature.transformation == "zscore":
            mean = df[feature.source_column].mean()
            std = df[feature.source_column].std()
            preview_df[output_column] = (preview_df[feature.source_column] - mean) / std if std != 0 else 0
        elif feature.transformation == "minmax":
            min_value = df[feature.source_column].min()
            max_value = df[feature.source_column].max()
            denominator = max_value - min_value
            preview_df[output_column] = (preview_df[feature.source_column] - min_value) / denominator if denominator != 0 else 0
        else:
            raise HTTPException(status_code=400, detail="Unsupported transformation.")

    else:
        grouped = df.groupby(feature.entity_column)[feature.source_column]

        if feature.aggregation_function == "count":
            result = grouped.count()
        elif feature.aggregation_function == "mean":
            result = grouped.mean()
        elif feature.aggregation_function == "sum":
            result = grouped.sum()
        elif feature.aggregation_function == "min":
            result = grouped.min()
        elif feature.aggregation_function == "max":
            result = grouped.max()
        elif feature.aggregation_function == "nunique":
            result = grouped.nunique()
        else:
            raise HTTPException(status_code=400, detail="Unsupported aggregation function.")

        output_column = f"{feature.name}_v{feature.version}"
        preview_df = result.reset_index().rename(columns={feature.source_column: output_column}).head(limit)

    preview_df = preview_df.where(pd.notnull(preview_df), None)

    return {
        "feature_id": feature.id,
        "name": feature.name,
        "version": feature.version,
        "feature_kind": feature.feature_kind,
        "transformation": feature.transformation,
        "aggregation_function": feature.aggregation_function,
        "rows_returned": int(preview_df.shape[0]),
        "preview": preview_df.to_dict(orient="records"),
    }
''',

    "backend/app/routers/features.py": '''from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.dataset import Dataset
from app.models.feature_definition import FeatureDefinition
from app.schemas.feature_definition import (
    FeatureCreate,
    FeatureListItem,
    FeatureResponse,
    FeatureValidationResponse,
)
from app.services.feature_service import (
    create_feature_definition,
    preview_feature_definition,
    validate_feature_definition,
)

router = APIRouter(prefix="/features", tags=["Features"])


@router.post("", response_model=FeatureResponse)
def create_feature(
    feature: FeatureCreate,
    db: Session = Depends(get_db),
):
    return create_feature_definition(db=db, feature=feature)


@router.get("", response_model=list[FeatureListItem])
def list_features(
    dataset_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(FeatureDefinition)

    if dataset_id is not None:
        query = query.filter(FeatureDefinition.dataset_id == dataset_id)

    return query.order_by(FeatureDefinition.created_at.desc()).all()


@router.get("/{feature_id}", response_model=FeatureResponse)
def get_feature(
    feature_id: int,
    db: Session = Depends(get_db),
):
    feature = (
        db.query(FeatureDefinition)
        .filter(FeatureDefinition.id == feature_id)
        .first()
    )

    if feature is None:
        raise HTTPException(status_code=404, detail="Feature definition not found.")

    return feature


@router.post("/validate", response_model=FeatureValidationResponse)
def validate_feature(
    feature: FeatureCreate,
    db: Session = Depends(get_db),
):
    dataset = db.query(Dataset).filter(Dataset.id == feature.dataset_id).first()

    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    available_columns = validate_feature_definition(dataset, feature)

    return {
        "valid": True,
        "message": "Feature definition is valid.",
        "available_columns": available_columns,
    }


@router.get("/{feature_id}/preview")
def preview_feature(
    feature_id: int,
    limit: int = 10,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return preview_feature_definition(
        db=db,
        feature_id=feature_id,
        limit=limit,
    )
''',

    "backend/tests/test_feature_registry.py": '''from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_dataset():
    csv_content = (
        b"user_id,amount,merchant,is_fraud\\n"
        b"1,100,amazon,0\\n"
        b"1,250,swiggy,0\\n"
        b"2,9000,unknown,1\\n"
        b"2,1000,amazon,0\\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "feature_test_transactions"},
    )

    assert response.status_code == 200

    return response.json()["id"]


def test_create_column_feature():
    dataset_id = _create_dataset()

    response = client.post(
        "/api/features",
        json={
            "name": "amount_log",
            "dataset_id": dataset_id,
            "description": "Log-transformed transaction amount.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "column",
            "transformation": "log1p",
            "output_dtype": "float",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "amount_log"
    assert body["feature_kind"] == "column"
    assert body["transformation"] == "log1p"


def test_create_aggregate_feature():
    dataset_id = _create_dataset()

    response = client.post(
        "/api/features",
        json={
            "name": "user_avg_amount",
            "dataset_id": dataset_id,
            "description": "Average transaction amount per user.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "aggregate",
            "aggregation_function": "mean",
            "transformation": "identity",
            "output_dtype": "float",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "user_avg_amount"
    assert body["feature_kind"] == "aggregate"
    assert body["aggregation_function"] == "mean"


def test_list_features():
    response = client.get("/api/features")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
''',
}


main_py = '''from fastapi import FastAPI

from app.db.database import Base, engine
from app.models import dataset, feature_definition
from app.routers.datasets import router as dataset_router
from app.routers.features import router as feature_router
from app.routers.health import router as health_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FeatureForge API",
    description="Lightweight ML Feature Store backend",
    version="0.3.0",
)

app.include_router(health_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(feature_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "FeatureForge",
        "message": "ML Feature Store API is running",
        "docs": "/docs",
    }
'''


def write_files():
    print("Adding Feature Registry to FeatureForge...")

    for file_path, content in files.items():
        path = ROOT / file_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            print(f"Skipped existing file: {file_path}")
            continue

        path.write_text(content, encoding="utf-8")
        print(f"Created file: {file_path}")

    main_path = ROOT / "backend" / "app" / "main.py"
    main_path.write_text(main_py, encoding="utf-8")
    print("Updated file: backend/app/main.py")

    print("\\nFeature Registry added successfully.")
    print("\\nNew API routes:")
    print("POST /api/features")
    print("POST /api/features/validate")
    print("GET  /api/features")
    print("GET  /api/features/{feature_id}")
    print("GET  /api/features/{feature_id}/preview")


if __name__ == "__main__":
    write_files()