from pathlib import Path

ROOT = Path.cwd()

files = {
    "backend/app/models/materialization.py": '''from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.db.database import Base


class Materialization(Base):
    __tablename__ = "materializations"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False, index=True)
    dataset_id = Column(Integer, nullable=False, index=True)

    feature_ids_json = Column(Text, nullable=False)
    feature_names_json = Column(Text, nullable=False)

    label_column = Column(String, nullable=True)

    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)

    stored_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
''',

    "backend/app/schemas/materialization.py": '''from datetime import datetime
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
''',

    "backend/app/services/materialization_service.py": '''import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.feature_definition import FeatureDefinition
from app.models.materialization import Materialization

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_\\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "materialization"


def _load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored dataset file not found.")

    return pd.read_csv(file_path)


def _output_feature_name(feature: FeatureDefinition) -> str:
    return f"{feature.name}_v{feature.version}"


def _apply_transformation(
    series: pd.Series,
    transformation: str,
) -> pd.Series:
    if transformation == "identity":
        return series

    if transformation == "log1p":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.clip(lower=0).apply(
            lambda value: None if pd.isna(value) else math.log1p(value)
        )

    if transformation == "abs":
        return pd.to_numeric(series, errors="coerce").abs()

    if transformation == "square":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric ** 2

    if transformation == "zscore":
        numeric = pd.to_numeric(series, errors="coerce")
        mean = numeric.mean()
        std = numeric.std()

        if std == 0 or pd.isna(std):
            return pd.Series([0] * len(series), index=series.index)

        return (numeric - mean) / std

    if transformation == "minmax":
        numeric = pd.to_numeric(series, errors="coerce")
        min_value = numeric.min()
        max_value = numeric.max()
        denominator = max_value - min_value

        if denominator == 0 or pd.isna(denominator):
            return pd.Series([0] * len(series), index=series.index)

        return (numeric - min_value) / denominator

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported transformation: {transformation}",
    )


def _materialize_column_feature(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature: FeatureDefinition,
) -> pd.DataFrame:
    if feature.source_column not in feature_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Source column '{feature.source_column}' not found for feature '{feature.name}'.",
        )

    output_name = _output_feature_name(feature)

    base_df[output_name] = _apply_transformation(
        feature_df[feature.source_column],
        feature.transformation,
    )

    return base_df


def _materialize_aggregate_feature(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature: FeatureDefinition,
) -> pd.DataFrame:
    if feature.entity_column not in feature_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Entity column '{feature.entity_column}' not found for feature '{feature.name}'.",
        )

    if feature.source_column not in feature_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Source column '{feature.source_column}' not found for feature '{feature.name}'.",
        )

    output_name = _output_feature_name(feature)

    grouped = feature_df.groupby(feature.entity_column)[feature.source_column]

    if feature.aggregation_function == "count":
        aggregate_series = grouped.count()
    elif feature.aggregation_function == "mean":
        aggregate_series = grouped.mean()
    elif feature.aggregation_function == "sum":
        aggregate_series = grouped.sum()
    elif feature.aggregation_function == "min":
        aggregate_series = grouped.min()
    elif feature.aggregation_function == "max":
        aggregate_series = grouped.max()
    elif feature.aggregation_function == "nunique":
        aggregate_series = grouped.nunique()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported aggregation function: {feature.aggregation_function}",
        )

    aggregate_df = aggregate_series.reset_index()
    aggregate_df = aggregate_df.rename(columns={feature.source_column: output_name})

    aggregate_df[output_name] = _apply_transformation(
        aggregate_df[output_name],
        feature.transformation,
    )

    base_df = base_df.merge(
        aggregate_df,
        on=feature.entity_column,
        how="left",
    )

    return base_df


def _get_features_for_materialization(
    db: Session,
    dataset_id: int,
    feature_ids: list[int] | None,
) -> list[FeatureDefinition]:
    query = db.query(FeatureDefinition).filter(
        FeatureDefinition.dataset_id == dataset_id,
        FeatureDefinition.status == "active",
    )

    if feature_ids:
        features = query.filter(FeatureDefinition.id.in_(feature_ids)).all()

        found_ids = {feature.id for feature in features}
        missing_ids = set(feature_ids) - found_ids

        if missing_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Feature IDs not found for this dataset: {sorted(missing_ids)}",
            )

        return features

    features = query.order_by(FeatureDefinition.created_at.asc()).all()

    if not features:
        raise HTTPException(
            status_code=400,
            detail="No active features found for this dataset. Create features first.",
        )

    return features


def create_materialization(
    db: Session,
    dataset_id: int,
    name: str | None = None,
    feature_ids: list[int] | None = None,
    label_column: str | None = None,
) -> Materialization:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _load_csv(dataset.stored_path)

    features = _get_features_for_materialization(
        db=db,
        dataset_id=dataset_id,
        feature_ids=feature_ids,
    )

    primary_entity_column = features[0].entity_column

    if primary_entity_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Primary entity column '{primary_entity_column}' not found in dataset.",
        )

    for feature in features:
        if feature.entity_column != primary_entity_column:
            raise HTTPException(
                status_code=400,
                detail="All features in one materialization must currently use the same entity_column.",
            )

    output_df = pd.DataFrame()
    output_df["__row_id"] = range(len(df))
    output_df[primary_entity_column] = df[primary_entity_column]

    if label_column is not None:
        if label_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Label column '{label_column}' not found in dataset.",
            )

        output_df[label_column] = df[label_column]

    for feature in features:
        if feature.feature_kind == "column":
            output_df = _materialize_column_feature(
                base_df=output_df,
                feature_df=df,
                feature=feature,
            )

        elif feature.feature_kind == "aggregate":
            output_df = _materialize_aggregate_feature(
                base_df=output_df,
                feature_df=df,
                feature=feature,
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported feature kind: {feature.feature_kind}",
            )

    clean_name = _safe_name(
        name or f"{dataset.name}_offline_features"
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stored_filename = f"{clean_name}_{timestamp}.csv"
    stored_path = PROCESSED_DATA_DIR / stored_filename

    output_df.to_csv(stored_path, index=False)

    feature_ids_used = [feature.id for feature in features]
    feature_names_used = [_output_feature_name(feature) for feature in features]

    materialization = Materialization(
        name=clean_name,
        dataset_id=dataset_id,
        feature_ids_json=json.dumps(feature_ids_used),
        feature_names_json=json.dumps(feature_names_used),
        label_column=label_column,
        rows=int(output_df.shape[0]),
        columns=int(output_df.shape[1]),
        stored_filename=stored_filename,
        stored_path=str(stored_path),
    )

    db.add(materialization)
    db.commit()
    db.refresh(materialization)

    return materialization


def materialization_to_response(materialization: Materialization) -> dict[str, Any]:
    return {
        "id": materialization.id,
        "name": materialization.name,
        "dataset_id": materialization.dataset_id,
        "feature_ids": json.loads(materialization.feature_ids_json),
        "feature_names": json.loads(materialization.feature_names_json),
        "label_column": materialization.label_column,
        "rows": materialization.rows,
        "columns": materialization.columns,
        "stored_filename": materialization.stored_filename,
        "stored_path": materialization.stored_path,
        "created_at": materialization.created_at,
    }


def preview_materialization(
    materialization: Materialization,
    limit: int = 10,
) -> dict[str, Any]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Preview limit must be between 1 and 100.")

    path = Path(materialization.stored_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored materialization file not found.")

    df = pd.read_csv(path).head(limit)
    df = df.where(pd.notnull(df), None)

    return {
        "materialization_id": materialization.id,
        "name": materialization.name,
        "rows_returned": int(df.shape[0]),
        "preview": df.to_dict(orient="records"),
    }
''',

    "backend/app/routers/materializations.py": '''from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.materialization import Materialization
from app.schemas.materialization import (
    MaterializationCreate,
    MaterializationListItem,
    MaterializationPreviewResponse,
    MaterializationResponse,
)
from app.services.materialization_service import (
    create_materialization,
    materialization_to_response,
    preview_materialization,
)

router = APIRouter(prefix="/materializations", tags=["Materializations"])


@router.post("", response_model=MaterializationResponse)
def materialize_features(
    request: MaterializationCreate,
    db: Session = Depends(get_db),
):
    materialization = create_materialization(
        db=db,
        dataset_id=request.dataset_id,
        name=request.name,
        feature_ids=request.feature_ids,
        label_column=request.label_column,
    )

    return materialization_to_response(materialization)


@router.get("", response_model=list[MaterializationListItem])
def list_materializations(
    dataset_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Materialization)

    if dataset_id is not None:
        query = query.filter(Materialization.dataset_id == dataset_id)

    return query.order_by(Materialization.created_at.desc()).all()


@router.get("/{materialization_id}", response_model=MaterializationResponse)
def get_materialization(
    materialization_id: int,
    db: Session = Depends(get_db),
):
    materialization = (
        db.query(Materialization)
        .filter(Materialization.id == materialization_id)
        .first()
    )

    if materialization is None:
        raise HTTPException(status_code=404, detail="Materialization not found.")

    return materialization_to_response(materialization)


@router.get("/{materialization_id}/preview", response_model=MaterializationPreviewResponse)
def get_materialization_preview(
    materialization_id: int,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    materialization = (
        db.query(Materialization)
        .filter(Materialization.id == materialization_id)
        .first()
    )

    if materialization is None:
        raise HTTPException(status_code=404, detail="Materialization not found.")

    return preview_materialization(
        materialization=materialization,
        limit=limit,
    )
''',

    "backend/tests/test_materialization.py": '''from io import BytesIO

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
        b"3,80,zomato,0\\n"
        b"3,300,amazon,0\\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "materialization_test_transactions"},
    )

    assert response.status_code == 200

    return response.json()["id"]


def _create_features(dataset_id: int):
    column_response = client.post(
        "/api/features",
        json={
            "name": "mat_amount_log",
            "dataset_id": dataset_id,
            "description": "Log-transformed transaction amount.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "column",
            "transformation": "log1p",
            "output_dtype": "float",
        },
    )

    assert column_response.status_code == 200

    aggregate_response = client.post(
        "/api/features",
        json={
            "name": "mat_user_avg_amount",
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

    assert aggregate_response.status_code == 200

    return [
        column_response.json()["id"],
        aggregate_response.json()["id"],
    ]


def test_create_materialization():
    dataset_id = _create_dataset()
    feature_ids = _create_features(dataset_id)

    response = client.post(
        "/api/materializations",
        json={
            "dataset_id": dataset_id,
            "name": "fraud_training_features",
            "feature_ids": feature_ids,
            "label_column": "is_fraud",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["dataset_id"] == dataset_id
    assert body["rows"] == 6
    assert body["columns"] >= 5
    assert body["label_column"] == "is_fraud"


def test_list_materializations():
    response = client.get("/api/materializations")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
''',
}


main_py = '''from fastapi import FastAPI

from app.db.database import Base, engine
from app.models import dataset, feature_definition, materialization
from app.routers.datasets import router as dataset_router
from app.routers.features import router as feature_router
from app.routers.health import router as health_router
from app.routers.materializations import router as materialization_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FeatureForge API",
    description="Lightweight ML Feature Store backend",
    version="0.4.0",
)

app.include_router(health_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(feature_router, prefix="/api")
app.include_router(materialization_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "FeatureForge",
        "message": "ML Feature Store API is running",
        "docs": "/docs",
    }
'''


def write_files():
    print("Adding Offline Feature Materialization to FeatureForge...")

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

    print("\\nOffline Feature Materialization added successfully.")
    print("\\nNew API routes:")
    print("POST /api/materializations")
    print("GET  /api/materializations")
    print("GET  /api/materializations/{materialization_id}")
    print("GET  /api/materializations/{materialization_id}/preview")


if __name__ == "__main__":
    write_files()