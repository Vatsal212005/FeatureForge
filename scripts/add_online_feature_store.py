from pathlib import Path

ROOT = Path.cwd()

files = {
    "backend/app/models/online_feature.py": '''from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint

from app.db.database import Base


class OnlineFeature(Base):
    __tablename__ = "online_features"

    id = Column(Integer, primary_key=True, index=True)

    materialization_id = Column(Integer, nullable=False, index=True)

    entity_column = Column(String, nullable=False, index=True)
    entity_value = Column(String, nullable=False, index=True)

    features_json = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "materialization_id",
            "entity_column",
            "entity_value",
            name="uq_online_feature_entity",
        ),
    )
''',

    "backend/app/schemas/online_feature.py": '''from datetime import datetime
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
''',

    "backend/app/services/online_feature_service.py": '''import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.materialization import Materialization
from app.models.online_feature import OnlineFeature
from app.models.trained_model import TrainedModel


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def _row_to_json_dict(row: pd.Series) -> dict[str, Any]:
    return {
        str(key): _json_safe_value(value)
        for key, value in row.to_dict().items()
    }


def _load_materialization_df(materialization: Materialization) -> pd.DataFrame:
    path = Path(materialization.stored_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Stored materialization file not found.",
        )

    return pd.read_csv(path)


def _resolve_entity_column(
    df: pd.DataFrame,
    requested_entity_column: str | None,
) -> str:
    if requested_entity_column:
        if requested_entity_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Entity column '{requested_entity_column}' not found in materialized table.",
            )

        return requested_entity_column

    ignored_columns = {"__row_id", "is_fraud", "label", "target", "y"}

    candidate_columns = [
        column for column in df.columns
        if column not in ignored_columns
    ]

    if not candidate_columns:
        raise HTTPException(
            status_code=400,
            detail="Could not infer entity column. Please provide entity_column.",
        )

    return candidate_columns[0]


def _deduplicate_entity_rows(
    df: pd.DataFrame,
    entity_column: str,
    strategy: str,
) -> pd.DataFrame:
    if strategy == "last":
        if "__row_id" in df.columns:
            df = df.sort_values("__row_id")

        return df.groupby(entity_column, as_index=False).tail(1)

    if strategy == "first":
        if "__row_id" in df.columns:
            df = df.sort_values("__row_id")

        return df.groupby(entity_column, as_index=False).head(1)

    if strategy == "mean":
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        if entity_column not in numeric_columns:
            numeric_columns = [entity_column] + numeric_columns

        aggregate_df = (
            df.groupby(entity_column, as_index=False)
            .mean(numeric_only=True)
        )

        return aggregate_df

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported deduplication strategy: {strategy}",
    )


def materialize_to_online_store(
    db: Session,
    materialization_id: int,
    entity_column: str | None,
    deduplication_strategy: str,
) -> dict[str, Any]:
    materialization = (
        db.query(Materialization)
        .filter(Materialization.id == materialization_id)
        .first()
    )

    if materialization is None:
        raise HTTPException(status_code=404, detail="Materialization not found.")

    df = _load_materialization_df(materialization)

    resolved_entity_column = _resolve_entity_column(
        df=df,
        requested_entity_column=entity_column,
    )

    label_column = materialization.label_column

    drop_columns = {"__row_id"}

    if label_column:
        drop_columns.add(label_column)

    feature_df = df.drop(
        columns=[column for column in drop_columns if column in df.columns]
    )

    deduped_df = _deduplicate_entity_rows(
        df=feature_df,
        entity_column=resolved_entity_column,
        strategy=deduplication_strategy,
    )

    # Refresh online store for this materialization/entity pair.
    db.query(OnlineFeature).filter(
        OnlineFeature.materialization_id == materialization_id,
        OnlineFeature.entity_column == resolved_entity_column,
    ).delete()

    stored_count = 0

    for _, row in deduped_df.iterrows():
        entity_value = str(row[resolved_entity_column])
        features = _row_to_json_dict(row)

        online_feature = OnlineFeature(
            materialization_id=materialization_id,
            entity_column=resolved_entity_column,
            entity_value=entity_value,
            features_json=json.dumps(features),
        )

        db.add(online_feature)
        stored_count += 1

    db.commit()

    feature_columns = [
        column for column in deduped_df.columns
        if column != resolved_entity_column
    ]

    return {
        "materialization_id": materialization_id,
        "entity_column": resolved_entity_column,
        "deduplication_strategy": deduplication_strategy,
        "entities_stored": stored_count,
        "feature_columns": feature_columns,
        "created_at": datetime.utcnow(),
    }


def lookup_online_feature(
    db: Session,
    materialization_id: int,
    entity_column: str,
    entity_value: str | int | float,
) -> dict[str, Any]:
    online_feature = (
        db.query(OnlineFeature)
        .filter(
            OnlineFeature.materialization_id == materialization_id,
            OnlineFeature.entity_column == entity_column,
            OnlineFeature.entity_value == str(entity_value),
        )
        .first()
    )

    if online_feature is None:
        raise HTTPException(
            status_code=404,
            detail="Online feature vector not found for this entity.",
        )

    return {
        "materialization_id": online_feature.materialization_id,
        "entity_column": online_feature.entity_column,
        "entity_value": online_feature.entity_value,
        "features": json.loads(online_feature.features_json),
        "updated_at": online_feature.updated_at,
    }


def batch_lookup_online_features(
    db: Session,
    materialization_id: int,
    entity_column: str,
    entity_values: list[str | int | float],
) -> dict[str, Any]:
    normalized_values = [str(value) for value in entity_values]

    rows = (
        db.query(OnlineFeature)
        .filter(
            OnlineFeature.materialization_id == materialization_id,
            OnlineFeature.entity_column == entity_column,
            OnlineFeature.entity_value.in_(normalized_values),
        )
        .all()
    )

    found_map = {
        row.entity_value: row
        for row in rows
    }

    results = []

    for value in normalized_values:
        row = found_map.get(value)

        if row is None:
            continue

        results.append(
            {
                "materialization_id": row.materialization_id,
                "entity_column": row.entity_column,
                "entity_value": row.entity_value,
                "features": json.loads(row.features_json),
                "updated_at": row.updated_at,
            }
        )

    return {
        "materialization_id": materialization_id,
        "entity_column": entity_column,
        "found": len(results),
        "missing": len(normalized_values) - len(results),
        "results": results,
    }


def get_online_store_stats(
    db: Session,
    materialization_id: int,
) -> dict[str, Any]:
    rows = (
        db.query(OnlineFeature)
        .filter(OnlineFeature.materialization_id == materialization_id)
        .all()
    )

    if not rows:
        return {
            "materialization_id": materialization_id,
            "entity_count": 0,
            "entity_columns": [],
            "sample_feature_keys": [],
        }

    entity_columns = sorted({row.entity_column for row in rows})
    sample_features = json.loads(rows[0].features_json)

    return {
        "materialization_id": materialization_id,
        "entity_count": len(rows),
        "entity_columns": entity_columns,
        "sample_feature_keys": list(sample_features.keys()),
    }


def _load_model_bundle(trained_model: TrainedModel) -> dict[str, Any]:
    path = Path(trained_model.artifact_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Saved model artifact not found.",
        )

    return joblib.load(path)


def _prepare_model_frame(
    feature_records: list[dict[str, Any]],
    expected_feature_columns: list[str],
) -> pd.DataFrame:
    df = pd.DataFrame(feature_records)

    X = pd.get_dummies(df, drop_first=False)

    for column in expected_feature_columns:
        if column not in X.columns:
            X[column] = 0

    X = X[expected_feature_columns]

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].fillna(0)
        else:
            X[column] = X[column].fillna("missing")

    return X


def _format_probabilities(model, X: pd.DataFrame) -> list[dict[str, float] | None]:
    if not hasattr(model, "predict_proba"):
        return [None] * len(X)

    try:
        probabilities = model.predict_proba(X)
    except Exception:
        return [None] * len(X)

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        inner_model = model.named_steps.get("model")
        classes = getattr(inner_model, "classes_", None)

    if classes is None:
        classes = list(range(probabilities.shape[1]))

    output = []

    for row in probabilities:
        output.append(
            {
                str(label): round(float(prob), 6)
                for label, prob in zip(classes, row)
            }
        )

    return output


def predict_from_online_store(
    db: Session,
    model_id: int,
    materialization_id: int | None,
    entity_column: str,
    entity_values: list[str | int | float],
) -> dict[str, Any]:
    trained_model = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id)
        .first()
    )

    if trained_model is None:
        raise HTTPException(status_code=404, detail="Model not found.")

    resolved_materialization_id = materialization_id or trained_model.materialization_id

    lookup_result = batch_lookup_online_features(
        db=db,
        materialization_id=resolved_materialization_id,
        entity_column=entity_column,
        entity_values=entity_values,
    )

    if lookup_result["found"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No online feature vectors found for provided entities.",
        )

    bundle = _load_model_bundle(trained_model)

    model = bundle["model"]
    expected_feature_columns = bundle["feature_columns"]

    feature_records = [
        row["features"]
        for row in lookup_result["results"]
    ]

    X = _prepare_model_frame(
        feature_records=feature_records,
        expected_feature_columns=expected_feature_columns,
    )

    raw_predictions = model.predict(X)
    probabilities = _format_probabilities(model, X)

    predictions = []

    for index, row in enumerate(lookup_result["results"]):
        prediction = raw_predictions[index]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        predictions.append(
            {
                "entity_value": row["entity_value"],
                "prediction": prediction,
                "probabilities": probabilities[index],
                "features_used": row["features"],
            }
        )

    return {
        "model_id": trained_model.id,
        "model_name": trained_model.name,
        "materialization_id": resolved_materialization_id,
        "entity_column": entity_column,
        "rows_predicted": len(predictions),
        "predictions": predictions,
        "created_at": datetime.utcnow(),
    }
''',

    "backend/app/routers/online_store.py": '''from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.online_feature import (
    BatchOnlineFeatureLookupRequest,
    BatchOnlineFeatureLookupResponse,
    OnlineFeatureLookupResponse,
    OnlinePredictionRequest,
    OnlinePredictionResponse,
    OnlineStoreMaterializeRequest,
    OnlineStoreMaterializeResponse,
    OnlineStoreStatsResponse,
)
from app.services.online_feature_service import (
    batch_lookup_online_features,
    get_online_store_stats,
    lookup_online_feature,
    materialize_to_online_store,
    predict_from_online_store,
)

router = APIRouter(prefix="/online-store", tags=["Online Store"])


@router.post("/materialize", response_model=OnlineStoreMaterializeResponse)
def materialize_online_store(
    request: OnlineStoreMaterializeRequest,
    db: Session = Depends(get_db),
):
    return materialize_to_online_store(
        db=db,
        materialization_id=request.materialization_id,
        entity_column=request.entity_column,
        deduplication_strategy=request.deduplication_strategy,
    )


@router.get(
    "/{materialization_id}/features/{entity_value}",
    response_model=OnlineFeatureLookupResponse,
)
def get_online_feature(
    materialization_id: int,
    entity_value: str,
    entity_column: str,
    db: Session = Depends(get_db),
):
    return lookup_online_feature(
        db=db,
        materialization_id=materialization_id,
        entity_column=entity_column,
        entity_value=entity_value,
    )


@router.post(
    "/{materialization_id}/batch-lookup",
    response_model=BatchOnlineFeatureLookupResponse,
)
def batch_lookup_features(
    materialization_id: int,
    request: BatchOnlineFeatureLookupRequest,
    db: Session = Depends(get_db),
):
    return batch_lookup_online_features(
        db=db,
        materialization_id=materialization_id,
        entity_column=request.entity_column,
        entity_values=request.entity_values,
    )


@router.get(
    "/{materialization_id}/stats",
    response_model=OnlineStoreStatsResponse,
)
def online_store_stats(
    materialization_id: int,
    db: Session = Depends(get_db),
):
    return get_online_store_stats(
        db=db,
        materialization_id=materialization_id,
    )


@router.post(
    "/models/{model_id}/predict",
    response_model=OnlinePredictionResponse,
)
def predict_using_online_features(
    model_id: int,
    request: OnlinePredictionRequest,
    db: Session = Depends(get_db),
):
    return predict_from_online_store(
        db=db,
        model_id=model_id,
        materialization_id=request.materialization_id,
        entity_column=request.entity_column,
        entity_values=request.entity_values,
    )
''',

    "backend/tests/test_online_store.py": '''from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _setup_pipeline():
    csv_content = (
        b"user_id,amount,merchant,is_fraud\\n"
        b"1,100,amazon,0\\n"
        b"1,250,swiggy,0\\n"
        b"2,9000,unknown,1\\n"
        b"2,1000,amazon,0\\n"
        b"3,80,zomato,0\\n"
        b"3,300,amazon,0\\n"
        b"4,120,amazon,0\\n"
        b"4,7000,unknown,1\\n"
        b"5,90,swiggy,0\\n"
        b"5,8500,unknown,1\\n"
    )

    dataset_response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "online_store_test_transactions"},
    )

    assert dataset_response.status_code == 200

    dataset_id = dataset_response.json()["id"]

    f1 = client.post(
        "/api/features",
        json={
            "name": "online_amount_log",
            "dataset_id": dataset_id,
            "description": "Log amount feature.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "column",
            "transformation": "log1p",
            "output_dtype": "float",
        },
    )

    assert f1.status_code == 200

    f2 = client.post(
        "/api/features",
        json={
            "name": "online_user_avg_amount",
            "dataset_id": dataset_id,
            "description": "Average amount per user.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "aggregate",
            "aggregation_function": "mean",
            "transformation": "identity",
            "output_dtype": "float",
        },
    )

    assert f2.status_code == 200

    materialization_response = client.post(
        "/api/materializations",
        json={
            "dataset_id": dataset_id,
            "name": "online_store_features",
            "feature_ids": [f1.json()["id"], f2.json()["id"]],
            "label_column": "is_fraud",
        },
    )

    assert materialization_response.status_code == 200

    materialization_id = materialization_response.json()["id"]

    model_response = client.post(
        "/api/models/train",
        json={
            "materialization_id": materialization_id,
            "name": "online_store_model",
            "label_column": "is_fraud",
            "algorithm": "random_forest",
            "problem_type": "classification",
            "test_size": 0.3,
            "random_state": 42,
        },
    )

    assert model_response.status_code == 200

    return materialization_id, model_response.json()["id"]


def test_online_store_materialize_and_lookup():
    materialization_id, _ = _setup_pipeline()

    materialize_response = client.post(
        "/api/online-store/materialize",
        json={
            "materialization_id": materialization_id,
            "entity_column": "user_id",
            "deduplication_strategy": "last",
        },
    )

    assert materialize_response.status_code == 200
    assert materialize_response.json()["entities_stored"] == 5

    lookup_response = client.get(
        f"/api/online-store/{materialization_id}/features/2?entity_column=user_id"
    )

    assert lookup_response.status_code == 200
    assert lookup_response.json()["entity_value"] == "2"


def test_online_prediction():
    materialization_id, model_id = _setup_pipeline()

    materialize_response = client.post(
        "/api/online-store/materialize",
        json={
            "materialization_id": materialization_id,
            "entity_column": "user_id",
            "deduplication_strategy": "last",
        },
    )

    assert materialize_response.status_code == 200

    prediction_response = client.post(
        f"/api/online-store/models/{model_id}/predict",
        json={
            "materialization_id": materialization_id,
            "entity_column": "user_id",
            "entity_values": [1, 2, 3],
        },
    )

    assert prediction_response.status_code == 200
    assert prediction_response.json()["rows_predicted"] == 3
''',
}


main_py = '''from fastapi import FastAPI

from app.db.database import Base, engine
from app.models import dataset, feature_definition, materialization, online_feature, trained_model
from app.routers.datasets import router as dataset_router
from app.routers.features import router as feature_router
from app.routers.health import router as health_router
from app.routers.materializations import router as materialization_router
from app.routers.models import router as model_router
from app.routers.online_store import router as online_store_router
from app.routers.predictions import router as prediction_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FeatureForge API",
    description="Lightweight ML Feature Store backend",
    version="0.7.0",
)

app.include_router(health_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(feature_router, prefix="/api")
app.include_router(materialization_router, prefix="/api")
app.include_router(model_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")
app.include_router(online_store_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "FeatureForge",
        "message": "ML Feature Store API is running",
        "docs": "/docs",
    }
'''


def write_files():
    print("Adding Online Feature Store to FeatureForge...")

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

    print("\\nOnline Feature Store added successfully.")
    print("\\nNew API routes:")
    print("POST /api/online-store/materialize")
    print("GET  /api/online-store/{materialization_id}/features/{entity_value}")
    print("POST /api/online-store/{materialization_id}/batch-lookup")
    print("GET  /api/online-store/{materialization_id}/stats")
    print("POST /api/online-store/models/{model_id}/predict")


if __name__ == "__main__":
    write_files()