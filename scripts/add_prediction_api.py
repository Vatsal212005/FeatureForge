from pathlib import Path

ROOT = Path.cwd()

files = {
    "backend/app/schemas/prediction.py": '''from datetime import datetime
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
''',

    "backend/app/services/prediction_service.py": '''import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.materialization import Materialization
from app.models.trained_model import TrainedModel


def _load_model_bundle(trained_model: TrainedModel) -> dict[str, Any]:
    artifact_path = Path(trained_model.artifact_path)

    if not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Saved model artifact not found.",
        )

    try:
        return joblib.load(artifact_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model artifact: {exc}",
        ) from exc


def _prepare_prediction_frame(
    records: list[dict[str, Any]],
    feature_columns: list[str],
) -> pd.DataFrame:
    if not records:
        raise HTTPException(status_code=400, detail="No prediction records provided.")

    df = pd.DataFrame(records)

    if df.empty:
        raise HTTPException(status_code=400, detail="Prediction dataframe is empty.")

    # Same strategy used during training.
    X = pd.get_dummies(df, drop_first=False)

    # Add missing training columns.
    for column in feature_columns:
        if column not in X.columns:
            X[column] = 0

    # Remove unknown extra columns and enforce training order.
    X = X[feature_columns]

    # Basic missing value handling.
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

    # Pipeline case, e.g. logistic regression.
    if classes is None and hasattr(model, "named_steps"):
        inner_model = model.named_steps.get("model")
        classes = getattr(inner_model, "classes_", None)

    if classes is None:
        classes = list(range(probabilities.shape[1]))

    formatted = []

    for row in probabilities:
        row_probs = {
            str(label): round(float(prob), 6)
            for label, prob in zip(classes, row)
        }

        formatted.append(row_probs)

    return formatted


def predict_records(
    db: Session,
    model_id: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    trained_model = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id)
        .first()
    )

    if trained_model is None:
        raise HTTPException(status_code=404, detail="Model not found.")

    bundle = _load_model_bundle(trained_model)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    X = _prepare_prediction_frame(
        records=records,
        feature_columns=feature_columns,
    )

    raw_predictions = model.predict(X)
    probabilities = _format_probabilities(model, X)

    predictions = []

    for index, prediction in enumerate(raw_predictions):
        if hasattr(prediction, "item"):
            prediction = prediction.item()

        predictions.append(
            {
                "row_index": index,
                "prediction": prediction,
                "probabilities": probabilities[index],
            }
        )

    return {
        "model_id": trained_model.id,
        "model_name": trained_model.name,
        "algorithm": trained_model.algorithm,
        "problem_type": trained_model.problem_type,
        "rows_predicted": len(predictions),
        "predictions": predictions,
        "created_at": datetime.utcnow(),
    }


def predict_from_materialization(
    db: Session,
    model_id: int,
    materialization_id: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    trained_model = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id)
        .first()
    )

    if trained_model is None:
        raise HTTPException(status_code=404, detail="Model not found.")

    resolved_materialization_id = materialization_id or trained_model.materialization_id

    materialization = (
        db.query(Materialization)
        .filter(Materialization.id == resolved_materialization_id)
        .first()
    )

    if materialization is None:
        raise HTTPException(status_code=404, detail="Materialization not found.")

    path = Path(materialization.stored_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Stored materialization file not found.",
        )

    df = pd.read_csv(path)

    if limit is not None:
        df = df.head(limit)

    # Remove label and internal row ID before prediction.
    drop_columns = {"__row_id"}

    if trained_model.label_column in df.columns:
        drop_columns.add(trained_model.label_column)

    records_df = df.drop(
        columns=[column for column in drop_columns if column in df.columns]
    )

    records = records_df.to_dict(orient="records")

    result = predict_records(
        db=db,
        model_id=model_id,
        records=records,
    )

    return {
        "model_id": result["model_id"],
        "model_name": result["model_name"],
        "source": f"materialization:{resolved_materialization_id}",
        "rows_predicted": result["rows_predicted"],
        "predictions": result["predictions"],
        "created_at": result["created_at"],
    }


def get_model_input_schema(
    db: Session,
    model_id: int,
) -> dict[str, Any]:
    trained_model = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id)
        .first()
    )

    if trained_model is None:
        raise HTTPException(status_code=404, detail="Model not found.")

    feature_columns = json.loads(trained_model.feature_columns_json)

    return {
        "model_id": trained_model.id,
        "model_name": trained_model.name,
        "algorithm": trained_model.algorithm,
        "problem_type": trained_model.problem_type,
        "label_column": trained_model.label_column,
        "expected_feature_columns": feature_columns,
        "example_request": {
            "records": [
                {
                    column: 0
                    for column in feature_columns
                }
            ]
        },
    }
''',

    "backend/app/routers/predictions.py": '''from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import (
    get_model_input_schema,
    predict_from_materialization,
    predict_records,
)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/models/{model_id}", response_model=PredictionResponse)
def predict_with_model(
    model_id: int,
    request: PredictionRequest,
    db: Session = Depends(get_db),
):
    return predict_records(
        db=db,
        model_id=model_id,
        records=request.records,
    )


@router.post("/models/{model_id}/batch", response_model=BatchPredictionResponse)
def batch_predict_with_model(
    model_id: int,
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
):
    return predict_from_materialization(
        db=db,
        model_id=model_id,
        materialization_id=request.materialization_id,
        limit=request.limit,
    )


@router.get("/models/{model_id}/input-schema")
def model_input_schema(
    model_id: int,
    db: Session = Depends(get_db),
):
    return get_model_input_schema(
        db=db,
        model_id=model_id,
    )
''',

    "backend/tests/test_predictions.py": '''from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _setup_model():
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
        data={"name": "prediction_test_transactions"},
    )

    assert dataset_response.status_code == 200

    dataset_id = dataset_response.json()["id"]

    f1 = client.post(
        "/api/features",
        json={
            "name": "pred_amount_log",
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
            "name": "pred_user_avg_amount",
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
            "name": "prediction_features",
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
            "name": "prediction_test_model",
            "label_column": "is_fraud",
            "algorithm": "random_forest",
            "problem_type": "classification",
            "test_size": 0.3,
            "random_state": 42,
        },
    )

    assert model_response.status_code == 200

    return model_response.json()["id"], materialization_id


def test_model_input_schema():
    model_id, _ = _setup_model()

    response = client.get(f"/api/predictions/models/{model_id}/input-schema")

    assert response.status_code == 200
    assert "expected_feature_columns" in response.json()


def test_batch_prediction():
    model_id, materialization_id = _setup_model()

    response = client.post(
        f"/api/predictions/models/{model_id}/batch",
        json={
            "materialization_id": materialization_id,
            "limit": 3,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["rows_predicted"] == 3
    assert len(body["predictions"]) == 3
''',
}


main_py = '''from fastapi import FastAPI

from app.db.database import Base, engine
from app.models import dataset, feature_definition, materialization, trained_model
from app.routers.datasets import router as dataset_router
from app.routers.features import router as feature_router
from app.routers.health import router as health_router
from app.routers.materializations import router as materialization_router
from app.routers.models import router as model_router
from app.routers.predictions import router as prediction_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FeatureForge API",
    description="Lightweight ML Feature Store backend",
    version="0.6.0",
)

app.include_router(health_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(feature_router, prefix="/api")
app.include_router(materialization_router, prefix="/api")
app.include_router(model_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "FeatureForge",
        "message": "ML Feature Store API is running",
        "docs": "/docs",
    }
'''


def write_files():
    print("Adding Prediction API to FeatureForge...")

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

    print("\\nPrediction API added successfully.")
    print("\\nNew API routes:")
    print("POST /api/predictions/models/{model_id}")
    print("POST /api/predictions/models/{model_id}/batch")
    print("GET  /api/predictions/models/{model_id}/input-schema")


if __name__ == "__main__":
    write_files()