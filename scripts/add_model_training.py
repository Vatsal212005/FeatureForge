from pathlib import Path

ROOT = Path.cwd()

files = {
    "backend/app/models/trained_model.py": '''from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.db.database import Base


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False, index=True)
    algorithm = Column(String, nullable=False)

    materialization_id = Column(Integer, nullable=False, index=True)
    dataset_id = Column(Integer, nullable=False, index=True)

    label_column = Column(String, nullable=False)
    problem_type = Column(String, nullable=False)

    feature_columns_json = Column(Text, nullable=False)
    metrics_json = Column(Text, nullable=False)

    artifact_filename = Column(String, nullable=False)
    artifact_path = Column(String, nullable=False)

    train_rows = Column(Integer, nullable=False)
    test_rows = Column(Integer, nullable=False)

    test_size = Column(Float, nullable=False)
    random_state = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
''',

    "backend/app/schemas/trained_model.py": '''from datetime import datetime
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
''',

    "backend/app/services/model_training_service.py": '''import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.models.materialization import Materialization
from app.models.trained_model import TrainedModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"

MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_\\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "model"


def _load_materialized_table(materialization: Materialization) -> pd.DataFrame:
    path = Path(materialization.stored_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Stored materialized feature table not found.",
        )

    return pd.read_csv(path)


def _infer_label_column(
    materialization: Materialization,
    label_column: str | None,
) -> str:
    if label_column:
        return label_column

    if materialization.label_column:
        return materialization.label_column

    raise HTTPException(
        status_code=400,
        detail="No label column provided and materialization has no stored label_column.",
    )


def _prepare_features(
    df: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if label_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Label column '{label_column}' not found in materialized table.",
        )

    drop_columns = {label_column, "__row_id"}

    candidate_df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Keep model features simple and robust: encode categoricals using one-hot encoding.
    X = pd.get_dummies(candidate_df, drop_first=False)

    # Remove columns that are entirely empty after encoding.
    X = X.dropna(axis=1, how="all")

    # Fill remaining missing values.
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].fillna(X[column].median())
        else:
            X[column] = X[column].fillna("missing")

    y = df[label_column]

    if y.isna().any():
        valid_mask = ~y.isna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

    if len(X) < 5:
        raise HTTPException(
            status_code=400,
            detail="Not enough rows to train a model. Need at least 5 usable rows.",
        )

    return X, y, list(X.columns)


def _build_model(
    algorithm: str,
    problem_type: str,
    random_state: int,
):
    if problem_type == "classification":
        if algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                random_state=random_state,
                class_weight="balanced",
            )

        if algorithm == "logistic_regression":
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            )

        if algorithm == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="xgboost is not installed. Run: pip install xgboost",
                ) from exc

            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
            )

    if problem_type == "regression":
        if algorithm == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
            )

        if algorithm == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="xgboost is not installed. Run: pip install xgboost",
                ) from exc

            return XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
            )

        raise HTTPException(
            status_code=400,
            detail="logistic_regression is only supported for classification.",
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported algorithm/problem_type combination: {algorithm}/{problem_type}",
    )


def _classification_metrics(model, X_test, y_test, predictions) -> dict[str, Any]:
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "precision_macro": round(
            float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            6,
        ),
        "recall_macro": round(
            float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            6,
        ),
        "f1_macro": round(
            float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            6,
        ),
    }

    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)

            if probabilities.shape[1] == 2:
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_test, probabilities[:, 1])),
                    6,
                )
    except Exception:
        metrics["roc_auc"] = None

    return metrics


def _regression_metrics(y_test, predictions) -> dict[str, Any]:
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))

    return {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 6),
        "mse": round(float(mse), 6),
        "rmse": round(rmse, 6),
        "r2": round(float(r2_score(y_test, predictions)), 6),
    }


def train_model_from_materialization(
    db: Session,
    materialization_id: int,
    name: str | None,
    label_column: str | None,
    algorithm: str,
    problem_type: str,
    test_size: float,
    random_state: int,
) -> TrainedModel:
    materialization = (
        db.query(Materialization)
        .filter(Materialization.id == materialization_id)
        .first()
    )

    if materialization is None:
        raise HTTPException(status_code=404, detail="Materialization not found.")

    resolved_label_column = _infer_label_column(
        materialization=materialization,
        label_column=label_column,
    )

    df = _load_materialized_table(materialization)

    X, y, feature_columns = _prepare_features(
        df=df,
        label_column=resolved_label_column,
    )

    stratify = None

    if problem_type == "classification":
        class_counts = y.value_counts()

        if len(class_counts) < 2:
            raise HTTPException(
                status_code=400,
                detail="Classification requires at least two target classes.",
            )

        if class_counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = _build_model(
        algorithm=algorithm,
        problem_type=problem_type,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if problem_type == "classification":
        metrics = _classification_metrics(
            model=model,
            X_test=X_test,
            y_test=y_test,
            predictions=predictions,
        )
    else:
        metrics = _regression_metrics(
            y_test=y_test,
            predictions=predictions,
        )

    clean_name = _safe_name(
        name or f"{materialization.name}_{algorithm}"
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    artifact_filename = f"{clean_name}_{timestamp}.joblib"
    artifact_path = MODEL_ARTIFACT_DIR / artifact_filename

    model_bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "label_column": resolved_label_column,
        "algorithm": algorithm,
        "problem_type": problem_type,
        "materialization_id": materialization_id,
        "dataset_id": materialization.dataset_id,
        "metrics": metrics,
    }

    joblib.dump(model_bundle, artifact_path)

    trained_model = TrainedModel(
        name=clean_name,
        algorithm=algorithm,
        materialization_id=materialization_id,
        dataset_id=materialization.dataset_id,
        label_column=resolved_label_column,
        problem_type=problem_type,
        feature_columns_json=json.dumps(feature_columns),
        metrics_json=json.dumps(metrics),
        artifact_filename=artifact_filename,
        artifact_path=str(artifact_path),
        train_rows=int(X_train.shape[0]),
        test_rows=int(X_test.shape[0]),
        test_size=float(test_size),
        random_state=int(random_state),
    )

    db.add(trained_model)
    db.commit()
    db.refresh(trained_model)

    return trained_model


def trained_model_to_response(model: TrainedModel) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "algorithm": model.algorithm,
        "materialization_id": model.materialization_id,
        "dataset_id": model.dataset_id,
        "label_column": model.label_column,
        "problem_type": model.problem_type,
        "feature_columns": json.loads(model.feature_columns_json),
        "metrics": json.loads(model.metrics_json),
        "artifact_filename": model.artifact_filename,
        "artifact_path": model.artifact_path,
        "train_rows": model.train_rows,
        "test_rows": model.test_rows,
        "test_size": model.test_size,
        "random_state": model.random_state,
        "created_at": model.created_at,
    }
''',

    "backend/app/routers/models.py": '''from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.trained_model import TrainedModel
from app.schemas.trained_model import (
    TrainModelRequest,
    TrainModelResponse,
    TrainedModelListItem,
)
from app.services.model_training_service import (
    train_model_from_materialization,
    trained_model_to_response,
)

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/train", response_model=TrainModelResponse)
def train_model(
    request: TrainModelRequest,
    db: Session = Depends(get_db),
):
    trained_model = train_model_from_materialization(
        db=db,
        materialization_id=request.materialization_id,
        name=request.name,
        label_column=request.label_column,
        algorithm=request.algorithm,
        problem_type=request.problem_type,
        test_size=request.test_size,
        random_state=request.random_state,
    )

    return trained_model_to_response(trained_model)


@router.get("", response_model=list[TrainedModelListItem])
def list_models(db: Session = Depends(get_db)):
    return (
        db.query(TrainedModel)
        .order_by(TrainedModel.created_at.desc())
        .all()
    )


@router.get("/{model_id}", response_model=TrainModelResponse)
def get_model(
    model_id: int,
    db: Session = Depends(get_db),
):
    trained_model = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id)
        .first()
    )

    if trained_model is None:
        raise HTTPException(status_code=404, detail="Model not found.")

    return trained_model_to_response(trained_model)


@router.get("/{model_id}/metrics")
def get_model_metrics(
    model_id: int,
    db: Session = Depends(get_db),
):
    trained_model = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id)
        .first()
    )

    if trained_model is None:
        raise HTTPException(status_code=404, detail="Model not found.")

    return {
        "model_id": trained_model.id,
        "name": trained_model.name,
        "algorithm": trained_model.algorithm,
        "problem_type": trained_model.problem_type,
        "metrics": trained_model_to_response(trained_model)["metrics"],
    }
''',

    "backend/tests/test_model_training.py": '''from io import BytesIO

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
        b"4,120,amazon,0\\n"
        b"4,7000,unknown,1\\n"
        b"5,90,swiggy,0\\n"
        b"5,8500,unknown,1\\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "model_training_test_transactions"},
    )

    assert response.status_code == 200

    return response.json()["id"]


def _create_features(dataset_id: int):
    first = client.post(
        "/api/features",
        json={
            "name": "train_amount_log",
            "dataset_id": dataset_id,
            "description": "Log amount feature.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "column",
            "transformation": "log1p",
            "output_dtype": "float",
        },
    )

    assert first.status_code == 200

    second = client.post(
        "/api/features",
        json={
            "name": "train_user_avg_amount",
            "dataset_id": dataset_id,
            "description": "User average transaction amount.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "aggregate",
            "aggregation_function": "mean",
            "transformation": "identity",
            "output_dtype": "float",
        },
    )

    assert second.status_code == 200

    return [first.json()["id"], second.json()["id"]]


def _create_materialization(dataset_id: int, feature_ids: list[int]):
    response = client.post(
        "/api/materializations",
        json={
            "dataset_id": dataset_id,
            "name": "model_training_features",
            "feature_ids": feature_ids,
            "label_column": "is_fraud",
        },
    )

    assert response.status_code == 200

    return response.json()["id"]


def test_train_model():
    dataset_id = _create_dataset()
    feature_ids = _create_features(dataset_id)
    materialization_id = _create_materialization(dataset_id, feature_ids)

    response = client.post(
        "/api/models/train",
        json={
            "materialization_id": materialization_id,
            "name": "fraud_model_test",
            "label_column": "is_fraud",
            "algorithm": "random_forest",
            "problem_type": "classification",
            "test_size": 0.3,
            "random_state": 42,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "fraud_model_test"
    assert body["algorithm"] == "random_forest"
    assert body["problem_type"] == "classification"
    assert "accuracy" in body["metrics"]
    assert len(body["feature_columns"]) > 0


def test_list_models():
    response = client.get("/api/models")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
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

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FeatureForge API",
    description="Lightweight ML Feature Store backend",
    version="0.5.0",
)

app.include_router(health_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(feature_router, prefix="/api")
app.include_router(materialization_router, prefix="/api")
app.include_router(model_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "FeatureForge",
        "message": "ML Feature Store API is running",
        "docs": "/docs",
    }
'''


def write_files():
    print("Adding Model Training and Model Registry to FeatureForge...")

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

    print("\\nModel Training and Model Registry added successfully.")
    print("\\nNew API routes:")
    print("POST /api/models/train")
    print("GET  /api/models")
    print("GET  /api/models/{model_id}")
    print("GET  /api/models/{model_id}/metrics")


if __name__ == "__main__":
    write_files()