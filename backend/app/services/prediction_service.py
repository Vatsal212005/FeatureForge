import json
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
