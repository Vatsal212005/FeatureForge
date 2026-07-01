import json
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
