import json
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
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
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
