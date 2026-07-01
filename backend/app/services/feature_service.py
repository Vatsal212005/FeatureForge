import hashlib
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
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
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
