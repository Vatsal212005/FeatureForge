import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.drift_report import DriftReport
from app.models.materialization import Materialization


def _safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "drift_report"


def _load_materialization(db: Session, materialization_id: int) -> Materialization:
    materialization = (
        db.query(Materialization)
        .filter(Materialization.id == materialization_id)
        .first()
    )

    if materialization is None:
        raise HTTPException(status_code=404, detail=f"Materialization {materialization_id} not found.")

    return materialization


def _load_materialized_df(materialization: Materialization) -> pd.DataFrame:
    path = Path(materialization.stored_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Stored materialization file not found: {materialization.stored_filename}",
        )

    return pd.read_csv(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)

    if isinstance(value, (np.floating, np.float64, np.float32)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)

    if pd.isna(value):
        return None

    return value


def _drift_level(score: float) -> str:
    if score >= 0.25:
        return "high"

    if score >= 0.1:
        return "medium"

    return "low"


def _candidate_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str] | None,
    reference_label_column: str | None,
    current_label_column: str | None,
) -> list[str]:
    common_columns = set(reference_df.columns).intersection(set(current_df.columns))

    ignored = {"__row_id"}

    if reference_label_column:
        ignored.add(reference_label_column)

    if current_label_column:
        ignored.add(current_label_column)

    if feature_columns:
        missing = [column for column in feature_columns if column not in common_columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Columns not found in both materializations: {missing}",
            )

        return [column for column in feature_columns if column not in ignored]

    return sorted([column for column in common_columns if column not in ignored])


def _is_numeric_pair(reference_series: pd.Series, current_series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(reference_series)
        and pd.api.types.is_numeric_dtype(current_series)
    )


def _psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference_clean = pd.to_numeric(reference, errors="coerce").dropna()
    current_clean = pd.to_numeric(current, errors="coerce").dropna()

    if len(reference_clean) == 0 or len(current_clean) == 0:
        return 0.0

    if reference_clean.nunique() <= 1:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(reference_clean, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 3:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_clean, bins=bin_edges)
    current_counts, _ = np.histogram(current_clean, bins=bin_edges)

    reference_percents = reference_counts / max(reference_counts.sum(), 1)
    current_percents = current_counts / max(current_counts.sum(), 1)

    epsilon = 1e-6

    reference_percents = np.where(reference_percents == 0, epsilon, reference_percents)
    current_percents = np.where(current_percents == 0, epsilon, current_percents)

    psi_values = (current_percents - reference_percents) * np.log(
        current_percents / reference_percents
    )

    return round(float(np.sum(psi_values)), 6)


def _numeric_drift_metric(column: str, reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    reference_numeric = pd.to_numeric(reference, errors="coerce")
    current_numeric = pd.to_numeric(current, errors="coerce")

    ref_mean = reference_numeric.mean()
    cur_mean = current_numeric.mean()
    ref_std = reference_numeric.std()
    cur_std = current_numeric.std()

    psi = _psi_numeric(reference_numeric, current_numeric)

    if pd.isna(ref_std) or ref_std == 0:
        normalized_mean_shift = 0.0
    else:
        normalized_mean_shift = abs(cur_mean - ref_mean) / ref_std

    normalized_mean_shift = round(float(normalized_mean_shift), 6)

    drift_score = round(float(max(psi, min(normalized_mean_shift / 3, 1.0))), 6)

    return {
        "column": column,
        "dtype": str(reference.dtype),
        "metric_type": "numeric",
        "drift_score": drift_score,
        "drift_level": _drift_level(drift_score),
        "details": {
            "psi": psi,
            "normalized_mean_shift": normalized_mean_shift,
            "reference": {
                "count": int(reference_numeric.count()),
                "missing_rate": round(float(reference_numeric.isna().mean()), 6),
                "mean": _json_safe(ref_mean),
                "std": _json_safe(ref_std),
                "min": _json_safe(reference_numeric.min()),
                "max": _json_safe(reference_numeric.max()),
            },
            "current": {
                "count": int(current_numeric.count()),
                "missing_rate": round(float(current_numeric.isna().mean()), 6),
                "mean": _json_safe(cur_mean),
                "std": _json_safe(cur_std),
                "min": _json_safe(current_numeric.min()),
                "max": _json_safe(current_numeric.max()),
            },
        },
    }


def _categorical_distribution(series: pd.Series) -> dict[str, float]:
    clean = series.fillna("__missing__").astype(str)

    if len(clean) == 0:
        return {}

    distribution = clean.value_counts(normalize=True).to_dict()

    return {
        key: round(float(value), 6)
        for key, value in distribution.items()
    }


def _categorical_drift_metric(column: str, reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    ref_dist = _categorical_distribution(reference)
    cur_dist = _categorical_distribution(current)

    all_keys = set(ref_dist.keys()).union(set(cur_dist.keys()))

    total_variation_distance = 0.5 * sum(
        abs(cur_dist.get(key, 0.0) - ref_dist.get(key, 0.0))
        for key in all_keys
    )

    drift_score = round(float(total_variation_distance), 6)

    new_categories = sorted([key for key in cur_dist.keys() if key not in ref_dist])
    missing_categories = sorted([key for key in ref_dist.keys() if key not in cur_dist])

    return {
        "column": column,
        "dtype": str(reference.dtype),
        "metric_type": "categorical",
        "drift_score": drift_score,
        "drift_level": _drift_level(drift_score),
        "details": {
            "total_variation_distance": drift_score,
            "new_categories": new_categories[:20],
            "missing_categories": missing_categories[:20],
            "reference_top_values": dict(list(ref_dist.items())[:10]),
            "current_top_values": dict(list(cur_dist.items())[:10]),
            "reference_missing_rate": round(float(reference.isna().mean()), 6),
            "current_missing_rate": round(float(current.isna().mean()), 6),
        },
    }


def create_drift_report(
    db: Session,
    reference_materialization_id: int,
    current_materialization_id: int,
    name: str | None = None,
    feature_columns: list[str] | None = None,
) -> DriftReport:
    reference_materialization = _load_materialization(db, reference_materialization_id)
    current_materialization = _load_materialization(db, current_materialization_id)

    reference_df = _load_materialized_df(reference_materialization)
    current_df = _load_materialized_df(current_materialization)

    columns = _candidate_columns(
        reference_df=reference_df,
        current_df=current_df,
        feature_columns=feature_columns,
        reference_label_column=reference_materialization.label_column,
        current_label_column=current_materialization.label_column,
    )

    if not columns:
        raise HTTPException(
            status_code=400,
            detail="No comparable columns found for drift report.",
        )

    metrics = []

    for column in columns:
        reference_series = reference_df[column]
        current_series = current_df[column]

        if _is_numeric_pair(reference_series, current_series):
            metric = _numeric_drift_metric(column, reference_series, current_series)
        else:
            metric = _categorical_drift_metric(column, reference_series, current_series)

        metrics.append(metric)

    overall_score = round(
        float(sum(metric["drift_score"] for metric in metrics) / len(metrics)),
        6,
    )

    clean_name = _safe_name(
        name
        or f"drift_ref_{reference_materialization_id}_cur_{current_materialization_id}"
    )

    drift_report = DriftReport(
        name=clean_name,
        reference_materialization_id=reference_materialization_id,
        current_materialization_id=current_materialization_id,
        compared_columns_json=json.dumps(columns),
        metrics_json=json.dumps(metrics),
        overall_drift_score=overall_score,
        drift_level=_drift_level(overall_score),
    )

    db.add(drift_report)
    db.commit()
    db.refresh(drift_report)

    return drift_report


def drift_report_to_response(report: DriftReport) -> dict[str, Any]:
    return {
        "id": report.id,
        "name": report.name,
        "reference_materialization_id": report.reference_materialization_id,
        "current_materialization_id": report.current_materialization_id,
        "compared_columns": json.loads(report.compared_columns_json),
        "metrics": json.loads(report.metrics_json),
        "overall_drift_score": report.overall_drift_score,
        "drift_level": report.drift_level,
        "created_at": report.created_at,
    }
