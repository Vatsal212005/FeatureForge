from pathlib import Path

ROOT = Path.cwd()

files = {
    "backend/app/models/drift_report.py": '''from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.db.database import Base


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False, index=True)

    reference_materialization_id = Column(Integer, nullable=False, index=True)
    current_materialization_id = Column(Integer, nullable=False, index=True)

    compared_columns_json = Column(Text, nullable=False)
    metrics_json = Column(Text, nullable=False)

    overall_drift_score = Column(Float, nullable=False)
    drift_level = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
''',

    "backend/app/schemas/drift_report.py": '''from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DriftReportCreate(BaseModel):
    reference_materialization_id: int
    current_materialization_id: int
    name: str | None = Field(default=None, min_length=2, max_length=100)
    feature_columns: list[str] | None = None


class DriftColumnMetric(BaseModel):
    column: str
    dtype: str
    metric_type: str
    drift_score: float
    drift_level: str
    details: dict[str, Any]


class DriftReportResponse(BaseModel):
    id: int
    name: str

    reference_materialization_id: int
    current_materialization_id: int

    compared_columns: list[str]
    metrics: list[DriftColumnMetric]

    overall_drift_score: float
    drift_level: str

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DriftReportListItem(BaseModel):
    id: int
    name: str
    reference_materialization_id: int
    current_materialization_id: int
    overall_drift_score: float
    drift_level: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
''',

    "backend/app/services/drift_service.py": '''import json
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
    name = re.sub(r"[^a-z0-9_\\-]+", "_", name)
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
''',

    "backend/app/routers/drift.py": '''from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.drift_report import DriftReport
from app.schemas.drift_report import (
    DriftReportCreate,
    DriftReportListItem,
    DriftReportResponse,
)
from app.services.drift_service import create_drift_report, drift_report_to_response

router = APIRouter(prefix="/drift", tags=["Drift Monitoring"])


@router.post("/reports", response_model=DriftReportResponse)
def generate_drift_report(
    request: DriftReportCreate,
    db: Session = Depends(get_db),
):
    report = create_drift_report(
        db=db,
        reference_materialization_id=request.reference_materialization_id,
        current_materialization_id=request.current_materialization_id,
        name=request.name,
        feature_columns=request.feature_columns,
    )

    return drift_report_to_response(report)


@router.get("/reports", response_model=list[DriftReportListItem])
def list_drift_reports(db: Session = Depends(get_db)):
    return (
        db.query(DriftReport)
        .order_by(DriftReport.created_at.desc())
        .all()
    )


@router.get("/reports/{report_id}", response_model=DriftReportResponse)
def get_drift_report(
    report_id: int,
    db: Session = Depends(get_db),
):
    report = (
        db.query(DriftReport)
        .filter(DriftReport.id == report_id)
        .first()
    )

    if report is None:
        raise HTTPException(status_code=404, detail="Drift report not found.")

    return drift_report_to_response(report)
''',

    "backend/tests/test_drift_monitoring.py": '''from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_materialized_dataset(name: str, rows: bytes):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(rows), "text/csv")},
        data={"name": name},
    )

    assert response.status_code == 200
    dataset_id = response.json()["id"]

    f1 = client.post(
        "/api/features",
        json={
            "name": f"{name}_amount_log",
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
            "name": f"{name}_user_avg_amount",
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

    mat = client.post(
        "/api/materializations",
        json={
            "dataset_id": dataset_id,
            "name": f"{name}_features",
            "feature_ids": [f1.json()["id"], f2.json()["id"]],
            "label_column": "is_fraud",
        },
    )

    assert mat.status_code == 200
    return mat.json()["id"]


def test_create_drift_report():
    reference_rows = (
        b"user_id,amount,merchant,is_fraud\\n"
        b"1,100,amazon,0\\n"
        b"1,250,swiggy,0\\n"
        b"2,900,amazon,0\\n"
        b"2,1000,amazon,0\\n"
        b"3,80,zomato,0\\n"
        b"3,300,amazon,0\\n"
    )

    current_rows = (
        b"user_id,amount,merchant,is_fraud\\n"
        b"1,1000,amazon,0\\n"
        b"1,2500,swiggy,0\\n"
        b"2,9000,unknown,1\\n"
        b"2,10000,unknown,1\\n"
        b"3,800,zomato,0\\n"
        b"3,3000,amazon,0\\n"
    )

    reference_id = _create_materialized_dataset("drift_reference", reference_rows)
    current_id = _create_materialized_dataset("drift_current", current_rows)

    response = client.post(
        "/api/drift/reports",
        json={
            "reference_materialization_id": reference_id,
            "current_materialization_id": current_id,
            "name": "amount_drift_report",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["reference_materialization_id"] == reference_id
    assert body["current_materialization_id"] == current_id
    assert len(body["metrics"]) > 0
    assert "overall_drift_score" in body


def test_list_drift_reports():
    response = client.get("/api/drift/reports")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
''',
}


main_py = '''from fastapi import FastAPI

from app.db.database import Base, engine
from app.models import dataset, drift_report, feature_definition, materialization, online_feature, trained_model
from app.routers.datasets import router as dataset_router
from app.routers.drift import router as drift_router
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
    version="0.8.0",
)

app.include_router(health_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(feature_router, prefix="/api")
app.include_router(materialization_router, prefix="/api")
app.include_router(model_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")
app.include_router(online_store_router, prefix="/api")
app.include_router(drift_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "FeatureForge",
        "message": "ML Feature Store API is running",
        "docs": "/docs",
    }
'''


def write_files():
    print("Adding Drift Monitoring to FeatureForge...")

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

    print("\\nDrift Monitoring added successfully.")
    print("\\nNew API routes:")
    print("POST /api/drift/reports")
    print("GET  /api/drift/reports")
    print("GET  /api/drift/reports/{report_id}")


if __name__ == "__main__":
    write_files()