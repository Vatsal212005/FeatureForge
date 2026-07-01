import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.models.dataset import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "dataset"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def infer_column_schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    schema = []

    total_rows = len(df)

    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())

        non_null_samples = (
            series.dropna()
            .head(5)
            .map(_json_safe_value)
            .tolist()
        )

        schema.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "nullable": bool(missing_count > 0),
                "missing_count": missing_count,
                "missing_percentage": round(
                    (missing_count / total_rows) * 100, 4
                )
                if total_rows > 0
                else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": non_null_samples,
            }
        )

    return schema


def _next_dataset_version(db: Session, dataset_name: str) -> int:
    latest_dataset = (
        db.query(Dataset)
        .filter(Dataset.name == dataset_name)
        .order_by(Dataset.version.desc())
        .first()
    )

    if latest_dataset is None:
        return 1

    return latest_dataset.version + 1


async def register_csv_dataset(
    db: Session,
    file: UploadFile,
    dataset_name: str | None = None,
) -> Dataset:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported right now.")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    file_hash = _sha256_bytes(content)

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV file: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV parsed successfully but contains no rows.")

    raw_name = dataset_name or Path(file.filename).stem
    clean_name = _safe_name(raw_name)

    version = _next_dataset_version(db, clean_name)

    stored_filename = f"{clean_name}_v{version}_{file_hash[:8]}.csv"
    stored_path = RAW_DATA_DIR / stored_filename

    stored_path.write_bytes(content)

    column_schema = infer_column_schema(df)

    dataset = Dataset(
        name=clean_name,
        version=version,
        original_filename=file.filename,
        stored_filename=stored_filename,
        stored_path=str(stored_path),
        file_hash=file_hash,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_schema_json=json.dumps(column_schema),
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset


def dataset_to_response(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "version": dataset.version,
        "original_filename": dataset.original_filename,
        "stored_filename": dataset.stored_filename,
        "file_hash": dataset.file_hash,
        "rows": dataset.rows,
        "columns": dataset.columns,
        "column_schema": json.loads(dataset.column_schema_json),
        "created_at": dataset.created_at,
    }


def preview_dataset(dataset: Dataset, limit: int = 10) -> dict[str, Any]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Preview limit must be between 1 and 100.")

    path = Path(dataset.stored_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored dataset file not found.")

    df = pd.read_csv(path).head(limit)

    df = df.where(pd.notnull(df), None)

    return {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "version": dataset.version,
        "rows_returned": int(df.shape[0]),
        "preview": df.to_dict(orient="records"),
    }
