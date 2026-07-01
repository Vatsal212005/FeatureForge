from pathlib import Path

ROOT = Path.cwd()

files = {
    "backend/app/models/dataset.py": '''from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint

from app.db.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)

    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)

    file_hash = Column(String, nullable=False, index=True)

    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)

    column_schema_json = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_dataset_name_version"),
    )
''',

    "backend/app/schemas/dataset.py": '''from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    nullable: bool
    missing_count: int
    missing_percentage: float
    unique_count: int
    sample_values: list[Any]


class DatasetResponse(BaseModel):
    id: int
    name: str
    version: int
    original_filename: str
    stored_filename: str
    file_hash: str
    rows: int
    columns: int
    column_schema: list[ColumnSchema]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetListItem(BaseModel):
    id: int
    name: str
    version: int
    original_filename: str
    rows: int
    columns: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetPreviewResponse(BaseModel):
    dataset_id: int
    name: str
    version: int
    rows_returned: int
    preview: list[dict[str, Any]]
''',

    "backend/app/services/dataset_service.py": '''import hashlib
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
    name = re.sub(r"[^a-z0-9_\\-]+", "_", name)
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
''',

    "backend/app/routers/datasets.py": '''from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetListItem, DatasetPreviewResponse, DatasetResponse
from app.services.dataset_service import (
    dataset_to_response,
    preview_dataset,
    register_csv_dataset,
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    dataset = await register_csv_dataset(
        db=db,
        file=file,
        dataset_name=name,
    )

    return dataset_to_response(dataset)


@router.get("", response_model=list[DatasetListItem])
def list_datasets(db: Session = Depends(get_db)):
    return (
        db.query(Dataset)
        .order_by(Dataset.created_at.desc())
        .all()
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    return dataset_to_response(dataset)


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def get_dataset_preview(
    dataset_id: int,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    return preview_dataset(dataset, limit=limit)
''',

    "backend/tests/test_dataset_registry.py": '''from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_dataset_upload():
    csv_content = b"user_id,amount,is_fraud\\n1,100,0\\n2,250,1\\n3,80,0\\n"

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "transactions"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "transactions"
    assert body["rows"] == 3
    assert body["columns"] == 3
    assert len(body["column_schema"]) == 3


def test_dataset_list():
    response = client.get("/api/datasets")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
''',
}


main_py = '''from fastapi import FastAPI

from app.db.database import Base, engine
from app.models import dataset
from app.routers.datasets import router as dataset_router
from app.routers.health import router as health_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FeatureForge API",
    description="Lightweight ML Feature Store backend",
    version="0.2.0",
)

app.include_router(health_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "FeatureForge",
        "message": "ML Feature Store API is running",
        "docs": "/docs",
    }
'''


def write_files():
    print("Adding Dataset Registry to FeatureForge...")

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

    print("\\nDataset Registry added successfully.")
    print("\\nNew API routes:")
    print("POST /api/datasets/upload")
    print("GET  /api/datasets")
    print("GET  /api/datasets/{dataset_id}")
    print("GET  /api/datasets/{dataset_id}/preview")


if __name__ == "__main__":
    write_files()