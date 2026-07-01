from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
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
