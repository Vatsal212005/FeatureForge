from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.materialization import Materialization
from app.schemas.materialization import (
    MaterializationCreate,
    MaterializationListItem,
    MaterializationPreviewResponse,
    MaterializationResponse,
)
from app.services.materialization_service import (
    create_materialization,
    materialization_to_response,
    preview_materialization,
)

router = APIRouter(prefix="/materializations", tags=["Materializations"])


@router.post("", response_model=MaterializationResponse)
def materialize_features(
    request: MaterializationCreate,
    db: Session = Depends(get_db),
):
    materialization = create_materialization(
        db=db,
        dataset_id=request.dataset_id,
        name=request.name,
        feature_ids=request.feature_ids,
        label_column=request.label_column,
    )

    return materialization_to_response(materialization)


@router.get("", response_model=list[MaterializationListItem])
def list_materializations(
    dataset_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Materialization)

    if dataset_id is not None:
        query = query.filter(Materialization.dataset_id == dataset_id)

    return query.order_by(Materialization.created_at.desc()).all()


@router.get("/{materialization_id}", response_model=MaterializationResponse)
def get_materialization(
    materialization_id: int,
    db: Session = Depends(get_db),
):
    materialization = (
        db.query(Materialization)
        .filter(Materialization.id == materialization_id)
        .first()
    )

    if materialization is None:
        raise HTTPException(status_code=404, detail="Materialization not found.")

    return materialization_to_response(materialization)


@router.get("/{materialization_id}/preview", response_model=MaterializationPreviewResponse)
def get_materialization_preview(
    materialization_id: int,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    materialization = (
        db.query(Materialization)
        .filter(Materialization.id == materialization_id)
        .first()
    )

    if materialization is None:
        raise HTTPException(status_code=404, detail="Materialization not found.")

    return preview_materialization(
        materialization=materialization,
        limit=limit,
    )
