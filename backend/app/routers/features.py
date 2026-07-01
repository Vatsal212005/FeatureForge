from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.dataset import Dataset
from app.models.feature_definition import FeatureDefinition
from app.schemas.feature_definition import (
    FeatureCreate,
    FeatureListItem,
    FeatureResponse,
    FeatureValidationResponse,
)
from app.services.feature_service import (
    create_feature_definition,
    preview_feature_definition,
    validate_feature_definition,
)

router = APIRouter(prefix="/features", tags=["Features"])


@router.post("", response_model=FeatureResponse)
def create_feature(
    feature: FeatureCreate,
    db: Session = Depends(get_db),
):
    return create_feature_definition(db=db, feature=feature)


@router.get("", response_model=list[FeatureListItem])
def list_features(
    dataset_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(FeatureDefinition)

    if dataset_id is not None:
        query = query.filter(FeatureDefinition.dataset_id == dataset_id)

    return query.order_by(FeatureDefinition.created_at.desc()).all()


@router.get("/{feature_id}", response_model=FeatureResponse)
def get_feature(
    feature_id: int,
    db: Session = Depends(get_db),
):
    feature = (
        db.query(FeatureDefinition)
        .filter(FeatureDefinition.id == feature_id)
        .first()
    )

    if feature is None:
        raise HTTPException(status_code=404, detail="Feature definition not found.")

    return feature


@router.post("/validate", response_model=FeatureValidationResponse)
def validate_feature(
    feature: FeatureCreate,
    db: Session = Depends(get_db),
):
    dataset = db.query(Dataset).filter(Dataset.id == feature.dataset_id).first()

    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    available_columns = validate_feature_definition(dataset, feature)

    return {
        "valid": True,
        "message": "Feature definition is valid.",
        "available_columns": available_columns,
    }


@router.get("/{feature_id}/preview")
def preview_feature(
    feature_id: int,
    limit: int = 10,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return preview_feature_definition(
        db=db,
        feature_id=feature_id,
        limit=limit,
    )
