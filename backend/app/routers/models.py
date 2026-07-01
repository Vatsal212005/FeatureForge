from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.trained_model import TrainedModel
from app.schemas.trained_model import (
    TrainModelRequest,
    TrainModelResponse,
    TrainedModelListItem,
)
from app.services.model_training_service import (
    train_model_from_materialization,
    trained_model_to_response,
)

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/train", response_model=TrainModelResponse)
def train_model(
    request: TrainModelRequest,
    db: Session = Depends(get_db),
):
    trained_model = train_model_from_materialization(
        db=db,
        materialization_id=request.materialization_id,
        name=request.name,
        label_column=request.label_column,
        algorithm=request.algorithm,
        problem_type=request.problem_type,
        test_size=request.test_size,
        random_state=request.random_state,
    )

    return trained_model_to_response(trained_model)


@router.get("", response_model=list[TrainedModelListItem])
def list_models(db: Session = Depends(get_db)):
    return (
        db.query(TrainedModel)
        .order_by(TrainedModel.created_at.desc())
        .all()
    )


@router.get("/{model_id}", response_model=TrainModelResponse)
def get_model(
    model_id: int,
    db: Session = Depends(get_db),
):
    trained_model = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id)
        .first()
    )

    if trained_model is None:
        raise HTTPException(status_code=404, detail="Model not found.")

    return trained_model_to_response(trained_model)


@router.get("/{model_id}/metrics")
def get_model_metrics(
    model_id: int,
    db: Session = Depends(get_db),
):
    trained_model = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id)
        .first()
    )

    if trained_model is None:
        raise HTTPException(status_code=404, detail="Model not found.")

    return {
        "model_id": trained_model.id,
        "name": trained_model.name,
        "algorithm": trained_model.algorithm,
        "problem_type": trained_model.problem_type,
        "metrics": trained_model_to_response(trained_model)["metrics"],
    }
