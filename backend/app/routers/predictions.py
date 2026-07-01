from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import (
    get_model_input_schema,
    predict_from_materialization,
    predict_records,
)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/models/{model_id}", response_model=PredictionResponse)
def predict_with_model(
    model_id: int,
    request: PredictionRequest,
    db: Session = Depends(get_db),
):
    return predict_records(
        db=db,
        model_id=model_id,
        records=request.records,
    )


@router.post("/models/{model_id}/batch", response_model=BatchPredictionResponse)
def batch_predict_with_model(
    model_id: int,
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
):
    return predict_from_materialization(
        db=db,
        model_id=model_id,
        materialization_id=request.materialization_id,
        limit=request.limit,
    )


@router.get("/models/{model_id}/input-schema")
def model_input_schema(
    model_id: int,
    db: Session = Depends(get_db),
):
    return get_model_input_schema(
        db=db,
        model_id=model_id,
    )
