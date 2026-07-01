from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.online_feature import (
    BatchOnlineFeatureLookupRequest,
    BatchOnlineFeatureLookupResponse,
    OnlineFeatureLookupResponse,
    OnlinePredictionRequest,
    OnlinePredictionResponse,
    OnlineStoreMaterializeRequest,
    OnlineStoreMaterializeResponse,
    OnlineStoreStatsResponse,
)
from app.services.online_feature_service import (
    batch_lookup_online_features,
    get_online_store_stats,
    lookup_online_feature,
    materialize_to_online_store,
    predict_from_online_store,
)

router = APIRouter(prefix="/online-store", tags=["Online Store"])


@router.post("/materialize", response_model=OnlineStoreMaterializeResponse)
def materialize_online_store(
    request: OnlineStoreMaterializeRequest,
    db: Session = Depends(get_db),
):
    return materialize_to_online_store(
        db=db,
        materialization_id=request.materialization_id,
        entity_column=request.entity_column,
        deduplication_strategy=request.deduplication_strategy,
    )


@router.get(
    "/{materialization_id}/features/{entity_value}",
    response_model=OnlineFeatureLookupResponse,
)
def get_online_feature(
    materialization_id: int,
    entity_value: str,
    entity_column: str,
    db: Session = Depends(get_db),
):
    return lookup_online_feature(
        db=db,
        materialization_id=materialization_id,
        entity_column=entity_column,
        entity_value=entity_value,
    )


@router.post(
    "/{materialization_id}/batch-lookup",
    response_model=BatchOnlineFeatureLookupResponse,
)
def batch_lookup_features(
    materialization_id: int,
    request: BatchOnlineFeatureLookupRequest,
    db: Session = Depends(get_db),
):
    return batch_lookup_online_features(
        db=db,
        materialization_id=materialization_id,
        entity_column=request.entity_column,
        entity_values=request.entity_values,
    )


@router.get(
    "/{materialization_id}/stats",
    response_model=OnlineStoreStatsResponse,
)
def online_store_stats(
    materialization_id: int,
    db: Session = Depends(get_db),
):
    return get_online_store_stats(
        db=db,
        materialization_id=materialization_id,
    )


@router.post(
    "/models/{model_id}/predict",
    response_model=OnlinePredictionResponse,
)
def predict_using_online_features(
    model_id: int,
    request: OnlinePredictionRequest,
    db: Session = Depends(get_db),
):
    return predict_from_online_store(
        db=db,
        model_id=model_id,
        materialization_id=request.materialization_id,
        entity_column=request.entity_column,
        entity_values=request.entity_values,
    )
