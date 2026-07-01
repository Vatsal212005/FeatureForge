from fastapi import APIRouter, Depends, HTTPException
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
