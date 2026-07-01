from datetime import datetime

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
