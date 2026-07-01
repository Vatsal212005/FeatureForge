from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.db.database import Base


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False, index=True)
    algorithm = Column(String, nullable=False)

    materialization_id = Column(Integer, nullable=False, index=True)
    dataset_id = Column(Integer, nullable=False, index=True)

    label_column = Column(String, nullable=False)
    problem_type = Column(String, nullable=False)

    feature_columns_json = Column(Text, nullable=False)
    metrics_json = Column(Text, nullable=False)

    artifact_filename = Column(String, nullable=False)
    artifact_path = Column(String, nullable=False)

    train_rows = Column(Integer, nullable=False)
    test_rows = Column(Integer, nullable=False)

    test_size = Column(Float, nullable=False)
    random_state = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
