from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

from app.db.database import Base


class FeatureDefinition(Base):
    __tablename__ = "feature_definitions"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)
    description = Column(Text, nullable=True)

    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)

    entity_column = Column(String, nullable=False)
    source_column = Column(String, nullable=True)
    timestamp_column = Column(String, nullable=True)

    feature_kind = Column(String, nullable=False, default="column")
    transformation = Column(String, nullable=False, default="identity")

    aggregation_function = Column(String, nullable=True)
    window_days = Column(Integer, nullable=True)

    output_dtype = Column(String, nullable=False, default="float")
    status = Column(String, nullable=False, default="active")

    definition_hash = Column(String, nullable=False, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_feature_name_version"),
    )
