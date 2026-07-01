from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint

from app.db.database import Base


class OnlineFeature(Base):
    __tablename__ = "online_features"

    id = Column(Integer, primary_key=True, index=True)

    materialization_id = Column(Integer, nullable=False, index=True)

    entity_column = Column(String, nullable=False, index=True)
    entity_value = Column(String, nullable=False, index=True)

    features_json = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "materialization_id",
            "entity_column",
            "entity_value",
            name="uq_online_feature_entity",
        ),
    )
