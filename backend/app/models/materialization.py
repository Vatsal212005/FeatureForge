from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.db.database import Base


class Materialization(Base):
    __tablename__ = "materializations"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False, index=True)
    dataset_id = Column(Integer, nullable=False, index=True)

    feature_ids_json = Column(Text, nullable=False)
    feature_names_json = Column(Text, nullable=False)

    label_column = Column(String, nullable=True)

    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)

    stored_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
