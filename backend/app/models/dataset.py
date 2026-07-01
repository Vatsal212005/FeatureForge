from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint

from app.db.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)

    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)

    file_hash = Column(String, nullable=False, index=True)

    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)

    column_schema_json = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_dataset_name_version"),
    )
