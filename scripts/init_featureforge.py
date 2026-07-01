from pathlib import Path

PROJECT_NAME = "featureforge"

ROOT = Path.cwd()

folders = [
    "backend/app",
    "backend/app/api",
    "backend/app/core",
    "backend/app/db",
    "backend/app/models",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/routers",
    "backend/tests",
    "frontend",
    "data/raw",
    "data/processed",
    "artifacts/models",
    "artifacts/reports",
    "notebooks",
]

files = {
    "README.md": """# FeatureForge

A lightweight ML Feature Store for reusable feature engineering, offline training datasets, online feature serving, feature versioning, lineage tracking, and drift monitoring.

## Goal

Build an industry-style ML infrastructure project that demonstrates:

- Dataset registry
- Feature registry
- Offline feature store
- Online feature store
- Feature versioning
- Point-in-time training dataset generation
- Model training and serving
- Drift monitoring
- Dashboard-ready APIs

## Tech Stack

- Python
- FastAPI
- SQLite / PostgreSQL
- Redis
- pandas
- scikit-learn / XGBoost
- Docker
- React
""",

    ".gitignore": """__pycache__/
*.pyc
.venv/
.env
*.db
data/raw/*
data/processed/*
artifacts/models/*
artifacts/reports/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!artifacts/models/.gitkeep
!artifacts/reports/.gitkeep
""",

    "backend/requirements.txt": """fastapi
uvicorn[standard]
sqlalchemy
pydantic
pydantic-settings
pandas
numpy
scikit-learn
xgboost
python-multipart
redis
joblib
pytest
""",

    "backend/app/__init__.py": "",

    "backend/app/main.py": """from fastapi import FastAPI

from app.db.database import Base, engine
from app.routers.health import router as health_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FeatureForge API",
    description="Lightweight ML Feature Store backend",
    version="0.1.0",
)

app.include_router(health_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "FeatureForge",
        "message": "ML Feature Store API is running",
        "docs": "/docs",
    }
""",

    "backend/app/core/config.py": """from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "FeatureForge"
    environment: str = "development"

    database_url: str = "sqlite:///./featureforge.db"

    redis_host: str = "localhost"
    redis_port: int = 6379

    class Config:
        env_file = ".env"


settings = Settings()
""",

    "backend/app/db/__init__.py": "",

    "backend/app/db/database.py": """from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
""",

    "backend/app/routers/__init__.py": "",

    "backend/app/routers/health.py": """from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FeatureForge API",
        "version": "0.1.0",
    }
""",

    "backend/app/models/__init__.py": "",

    "backend/app/schemas/__init__.py": "",

    "backend/app/services/__init__.py": "",

    "backend/tests/test_health.py": """from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
""",

    "data/raw/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "artifacts/models/.gitkeep": "",
    "artifacts/reports/.gitkeep": "",
}


def create_project_structure():
    print("Creating FeatureForge project structure...")

    for folder in folders:
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder}")

    for file_path, content in files.items():
        path = ROOT / file_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            print(f"Skipped existing file: {file_path}")
            continue

        path.write_text(content, encoding="utf-8")
        print(f"Created file: {file_path}")

    print("\\nFeatureForge base project created successfully.")
    print("\\nNext steps:")
    print('1. cd "backend"')
    print("2. python -m venv .venv")
    print("3. Activate the virtual environment")
    print("4. pip install -r requirements.txt")
    print("5. uvicorn app.main:app --reload")
    print("6. Open http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    create_project_structure()