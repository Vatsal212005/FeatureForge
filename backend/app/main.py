from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import Base, engine
from app.models import dataset, drift_report, feature_definition, materialization, online_feature, trained_model
from app.routers.datasets import router as dataset_router
from app.routers.drift import router as drift_router
from app.routers.features import router as feature_router
from app.routers.health import router as health_router
from app.routers.materializations import router as materialization_router
from app.routers.models import router as model_router
from app.routers.online_store import router as online_store_router
from app.routers.predictions import router as prediction_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FeatureForge API",
    description="Lightweight ML Feature Store backend",
    version="0.9.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(feature_router, prefix="/api")
app.include_router(materialization_router, prefix="/api")
app.include_router(model_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")
app.include_router(online_store_router, prefix="/api")
app.include_router(drift_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "FeatureForge",
        "message": "ML Feature Store API is running",
        "docs": "/docs",
    }
