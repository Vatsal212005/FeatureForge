# FeatureForge Docker Run Guide

## Start the full app

```bash
docker compose up --build
```

The backend will be available at:

- API: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

The frontend will be available at:

- Dashboard: http://localhost:5173

## Seed demo data

In another terminal, after the backend is healthy, run:

```bash
python scripts/seed_demo.py
```

This creates sample reference/current transaction datasets, feature definitions, materializations, a model, online-store entries, and a drift report.

## Stop the app

```bash
docker compose down
```

## Reset local generated state

```bash
docker compose down
rm -f backend/featureforge.db
rm -rf sample_data artifacts/reports artifacts/models data/raw data/processed
```
