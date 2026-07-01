from pathlib import Path

ROOT = Path.cwd()

files = {
    "backend/Dockerfile": '''FROM python:3.11-slim

WORKDIR /workspace/backend

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /workspace/backend/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY backend /workspace/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
''',

    "frontend/Dockerfile": '''FROM node:20-alpine

WORKDIR /workspace/frontend

COPY frontend/package.json /workspace/frontend/package.json

RUN npm install

COPY frontend /workspace/frontend

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
''',

    "docker-compose.yml": '''services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    container_name: featureforge-backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/workspace/backend
      - ./data:/workspace/data
      - ./artifacts:/workspace/artifacts
    environment:
      DATABASE_URL: sqlite:////workspace/backend/featureforge.db
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    container_name: featureforge-frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/workspace/frontend
      - /workspace/frontend/node_modules
    depends_on:
      - backend
    restart: unless-stopped
''',

    ".dockerignore": '''__pycache__/
*.pyc
.venv/
.env
.git/
.gitignore
node_modules/
frontend/node_modules/
*.db
data/raw/*.csv
data/processed/*.csv
artifacts/models/*.joblib
artifacts/reports/*
''',

    ".env.example": '''DATABASE_URL=sqlite:///./featureforge.db
REDIS_HOST=localhost
REDIS_PORT=6379
''',

    "scripts/seed_demo.py": r'''import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "http://127.0.0.1:8000/api"

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


REFERENCE_CSV = """user_id,amount,merchant,is_fraud
1,100,amazon,0
1,250,swiggy,0
2,900,amazon,0
2,1200,amazon,0
3,80,zomato,0
3,300,amazon,0
4,140,swiggy,0
4,650,amazon,0
5,90,zomato,0
5,400,swiggy,0
6,150,amazon,0
6,700,swiggy,0
7,110,zomato,0
7,500,amazon,0
8,130,amazon,0
8,600,zomato,0
9,170,swiggy,0
9,850,amazon,0
10,120,zomato,0
10,450,swiggy,0
11,8000,unknown_store,1
11,9500,foreign_merchant,1
12,7200,unknown_store,1
12,8800,foreign_merchant,1
13,6900,unknown_store,1
13,9100,foreign_merchant,1
14,7600,unknown_store,1
14,9900,foreign_merchant,1
15,8100,unknown_store,1
15,10300,foreign_merchant,1
"""

CURRENT_CSV = """user_id,amount,merchant,is_fraud
1,180,amazon,0
1,360,swiggy,0
2,1400,amazon,0
2,2200,amazon,0
3,120,zomato,0
3,500,amazon,0
4,210,swiggy,0
4,900,amazon,0
5,160,zomato,0
5,700,swiggy,0
6,300,amazon,0
6,1200,swiggy,0
7,250,zomato,0
7,950,amazon,0
8,280,amazon,0
8,1100,zomato,0
9,350,swiggy,0
9,1600,amazon,0
10,290,zomato,0
10,1000,swiggy,0
11,10000,unknown_store,1
11,12500,foreign_merchant,1
12,9800,unknown_store,1
12,11800,foreign_merchant,1
13,8900,unknown_store,1
13,12100,foreign_merchant,1
14,10600,unknown_store,1
14,13900,foreign_merchant,1
15,11200,unknown_store,1
15,14500,foreign_merchant,1
"""


def request_json(method, path, payload=None):
    url = f"{API_BASE}{path}"
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise RuntimeError(f"{method} {path} failed: {exc.code} {error_body}") from exc


def upload_csv(path, dataset_name):
    boundary = "----FeatureForgeBoundary"
    file_bytes = path.read_bytes()

    body = b""

    body += f"--{boundary}\\r\\n".encode()
    body += b'Content-Disposition: form-data; name="name"\\r\\n\\r\\n'
    body += dataset_name.encode()
    body += b"\\r\\n"

    body += f"--{boundary}\\r\\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{path.name}"\\r\\n'.encode()
    body += b"Content-Type: text/csv\\r\\n\\r\\n"
    body += file_bytes
    body += b"\\r\\n"

    body += f"--{boundary}--\\r\\n".encode()

    request = urllib.request.Request(
        url=f"{API_BASE}/datasets/upload",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise RuntimeError(f"Dataset upload failed: {exc.code} {error_body}") from exc


def wait_for_backend():
    print("Checking backend health...")

    for _ in range(30):
        try:
            response = request_json("GET", "/health")
            if response.get("status") == "healthy":
                print("Backend is healthy.")
                return
        except Exception:
            time.sleep(1)

    raise RuntimeError("Backend did not become healthy at http://127.0.0.1:8000")


def create_features(dataset_id, prefix):
    print(f"Creating features for dataset {dataset_id}...")

    amount_log = request_json(
        "POST",
        "/features",
        {
            "name": f"{prefix}_amount_log",
            "dataset_id": dataset_id,
            "description": "Log-transformed transaction amount.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "column",
            "transformation": "log1p",
            "output_dtype": "float",
        },
    )

    user_avg_amount = request_json(
        "POST",
        "/features",
        {
            "name": f"{prefix}_user_avg_amount",
            "dataset_id": dataset_id,
            "description": "Average transaction amount per user.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "aggregate",
            "aggregation_function": "mean",
            "transformation": "identity",
            "output_dtype": "float",
        },
    )

    user_transaction_count = request_json(
        "POST",
        "/features",
        {
            "name": f"{prefix}_user_transaction_count",
            "dataset_id": dataset_id,
            "description": "Number of transactions per user.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "aggregate",
            "aggregation_function": "count",
            "transformation": "identity",
            "output_dtype": "int",
        },
    )

    return [
        amount_log["id"],
        user_avg_amount["id"],
        user_transaction_count["id"],
    ]


def materialize(dataset_id, feature_ids, name):
    print(f"Materializing feature table: {name}")

    return request_json(
        "POST",
        "/materializations",
        {
            "dataset_id": dataset_id,
            "name": name,
            "feature_ids": feature_ids,
            "label_column": "is_fraud",
        },
    )


def train_model(materialization_id):
    print("Training demo model...")

    return request_json(
        "POST",
        "/models/train",
        {
            "materialization_id": materialization_id,
            "name": "demo_fraud_detection_rf",
            "label_column": "is_fraud",
            "algorithm": "random_forest",
            "problem_type": "classification",
            "test_size": 0.3,
            "random_state": 42,
        },
    )


def push_online_store(materialization_id):
    print("Pushing features to online store...")

    return request_json(
        "POST",
        "/online-store/materialize",
        {
            "materialization_id": materialization_id,
            "entity_column": "user_id",
            "deduplication_strategy": "last",
        },
    )


def create_drift_report(reference_materialization_id, current_materialization_id):
    print("Creating drift report...")

    return request_json(
        "POST",
        "/drift/reports",
        {
            "reference_materialization_id": reference_materialization_id,
            "current_materialization_id": current_materialization_id,
            "name": "demo_reference_vs_current_drift",
            "feature_columns": None,
        },
    )


def main():
    wait_for_backend()

    reference_path = SAMPLE_DIR / "demo_reference_transactions.csv"
    current_path = SAMPLE_DIR / "demo_current_transactions.csv"

    reference_path.write_text(REFERENCE_CSV, encoding="utf-8")
    current_path.write_text(CURRENT_CSV, encoding="utf-8")

    print("Uploading reference dataset...")
    reference_dataset = upload_csv(reference_path, "demo_reference_transactions")

    print("Uploading current dataset...")
    current_dataset = upload_csv(current_path, "demo_current_transactions")

    reference_feature_ids = create_features(
        dataset_id=reference_dataset["id"],
        prefix=f"demo_ref_{reference_dataset['id']}",
    )

    current_feature_ids = create_features(
        dataset_id=current_dataset["id"],
        prefix=f"demo_cur_{current_dataset['id']}",
    )

    reference_materialization = materialize(
        dataset_id=reference_dataset["id"],
        feature_ids=reference_feature_ids,
        name="demo_reference_features",
    )

    current_materialization = materialize(
        dataset_id=current_dataset["id"],
        feature_ids=current_feature_ids,
        name="demo_current_features",
    )

    model = train_model(reference_materialization["id"])
    online_store = push_online_store(reference_materialization["id"])
    drift_report = create_drift_report(
        reference_materialization_id=reference_materialization["id"],
        current_materialization_id=current_materialization["id"],
    )

    print()
    print("Demo seed completed.")
    print()
    print("Created resources:")
    print(f"Reference dataset ID:       {reference_dataset['id']}")
    print(f"Current dataset ID:         {current_dataset['id']}")
    print(f"Reference materialization:  {reference_materialization['id']}")
    print(f"Current materialization:    {current_materialization['id']}")
    print(f"Model ID:                   {model['id']}")
    print(f"Online entities stored:     {online_store['entities_stored']}")
    print(f"Drift report ID:            {drift_report['id']}")
    print()
    print("Open dashboard:")
    print("http://localhost:5173")
    print()
    print("Open API docs:")
    print("http://127.0.0.1:8000/docs")
    print()
    print("Try online prediction:")
    print(f"POST /api/online-store/models/{model['id']}/predict")
    print(
        json.dumps(
            {
                "materialization_id": reference_materialization["id"],
                "entity_column": "user_id",
                "entity_values": [1, 11, 15],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
''',

    "README_DOCKER.md": '''# FeatureForge Docker Run Guide

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
''',
}


def write_files(root: Path = ROOT) -> None:
    """Create/update the project files declared in the files mapping."""
    for relative_path, content in files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip() + "\n", encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    write_files()
