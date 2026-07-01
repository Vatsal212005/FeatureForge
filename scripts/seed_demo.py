import json
import time
import requests
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

    response = requests.request(
        method=method,
        url=url,
        json=payload,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(f"{method} {path} failed: {response.status_code} {response.text}")

    return response.json()


def upload_csv(path, dataset_name):
    with path.open("rb") as file:
        response = requests.post(
            f"{API_BASE}/datasets/upload",
            files={"file": (path.name, file, "text/csv")},
            data={"name": dataset_name},
            timeout=60,
        )

    if not response.ok:
        raise RuntimeError(f"Dataset upload failed: {response.status_code} {response.text}")

    return response.json()

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
