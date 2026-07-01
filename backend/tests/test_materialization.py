from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_dataset():
    csv_content = (
        b"user_id,amount,merchant,is_fraud\n"
        b"1,100,amazon,0\n"
        b"1,250,swiggy,0\n"
        b"2,9000,unknown,1\n"
        b"2,1000,amazon,0\n"
        b"3,80,zomato,0\n"
        b"3,300,amazon,0\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "materialization_test_transactions"},
    )

    assert response.status_code == 200

    return response.json()["id"]


def _create_features(dataset_id: int):
    column_response = client.post(
        "/api/features",
        json={
            "name": "mat_amount_log",
            "dataset_id": dataset_id,
            "description": "Log-transformed transaction amount.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "column",
            "transformation": "log1p",
            "output_dtype": "float",
        },
    )

    assert column_response.status_code == 200

    aggregate_response = client.post(
        "/api/features",
        json={
            "name": "mat_user_avg_amount",
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

    assert aggregate_response.status_code == 200

    return [
        column_response.json()["id"],
        aggregate_response.json()["id"],
    ]


def test_create_materialization():
    dataset_id = _create_dataset()
    feature_ids = _create_features(dataset_id)

    response = client.post(
        "/api/materializations",
        json={
            "dataset_id": dataset_id,
            "name": "fraud_training_features",
            "feature_ids": feature_ids,
            "label_column": "is_fraud",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["dataset_id"] == dataset_id
    assert body["rows"] == 6
    assert body["columns"] >= 5
    assert body["label_column"] == "is_fraud"


def test_list_materializations():
    response = client.get("/api/materializations")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
