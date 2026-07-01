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
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "feature_test_transactions"},
    )

    assert response.status_code == 200

    return response.json()["id"]


def test_create_column_feature():
    dataset_id = _create_dataset()

    response = client.post(
        "/api/features",
        json={
            "name": "amount_log",
            "dataset_id": dataset_id,
            "description": "Log-transformed transaction amount.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "column",
            "transformation": "log1p",
            "output_dtype": "float",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "amount_log"
    assert body["feature_kind"] == "column"
    assert body["transformation"] == "log1p"


def test_create_aggregate_feature():
    dataset_id = _create_dataset()

    response = client.post(
        "/api/features",
        json={
            "name": "user_avg_amount",
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

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "user_avg_amount"
    assert body["feature_kind"] == "aggregate"
    assert body["aggregation_function"] == "mean"


def test_list_features():
    response = client.get("/api/features")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
