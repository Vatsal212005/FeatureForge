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
        b"4,120,amazon,0\n"
        b"4,7000,unknown,1\n"
        b"5,90,swiggy,0\n"
        b"5,8500,unknown,1\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "model_training_test_transactions"},
    )

    assert response.status_code == 200

    return response.json()["id"]


def _create_features(dataset_id: int):
    first = client.post(
        "/api/features",
        json={
            "name": "train_amount_log",
            "dataset_id": dataset_id,
            "description": "Log amount feature.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "column",
            "transformation": "log1p",
            "output_dtype": "float",
        },
    )

    assert first.status_code == 200

    second = client.post(
        "/api/features",
        json={
            "name": "train_user_avg_amount",
            "dataset_id": dataset_id,
            "description": "User average transaction amount.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "aggregate",
            "aggregation_function": "mean",
            "transformation": "identity",
            "output_dtype": "float",
        },
    )

    assert second.status_code == 200

    return [first.json()["id"], second.json()["id"]]


def _create_materialization(dataset_id: int, feature_ids: list[int]):
    response = client.post(
        "/api/materializations",
        json={
            "dataset_id": dataset_id,
            "name": "model_training_features",
            "feature_ids": feature_ids,
            "label_column": "is_fraud",
        },
    )

    assert response.status_code == 200

    return response.json()["id"]


def test_train_model():
    dataset_id = _create_dataset()
    feature_ids = _create_features(dataset_id)
    materialization_id = _create_materialization(dataset_id, feature_ids)

    response = client.post(
        "/api/models/train",
        json={
            "materialization_id": materialization_id,
            "name": "fraud_model_test",
            "label_column": "is_fraud",
            "algorithm": "random_forest",
            "problem_type": "classification",
            "test_size": 0.3,
            "random_state": 42,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "fraud_model_test"
    assert body["algorithm"] == "random_forest"
    assert body["problem_type"] == "classification"
    assert "accuracy" in body["metrics"]
    assert len(body["feature_columns"]) > 0


def test_list_models():
    response = client.get("/api/models")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
