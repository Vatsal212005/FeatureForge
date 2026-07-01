from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _setup_model():
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

    dataset_response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "prediction_test_transactions"},
    )

    assert dataset_response.status_code == 200

    dataset_id = dataset_response.json()["id"]

    f1 = client.post(
        "/api/features",
        json={
            "name": "pred_amount_log",
            "dataset_id": dataset_id,
            "description": "Log amount feature.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "column",
            "transformation": "log1p",
            "output_dtype": "float",
        },
    )

    assert f1.status_code == 200

    f2 = client.post(
        "/api/features",
        json={
            "name": "pred_user_avg_amount",
            "dataset_id": dataset_id,
            "description": "Average amount per user.",
            "entity_column": "user_id",
            "source_column": "amount",
            "feature_kind": "aggregate",
            "aggregation_function": "mean",
            "transformation": "identity",
            "output_dtype": "float",
        },
    )

    assert f2.status_code == 200

    materialization_response = client.post(
        "/api/materializations",
        json={
            "dataset_id": dataset_id,
            "name": "prediction_features",
            "feature_ids": [f1.json()["id"], f2.json()["id"]],
            "label_column": "is_fraud",
        },
    )

    assert materialization_response.status_code == 200

    materialization_id = materialization_response.json()["id"]

    model_response = client.post(
        "/api/models/train",
        json={
            "materialization_id": materialization_id,
            "name": "prediction_test_model",
            "label_column": "is_fraud",
            "algorithm": "random_forest",
            "problem_type": "classification",
            "test_size": 0.3,
            "random_state": 42,
        },
    )

    assert model_response.status_code == 200

    return model_response.json()["id"], materialization_id


def test_model_input_schema():
    model_id, _ = _setup_model()

    response = client.get(f"/api/predictions/models/{model_id}/input-schema")

    assert response.status_code == 200
    assert "expected_feature_columns" in response.json()


def test_batch_prediction():
    model_id, materialization_id = _setup_model()

    response = client.post(
        f"/api/predictions/models/{model_id}/batch",
        json={
            "materialization_id": materialization_id,
            "limit": 3,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["rows_predicted"] == 3
    assert len(body["predictions"]) == 3
