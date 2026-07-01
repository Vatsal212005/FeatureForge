from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_materialized_dataset(name: str, rows: bytes):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(rows), "text/csv")},
        data={"name": name},
    )

    assert response.status_code == 200
    dataset_id = response.json()["id"]

    f1 = client.post(
        "/api/features",
        json={
            "name": f"{name}_amount_log",
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
            "name": f"{name}_user_avg_amount",
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

    mat = client.post(
        "/api/materializations",
        json={
            "dataset_id": dataset_id,
            "name": f"{name}_features",
            "feature_ids": [f1.json()["id"], f2.json()["id"]],
            "label_column": "is_fraud",
        },
    )

    assert mat.status_code == 200
    return mat.json()["id"]


def test_create_drift_report():
    reference_rows = (
        b"user_id,amount,merchant,is_fraud\n"
        b"1,100,amazon,0\n"
        b"1,250,swiggy,0\n"
        b"2,900,amazon,0\n"
        b"2,1000,amazon,0\n"
        b"3,80,zomato,0\n"
        b"3,300,amazon,0\n"
    )

    current_rows = (
        b"user_id,amount,merchant,is_fraud\n"
        b"1,1000,amazon,0\n"
        b"1,2500,swiggy,0\n"
        b"2,9000,unknown,1\n"
        b"2,10000,unknown,1\n"
        b"3,800,zomato,0\n"
        b"3,3000,amazon,0\n"
    )

    reference_id = _create_materialized_dataset("drift_reference", reference_rows)
    current_id = _create_materialized_dataset("drift_current", current_rows)

    response = client.post(
        "/api/drift/reports",
        json={
            "reference_materialization_id": reference_id,
            "current_materialization_id": current_id,
            "name": "amount_drift_report",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["reference_materialization_id"] == reference_id
    assert body["current_materialization_id"] == current_id
    assert len(body["metrics"]) > 0
    assert "overall_drift_score" in body


def test_list_drift_reports():
    response = client.get("/api/drift/reports")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
