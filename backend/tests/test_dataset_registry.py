from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_dataset_upload():
    csv_content = b"user_id,amount,is_fraud\n1,100,0\n2,250,1\n3,80,0\n"

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("transactions.csv", BytesIO(csv_content), "text/csv")},
        data={"name": "transactions"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "transactions"
    assert body["rows"] == 3
    assert body["columns"] == 3
    assert len(body["column_schema"]) == 3


def test_dataset_list():
    response = client.get("/api/datasets")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
