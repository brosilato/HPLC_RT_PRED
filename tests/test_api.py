from fastapi.testclient import TestClient

from dl_hplc_smrt.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint_returns_api_message() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "HPLC RT prediction API"}


def test_models_info_endpoint_lists_available_models() -> None:
    client = TestClient(app)
    response = client.get("/models_info")

    assert response.status_code == 200
    assert "mlp_fp_smart_retained" in response.json()["message"]


def test_predict_endpoint_returns_prediction_for_valid_model() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict/mlp_fp_smart_retained",
        json={"smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["smiles"] == "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O"
    assert body["model_name"] == "mlp_fp_smart_retained"
    assert isinstance(body["prediction"], (int, float))
    assert body["model_notes"]


def test_predict_endpoint_returns_404_for_unknown_model() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict/unknown_model",
        json={"smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"


def test_batch_predict_endpoint_returns_predictions_for_valid_model() -> None:
    client = TestClient(app)
    response = client.post(
        "/batch_predict/mlp_fp_smart_retained",
        json={
            "smiles": [
                "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O",
                "CCO",
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["smiles"] == [
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O",
        "CCO",
    ]
    assert body["model_name"] == "mlp_fp_smart_retained"
    assert len(body["prediction"]) == 2
    assert all(isinstance(value, (int, float)) for row in body["prediction"] for value in row)
    assert body["model_notes"]
