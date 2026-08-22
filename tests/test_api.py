import os
import pytest
from fastapi.testclient import TestClient
from src.serving.main import app

@pytest.fixture(scope="module")
def client():
    # Use context manager to trigger FastAPI startup events
    with TestClient(app) as c:
        yield c

def test_model_info_endpoint(client):
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "churn_predictor"
    assert "model_version" in data
    assert "optimal_threshold" in data
    assert "evaluation_metrics" in data

def test_predict_endpoint_with_request_data(client):
    # Submit raw customer attributes in request body
    payload = {
        "customerID": "9999-XYZ",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.00,
        "TotalCharges": "255.00"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "9999-XYZ"
    assert "churn_score" in data
    assert "confidence" in data
    assert "top_3_shap_drivers" in data
    assert len(data["top_3_shap_drivers"]) == 3
    assert "recommended_action" in data

def test_outcome_endpoint(client):
    payload = {
        "customer_id": "0001-ABCD",
        "intervention_type": "Loyalty Discount",
        "outcome": 1
    }
    response = client.post("/outcome", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["logged"] is True
    assert "outcome_id" in data
