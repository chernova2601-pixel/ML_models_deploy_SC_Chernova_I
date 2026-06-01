import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("✅ /health passed")

def test_predict():
    payload = {"features": [1]*23}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in (0, 1)
    assert 0 <= data["probability"] <= 1
    print("✅ /predict passed")

if __name__ == "__main__":
    test_health()
    test_predict()
