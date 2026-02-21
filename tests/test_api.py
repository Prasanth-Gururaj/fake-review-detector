from fastapi.testclient import TestClient
from api.serve import app

client = TestClient(app)

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_predict_endpoint_exists():
    response = client.post('/predict', json={'text': 'This is a test review'})
    assert response.status_code == 200