import os
import pytest
from fastapi.testclient import TestClient

# Set temporary test database
os.environ["DB_PATH"] = "test_face_recognition.db"

from app.main import app
from app.database import init_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_teardown():
    if os.path.exists("test_face_recognition.db"):
        try:
            os.remove("test_face_recognition.db")
        except Exception:
            pass
    init_db()
    yield
    if os.path.exists("test_face_recognition.db"):
        try:
            os.remove("test_face_recognition.db")
        except Exception:
            pass

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "registered_users" in data

def test_user_registration_and_login():
    # 128-dimensional mock face vector (normalized)
    mock_vector = [0.05] * 128
    
    # 1. Register User
    reg_payload = {
        "username": "lagan",
        "full_name": "Lagan Arora",
        "email": "lagan@example.com",
        "face_descriptor": mock_vector,
        "snapshot": "data:image/png;base64,mock"
    }
    reg_res = client.post("/api/register", json=reg_payload)
    assert reg_res.status_code == 201
    assert reg_res.json()["success"] is True

    # 2. Duplicate registration rejected
    dup_res = client.post("/api/register", json=reg_payload)
    assert dup_res.status_code == 409

    # 3. Successful Login (exact same vector -> 100% confidence)
    login_res = client.post("/api/login", json={"face_descriptor": mock_vector})
    assert login_res.status_code == 200
    login_data = login_res.json()
    assert login_data["authenticated"] is True
    assert login_data["user"]["username"] == "lagan"
    assert login_data["distance"] == 0.0
    assert login_data["confidence"] >= 95.0

    # 4. Successful Login with slight variation (< 0.50 threshold)
    similar_vector = [0.05 + 0.005] * 128
    sim_res = client.post("/api/login", json={"face_descriptor": similar_vector})
    assert sim_res.status_code == 200
    assert sim_res.json()["authenticated"] is True

    # 5. Rejected Login with distant vector (> 0.50 threshold)
    distant_vector = [0.85] * 128
    dist_res = client.post("/api/login", json={"face_descriptor": distant_vector})
    assert dist_res.status_code == 401

    # 6. Check Users List
    users_res = client.get("/api/users")
    assert users_res.status_code == 200
    users = users_res.json()
    assert len(users) == 1
    assert users[0]["username"] == "lagan"

    # 7. Check Audit Logs
    logs_res = client.get("/api/logs")
    assert logs_res.status_code == 200
    logs = logs_res.json()
    assert len(logs) >= 3 # REGISTERED, SUCCESS, SUCCESS, FAILED
