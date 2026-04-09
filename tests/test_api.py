from fastapi.testclient import TestClient
from src.api.main import app
import io

client = TestClient(app)


def test_upload_video():
    # Create a dummy file to upload
    dummy_file = io.BytesIO(b"fake video content")
    response = client.post(
        "/api/upload", files={"file": ("test_video.mp4", dummy_file, "video/mp4")}
    )
    assert response.status_code == 200
    assert "job_id" in response.json()


def test_get_status():
    response = client.get("/api/status/fake-job-123")
    assert response.status_code == 200
    assert response.json()["status"] == "processing"


def test_get_results():
    response = client.get("/api/results/fake-job-123")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "complete"
    assert len(data["strokes"]) > 0


def test_get_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json() == {"models": ["Model A", "Model B"]}
