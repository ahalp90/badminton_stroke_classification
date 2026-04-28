import io
import time

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

_DUMMY_VIDEO = ("test_video.mp4", io.BytesIO(b"fake video content"), "video/mp4")


def test_upload_returns_queued():
    response = client.post("/api/upload", files={"file": _DUMMY_VIDEO})
    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "queued"


def test_upload_rejects_bad_extension():
    bad_file = ("test.txt", io.BytesIO(b"not a video"), "text/plain")
    response = client.post("/api/upload", files={"file": bad_file})
    assert response.status_code == 400


def test_upload_rejects_unknown_model():
    dummy = ("clip.mp4", io.BytesIO(b"fake video content"), "video/mp4")
    response = client.post("/api/upload", files={"file": dummy}, params={"model": "../../etc/passwd"})
    assert response.status_code == 400


def test_upload_rejects_partial_spatial_crop():
    dummy = ("clip.mp4", io.BytesIO(b"fake video content"), "video/mp4")
    response = client.post("/api/upload", files={"file": dummy}, params={"crop_x": 10, "crop_y": 10})
    assert response.status_code == 400


def test_upload_rejects_invalid_temporal_crop():
    dummy = ("clip.mp4", io.BytesIO(b"fake video content"), "video/mp4")
    response = client.post(
        "/api/upload",
        files={"file": dummy},
        params={"start_sec": 10.0, "end_sec": 5.0},
    )
    assert response.status_code == 400


def test_status_unknown_job_returns_404():
    response = client.get("/api/status/does-not-exist")
    assert response.status_code == 404


def test_results_unknown_job_returns_404():
    response = client.get("/api/results/does-not-exist")
    assert response.status_code == 404


def test_delete_unknown_job_returns_404():
    response = client.delete("/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_full_job_lifecycle():
    dummy = ("clip.mp4", io.BytesIO(b"fake video content"), "video/mp4")
    upload = client.post("/api/upload", files={"file": dummy})
    assert upload.status_code == 200
    job_id = upload.json()["job_id"]

    # Poll until complete (inference stub sleeps 3s, timeout after 15s)
    deadline = time.time() + 15
    while time.time() < deadline:
        status_resp = client.get(f"/api/status/{job_id}")
        assert status_resp.status_code == 200
        if status_resp.json()["status"] == "complete":
            break
        time.sleep(0.5)
    else:
        raise AssertionError("Job did not complete within timeout")

    results = client.get(f"/api/results/{job_id}")
    assert results.status_code == 200
    body = results.json()
    assert body["status"] == "complete"
    assert "strokes" in body
    assert "rally_summary" in body
    assert len(body["strokes"]) > 0

    deleted = client.delete(f"/api/jobs/{job_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    assert client.get(f"/api/status/{job_id}").status_code == 404


def test_get_models_returns_list():
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    for model in body["models"]:
        assert "path" not in model, "Internal file paths must not be exposed"
