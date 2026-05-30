import io
import time

from fastapi.testclient import TestClient

from src.api.main import app

# TODO(coverage): this suite only exercises the upload flow (which always
# routes to the smart stub in inference.py) plus the validation/404 paths.
# The real ML and registry-serving logic is untested here:
#   - POST /api/library_predict  -> live BST forward pass (bst_inference.predict)
#   - GET  /api/registry[/...]   -> registry/manifest/sidecar JSON serving
#   - GET  /api/clips/{stem}/video -> FileResponse with Range support
# The stub-fallback paths for these are testable without the heavy
# checkpoint/tensors. Also note: test_upload_returns_queued and
# test_full_job_lifecycle currently fail unless UPLOAD_DIR exists — it
# defaults to the Docker path /app/uploads and the module-level TestClient
# doesn't trigger the lifespan startup that mkdir's it.

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


def test_available_splits_empty_without_inputs(monkeypatch, tmp_path):
    import src.api.bst_inference as bi
    monkeypatch.setattr(bi, "BST_INPUTS_DIR", tmp_path)
    assert bi.available_splits() == set()


def test_available_splits_detects_present_split(monkeypatch, tmp_path):
    import src.api.bst_inference as bi
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "JnB_bone.npy").write_bytes(b"")
    monkeypatch.setattr(bi, "BST_INPUTS_DIR", tmp_path)
    assert bi.available_splits() == {"test"}


def test_registry_bst_status_and_live_predictions():
    resp = client.get("/api/registry")
    assert resp.status_code == 200
    models = {m["id"]: m for m in resp.json()["models"]}

    bst = models["bst_x_v1_wipe_drop_s5"]
    assert bst["status"] == "available"
    # No scratch/bst_inputs in the test env => no live predictions, metrics still real.
    assert bst["live_predictions"] == {"test": False, "val": False}
    assert bst["test_metrics"]["macro_f1"] == 0.7479


def test_registry_bric_status_and_live_predictions():
    resp = client.get("/api/registry")
    models = {m["id"]: m for m in resp.json()["models"]}
    bric = models["bric_rgb_shuttle_tcn_outgoing_only_v1"]
    assert bric["status"] == "available"
    assert bric["live_predictions"] == {"test": False, "val": False}
    assert bric["test_metrics"]["macro_f1"] == 0.7305


def test_list_clips_serves_live_predictions(monkeypatch):
    from src.api import registry as reg
    import src.api.bst_inference as bi
    monkeypatch.setattr(reg, "_live_splits", lambda: {"test"})
    monkeypatch.setattr(
        bi, "predict",
        lambda stem, split: {
            "predicted_class": "smash", "true_class": "smash", "confidence_pct": 88,
        },
    )
    resp = client.get("/api/registry/bst_x_v1_wipe_drop_s5/splits/test/clips?limit=3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["live"] is True
    assert len(body["clips"]) > 0
    for c in body["clips"]:
        assert c["predicted_class"] == "smash"
        assert c["is_correct"] is True
        assert c["confidence_pct"] == 88


def test_list_clips_not_live_omits_predictions(monkeypatch):
    from src.api import registry as reg
    monkeypatch.setattr(reg, "_live_splits", lambda: set())
    resp = client.get("/api/registry/bst_x_v1_wipe_drop_s5/splits/test/clips?limit=3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["live"] is False
    assert "_mock_data" not in body
    for c in body["clips"]:
        assert c["predicted_class"] is None
        assert c["is_correct"] is None
        assert c["confidence_pct"] is None
        assert c["true_class"] is not None  # ground truth is real
