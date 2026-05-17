"""Timed end-to-end programmatic upload test against the live backend."""
from __future__ import annotations
import json
import time
import urllib.request
import urllib.parse
import mimetypes
from pathlib import Path

BASE = "http://localhost:24082"
SAMPLE = Path("E:/bsc-tier1/scratch/inspect_clips/train/Top_smash/11_1_17_9.mp4")


def post_multipart(url: str, file_path: Path) -> dict:
    boundary = "----formdata-boundary-87432"
    mime, _ = mimetypes.guess_type(str(file_path))
    body = []
    body.append(f"--{boundary}\r\n".encode())
    body.append(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode()
    )
    body.append(f"Content-Type: {mime or 'video/mp4'}\r\n\r\n".encode())
    body.append(file_path.read_bytes())
    body.append(f"\r\n--{boundary}--\r\n".encode())
    data = b"".join(body)
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def http_get(url: str):
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    print(f"sample: {SAMPLE.name} ({SAMPLE.stat().st_size} bytes)")
    t0 = time.perf_counter()
    up = post_multipart(f"{BASE}/api/upload?model=default", SAMPLE)
    t_upload = time.perf_counter() - t0
    print(f"upload: {up}  (took {t_upload*1000:.0f} ms)")

    job_id = up["job_id"]

    poll_t0 = time.perf_counter()
    for i in range(60):
        code, status = http_get(f"{BASE}/api/status/{job_id}")
        if status.get("status") in ("complete", "failed"):
            print(f"poll {i+1}: {status}  (after {(time.perf_counter()-poll_t0)*1000:.0f} ms)")
            break
        time.sleep(0.2)
    else:
        print("timed out polling")
        return

    t_results0 = time.perf_counter()
    code, results = http_get(f"{BASE}/api/results/{job_id}")
    t_results = time.perf_counter() - t_results0
    t_total = time.perf_counter() - t0
    print(f"results (took {t_results*1000:.0f} ms):")
    print(json.dumps(results, indent=2))
    print(f"\nEND-TO-END TOTAL: {t_total*1000:.0f} ms  ({'PASS' if t_total <= 5 else 'OVER 5s'})")

    # Frontend-expectation check (per task 4)
    print("\n=== shape vs Results screen expectations ===")
    expectations = [
        ("strokes (list)", isinstance(results.get("strokes"), list)),
        ("strokes[0].timestamp_sec", isinstance(results.get("strokes", [{}])[0].get("timestamp_sec"), (int, float))),
        ("strokes[0].stroke_type",   isinstance(results.get("strokes", [{}])[0].get("stroke_type"), str)),
        ("strokes[0].confidence",    isinstance(results.get("strokes", [{}])[0].get("confidence"), (int, float))),
        ("rally_summary",            isinstance(results.get("rally_summary"), dict)),
        ("rally_summary.total_strokes",          isinstance(results.get("rally_summary", {}).get("total_strokes"), int)),
        ("rally_summary.rally_length_seconds",   isinstance(results.get("rally_summary", {}).get("rally_length_seconds"), (int, float))),
        # Fields mentioned in user prompt but NOT present in the canned stub:
        ("top_k (NOT present in canned stub)",   "top_k" in results),
        ("court_position (NOT present in canned stub)", "court_position" in results),
    ]
    for label, ok in expectations:
        print(f"  [{ 'ok' if ok else '--' }] {label}")


if __name__ == "__main__":
    main()
