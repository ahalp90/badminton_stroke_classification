#!/usr/bin/env python3
"""Advisory (non-blocking) PR-quality review using the Gemini API.

Called by the `advisory` job in .github/workflows/pr-quality.yml. Reads the PR's
commit messages, description and changed-file list, asks a cheap/fast LLM whether
the messages are human-legible and the description substantively (and readably)
explains the change, then posts a friendly comment.

Design goals (see docs/ci.md):
  * NEVER blocks a merge -- this script always exits 0.
  * If the model is rate-limited, unreachable, mis-named or quota-exhausted, that
    is surfaced as a GitHub ``::warning::`` annotation, not a failure.
  * Stdlib only -- no pip install in the workflow.

Environment:
  GEMINI_API_KEY     required; if empty the script no-ops (the workflow already
                     guards on this, this is just belt-and-suspenders).
  GEMINI_MODEL       optional model id; defaults to ``gemini-2.5-flash``.
  GITHUB_EVENT_PATH  path to the PR event payload (provided by Actions).
  GITHUB_TOKEN       optional; if present, post/update a sticky PR comment.
  GITHUB_REPOSITORY  "owner/repo" (provided by Actions).
  GITHUB_STEP_SUMMARY  optional; the advisory is also written here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "gemini-2.5-flash"
COMMENT_MARKER = "<!-- pr-advisory-bot -->"
HTTP_TIMEOUT = 30  # seconds

RUBRIC = """\
You are a friendly, concise reviewer bot. Your job is ADVISORY only -- you never
block a merge. Judge two things about this pull request:

1. Commit messages -- is each one human-legible and does it describe *what
   changed*? Call out vague/low-signal ones by their short hash (e.g. `a1b2c3d`)
   with a one-line suggestion. Ignore messages that are already clear.
2. PR description -- does it substantively and readably explain WHAT changed,
   WHY, and HOW it was tested? Note anything important that's missing.

Keep it under ~180 words, specific and kind (cite hashes/sections, don't scold).
Use GitHub-flavoured markdown. End with a single **Verdict:** line. Open with a
one-sentence summary. Do not restate these instructions.
"""


def warn(message: str) -> None:
    """Emit a non-fatal GitHub Actions warning annotation (single line)."""
    one_line = " ".join(message.split())
    print(f"::warning title=PR advisory unavailable::{one_line}")
    _write_summary(f"### 🤖 PR advisory\n\n> ⚠️ Skipped: {one_line}\n")


def _write_summary(markdown: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
    except OSError:
        pass  # summary is best-effort


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return ""


def gather_context(pr: dict) -> str:
    """Build the prompt input from commits, PR body and the changed-file list."""
    base = pr.get("base", {}).get("sha", "")
    head = pr.get("head", {}).get("sha", "")
    rng = f"{base}..{head}" if base and head else "HEAD~20..HEAD"

    # Commit subjects + bodies, one block each, capped to keep the request small.
    commits = _git(["log", "--no-merges", "--format=%h%x1f%s%x1f%b%x1e", rng])
    blocks = []
    for entry in filter(None, commits.split("\x1e")):
        parts = entry.strip("\n").split("\x1f")
        if len(parts) < 2:
            continue
        short, subject = parts[0].strip(), parts[1].strip()
        body = (parts[2].strip() if len(parts) > 2 else "")[:500]
        blocks.append(f"- {short} {subject}" + (f"\n    {body}" if body else ""))
        if len(blocks) >= 50:
            break
    commit_text = "\n".join(blocks) or "(no commits found in range)"

    diffstat = _git(["diff", "--stat", f"{base}...{head}"]) if base and head else ""
    diffstat = "\n".join(diffstat.splitlines()[:100]) or "(diffstat unavailable)"

    title = pr.get("title", "") or "(no title)"
    body = (pr.get("body") or "(empty PR description)")[:4000]

    return (
        f"## PR title\n{title}\n\n"
        f"## PR description\n{body}\n\n"
        f"## Commits\n{commit_text}\n\n"
        f"## Changed files (diffstat)\n{diffstat}\n"
    )


def call_gemini(model: str, api_key: str, prompt: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback", {})
        raise RuntimeError(f"no candidates returned (feedback: {feedback})")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError("empty response text")
    return text


def post_comment(repo: str, number: int, token: str, body: str) -> None:
    """Create or update a single sticky advisory comment (best-effort)."""
    api = "https://api.github.com"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }
    marked = f"{COMMENT_MARKER}\n{body}"

    def _request(method: str, url: str, data: dict | None = None):
        raw = json.dumps(data).encode("utf-8") if data is not None else None
        req = urllib.request.Request(url, data=raw, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        existing = _request(
            "GET", f"{api}/repos/{repo}/issues/{number}/comments?per_page=100"
        )
        for comment in existing:
            if COMMENT_MARKER in (comment.get("body") or ""):
                _request("PATCH", comment["url"], {"body": marked})
                return
        _request(
            "POST",
            f"{api}/repos/{repo}/issues/{number}/comments",
            {"body": marked},
        )
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        # Comment posting is a nicety; the step summary still carries the review.
        warn(f"could not post PR comment ({exc}); see the step summary instead")


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY not set -- advisory is dormant, nothing to do.")
        return 0

    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    try:
        with open(event_path, encoding="utf-8") as fh:
            pr = json.load(fh).get("pull_request", {})
    except (OSError, ValueError) as exc:
        warn(f"could not read PR event payload ({exc})")
        return 0
    if not pr:
        warn("event payload has no pull_request object")
        return 0

    prompt = RUBRIC + "\n\n---\n\n" + gather_context(pr)

    try:
        review = call_gemini(model, api_key, prompt)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001 - best-effort detail only
            pass
        if exc.code == 429:
            warn(f"Gemini API rate-limited (HTTP 429) for model '{model}'. {detail}")
        elif exc.code in (400, 404):
            warn(
                f"Gemini rejected the request (HTTP {exc.code}) -- the model name "
                f"'{model}' may be wrong; set the GEMINI_MODEL repo variable to a "
                f"current free-tier model. {detail}"
            )
        elif exc.code in (401, 403):
            warn(f"Gemini auth/quota problem (HTTP {exc.code}) for model '{model}'. {detail}")
        else:
            warn(f"Gemini API error (HTTP {exc.code}) for model '{model}'. {detail}")
        return 0
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        warn(f"Gemini API unreachable ({exc}).")
        return 0
    except (ValueError, RuntimeError, KeyError) as exc:
        warn(f"Unexpected response from Gemini ({exc}).")
        return 0

    note = (
        "🤖 **PR advisory** (non-blocking, AI-generated — use your judgement)\n\n"
        f"{review}\n"
    )
    _write_summary("### " + note)

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    number = pr.get("number")
    if token and repo and number:
        post_comment(repo, int(number), token, note)
    else:
        print("No GITHUB_TOKEN/repo/number -- advisory written to the step summary only.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
