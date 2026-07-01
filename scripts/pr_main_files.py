#!/usr/bin/env python3
"""Insert a short "Main files changed" block into the PR description.

Run by the `main-files` job in .github/workflows/pr-quality.yml. Ranks changed
files by churn x path relevance (WEIGHTS), skips trivial and noise files, and
edits the PR body between the markers below. Deterministic, stdlib-only, and
best-effort: it only PATCHes when the block changes (so it can't retrigger the
`edited` run) and any failure is a ::warning::, never a merge gate. Needs
GITHUB_EVENT_PATH, GITHUB_REPOSITORY, and a write-scoped GITHUB_TOKEN (read-only
on fork PRs -> it warns and no-ops).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

MAX_FILES = 8   # cap the list length
MIN_CHURN = 3   # skip files with fewer changed lines
START, END = "<!-- main-files-start -->", "<!-- main-files-end -->"

# Path relevance: score = churn x weight, first match wins, default 1. Weight 0
# never shows (research/data churn is noise to a code reviewer). Tune freely.
WEIGHTS = [
    (re.compile(r"^(data|experiments|notebooks|clips_local|scratch)/"), 0),
    (re.compile(r"^(src|training)/|^frontend/src/"), 3),
    (re.compile(r"^(scripts|tests)/|^\.github/|^nginx/"), 2),
]
# Files whose churn misleads (lockfiles, generated, binaries); never shown.
NOISE = re.compile(
    r"\.lock$|-lock\.(json|yaml)$|(^|/)requirements[\w.-]*\.txt$"
    r"|\.(min\.(js|css)|map|png|jpe?g|gif|svg|pdf|mp4|mov|wav|mp3"
    r"|pt|pth|onnx|npz|npy|bin|ckpt|ipynb)$"
    r"|(^|/)(__pycache__|node_modules|dist|build)/"
)


def weight_for(path: str) -> int:
    for pattern, weight in WEIGHTS:
        if pattern.search(path):
            return weight
    return 1


def warn(message: str) -> None:
    print(f"::warning title=Main-files block skipped::{' '.join(message.split())}")


def git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return ""


def build_block(base: str, head: str) -> str | None:
    """Markdown block for the top files, or None if the PR changed nothing."""
    numstat = git(["diff", "--numstat", f"{base}...{head}"]) if base and head else ""
    scored: list[tuple[int, str]] = []  # (score, "- `path` (+a/-d)")
    total = 0
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        total += 1
        a = int(added) if added.isdigit() else 0   # binary files report "-"
        d = int(deleted) if deleted.isdigit() else 0
        weight = weight_for(path)
        if weight == 0 or a + d < MIN_CHURN or NOISE.search(path):
            continue
        scored.append(((a + d) * weight, f"- `{path}` (+{a}/-{d})"))

    if total == 0:
        return None

    scored.sort(reverse=True)
    shown = [entry for _, entry in scored[:MAX_FILES]] or [
        "_No code files stood out; this PR is docs/data/config only._"
    ]
    remaining = total - sum(s.startswith("- ") for s in shown)
    if remaining > 0:
        shown.append(f"_...and {remaining} more file(s)._")
    return "\n".join([START, "## Main files changed", *shown, END])


def splice(body: str, block: str) -> str:
    """Insert or replace the marked block in the PR body."""
    if START in body and END in body:
        pattern = re.escape(START) + r".*?" + re.escape(END)
        return re.sub(pattern, lambda _: block, body, count=1, flags=re.DOTALL)
    return (body.rstrip() + "\n\n" if body.strip() else "") + block + "\n"


def patch_body(repo: str, number: int, token: str, body: str) -> None:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/pulls/{number}",
        data=json.dumps({"body": body}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="PATCH",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def main() -> int:
    try:
        with open(os.environ.get("GITHUB_EVENT_PATH", ""), encoding="utf-8") as fh:
            pr = json.load(fh).get("pull_request", {})
    except (OSError, ValueError) as exc:
        warn(f"could not read PR event payload ({exc})")
        return 0
    if not pr:
        warn("event payload has no pull_request object")
        return 0

    block = build_block(
        pr.get("base", {}).get("sha", ""), pr.get("head", {}).get("sha", "")
    )
    if block is None:
        print("No changed files; nothing to do.")
        return 0

    body = pr.get("body") or ""
    new_body = splice(body, block)
    if new_body == body:
        print("Main-files block already current; no edit needed.")
        return 0

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    number = pr.get("number")
    if not (token and repo and number):
        warn("no write token/repo/number (fork PR?) -- cannot update the body.")
        return 0
    try:
        patch_body(repo, int(number), token, new_body)
        print("Updated PR body.")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        warn(f"could not update PR body ({exc}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
