#!/usr/bin/env python3
"""Build matches.json — the match library manifest the frontend consumes.

Output: frontend/hba-stroke-classifier/data/matches.json

Each entry: { id, title, tournament, year, round, sets, strokes, youtubeId, curated }

`curated: true` flags 6 deterministically-picked matches for the homepage cards.
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
SHUTTLESET = REPO / "src/bst_refactor/ShuttleSet"
META_CSV = SHUTTLESET / "video_metadata.csv"
SET_DIR = SHUTTLESET / "set"
OUT = REPO / "frontend/hba-stroke-classifier/data/matches.json"

TOURNAMENTS = [
    "HSBC BWF World Tour Finals",
    "Toyota Thailand Open",
    "Yonex Thailand Open",
    "World Tour Finals",
    "Sudirman Cup",
    "Indonesia Masters",
    "Indonesia Open",
    "Malaysia Masters",
    "Malaysia Open",
    "Thailand Masters",
    "Denmark Open",
    "Fuzhou Open",
    "Korea Open",
    "All England Open",
]


def youtube_id(url: str) -> str | None:
    q = urlparse(url)
    if q.hostname == "youtu.be":
        return q.path.lstrip("/")
    if q.hostname and "youtube.com" in q.hostname:
        return parse_qs(q.query).get("v", [None])[0]
    return None


def title_case_players(raw: str) -> str:
    words = raw.replace("_", " ").split()
    out = []
    for w in words:
        if "-" in w:
            out.append("-".join(p.capitalize() for p in w.split("-")))
        elif w.endswith("."):
            out.append(w.capitalize())
        else:
            out.append(w.capitalize())
    return " ".join(out)


def parse_folder(folder: str) -> dict:
    s = folder.replace("_", " ").strip()
    for t in sorted(TOURNAMENTS, key=len, reverse=True):
        idx = s.lower().find(t.lower())
        if idx == -1:
            continue
        players = s[:idx].strip()
        rest = s[idx + len(t):].strip()
        m = re.match(r"(\d{4})\s*(.*)", rest)
        year = int(m.group(1)) if m else None
        round_ = (m.group(2) if m else rest).replace("-", " ").strip()
        return {
            "title": title_case_players(players),
            "tournament": t,
            "year": year,
            "round": round_,
        }
    return {"title": title_case_players(s), "tournament": "", "year": None, "round": ""}


def parse_time(s: str) -> float | None:
    parts = s.strip().split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    except ValueError:
        return None
    return None


def count_strokes(folder: Path) -> tuple[int, int, list[float]]:
    sets = 0
    total = 0
    times: list[float] = []
    for p in sorted(folder.glob("set*.csv")):
        sets += 1
        with p.open() as f:
            for row in csv.DictReader(f):
                total += 1
                t = parse_time(row.get("time", "") or "")
                if t is not None:
                    times.append(t)
    times = sorted(set(round(x, 1) for x in times))
    return sets, total, times


def main() -> int:
    with META_CSV.open() as f:
        rows = list(csv.DictReader(f))

    matches: list[dict] = []
    for r in rows:
        vid = youtube_id(r["url"])
        if not vid:
            continue
        folder = SET_DIR / r["video"]
        if not folder.is_dir():
            continue
        sets, strokes, times = count_strokes(folder)
        parsed = parse_folder(r["video"])
        matches.append({
            "id": vid,
            "youtubeId": vid,
            "url": r["url"],
            "fps": int(r["fps"]) if r.get("fps") else 25,
            **parsed,
            "sets": sets,
            "strokes": strokes,
            "strokeTimes": times,
        })

    rng = random.Random(42)
    curated_ids = set(m["id"] for m in rng.sample(matches, 6))
    for m in matches:
        m["curated"] = m["id"] in curated_ids

    matches.sort(key=lambda m: (-(m["year"] or 0), m["title"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(matches, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(matches)} matches → {OUT.relative_to(REPO)}")
    print("Curated 6:")
    for m in matches:
        if m["curated"]:
            print(f"  {m['id']} · {m['title']} · {m['tournament']} {m['year']} {m['round']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
