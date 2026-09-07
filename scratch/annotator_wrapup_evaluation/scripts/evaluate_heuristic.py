"""Score saved native heuristic contacts and rallies without rerunning vision models."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import pandas as pd

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_sections,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_closing_pass.scripts.summarise_metrics import load_populations

ROOT = Path(__file__).resolve().parents[1]
OTHER_SIDE = {"Top": "Bot", "Bot": "Top", None: None}


def read_json(path: Path) -> dict:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as source:
        return json.load(source)


def load_native(saved_root: Path, fixtures: list[str]) -> tuple[list[FixedSpan], dict, list[dict]]:
    spans, videos, records = [], {}, []
    for fixture in fixtures:
        folder = saved_root / "videos" / f"ss22_{int(fixture):02d}"
        receipt = read_json(folder / "result.json")
        prediction = read_json(folder / "predictions.json.gz")
        annotation = read_json(folder / "annotation/annotator_result.json.gz")
        assert receipt["status"] == "complete" and receipt["fixture"] == fixture
        assert annotation["schema"] == "annotator-result/0.1" and str(annotation["video_id"]) == fixture
        assert prediction["fps"] == 30 and receipt["frame_count"] == prediction["frame_count"]
        result = annotation["result"]
        raw, filtered = result["contacts"], result["filtered_contacts"]
        retained = [row for row in raw if row["wrist_near"] is not False and row["suppressed"] is not True]
        assert retained == filtered, fixture
        count = len(result["spans"])
        assert len(result["fitted_first_all"]) == len(result["striker_halves"]) == len(result["n_strokes_list"]) == count
        by_rally = defaultdict(list)
        for row in filtered:
            by_rally[row["rally_id"]].append(row["contact_frame"])
        events = []
        for rally_id, (start, end) in enumerate(result["spans"]):
            frames = by_rally[rally_id]
            assert frames == sorted(set(frames))
            assert frames == result["filtered_by_rally"].get(str(rally_id), [])
            assert len(frames) == result["n_strokes_list"][rally_id]
            first, last = result["fitted_first_all"][rally_id], result["striker_halves"][rally_id]
            assert first in OTHER_SIDE and last in OTHER_SIDE
            expected_last = first if len(frames) % 2 else OTHER_SIDE[first]
            assert expected_last == last, (fixture, rally_id)
            rally_events = []
            for index, frame in enumerate(frames):
                assert 0 <= start <= frame < end <= receipt["frame_count"]
                side = first if index % 2 == 0 else OTHER_SIDE[first]
                event = FixedEvent(fixture, frame, 1.0, side)
                rally_events.append(event)
                events.append(event)
            spans.append(FixedSpan(fixture, rally_id, start, end, tuple(rally_events)))
        events.sort(key=lambda event: event.frame)
        raw_frames = sorted(row["contact_frame"] for row in raw)
        assert len(raw_frames) == len(set(raw_frames)), fixture
        assert len(events) == len({event.frame for event in events}), fixture
        videos[fixture] = {"raw_frames": raw_frames, "events": events, "raw": raw}
        records.append({"fixture": fixture, "fps": 30, "frame_count": receipt["frame_count"],
                        "source_commit": receipt["source_commit"], "spans": count,
                        "raw_contacts": len(raw), "filtered_contacts": len(filtered),
                        "wrist_rejected": sum(row["wrist_near"] is False for row in raw),
                        "suppressed": sum(row["suppressed"] is True for row in raw),
                        "unresolved_player_rallies": sum(side is None for side in result["fitted_first_all"]),
                        "empty_rallies": sum(value == 0 for value in result["n_strokes_list"])})
    return spans, videos, records


def score_proposals(spans: list, labels: object, sides: dict, tolerance: int) -> list[dict]:
    rows = score_sections(spans, labels, {span.fixture: 30 for span in spans}, tolerance)
    for row, span in zip(rows, spans, strict=True):
        targets = sides.get((span.fixture, row["rally_id"]), ())
        known = sum(side is not None for side in targets)
        correct = sum(targets[gt] is not None and span.events[pred].predicted_side == targets[gt]
                      for gt, pred, _ in row["matches"])
        matched_known = sum(targets[gt] is not None for gt, _, _ in row["matches"])
        row["correct_sides"] = correct
        row["fully_correct"] = row["timing_complete"] and correct == row["labelled_contacts"]
        one = row["overlapping_rallies"] == 1
        row["missing"] = row["labelled_contacts"] - len(row["matches"]) if one else None
        row["extra"] = row["events"] - len(row["matches"]) if one else None
        row["wrong_player"] = matched_known - correct if one else None
        row["boundary_error"] = one and not row["whole_rally_contained"]
        if row["overlapping_rallies"] == 0:
            outcome = "unknown"
        elif not one or not row["whole_rally_contained"] or row["missing"] or row["extra"] or row["wrong_player"]:
            outcome = "wrong"
        elif known < row["labelled_contacts"]:
            outcome = "unknown"
        else:
            outcome = "correct"
        row["outcome"] = outcome
        row["known_label_sides"] = known
        row["matched"] = len(row.pop("matches"))
        del row["voted_correct_sides"], row["side_rule_fully_correct"]
    return rows


def score_events(videos: dict, labels: object, sides: dict, tolerance: int) -> tuple[list[dict], dict]:
    records, totals = [], Counter()
    for fixture, video in videos.items():
        human = []
        for rally in labels.rallies[fixture]:
            for index, frame in enumerate(rally.frames):
                human.append((frame, rally.rally_id, index, sides[fixture, rally.rally_id][index], len(rally.frames)))
        human.sort(key=lambda row: row[:3])
        target_frames = [row[0] for row in human]
        events = video["events"]
        matches = match_contacts(target_frames, [event.frame for event in events], tolerance)
        raw_matches = match_contacts(target_frames, video["raw_frames"], tolerance)
        by_label = {gt: (pred, offset) for gt, pred, offset in matches}
        raw_by_label = {gt: (pred, offset) for gt, pred, offset in raw_matches}
        totals.update(labelled=len(human), raw=len(video["raw_frames"]), filtered=len(events),
                      raw_matched=len(raw_matches), matched=len(matches))
        for index, (frame, rally_id, contact_index, side, count) in enumerate(human):
            pair = by_label.get(index)
            event = events[pair[0]] if pair is not None else None
            correct = event is not None and side is not None and side == event.predicted_side
            totals["side_correct"] += correct
            records.append({"fixture": fixture, "fps": 30, "rally_id": rally_id, "label_index": contact_index,
                            "source_frame": frame, "labelled_contacts": count,
                            "position": "serve" if contact_index == 0 else "last" if contact_index == count - 1 else "middle",
                            "target_side": side, "matched": pair is not None, "raw_matched": index in raw_by_label,
                            "player_correct": correct, "prediction_frame": event.frame if event else None,
                            "predicted_side": event.predicted_side if event else None,
                            "offset_base30": pair[1] if pair else None})
    return records, dict(totals)


def run(annotations: Path, saved_root: Path, output: Path, limit: int | None) -> None:
    started = perf_counter()
    populations, sides = load_populations(annotations)
    fixtures = list(populations["retained"].rallies)[:limit]
    spans, videos, receipts = load_native(saved_root, fixtures)
    proposal_rows, contact_rows, rally_rows, summaries = [], [], [], []
    for population, labels in populations.items():
        for tolerance in (10, 5):
            rows = score_proposals(spans, labels, sides, tolerance)
            contacts, totals = score_events(videos, labels, sides, tolerance)
            common = {"population": population, "tolerance_base30": tolerance}
            proposal_rows.extend({**common, **row} for row in rows)
            contact_rows.extend({**common, **row} for row in contacts)
            correct_ids = {(row["fixture"], row["rally_id"]) for row in rows if row["fully_correct"]}
            timing_ids = {(row["fixture"], row["rally_id"]) for row in rows if row["timing_complete"]}
            for fixture in fixtures:
                fixture_spans = [span for span in spans if span.fixture == fixture]
                for rally in labels.rallies[fixture]:
                    touching = [span for span in fixture_spans if any(span.start_frame <= frame < span.end_frame
                                                                     for frame in rally.frames)]
                    rally_rows.append({**common, "fixture": fixture, "rally_id": rally.rally_id,
                                       "labelled_contacts": len(rally.frames),
                                       "fully_correct": (fixture, rally.rally_id) in correct_ids,
                                       "timing_complete": (fixture, rally.rally_id) in timing_ids,
                                       "overlapping_proposals": len(touching),
                                       "contained": any(all(span.start_frame <= frame < span.end_frame
                                                            for frame in rally.frames) for span in touching)})
            summaries.append({**common, **totals, "proposals": len(rows),
                              "rallies": sum(len(labels.rallies[fixture]) for fixture in fixtures),
                              "correct_rallies": len(correct_ids), "timing_complete_rallies": len(timing_ids),
                              **Counter(row["outcome"] for row in rows)})
            print(population, tolerance, summaries[-1], flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("heuristic_proposals", proposal_rows), ("heuristic_contacts", contact_rows),
                       ("heuristic_rallies", rally_rows), ("heuristic_receipts", receipts)):
        pd.DataFrame(rows).to_csv(output / f"{name}.csv.gz", index=False)
    write_json(output / "heuristic_summary.json.gz", {"schema": "native-heuristic-evaluation/1",
               "fixtures": fixtures, "smoke": limit is not None, "summary": summaries,
               "seconds": perf_counter() - started,
               "player_rule": "saved fitted_first_all alternated over the saved filtered_by_rally sequence",
               "raw_contacts_are_timing_only": True})
    print("Complete", round(perf_counter() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.annotations, args.saved_root, args.output, args.limit)
