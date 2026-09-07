"""Serve recovery and proposed-start measures for a fixed contact stream."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from annotator.fps_constants import ScalingKind
from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.evaluation import score_contacts
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
)

SectionIdentity = tuple[str, int]
ContactIdentity = tuple[str, int]


def _side_status(label_side: str | None, predicted_side: str | None, matched: bool) -> str:
    if label_side is None:
        return "missing_label"
    if not matched or predicted_side is None:
        return "missing_prediction"
    return "correct" if predicted_side == label_side else "wrong"


def _timing_error(offset: int | None, fps: float) -> float | None:
    return None if offset is None else float(offset) * 30.0 / fps


def _event_lookup(stream: ContactStreams) -> dict[ContactIdentity, FixedEvent]:
    return {
        (fixture, event.frame): event
        for fixture, events in stream.events_by_fixture.items()
        for event in events
    }


def _span_memberships(
    spans: Sequence[FixedSpan],
) -> tuple[
    dict[ContactIdentity, tuple[SectionIdentity, ...]],
    dict[ContactIdentity, tuple[SectionIdentity, ...]],
]:
    members: dict[ContactIdentity, list[SectionIdentity]] = defaultdict(list)
    starts: dict[ContactIdentity, list[SectionIdentity]] = defaultdict(list)
    for span in spans:
        identity = (span.fixture, span.span_id)
        for event_index, event in enumerate(span.events):
            event_identity = (span.fixture, event.frame)
            members[event_identity].append(identity)
            if event_index == 0:
                starts[event_identity].append(identity)
    return (
        {identity: tuple(values) for identity, values in members.items()},
        {identity: tuple(values) for identity, values in starts.items()},
    )


def _pair_indexes(
    raw_scores: Mapping[str, Any],
) -> tuple[dict[ContactIdentity, dict[str, Any]], dict[ContactIdentity, dict[str, Any]]]:
    """Index the one-to-one pairs returned by the full-stream scorer."""
    by_gt: dict[ContactIdentity, dict[str, Any]] = {}
    by_prediction: dict[ContactIdentity, dict[str, Any]] = {}
    for video in raw_scores["by_video"]:
        fixture = str(video["fixture"])
        for pair in video["pairs"]:
            gt_frame, pred_frame, rally_id, first, offset, target_side, raw_side = pair
            record = {
                "gt_frame": int(gt_frame),
                "pred_frame": int(pred_frame),
                "rally_id": str(rally_id),
                "first": bool(first),
                "offset": int(offset),
                "target_side": target_side,
                "raw_side": raw_side,
            }
            gt_identity = (fixture, int(gt_frame))
            prediction_identity = (fixture, int(pred_frame))
            if gt_identity in by_gt or prediction_identity in by_prediction:
                raise ValueError("full-stream timing pairs reuse a contact identity")
            by_gt[gt_identity] = record
            by_prediction[prediction_identity] = record
    return by_gt, by_prediction


def _retained_rally(
    span: FixedSpan,
    labels: HumanLabels,
    pairs_by_prediction: Mapping[ContactIdentity, Mapping[str, Any]],
) -> RallyReference | None:
    matched_ids = {
        str(pairs_by_prediction[(span.fixture, event.frame)]["rally_id"])
        for event in span.events
        if (span.fixture, event.frame) in pairs_by_prediction
    }
    if len(matched_ids) == 1:
        rally_id = next(iter(matched_ids))
        for rally in labels.rallies.get(span.fixture, ()):
            if rally.rally_id == rally_id:
                return rally
    overlapping = tuple(
        rally
        for rally in labels.rallies.get(span.fixture, ())
        if any(span.start_frame <= frame < span.end_frame for frame in rally.frames)
    )
    return overlapping[0] if len(overlapping) == 1 else None


def _accepted_events(
    stream: ContactStreams,
    accepted: set[SectionIdentity] | None,
) -> set[ContactIdentity] | None:
    if accepted is None:
        return None
    return {
        (span.fixture, event.frame)
        for span in stream.spans
        if (span.fixture, span.span_id) in accepted
        for event in span.events
    }


def _serve_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "firsts": len(rows),
        "known_side_firsts": 0,
        "matched": 0,
        "accepted_matched": 0,
        "raw_correct": 0,
        "raw_wrong": 0,
        "raw_missing_prediction": 0,
        "raw_missing_label": 0,
        "final_correct": 0,
        "final_wrong": 0,
        "final_missing_prediction": 0,
        "final_missing_label": 0,
        "all_raw_correct": 0,
        "all_raw_wrong": 0,
        "all_raw_missing_prediction": 0,
        "all_raw_missing_label": 0,
        "all_final_correct": 0,
        "all_final_wrong": 0,
        "all_final_missing_prediction": 0,
        "all_final_missing_label": 0,
        "raw_joint_correct": 0,
        "joint_correct": 0,
        "all_joint_correct": 0,
        "all_known_gt": 0,
        "serves_outside_all_spans": 0,
        "recovered_but_preceded_serves": 0,
    }
    for row in rows:
        known = row["label_side"] is not None
        counts["known_side_firsts"] += int(known)
        counts["all_known_gt"] += int(known)
        counts["matched"] += int(row["timing_matched"])
        counts["accepted_matched"] += int(row["accepted_timing_matched"])
        if row["timing_matched"]:
            for prefix in ("raw", "final"):
                counts[f"all_{prefix}_{row[f'{prefix}_status']}"] += 1
                counts[f"{prefix}_{row[f'accepted_{prefix}_status']}"] += int(
                    row["accepted_timing_matched"]
                )
        counts["all_joint_correct"] += int(
            row["timing_matched"] and row["final_status"] == "correct"
        )
        counts["raw_joint_correct"] += int(
            row["accepted_timing_matched"] and row["accepted_raw_status"] == "correct"
        )
        counts["joint_correct"] += int(
            row["accepted_timing_matched"] and row["accepted_final_status"] == "correct"
        )
        counts["serves_outside_all_spans"] += int(
            row["timing_matched"] and not row["matched_span_ids"]
        )
        counts["recovered_but_preceded_serves"] += int(
            row["unmatched_before_recovered_serve"]
        )
    return counts


def _start_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "all_sections": len(rows),
        "accepted_sections_count": 0,
        "all_starts": 0,
        "accepted_starts": 0,
        "judgeable_timing_starts": 0,
        "judgeable_side_starts": 0,
        "timing_correct_starts": 0,
        "side_correct_starts": 0,
        "raw_joint_correct_starts": 0,
        "joint_correct_starts": 0,
        "serve": 0,
        "later_hit": 0,
        "extra_leading": 0,
        "unknown": 0,
        "empty": 0,
        "recovered_but_preceded_starts": 0,
    }
    counts.update({f"accepted_{status}": 0 for status in ("serve", "later_hit", "extra_leading", "unknown", "empty")})
    for row in rows:
        status = str(row["status"])
        counts[status] += 1
        counts["accepted_sections_count"] += int(row["accepted"])
        counts[f"accepted_{status}"] += int(row["accepted"])
        counts["recovered_but_preceded_starts"] += int(
            row["unmatched_before_recovered_serve"]
        )
        if status == "empty":
            continue
        accepted = bool(row["accepted"])
        counts["all_starts"] += 1
        counts["accepted_starts"] += int(accepted)
        counts["judgeable_timing_starts"] += int(
            accepted and row["judgeable_timing"]
        )
        counts["judgeable_side_starts"] += int(
            accepted
            and (
                status in {"later_hit", "extra_leading"}
                or (status == "serve" and row["label_side"] is not None)
            )
        )
        counts["timing_correct_starts"] += int(accepted and row["timing_correct"])
        counts["side_correct_starts"] += int(
            accepted and row["timing_correct"] and row["final_status"] == "correct"
        )
        counts["raw_joint_correct_starts"] += int(
            accepted and row["raw_joint_correct"]
        )
        counts["joint_correct_starts"] += int(accepted and row["joint_correct"])
    return counts


def _contact_totals(
    raw_scores: Mapping[str, Any],
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    keys = (
        "labelled", "predicted", "matched", "side_answered", "side_correct",
        "first_contacts", "first_matched", "precision", "recall", "f1",
    )
    total = {f"contact_{key}": raw_scores["total"][key] for key in keys}
    by_video = [
        {
            "fixture": str(video["fixture"]),
            **{f"contact_{key}": video[key] for key in keys if key in video},
        }
        for video in raw_scores["by_video"]
    ]
    return total, by_video


def analyse_serves(
    stream: ContactStreams,
    labels: HumanLabels,
    fps: Mapping[str, float],
    tolerance_base30: int,
    accepted: set[SectionIdentity] | None = None,
) -> dict[str, Any]:
    """Measure every labelled first contact and every proposed section start."""
    raw_scores = score_contacts(stream.events_by_fixture, labels, fps, tolerance_base30)
    voted = start.apply_whole_rally_alternation(stream)
    raw_events = _event_lookup(stream)
    voted_events = _event_lookup(voted)
    memberships, proposed_starts = _span_memberships(stream.spans)
    accepted_events = _accepted_events(stream, accepted)
    pairs_by_gt, pairs_by_prediction = _pair_indexes(raw_scores)

    serve_rows: list[dict[str, Any]] = []
    for fixture in sorted(labels.rallies):
        for rally in labels.rallies[fixture]:
            gt_frame = int(rally.frames[0])
            pair = pairs_by_gt.get((fixture, gt_frame))
            prediction_identity = None if pair is None else (fixture, pair["pred_frame"])
            matched_spans = () if pair is None else memberships.get(prediction_identity, ())
            matched_starts = () if pair is None else proposed_starts.get(prediction_identity, ())
            accepted_match = pair is not None and (
                accepted_events is None or prediction_identity in accepted_events
            )
            label_side = labels.target_sides.get((fixture, gt_frame))
            raw_side = None if pair is None else raw_events[prediction_identity].predicted_side
            final_side = None if pair is None else voted_events[prediction_identity].predicted_side
            raw_status = _side_status(label_side, raw_side, pair is not None)
            final_status = _side_status(label_side, final_side, pair is not None)
            accepted_raw_status = _side_status(label_side, raw_side, accepted_match)
            accepted_final_status = _side_status(label_side, final_side, accepted_match)
            offset = None if pair is None else int(pair["offset"])
            serve_rows.append({
                "fixture": fixture,
                "rally_id": rally.rally_id,
                "gt_frame": gt_frame,
                "label_side": label_side,
                "matched_pred_frame": None if pair is None else pair["pred_frame"],
                "timing_matched": pair is not None,
                "accepted_timing_matched": accepted_match,
                "raw_side": raw_side,
                "final_side": final_side,
                "raw_status": raw_status,
                "final_status": final_status,
                "accepted_raw_status": accepted_raw_status,
                "accepted_final_status": accepted_final_status,
                "timing_error_frames": offset,
                "timing_error_base30": _timing_error(offset, fps[fixture]),
                "matched_span_ids": [identity[1] for identity in matched_spans],
                "proposed_start_ids": [identity[1] for identity in matched_starts],
                "unmatched_before_recovered_serve": False,
            })

    serve_by_identity = {
        (str(row["fixture"]), str(row["rally_id"])): row for row in serve_rows
    }
    voted_spans = {(span.fixture, span.span_id): span for span in voted.spans}
    start_rows: list[dict[str, Any]] = []
    for span in stream.spans:
        section_identity = (span.fixture, span.span_id)
        accepted_span = accepted is None or section_identity in accepted
        if not span.events:
            start_rows.append({
                "fixture": span.fixture, "span_id": span.span_id, "first_frame": None,
                "status": "empty", "accepted": accepted_span,
                "matched_rally_id": None, "matched_gt_frame": None, "label_side": None,
                "raw_side": None, "final_side": None,
                "raw_status": "missing_prediction", "final_status": "missing_prediction",
                "judgeable_timing": False, "judgeable_side": False,
                "timing_correct": False, "raw_joint_correct": False, "joint_correct": False,
                "timing_error_frames": None, "timing_error_base30": None,
                "envelope_rally_id": None, "unmatched_before_recovered_serve": False,
            })
            continue

        first_event = span.events[0]
        first_identity = (span.fixture, first_event.frame)
        pair = pairs_by_prediction.get(first_identity)
        matched = pair is not None
        matched_rally_id = None if pair is None else str(pair["rally_id"])
        matched_gt_frame = None if pair is None else int(pair["gt_frame"])
        label_side = None if pair is None else labels.target_sides.get(
            (span.fixture, matched_gt_frame)
        )
        raw_side = first_event.predicted_side
        final_side = voted_spans[section_identity].events[0].predicted_side
        if matched:
            status = "serve" if pair["first"] else "later_hit"
            envelope_rally_id = matched_rally_id
        else:
            retained_rally = _retained_rally(span, labels, pairs_by_prediction)
            inside = retained_rally is not None and (
                retained_rally.frames[0] <= first_event.frame <= retained_rally.frames[-1]
            )
            status = "extra_leading" if inside else "unknown"
            envelope_rally_id = None if retained_rally is None else retained_rally.rally_id
        recovered = not matched and any(
            event.frame > first_event.frame
            and (span.fixture, event.frame) in pairs_by_prediction
            and pairs_by_prediction[(span.fixture, event.frame)]["first"]
            for event in span.events[1:]
        )
        if recovered:
            for event in span.events[1:]:
                later_pair = pairs_by_prediction.get((span.fixture, event.frame))
                if later_pair is not None and later_pair["first"]:
                    serve_by_identity[(span.fixture, str(later_pair["rally_id"]))][
                        "unmatched_before_recovered_serve"
                    ] = True
        raw_status = _side_status(label_side, raw_side, matched)
        final_status = _side_status(label_side, final_side, matched)
        offset = None if pair is None else int(pair["offset"])
        start_rows.append({
            "fixture": span.fixture, "span_id": span.span_id, "first_frame": first_event.frame,
            "status": status, "accepted": accepted_span,
            "matched_rally_id": matched_rally_id, "matched_gt_frame": matched_gt_frame,
            "label_side": label_side, "raw_side": raw_side, "final_side": final_side,
            "raw_status": raw_status, "final_status": final_status,
            "judgeable_timing": status in {"serve", "later_hit", "extra_leading"},
            "judgeable_side": status in {"later_hit", "extra_leading"} or (status == "serve" and label_side is not None),
            "timing_correct": status == "serve",
            "raw_joint_correct": status == "serve" and raw_status == "correct",
            "joint_correct": status == "serve" and final_status == "correct",
            "timing_error_frames": offset,
            "timing_error_base30": _timing_error(offset, fps[span.fixture]),
            "envelope_rally_id": envelope_rally_id,
            "unmatched_before_recovered_serve": recovered,
        })

    serve_counts = _serve_counts(serve_rows)
    start_counts = _start_counts(start_rows)
    contact_total, contact_by_video = _contact_totals(raw_scores)
    total: dict[str, int | float] = {**serve_counts, **start_counts, **contact_total}
    fixtures = sorted(set(fps) | set(labels.rallies) | set(stream.events_by_fixture))
    by_video = []
    for fixture in fixtures:
        contact = next((row for row in contact_by_video if row["fixture"] == fixture), {})
        by_video.append({
            "fixture": fixture,
            **_serve_counts([row for row in serve_rows if row["fixture"] == fixture]),
            **_start_counts([row for row in start_rows if row["fixture"] == fixture]),
            **{key: value for key, value in contact.items() if key != "fixture"},
        })
    return {
        "tolerance_base30": tolerance_base30,
        "tolerance_by_fixture": {
            fixture: ScalingKind.FRAME_COUNT.scale(tolerance_base30, fps[fixture])
            for fixture in sorted(fps)
        },
        "accepted_sections": None if accepted is None else [list(value) for value in sorted(accepted)],
        "total": total, "by_video": by_video,
        "serve_rows": serve_rows, "start_rows": start_rows,
        "rows": {"serves": serve_rows, "starts": start_rows},
        "contact_totals": raw_scores["total"], "contact_by_video": raw_scores["by_video"],
    }


def _rows_by_identity(
    rows: Sequence[Mapping[str, Any]], keys: tuple[str, ...]
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output = {}
    for row in rows:
        identity = tuple(row[key] for key in keys)
        if identity in output:
            raise ValueError(f"duplicate row identity {identity}")
        output[identity] = row
    return output


def compare_serves(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Compare serve and proposed-start identities between two analyses."""
    before_serves = _rows_by_identity(before["serve_rows"], ("fixture", "rally_id"))
    after_serves = _rows_by_identity(after["serve_rows"], ("fixture", "rally_id"))
    recovered, lost, right_to_wrong, wrong_to_right = [], [], [], []
    for identity in sorted(set(before_serves) | set(after_serves)):
        old, new = before_serves.get(identity), after_serves.get(identity)
        old_matched = old is not None and old["accepted_timing_matched"]
        new_matched = new is not None and new["accepted_timing_matched"]
        if not old_matched and new_matched:
            recovered.append({"identity": list(identity), "before": old, "after": new})
        if old_matched and not new_matched:
            lost.append({"identity": list(identity), "before": old, "after": new})
        if old is None or new is None or not old_matched or not new_matched:
            continue
        if old["accepted_final_status"] == "correct" and new["accepted_final_status"] == "wrong":
            right_to_wrong.append({"identity": list(identity), "before": old, "after": new})
        if old["accepted_final_status"] == "wrong" and new["accepted_final_status"] == "correct":
            wrong_to_right.append({"identity": list(identity), "before": old, "after": new})

    before_starts = _rows_by_identity(before["start_rows"], ("fixture", "span_id"))
    after_starts = _rows_by_identity(after["start_rows"], ("fixture", "span_id"))
    starts = {
        "newly_timing_correct": [], "spoiled_timing_correct": [],
        "newly_joint_correct": [], "spoiled_joint_correct": [],
    }
    for identity in sorted(set(before_starts) | set(after_starts)):
        old, new = before_starts.get(identity), after_starts.get(identity)
        for name, field in (("timing", "timing_correct"), ("joint", "joint_correct")):
            old_value = old is not None and old["accepted"] and old[field]
            new_value = new is not None and new["accepted"] and new[field]
            if not old_value and new_value:
                starts[f"newly_{name}_correct"].append(
                    {"identity": list(identity), "before": old, "after": new}
                )
            if old_value and not new_value:
                starts[f"spoiled_{name}_correct"].append(
                    {"identity": list(identity), "before": old, "after": new}
                )

    rows = {
        "recovered": recovered, "lost": lost,
        "right_to_wrong": right_to_wrong, "wrong_to_right": wrong_to_right,
        **starts,
    }
    return {
        "counts": {name: len(values) for name, values in rows.items()},
        **rows, "starts": starts, "rows": rows,
    }


def accepted_serves(result: Mapping[str, Any], accepted: set[SectionIdentity]) -> dict[str, Any]:
    """Recount accepted output using the already saved full-stream matches.

    Label denominators remain unchanged. All proposed starts remain available
    in the rows and all-output counts; accepted status counts have their own prefix.
    """
    serve_rows = []
    for source in result["serve_rows"]:
        matched = source["timing_matched"] and any(
            (source["fixture"], span_id) in accepted for span_id in source["matched_span_ids"]
        )
        row = {**source, "accepted_timing_matched": matched}
        for side in ("raw", "final"):
            row[f"accepted_{side}_status"] = _side_status(row["label_side"], row[f"{side}_side"], matched)
        serve_rows.append(row)
    start_rows = []
    for source in result["start_rows"]:
        row = {**source, "accepted": (source["fixture"], source["span_id"]) in accepted}
        start_rows.append(row)
    total = {**_serve_counts(serve_rows), **_start_counts(start_rows)}
    by_video = []
    for fixture in sorted({row["fixture"] for row in serve_rows} | {row["fixture"] for row in start_rows}):
        by_video.append({
            "fixture": fixture,
            **_serve_counts([row for row in serve_rows if row["fixture"] == fixture]),
            **_start_counts([row for row in start_rows if row["fixture"] == fixture]),
        })
    return {
        "tolerance_base30": result["tolerance_base30"], "matching_reused": True,
        "accepted_sections": [list(identity) for identity in sorted(accepted)],
        "total": total, "by_video": by_video, "serve_rows": serve_rows, "start_rows": start_rows,
    }
