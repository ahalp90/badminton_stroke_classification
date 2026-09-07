"""Recount chooser targets after the existing fixed-membership padding pass."""

from __future__ import annotations

import argparse
import lzma
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.evaluation import (
    overlapping_rallies,
    section_result,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    apply_options,
    option_record,
)
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
    scale_base30_frames,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"
DEFAULT_PREPARED = ROOT / "raw/later_run/prepared.joblib"
DEFAULT_REFERENCE = ROOT / "results/later/later_margin_predictions.json.gz"
DEFAULT_CURRENT = ROOT / "raw/followups/development_predictions/local_predictions.json.gz"
GROUPS = ("A", "B", "C", "D")
TOLERANCE_BASE30 = 10
SECONDARY_TOLERANCE_BASE30 = 5
SectionIdentity = tuple[str, int]
OptionRow = tuple[int, LaterOption, int]


@dataclass(frozen=True)
class FixtureResult:
    """Target updates and audit rows returned by one independent video task."""

    group: str
    updates: tuple[tuple[int, int], ...]
    changes: tuple[dict[str, Any], ...]
    proposal_changes: tuple[dict[str, Any], ...]
    stats: Mapping[str, Mapping[str, int]]


def _fixture_labels(labels: HumanLabels, fixture: str) -> HumanLabels:
    """Keep only one fixture so a worker carries no unrelated label records."""
    target_sides = {
        identity: side
        for identity, side in labels.target_sides.items()
        if identity[0] == fixture
    }
    return HumanLabels({fixture: tuple(labels.rallies.get(fixture, ()))}, target_sides)


def _proposal_key(option: LaterOption) -> tuple[Any, ...]:
    """Identify the baseline proposal that owns an alternative."""
    return option.base.identity


def _bucket(option: LaterOption) -> str:
    return f"kind={option.base.kind};later_insertion={str(bool(option.inserted_events)).lower()}"


def _empty_stats() -> dict[str, int]:
    return {
        "option_rows": 0,
        "old_target_minus_one": 0,
        "old_target_zero": 0,
        "old_target_one": 0,
        "feasible_options": 0,
        "reconstructed_options": 0,
        "reconstruction_calls": 0,
        "changed_rows": 0,
    }


def _target_stat_name(value: int) -> str:
    return {
        -1: "old_target_minus_one",
        0: "old_target_zero",
        1: "old_target_one",
    }[value]


def _rally_for_baseline(span: FixedSpan, labels: HumanLabels) -> RallyReference:
    rallies = overlapping_rallies(span, labels)
    if len(rallies) != 1:
        raise ValueError(
            f"{span.fixture}/{span.span_id}: saved non-unknown target has {len(rallies)} baseline rallies"
        )
    return rallies[0]


def _timing_possible(span: FixedSpan, rally: RallyReference, tolerance: int) -> bool:
    predicted = sorted(event.frame for event in span.events)
    labelled = sorted(rally.frames)
    return len(predicted) == len(labelled) and all(
        abs(predicted_frame - labelled_frame) <= tolerance
        for predicted_frame, labelled_frame in zip(predicted, labelled, strict=True)
    )


def _span_identity(span: FixedSpan) -> SectionIdentity:
    return span.fixture, span.span_id


def evaluate_option_with_padding(
    option: LaterOption,
    baseline_spans: Sequence[FixedSpan],
    full_events: Sequence[FixedEvent],
    reference: Mapping[SectionIdentity, LaterOption],
    labels: HumanLabels,
    fps: float,
) -> tuple[FixedSpan, FixedSpan, dict[str, Any], dict[str, Any]]:
    """Apply one option in its complete fixed reference stream, then score both tolerances."""
    fixture = option.span.fixture
    selected = dict(reference)
    selected[option.base.identity] = option
    raw = apply_options(baseline_spans, {fixture: tuple(full_events)}, selected)
    padded = pad_contact_boundaries(
        raw.spans,
        raw.events_by_fixture,
        {fixture: fps},
        padding_base30=TOLERANCE_BASE30,
        preserve_membership=True,
    )
    candidate = next((span for span in padded.spans if _span_identity(span) == option.base.identity), None)
    if candidate is None:
        raise ValueError(f"{option.base.identity}: padded option span is missing")
    raw_candidate = next((span for span in raw.spans if _span_identity(span) == option.base.identity), None)
    if raw_candidate is None:
        raise ValueError(f"{option.base.identity}: reconstructed option span is missing")
    tolerance = scale_base30_frames(TOLERANCE_BASE30, fps)
    secondary_tolerance = scale_base30_frames(SECONDARY_TOLERANCE_BASE30, fps)
    return (
        raw_candidate,
        candidate,
        section_result(candidate, labels, tolerance),
        section_result(candidate, labels, secondary_tolerance),
    )


def _matched_label_frames(result: Mapping[str, Any], labels: HumanLabels, fixture: str) -> list[int]:
    rally_id = result["rally_id"]
    if rally_id is None:
        return []
    rally = next(rally for rally in labels.rallies[fixture] if rally.rally_id == rally_id)
    return [int(rally.frames[pair[0]]) for pair in result["matches"]]


def _change_record(
    index: int,
    option: LaterOption,
    old_target: int,
    new_target: int,
    raw_span: FixedSpan,
    padded_span: FixedSpan,
    result: Mapping[str, Any],
    secondary_result: Mapping[str, Any],
    labels: HumanLabels,
    group: str,
) -> dict[str, Any]:
    return {
        "index": int(index),
        "option_record": option_record(option),
        "old_target": old_target,
        "new_target": new_target,
        "group": group,
        "base_kind": option.base.kind,
        "later_insertion": bool(option.inserted_events),
        "original_bounds": [raw_span.start_frame, raw_span.end_frame],
        "padded_bounds": [padded_span.start_frame, padded_span.end_frame],
        "predicted_frames": [event.frame for event in raw_span.events],
        "predicted_raw_sides": [event.predicted_side for event in raw_span.events],
        "matched_label_frames": _matched_label_frames(result, labels, option.span.fixture),
        "new_scorer_result": dict(result),
        "new_target_5": int(bool(secondary_result["side_rule_fully_correct"])),
        "new_scorer_result_5": dict(secondary_result),
    }


def _evaluate_fixture(
    fixture: str,
    group: str,
    spans: tuple[FixedSpan, ...],
    events: tuple[FixedEvent, ...],
    fps: float,
    option_rows: tuple[OptionRow, ...],
    reference: Mapping[SectionIdentity, LaterOption],
    labels: HumanLabels,
    old_positive_sections: frozenset[SectionIdentity],
    current_correct_sections: frozenset[SectionIdentity],
) -> FixtureResult:
    started = perf_counter()
    eligible_sections = {
        option.base.identity for _index, option, old_target in option_rows if old_target >= 0
    }
    baseline_rallies = {
        _span_identity(span): _rally_for_baseline(span, labels)
        for span in spans
        if _span_identity(span) in eligible_sections
    }
    cache: dict[FixedSpan, tuple[FixedSpan, FixedSpan, dict[str, Any], dict[str, Any]]] = {}
    updates: list[tuple[int, int]] = []
    changes: list[dict[str, Any]] = []
    proposal_changes: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = defaultdict(_empty_stats)
    tolerance = scale_base30_frames(TOLERANCE_BASE30, fps)

    for index, option, old_target in option_rows:
        bucket = _bucket(option)
        counters = stats[bucket]
        counters["option_rows"] += 1
        if old_target not in (-1, 0, 1):
            raise ValueError(f"{index}: target must be -1, 0 or 1")
        counters[_target_stat_name(old_target)] += 1
        if old_target == -1:
            continue
        rally = baseline_rallies[option.base.identity]
        if old_target == 0 and not _timing_possible(option.span, rally, tolerance):
            continue
        counters["feasible_options"] += 1
        if option.span not in cache:
            cache[option.span] = evaluate_option_with_padding(
                option, spans, events, reference, labels, fps,
            )
            counters["reconstruction_calls"] += 1
        raw_span, padded_span, result, secondary_result = cache[option.span]
        counters["reconstructed_options"] += 1
        new_target = int(bool(result["side_rule_fully_correct"]))
        updates.append((index, new_target))
        if new_target == old_target:
            continue
        counters["changed_rows"] += 1
        change = _change_record(
            index, option, old_target, new_target, raw_span, padded_span,
            result, secondary_result, labels, group,
        )
        changes.append(change)
        proposal_changes.append({
            "key": _proposal_key(option),
            "group": group,
            "bucket": bucket,
            "old_target": old_target,
            "new_target": new_target,
            "currently_correct": option.base.identity in current_correct_sections,
            "has_old_positive": option.base.identity in old_positive_sections,
        })
    print(f"{fixture}: {len(changes)} changed answers in {perf_counter() - started:.1f}s", flush=True)
    return FixtureResult(group, tuple(updates), tuple(changes), tuple(proposal_changes), dict(stats))


def _score_current_padding(
    spans: Sequence[FixedSpan],
    events: Mapping[str, Sequence[FixedEvent]],
    current: Mapping[SectionIdentity, LaterOption],
    labels: HumanLabels,
    fps: Mapping[str, float],
    groups: Mapping[str, str],
    fixtures: Sequence[str],
) -> tuple[frozenset[SectionIdentity], frozenset[tuple[str, str]], dict[str, int]]:
    raw = apply_options(spans, events, current)
    correct_sections: set[SectionIdentity] = set()
    correct_rallies: set[tuple[str, str]] = set()
    by_group: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for fixture in fixtures:
        fixture_spans = tuple(span for span in raw.spans if span.fixture == fixture)
        padded = pad_contact_boundaries(
            fixture_spans,
            {fixture: tuple(raw.events_by_fixture[fixture])},
            {fixture: fps[fixture]},
            padding_base30=TOLERANCE_BASE30,
            preserve_membership=True,
        )
        fixture_labels = _fixture_labels(labels, fixture)
        tolerance = scale_base30_frames(TOLERANCE_BASE30, fps[fixture])
        for span in padded.spans:
            result = section_result(span, fixture_labels, tolerance)
            if not result["side_rule_fully_correct"]:
                continue
            identity = span.fixture, span.span_id
            correct_sections.add(identity)
            rally_id = result["rally_id"]
            if rally_id is not None:
                rally_identity = (fixture, rally_id)
                correct_rallies.add(rally_identity)
                by_group[groups[fixture]].add(rally_identity)
    return frozenset(correct_sections), frozenset(correct_rallies), {
        group: len(rally_ids) for group, rally_ids in by_group.items()
    }


def _merge_stats(target: dict[str, int], source: Mapping[str, int]) -> None:
    for name, value in source.items():
        target[name] = target.get(name, 0) + int(value)


def _change_counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "changed_unique_proposals": len(entries),
        "zero_to_one": sum(entry["zero_to_one"] for entry in entries),
        "one_to_zero": sum(entry["one_to_zero"] for entry in entries),
        "changed_currently_wrong_proposals": sum(not entry["currently_correct"] for entry in entries),
        "proposals_without_any_old_positive_gaining_positive": sum(
            entry["zero_to_one"]
            and not entry["has_old_positive"]
            for entry in entries
        ),
    }


def _unique_changes(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in entries:
        key = entry["key"]
        previous = unique.get(key)
        if previous is None:
            unique[key] = {
                "key": key,
                "group": entry["group"],
                "currently_correct": bool(entry["currently_correct"]),
                "has_old_positive": bool(entry["has_old_positive"]),
                "zero_to_one": entry["old_target"] == 0 and entry["new_target"] == 1,
                "one_to_zero": entry["old_target"] == 1 and entry["new_target"] == 0,
            }
            continue
        previous["zero_to_one"] |= entry["old_target"] == 0 and entry["new_target"] == 1
        previous["one_to_zero"] |= entry["old_target"] == 1 and entry["new_target"] == 0
    return list(unique.values())


def _compress_targets(path: Path, targets: np.ndarray) -> None:
    with lzma.open(path, "wb", format=lzma.FORMAT_XZ, preset=9) as handle:
        np.save(handle, targets)


def _input_identifier(path: Path) -> str:
    """Return a stable repository-relative identifier without exposing host paths."""
    try:
        return path.resolve().relative_to(prediction_io.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _fresh_output_root(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output root must be a fresh directory: {path}")
    path.mkdir(parents=True)


def run(
    prepared_path: Path = DEFAULT_PREPARED,
    reference_path: Path = DEFAULT_REFERENCE,
    current_path: Path = DEFAULT_CURRENT,
    output_root: Path | None = None,
    jobs: int = 4,
    limit_fixtures: int | None = None,
) -> dict[str, Any]:
    """Run the target census and write ``targets.npy.xz`` and ``census.json.gz``."""
    if output_root is None:
        raise ValueError("output_root is required")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if limit_fixtures is not None and limit_fixtures <= 0:
        raise ValueError("limit_fixtures must be positive")
    started = perf_counter()
    _fresh_output_root(output_root)
    prepared = joblib.load(prepared_path)
    for key in ("base_population", "options", "targets"):
        if key not in prepared:
            raise KeyError(f"prepared cache is missing {key}")
    population = prepared["base_population"]
    options = tuple(prepared["options"])
    old_targets = np.asarray(prepared["targets"], dtype=np.int8)
    if old_targets.ndim != 1 or len(old_targets) != len(options) or not np.isin(old_targets, (-1, 0, 1)).all():
        raise ValueError("prepared targets do not align with options")
    if any(group not in GROUPS for group in population.groups.values()):
        raise ValueError("prepared population contains a group outside A-D")
    videos = tuple(population.videos)
    if not videos:
        raise ValueError("prepared population has no videos")
    all_fixtures = tuple(video.fixture for video in videos)
    fixture_limit = len(all_fixtures) if limit_fixtures is None else min(limit_fixtures, len(all_fixtures))
    fixtures = all_fixtures[:fixture_limit]
    fixture_set = frozenset(fixtures)

    reference_payload = prediction_io.read_json(reference_path)
    current_payload = prediction_io.read_json(current_path)
    reference = restore_choices(options, reference_payload["selected_actions"])
    current = restore_choices(options, current_payload["selected_actions"])
    reference_stream = apply_options(population.spans, population.events, reference)
    current_stream = apply_options(population.spans, population.events, current)
    for payload, stream, name in (
        (reference_payload, reference_stream, "reference"),
        (current_payload, current_stream, "current"),
    ):
        if "outputs" in payload and stream_records(stream) != payload["outputs"]:
            raise ValueError(f"saved {name} choices do not reproduce their contact stream")

    labels = load_human_labels(start.LABEL_PATH, population.videos)
    current_correct_sections, current_correct_rallies, current_by_group = _score_current_padding(
        population.spans, population.events, current, labels, population.fps, population.groups, fixtures,
    )
    old_positive_sections = frozenset(
        option.base.identity for option, target in zip(options, old_targets, strict=True) if target == 1
    )
    options_by_fixture: dict[str, list[OptionRow]] = defaultdict(list)
    for index, (option, old_target) in enumerate(zip(options, old_targets, strict=True)):
        if option.span.fixture in fixture_set:
            options_by_fixture[option.span.fixture].append((index, option, int(old_target)))

    tasks = []
    for fixture in fixtures:
        fixture_spans = tuple(span for span in population.spans if span.fixture == fixture)
        if not fixture_spans:
            raise ValueError(f"{fixture}: population has no sections")
        tasks.append((
            fixture,
            population.groups[fixture],
            fixture_spans,
            tuple(population.events[fixture]),
            float(population.fps[fixture]),
            tuple(options_by_fixture[fixture]),
            {identity: reference[identity] for identity in reference if identity[0] == fixture},
            _fixture_labels(labels, fixture),
            frozenset(identity for identity in old_positive_sections if identity[0] == fixture),
            frozenset(identity for identity in current_correct_sections if identity[0] == fixture),
        ))
    worker_count = min(jobs, len(tasks))
    if worker_count == 1:
        fixture_results = [_evaluate_fixture(*task) for task in tasks]
    else:
        with joblib.parallel_config(backend="loky", n_jobs=worker_count, inner_max_num_threads=1):
            fixture_results = joblib.Parallel()(
                joblib.delayed(_evaluate_fixture)(*task) for task in tasks
            )

    new_targets = old_targets.copy()
    changes: list[dict[str, Any]] = []
    proposal_changes: list[dict[str, Any]] = []
    stats_by_group: dict[str, dict[str, int]] = defaultdict(_empty_stats)
    stats_by_bucket: dict[str, dict[str, int]] = defaultdict(_empty_stats)
    stats_by_group_bucket: dict[tuple[str, str], dict[str, int]] = defaultdict(_empty_stats)
    for result in fixture_results:
        for index, value in result.updates:
            new_targets[index] = value
        changes.extend(result.changes)
        proposal_changes.extend(result.proposal_changes)
        for bucket, values in result.stats.items():
            _merge_stats(stats_by_bucket[bucket], values)
            _merge_stats(stats_by_group_bucket[(result.group, bucket)], values)
        for values in result.stats.values():
            _merge_stats(stats_by_group[result.group], values)

    unique_changes = _unique_changes(proposal_changes)
    unique_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in proposal_changes:
        unique_by_group[entry["group"]].append(entry)
    buckets = sorted(set(stats_by_bucket) | {entry["bucket"] for entry in proposal_changes})
    bucket_counts = {}
    for bucket in buckets:
        bucket_counts[bucket] = {
            **stats_by_bucket[bucket],
            **_change_counts(_unique_changes([
                entry for entry in proposal_changes if entry["bucket"] == bucket
            ])),
        }
    group_changes = {
        group: _change_counts(_unique_changes(entries)) for group, entries in unique_by_group.items()
    }
    scope_indices = [index for index, option in enumerate(options) if option.span.fixture in fixture_set]
    scope_targets_old = old_targets[scope_indices]
    scope_targets_new = new_targets[scope_indices]
    scope_counts = _empty_stats()
    for result in fixture_results:
        for values in result.stats.values():
            _merge_stats(scope_counts, values)
    scope_counts.update({
        "old_target_counts": {str(value): int(np.count_nonzero(scope_targets_old == value)) for value in (-1, 0, 1)},
        "new_target_counts": {str(value): int(np.count_nonzero(scope_targets_new == value)) for value in (-1, 0, 1)},
        "changed_rows": len(changes),
        **_change_counts(unique_changes),
    })
    scope_counts["changed_option_rows_zero_to_one"] = sum(
        row["old_target"] == 0 and row["new_target"] == 1 for row in changes
    )
    scope_counts["changed_option_rows_one_to_zero"] = sum(
        row["old_target"] == 1 and row["new_target"] == 0 for row in changes
    )

    _compress_targets(output_root / "targets.npy.xz", new_targets)
    seconds = perf_counter() - started
    status = "partial" if limit_fixtures is not None else "complete"
    census = {
        "schema": "contact-closing-padded-target-census/1",
        "status": status,
        "labels_used_for_scoring": True,
        "selection_uses_labels": False,
        "reference_choices_fixed": True,
        "eligibility": "saved targets; original -1 values remain -1 exactly",
        "target_tolerance_base30": TOLERANCE_BASE30,
        "secondary_tolerance_base30": SECONDARY_TOLERANCE_BASE30,
        "input_identifiers": {
            "prepared": _input_identifier(prepared_path),
            "reference": _input_identifier(reference_path),
            "current": _input_identifier(current_path),
        },
        "fixtures_processed": list(fixtures),
        "fps": {fixture: population.fps[fixture] for fixture in fixtures},
        "groups_processed": sorted({population.groups[fixture] for fixture in fixtures}),
        "option_count": len(options),
        "processed_option_count": len(scope_indices),
        "feasible_option_count": scope_counts["feasible_options"],
        "reconstructed_option_count": scope_counts["reconstructed_options"],
        "reconstruction_call_count": scope_counts["reconstruction_calls"],
        "seconds": seconds,
        "estimated_full_seconds": seconds * len(all_fixtures) / len(fixtures),
        "counts": scope_counts,
        "by_group": {
            group: {
                "counts": {
                    **stats_by_group[group],
                    **group_changes.get(group, {}),
                },
                "by_base_kind_and_later_insertion": {
                    bucket: {
                        **stats_by_group_bucket[(group, bucket)],
                        **_change_counts(_unique_changes([
                            entry for entry in proposal_changes
                            if entry["group"] == group and entry["bucket"] == bucket
                        ])),
                    }
                    for bucket in buckets
                    if stats_by_group_bucket[(group, bucket)]["option_rows"]
                },
            }
            for group in sorted({population.groups[fixture] for fixture in fixtures})
        },
        "by_base_kind_and_later_insertion": bucket_counts,
        "current": {
            "correct_unique_rally_identities_after_padding": len(current_correct_rallies),
            "correct_unique_rally_identities_by_group": current_by_group,
            "correct_sections_after_padding": len(current_correct_sections),
            "scored_fixtures": len(fixtures),
        },
        "changed_rows": sorted(changes, key=lambda row: row["index"]),
    }
    write_json(output_root / "census.json.gz", census)
    print(
        f"Padded target census {status}: {len(changes)} changed rows, "
        f"{scope_counts['changed_unique_proposals']} unique proposals; "
        f"{seconds:.1f}s",
        flush=True,
    )
    return census


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit-fixtures", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    run(
        arguments.prepared,
        arguments.reference,
        arguments.current,
        arguments.output_root,
        arguments.jobs,
        arguments.limit_fixtures,
    )


if __name__ == "__main__":
    main()
