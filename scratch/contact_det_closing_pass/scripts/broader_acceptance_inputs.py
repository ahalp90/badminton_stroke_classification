"""Prepare frozen broader detector inputs for acceptance scoring."""

from __future__ import annotations

import lzma
from collections.abc import Mapping, Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.chosen_acceptance import (
    refresh_boundary_features,
)
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.gap_evidence import gap_evidence
from scratch.contact_det_closing_pass.scripts.later_acceptance_features import (
    acceptance_features,
)
from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    apply_options,
    build_later_options,
)
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    ROOT,
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_insertion_broader import (
    _candidate_inputs,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    _merge_measurements,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    load_measurements,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_options import build_options
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    build_candidate_rows,
)

FEATURE_ROOT = ROOT / "raw/broader_inputs/features"
OPTION_SCORE_ROOT = ROOT / "raw/followups"
SectionIdentity = tuple[str, int]


def _scores(fixture: str, option_count: int) -> np.ndarray:
    path = OPTION_SCORE_ROOT / f"local_{fixture}_option_scores.npy.xz"
    with lzma.open(path, "rb") as handle:
        values = np.load(handle, allow_pickle=False)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (option_count,):
        raise ValueError(f"{fixture}: saved option score count differs from rebuilt options")
    if not np.isfinite(values).all():
        raise ValueError(f"{fixture}: saved option scores are incomplete")
    return values


def _restored_current(
    options: Sequence[LaterOption], spans: Sequence[FixedSpan], events: Sequence[FixedEvent],
    current_record: Mapping[str, Any], fixture: str,
) -> tuple[dict[SectionIdentity, LaterOption], Any]:
    selected = restore_choices(options, current_record["selected_actions"])
    stream = apply_options(spans, {fixture: tuple(events)}, selected)
    if stream_records(stream) != current_record["output"]:
        raise ValueError(f"{fixture}: current selected output does not replay exactly")
    return selected, stream


def prepare_video(
    opening_video: dict,
    later_video: dict,
    spans: Sequence[FixedSpan],
    events: Sequence[FixedEvent],
    physical_names: tuple,
    current_record: dict,
    guarded_record: dict,
    local_model: Any,
) -> dict:
    """Rebuild one frozen option pool and acceptance inputs without detector fits."""
    started = perf_counter()
    fixture = str(later_video["fixture"])
    fps = {fixture: float(later_video["fps"])}
    candidates, later_physical = _candidate_inputs(later_video, spans)
    actions = start.build_action_rows(
        build_candidate_rows([opening_video], default_group="ShuttleSet22", max_earlier_candidates=2),
    )
    base_by_section = build_options(spans, [opening_video], {fixture: tuple(events)})
    base_options = tuple(option for section in base_by_section.values() for option in section)
    options = build_later_options(base_options, candidates, fps, max_insertions=1)
    if int(current_record.get("option_count", len(options))) != len(options):
        raise ValueError(f"{fixture}: rebuilt option count differs from saved detector record")
    scores = _scores(fixture, len(options))
    current_selected, current_stream = _restored_current(
        options, spans, events, current_record, fixture,
    )
    guarded_stream = restore_stream(guarded_record["output"])
    guarded_by_identity = {
        (span.fixture, span.span_id): span for span in guarded_stream.spans
    }
    current_identities = set(current_selected)
    if set(guarded_by_identity) != current_identities:
        raise ValueError(f"{fixture}: guarded output section coverage differs")
    if guarded_stream.events_by_fixture != current_stream.events_by_fixture:
        raise ValueError(f"{fixture}: guarded output changed the full event stream")
    guarded_selected = {
        identity: replace(option, span=guarded_by_identity[identity])
        for identity, option in current_selected.items()
    }
    measurements = load_measurements(actions, {fixture: tuple(events)}, FEATURE_ROOT)
    if measurements.audit["missing_identity_count"]:
        raise ValueError(f"{fixture}: saved broader measurements are missing")
    measurements = _merge_measurements(measurements, later_physical, physical_names)
    base, base_names, identities = acceptance_features(
        options, scores, current_selected, candidates, fps, measurements,
    )
    refresh_boundary_features(base, base_names, identities, guarded_selected, fps)
    gap, gap_names, gap_identities = gap_evidence(
        guarded_selected, candidates, fps, measurements, local_model,
    )
    if tuple(gap_identities) != tuple(identities):
        by_identity = {identity: index for index, identity in enumerate(gap_identities)}
        gap = np.asarray([gap[by_identity[identity]] for identity in identities], dtype=np.float64)
    selected_scores = {
        identity: float(base[index, 0]) for index, identity in enumerate(identities)
    }
    return {
        "fixture": fixture,
        "seconds": perf_counter() - started,
        "base_features": base,
        "gap_features": np.column_stack((base, gap)),
        "base_feature_names": tuple(base_names),
        "gap_feature_names": (*base_names, *gap_names),
        "selected_scores": selected_scores,
        "identities": tuple(identities),
        "guarded_stream": guarded_stream,
    }
