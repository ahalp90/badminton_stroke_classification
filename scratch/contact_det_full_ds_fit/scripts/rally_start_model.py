"""Build, fit and score the fixed rally-start contact models."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_evidence import scale_base30_frames
from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
    evaluate_span,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    ModelKind,
    ModelSpec,
    RallyStartModelConfig,
    ResultGate,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    ContactLabels,
    contact_counts,
)

CandidateIdentity = tuple[str, int, int]
SectionIdentity = tuple[str, int]


@dataclass(frozen=True)
class HumanLabels:
    """Rally timing and player-side labels for an allowed video set."""

    rallies: Mapping[str, tuple[RallyReference, ...]]
    target_sides: Mapping[tuple[str, int], str]

    @property
    def contact_labels(self) -> ContactLabels:
        """Return the contact-only view used by the existing timing scorer."""
        frames = {
            fixture: np.asarray(
                [frame for rally in rallies for frame in rally.frames],
                dtype=np.int32,
            )
            for fixture, rallies in self.rallies.items()
        }
        first_contacts = {
            fixture: frozenset(rally.frames[0] for rally in rallies)
            for fixture, rallies in self.rallies.items()
        }
        rally_counts = {
            fixture: len(rallies) for fixture, rallies in self.rallies.items()
        }
        return ContactLabels(frames, first_contacts, rally_counts)


@dataclass(frozen=True)
class CandidateRow:
    """One label-free earlier contact and its fixed model inputs."""

    fixture: str
    group: str
    fps: float
    span_id: int
    section_start_frame: int
    section_end_frame: int
    prefix_start_frame: int
    fixed_contact_frame: int
    frame: int
    contact_score: float
    fixed_contact_score: float
    kept: bool
    predicted_side: str | None
    fixed_predicted_side: str | None
    features: tuple[float, ...]

    @property
    def identity(self) -> CandidateIdentity:
        return (self.fixture, self.span_id, self.frame)

    @property
    def section_identity(self) -> SectionIdentity:
        return (self.fixture, self.span_id)


@dataclass(frozen=True)
class CandidateTarget:
    """The checked training answer for one earlier candidate."""

    included_in_training: bool
    positive: bool
    section_status: str
    rally_id: str | None
    first_contact_frame: int | None
    timing_match: bool
    side_match: bool


@dataclass(frozen=True)
class TargetAssignments:
    """Training answers and section-level label joins."""

    by_candidate: Mapping[CandidateIdentity, CandidateTarget]
    section_statuses: Mapping[SectionIdentity, str]

    @property
    def recoverable_sections(self) -> frozenset[SectionIdentity]:
        return frozenset(
            identity[:2]
            for identity, target in self.by_candidate.items()
            if target.positive
        )


@dataclass(frozen=True)
class ContactStreams:
    """Events and section bounds after applying selected additions."""

    spans: tuple[FixedSpan, ...]
    events_by_fixture: Mapping[str, tuple[FixedEvent, ...]]


@dataclass(frozen=True)
class _TimingVideo:
    """The two video fields needed by the contact timing scorer."""

    fixture: str
    fps: float


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _normalise_side(value: object, label: str) -> str | None:
    if value is None:
        return None
    if value == "Top":
        return "Top"
    if value in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{label}: player side differs")


def _finite_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return score


def _video_identity(
    video: Mapping[str, Any], default_group: str
) -> tuple[str, str, float]:
    raw_identity = video.get("video")
    identity = (
        _mapping(raw_identity, "training video identity") if raw_identity else video
    )
    fixture = identity.get("fixture")
    fps = identity.get("fps")
    group = video.get("group", default_group)
    if (
        not isinstance(fixture, str)
        or not fixture
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or float(fps) <= 0.0
        or not isinstance(group, str)
        or not group
    ):
        raise ValueError("saved video identity differs")
    return fixture, group, float(fps)


def _at_30_fps(frame_count: int, fps: float) -> float:
    return float(frame_count) * 30.0 / fps


def build_candidate_rows(
    videos: Sequence[Mapping[str, Any]],
    *,
    default_group: str,
    max_earlier_candidates: int = 2,
) -> tuple[CandidateRow, ...]:
    """Build nine model inputs, retaining the historical two-candidate default.

    A larger experimental shortlist requires an explicit opt-in.
    """
    rows: list[CandidateRow] = []
    candidate_identities: set[CandidateIdentity] = set()
    candidate_frames: set[tuple[str, int]] = set()
    video_names: set[str] = set()
    for video in videos:
        fixture, group, fps = _video_identity(video, default_group)
        if fixture in video_names:
            raise ValueError("saved video identities repeat")
        video_names.add(fixture)
        raw_spans = video.get("spans")
        raw_lists = video.get("candidate_lists")
        if not isinstance(raw_spans, list) or not isinstance(raw_lists, list):
            raise TypeError(f"{fixture}: spans and candidate lists must be lists")
        spans_by_id = {
            int(_mapping(span, f"{fixture}: span")["span_id"]): _mapping(
                span, f"{fixture}: span"
            )
            for span in raw_spans
        }
        model_training_videos = video.get("model_training_videos")
        if model_training_videos is not None and (
            not isinstance(model_training_videos, list)
            or len(model_training_videos) != 24
            or any(not isinstance(name, str) for name in model_training_videos)
            or len(set(model_training_videos)) != 24
            or fixture in model_training_videos
        ):
            raise ValueError(f"{fixture}: first-model training videos differ")
        previous_span_id = -1
        for raw_list in raw_lists:
            candidate_list = _mapping(raw_list, f"{fixture}: candidate list")
            if candidate_list.get("fixture") != fixture:
                raise ValueError(f"{fixture}: candidate-list video differs")
            span_id = candidate_list.get("span_id")
            section_start = candidate_list.get("section_start_frame")
            section_end = candidate_list.get("section_end_frame")
            prefix_start = candidate_list.get("prefix_start_frame")
            fixed_frame = candidate_list.get("fixed_contact_frame")
            duplicate_distance = candidate_list.get("duplicate_distance_frames")
            values = (
                span_id,
                section_start,
                section_end,
                prefix_start,
                fixed_frame,
                duplicate_distance,
            )
            if any(type(value) is not int for value in values):
                raise ValueError(f"{fixture}: candidate-list frames differ")
            assert isinstance(span_id, int)
            assert isinstance(section_start, int)
            assert isinstance(section_end, int)
            assert isinstance(prefix_start, int)
            assert isinstance(fixed_frame, int)
            assert isinstance(duplicate_distance, int)
            if (
                span_id <= previous_span_id
                or prefix_start < 0
                or prefix_start >= fixed_frame
                or section_start < 0
                or section_end <= section_start
                or not section_start <= fixed_frame < section_end
                or duplicate_distance < 0
            ):
                raise ValueError(f"{fixture}/{span_id}: candidate-list bounds differ")
            saved_span = spans_by_id.get(span_id)
            if saved_span is None or (
                saved_span.get("start_frame") != section_start
                or saved_span.get("end_frame") != section_end
            ):
                raise ValueError(
                    f"{fixture}/{span_id}: candidate section bounds differ"
                )
            previous_span_id = span_id
            raw_candidates = candidate_list.get("candidates")
            if not isinstance(raw_candidates, list) or not 3 <= len(raw_candidates) <= max_earlier_candidates + 1:
                raise ValueError(f"{fixture}/{span_id}: candidate-list size differs")
            fixed = _mapping(raw_candidates[0], f"{fixture}/{span_id}: fixed contact")
            if (
                fixed.get("is_fixed_contact") is not True
                or fixed.get("kept") is not True
                or fixed.get("frame") != fixed_frame
            ):
                raise ValueError(f"{fixture}/{span_id}: fixed contact differs")
            fixed_score = _finite_score(
                fixed.get("contact_score"),
                f"{fixture}/{span_id}: fixed contact score",
            )
            fixed_side = _normalise_side(
                fixed.get("predicted_side"),
                f"{fixture}/{span_id}: fixed contact",
            )
            fixed_identity = (fixture, fixed_frame)
            if fixed_identity in candidate_frames:
                raise ValueError("a candidate frame appears in more than one section")
            candidate_frames.add(fixed_identity)
            for raw_candidate in raw_candidates[1:]:
                candidate = _mapping(
                    raw_candidate,
                    f"{fixture}/{span_id}: earlier candidate",
                )
                frame = candidate.get("frame")
                if (
                    type(frame) is not int
                    or candidate.get("is_fixed_contact") is not False
                ):
                    raise ValueError(f"{fixture}/{span_id}: earlier candidate differs")
                if not prefix_start <= frame < fixed_frame:
                    raise ValueError(
                        f"{fixture}/{span_id}/{frame}: candidate frame differs"
                    )
                if fixed_frame - frame <= duplicate_distance:
                    raise ValueError(
                        f"{fixture}/{span_id}/{frame}: candidate is within the nearby-contact distance"
                    )
                contact_score = _finite_score(
                    candidate.get("contact_score"),
                    f"{fixture}/{span_id}/{frame}: candidate score",
                )
                kept = candidate.get("kept")
                if type(kept) is not bool:
                    raise ValueError(f"{fixture}/{span_id}/{frame}: kept flag differs")
                predicted_side = _normalise_side(
                    candidate.get("predicted_side"),
                    f"{fixture}/{span_id}/{frame}: candidate",
                )
                identity = (fixture, span_id, frame)
                if identity in candidate_identities:
                    raise ValueError("candidate identities repeat")
                candidate_identities.add(identity)
                frame_identity = (fixture, frame)
                if frame_identity in candidate_frames:
                    raise ValueError(
                        "a candidate frame appears in more than one section"
                    )
                candidate_frames.add(frame_identity)
                both_known = predicted_side is not None and fixed_side is not None
                features = (
                    contact_score,
                    fixed_score,
                    _at_30_fps(fixed_frame - frame, fps),
                    _at_30_fps(frame - section_start, fps),
                    _at_30_fps(section_end - section_start, fps),
                    float(kept),
                    float(predicted_side is not None),
                    float(fixed_side is not None),
                    float(both_known and predicted_side == fixed_side),
                )
                rows.append(
                    CandidateRow(
                        fixture,
                        group,
                        fps,
                        span_id,
                        section_start,
                        section_end,
                        prefix_start,
                        fixed_frame,
                        frame,
                        contact_score,
                        fixed_score,
                        kept,
                        predicted_side,
                        fixed_side,
                        features,
                    )
                )
    return tuple(rows)


def _spans_by_fixture(
    videos: Sequence[Mapping[str, Any]],
    default_group: str,
) -> tuple[
    dict[str, tuple[Mapping[str, Any], ...]],
    dict[str, tuple[Mapping[str, Any], ...]],
]:
    spans_by_fixture: dict[str, tuple[Mapping[str, Any], ...]] = {}
    contacts_by_fixture: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for video in videos:
        fixture, _group, _fps = _video_identity(video, default_group)
        raw_spans = video.get("spans")
        raw_contacts = video.get("kept_contacts")
        if not isinstance(raw_spans, list) or not isinstance(raw_contacts, list):
            raise TypeError(f"{fixture}: spans or kept contacts must be a list")
        spans_by_fixture[fixture] = tuple(
            _mapping(span, f"{fixture}: span") for span in raw_spans
        )
        contacts_by_fixture[fixture] = tuple(
            _mapping(contact, f"{fixture}: kept contact") for contact in raw_contacts
        )
    return spans_by_fixture, contacts_by_fixture


def assign_candidate_targets(
    rows: Sequence[CandidateRow],
    videos: Sequence[Mapping[str, Any]],
    labels: HumanLabels,
    *,
    default_group: str,
    tolerance_at_30_fps: int = 10,
) -> TargetAssignments:
    """Join allowed labels and assign at most one positive candidate per section."""
    spans_by_fixture, contacts_by_fixture = _spans_by_fixture(videos, default_group)
    rally_sections: Counter[tuple[str, str]] = Counter()
    rallies_for_section: dict[SectionIdentity, tuple[RallyReference, ...]] = {}
    for fixture, spans in spans_by_fixture.items():
        fixture_rallies = labels.rallies.get(fixture, ())
        for raw_span in spans:
            span_id = int(raw_span["span_id"])
            start = int(raw_span["start_frame"])
            end = int(raw_span["end_frame"])
            matches = tuple(
                rally
                for rally in fixture_rallies
                if any(start <= frame < end for frame in rally.frames)
            )
            rallies_for_section[(fixture, span_id)] = matches
            for rally in matches:
                rally_sections[(fixture, rally.rally_id)] += 1

    rows_by_section: dict[SectionIdentity, list[CandidateRow]] = {}
    for row in rows:
        rows_by_section.setdefault(row.section_identity, []).append(row)
    targets: dict[CandidateIdentity, CandidateTarget] = {}
    statuses: dict[SectionIdentity, str] = {}
    for section_identity, section_rows in rows_by_section.items():
        fixture, span_id = section_identity
        matching_rallies = rallies_for_section.get(section_identity)
        if matching_rallies is None:
            raise ValueError(f"{fixture}/{span_id}: candidate section is missing")
        if not matching_rallies:
            status = "no_labelled_rally"
            statuses[section_identity] = status
            for row in section_rows:
                targets[row.identity] = CandidateTarget(
                    True, False, status, None, None, False, False
                )
            continue
        if len(matching_rallies) > 1:
            status = "more_than_one_labelled_rally"
            statuses[section_identity] = status
            for row in section_rows:
                targets[row.identity] = CandidateTarget(
                    False, False, status, None, None, False, False
                )
            continue
        rally = matching_rallies[0]
        if rally_sections[(fixture, rally.rally_id)] > 1:
            status = "labelled_rally_touches_more_than_one_section"
            statuses[section_identity] = status
            for row in section_rows:
                targets[row.identity] = CandidateTarget(
                    False,
                    False,
                    status,
                    rally.rally_id,
                    rally.frames[0],
                    False,
                    False,
                )
            continue

        first_contact_frame = rally.frames[0]
        target_side = labels.target_sides[(fixture, first_contact_frame)]
        tolerance = scale_base30_frames(tolerance_at_30_fps, section_rows[0].fps)
        section_start = section_rows[0].section_start_frame
        section_end = section_rows[0].section_end_frame
        existing_frames = [
            int(contact["frame"])
            for contact in contacts_by_fixture[fixture]
            if section_start <= int(contact["frame"]) < section_end
        ]
        if any(
            abs(frame - first_contact_frame) <= tolerance for frame in existing_frames
        ):
            status = "first_contact_already_matched"
            chosen_identity = None
        else:
            usable = [
                row
                for row in section_rows
                if abs(row.frame - first_contact_frame) <= tolerance
                and row.predicted_side == target_side
            ]
            chosen = min(
                usable,
                key=lambda row: (
                    abs(row.frame - first_contact_frame),
                    -row.contact_score,
                    row.frame,
                ),
                default=None,
            )
            chosen_identity = None if chosen is None else chosen.identity
            status = "usable_candidate" if chosen is not None else "no_usable_candidate"
        statuses[section_identity] = status
        for row in section_rows:
            timing_match = abs(row.frame - first_contact_frame) <= tolerance
            side_match = (
                row.predicted_side is not None and row.predicted_side == target_side
            )
            targets[row.identity] = CandidateTarget(
                True,
                row.identity == chosen_identity,
                status,
                rally.rally_id,
                first_contact_frame,
                timing_match,
                side_match,
            )
    if set(targets) != {row.identity for row in rows}:
        raise ValueError("candidate target coverage differs")
    return TargetAssignments(
        MappingProxyType(targets),
        MappingProxyType(statuses),
    )


def _feature_array(rows: Sequence[CandidateRow]) -> np.ndarray:
    values = np.asarray([row.features for row in rows], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
        raise ValueError("candidate model inputs differ")
    return values


def make_candidate_model(spec: ModelSpec) -> Any:
    """Construct one model from the exact fixed settings."""
    settings = dict(spec.settings)
    if spec.kind is ModelKind.LOGISTIC_REGRESSION:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        standardise = settings.pop("standardise_numeric_inputs")
        if standardise is not True:
            raise ValueError("logistic regression scaling setting differs")
        return make_pipeline(StandardScaler(), LogisticRegression(**settings))
    if spec.kind is ModelKind.HISTOGRAM_GRADIENT_BOOSTING:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(**settings)
    raise ValueError(f"unsupported candidate model: {spec.kind}")


def _fit_model(
    spec: ModelSpec,
    training_rows: Sequence[CandidateRow],
    targets: TargetAssignments,
) -> Any:
    included_rows = [
        row
        for row in training_rows
        if targets.by_candidate[row.identity].included_in_training
    ]
    target_values = np.asarray(
        [targets.by_candidate[row.identity].positive for row in included_rows],
        dtype=np.uint8,
    )
    if not len(included_rows) or set(target_values.tolist()) != {0, 1}:
        raise ValueError(
            "candidate model training needs positive and negative examples"
        )
    model = make_candidate_model(spec)
    model.fit(_feature_array(included_rows), target_values)
    return model


def predict_candidate_scores(
    model: Any,
    rows: Sequence[CandidateRow],
) -> dict[CandidateIdentity, float]:
    """Predict the positive probability for each candidate in fixed order."""
    if not rows:
        return {}
    classes = np.asarray(model.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise ValueError("candidate model positive class differs")
    probabilities = np.asarray(model.predict_proba(_feature_array(rows)))
    scores = probabilities[:, int(positive_positions[0])]
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise ValueError("candidate model scores differ")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("candidate model scores must be between zero and one")
    return {row.identity: float(score) for row, score in zip(rows, scores, strict=True)}


def held_out_candidate_scores(
    rows: Sequence[CandidateRow],
    targets: TargetAssignments,
    config: RallyStartModelConfig,
) -> dict[str, dict[CandidateIdentity, float]]:
    """Fit on three video groups and score every row in the fourth."""
    expected_groups = set(config.training_groups)
    if {row.group for row in rows} != expected_groups:
        raise ValueError("candidate row groups differ")
    output: dict[str, dict[CandidateIdentity, float]] = {
        spec.model_id: {} for spec in config.models
    }
    for held_out_group in config.training_groups:
        training_rows = [row for row in rows if row.group != held_out_group]
        held_out_rows = [row for row in rows if row.group == held_out_group]
        for spec in config.models:
            model = _fit_model(spec, training_rows, targets)
            group_scores = predict_candidate_scores(model, held_out_rows)
            if set(output[spec.model_id]) & set(group_scores):
                raise ValueError("held-out candidate score identities repeat")
            output[spec.model_id].update(group_scores)
    expected_identities = {row.identity for row in rows}
    if any(set(scores) != expected_identities for scores in output.values()):
        raise ValueError("held-out candidate score coverage differs")
    return output


def fit_final_candidate_model(
    spec: ModelSpec,
    rows: Sequence[CandidateRow],
    targets: TargetAssignments,
) -> Any:
    """Fit the fixed candidate model on every allowed training video."""
    return _fit_model(spec, rows, targets)


def select_candidates(
    rows: Sequence[CandidateRow],
    scores: Mapping[CandidateIdentity, float],
    cutoff: float,
) -> dict[SectionIdentity, CandidateRow]:
    """Select at most one side-answerable candidate for each section."""
    if not 0.0 < cutoff < 1.0:
        raise ValueError("candidate selection cut-off must be between zero and one")
    expected_identities = {row.identity for row in rows}
    if set(scores) != expected_identities:
        raise ValueError("candidate selection score coverage differs")
    rows_by_section: dict[SectionIdentity, list[CandidateRow]] = {}
    for row in rows:
        rows_by_section.setdefault(row.section_identity, []).append(row)
    selected: dict[SectionIdentity, CandidateRow] = {}
    for section_identity, section_rows in rows_by_section.items():
        eligible = [
            row
            for row in section_rows
            if row.predicted_side is not None and scores[row.identity] >= cutoff
        ]
        if eligible:
            selected[section_identity] = max(
                eligible,
                key=lambda row: (
                    scores[row.identity],
                    row.contact_score,
                    -row.frame,
                ),
            )
    return selected


def _saved_videos_by_fixture(
    videos: Sequence[Mapping[str, Any]],
    default_group: str,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for video in videos:
        fixture, _group, _fps = _video_identity(video, default_group)
        if fixture in output:
            raise ValueError("saved video identities repeat")
        output[fixture] = video
    return output


def apply_selected_candidates(
    videos: Sequence[Mapping[str, Any]],
    selections: Mapping[SectionIdentity, CandidateRow],
    *,
    default_group: str,
) -> ContactStreams:
    """Keep every baseline event and apply the selected addition-only actions."""
    videos_by_fixture = _saved_videos_by_fixture(videos, default_group)
    events_by_fixture: dict[str, list[FixedEvent]] = {}
    original_span_by_event: dict[tuple[str, int], int | None] = {}
    for fixture, video in videos_by_fixture.items():
        raw_contacts = video.get("kept_contacts")
        if not isinstance(raw_contacts, list):
            raise TypeError(f"{fixture}: kept contacts must be a list")
        events: list[FixedEvent] = []
        previous_frame = -1
        for raw_contact in raw_contacts:
            contact = _mapping(raw_contact, f"{fixture}: kept contact")
            frame = int(contact["frame"])
            if frame <= previous_frame:
                raise ValueError(f"{fixture}: kept-contact order differs")
            previous_frame = frame
            event = FixedEvent(
                fixture,
                frame,
                _finite_score(contact.get("contact_score"), f"{fixture}/{frame}"),
                _normalise_side(contact.get("predicted_side"), f"{fixture}/{frame}"),
            )
            events.append(event)
            span_id = contact.get("span_id")
            if span_id is not None and type(span_id) is not int:
                raise ValueError(f"{fixture}/{frame}: kept-contact section differs")
            original_span_by_event[(fixture, frame)] = span_id
        events_by_fixture[fixture] = events

    for section_identity, candidate in selections.items():
        fixture, span_id = section_identity
        if (
            candidate.section_identity != section_identity
            or fixture not in events_by_fixture
        ):
            raise ValueError("selected candidate identity differs")
        if candidate.predicted_side is None:
            raise ValueError(
                f"{fixture}/{span_id}: selected candidate has no player side"
            )
        existing = [
            event
            for event in events_by_fixture[fixture]
            if event.frame == candidate.frame
        ]
        if candidate.kept:
            if len(existing) != 1:
                raise ValueError(f"{fixture}/{candidate.frame}: kept candidate differs")
            original_span = original_span_by_event[(fixture, candidate.frame)]
            if original_span is not None and original_span != span_id:
                raise ValueError(
                    f"{fixture}/{candidate.frame}: candidate belongs to another section"
                )
        else:
            if existing:
                raise ValueError(
                    f"{fixture}/{candidate.frame}: new candidate already exists"
                )
            events_by_fixture[fixture].append(
                FixedEvent(
                    fixture,
                    candidate.frame,
                    candidate.contact_score,
                    candidate.predicted_side,
                )
            )

    frozen_events = {
        fixture: tuple(sorted(events, key=lambda event: event.frame))
        for fixture, events in events_by_fixture.items()
    }
    spans: list[FixedSpan] = []
    assigned_event_identities: set[tuple[str, int]] = set()
    for fixture, video in videos_by_fixture.items():
        raw_spans = video.get("spans")
        if not isinstance(raw_spans, list):
            raise TypeError(f"{fixture}: spans must be a list")
        previous_end = -1
        for expected_span_id, raw_span in enumerate(raw_spans):
            span = _mapping(raw_span, f"{fixture}: span")
            span_id = int(span["span_id"])
            start = int(span["start_frame"])
            end = int(span["end_frame"])
            if span_id != expected_span_id or start < previous_end or end <= start:
                raise ValueError(f"{fixture}: section bounds differ")
            candidate = selections.get((fixture, span_id))
            output_start = start if candidate is None else min(start, candidate.frame)
            if output_start < previous_end:
                raise ValueError(
                    f"{fixture}/{span_id}: moved section overlaps its predecessor"
                )
            section_events = tuple(
                event
                for event in frozen_events[fixture]
                if output_start <= event.frame < end
            )
            for event in section_events:
                identity = (fixture, event.frame)
                if identity in assigned_event_identities:
                    raise ValueError(
                        f"{fixture}/{event.frame}: event belongs to two sections"
                    )
                assigned_event_identities.add(identity)
            spans.append(
                FixedSpan(
                    fixture,
                    span_id,
                    output_start,
                    end,
                    section_events,
                )
            )
            previous_end = end
    return ContactStreams(tuple(spans), MappingProxyType(frozen_events))


def _fully_correct(
    streams: ContactStreams,
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    *,
    tolerance_at_30_fps: int,
    confidence_requirement: float,
) -> tuple[set[SectionIdentity], dict[SectionIdentity, str]]:
    identities: set[SectionIdentity] = set()
    rallies: dict[SectionIdentity, str] = {}
    for span in streams.spans:
        tolerance = scale_base30_frames(
            tolerance_at_30_fps,
            fps_by_fixture[span.fixture],
        )
        score = evaluate_span(
            span,
            labels.rallies.get(span.fixture, ()),
            labels.target_sides,
            tolerance,
            confidence_requirement,
        )
        if score.fully_correct:
            if score.rally_id is None:
                raise ValueError("fully correct section has no labelled rally")
            identity = (span.fixture, span.span_id)
            identities.add(identity)
            rallies[identity] = score.rally_id
    rally_counts = Counter(
        (fixture, rally_id) for (fixture, _span), rally_id in rallies.items()
    )
    if any(count > 1 for count in rally_counts.values()):
        raise ValueError("one labelled rally is fully correct in more than one section")
    return identities, rallies


def _contact_predictions(streams: ContactStreams) -> dict[str, np.ndarray]:
    return {
        fixture: np.asarray([event.frame for event in events], dtype=np.int32)
        for fixture, events in streams.events_by_fixture.items()
    }


def score_candidate_choice(
    baseline: ContactStreams,
    alternative: ContactStreams,
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    targets: TargetAssignments,
    selections: Mapping[SectionIdentity, CandidateRow],
) -> dict[str, object]:
    """Score strict rally changes, selected actions and contact timing."""
    fully_correct: dict[str, dict[str, object]] = {}
    for confidence in (0.0, 0.9):
        baseline_ids, baseline_rallies = _fully_correct(
            baseline,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=10,
            confidence_requirement=confidence,
        )
        alternative_ids, alternative_rallies = _fully_correct(
            alternative,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=10,
            confidence_requirement=confidence,
        )
        new_ids = alternative_ids - baseline_ids
        lost_ids = baseline_ids - alternative_ids
        fully_correct[f"{confidence:.1f}"] = {
            "baseline_count": len(baseline_ids),
            "alternative_count": len(alternative_ids),
            "new_identities": sorted(new_ids),
            "lost_identities": sorted(lost_ids),
            "new_rally_ids": sorted(
                alternative_rallies[identity] for identity in new_ids
            ),
            "lost_rally_ids": sorted(
                baseline_rallies[identity] for identity in lost_ids
            ),
            "lost_by_fixture": dict(
                sorted(Counter(fixture for fixture, _span in lost_ids).items())
            ),
        }

    selected_rows = list(selections.values())
    correct_selections = [
        row for row in selected_rows if targets.by_candidate[row.identity].positive
    ]
    recoverable_sections = targets.recoverable_sections
    recovered_sections = {row.section_identity for row in correct_selections}
    if not recovered_sections <= recoverable_sections:
        raise ValueError("recovered candidate sections differ")
    selected_count = len(selected_rows)
    correct_count = len(correct_selections)
    recoverable_count = len(recoverable_sections)
    recovered_count = len(recovered_sections)

    contact_timing: dict[str, dict[str, object]] = {}
    contact_labels = labels.contact_labels
    video_specs = [
        _TimingVideo(fixture, fps) for fixture, fps in fps_by_fixture.items()
    ]
    for tolerance in (5, 10):
        baseline_counts = contact_counts(
            contact_labels,
            _contact_predictions(baseline),
            video_specs,
            tolerance,
        )
        alternative_counts = contact_counts(
            contact_labels,
            _contact_predictions(alternative),
            video_specs,
            tolerance,
        )
        contact_timing[str(tolerance)] = {
            "baseline": baseline_counts,
            "alternative": alternative_counts,
            "f1_change": float(alternative_counts["f1"]) - float(baseline_counts["f1"]),
        }

    timing_only_selected = sum(
        targets.by_candidate[row.identity].timing_match
        and not targets.by_candidate[row.identity].side_match
        for row in selected_rows
    )
    return {
        "fully_correct_at_10_frames": fully_correct,
        "selected_actions": selected_count,
        "newly_added_contacts": sum(not row.kept for row in selected_rows),
        "correct_additions": correct_count,
        "correct_addition_rate": correct_count / selected_count
        if selected_count
        else 0.0,
        "recoverable_sections": recoverable_count,
        "recovered_sections": recovered_count,
        "recovery_rate": recovered_count / recoverable_count
        if recoverable_count
        else 0.0,
        "timing_only_selected_actions": timing_only_selected,
        "selected_candidate_identities": [row.identity for row in selected_rows],
        "contact_timing": contact_timing,
    }


def passes_result_gate(
    metrics: Mapping[str, Any],
    gate: ResultGate,
    *,
    validation: bool,
) -> bool:
    """Apply the fixed training or validation result checks."""
    fully_correct = _mapping(metrics["fully_correct_at_10_frames"], "fully correct")
    at_zero = _mapping(fully_correct["0.0"], "fully correct at zero")
    at_ninety = _mapping(fully_correct["0.9"], "fully correct at 0.9")
    new_count = len(at_zero["new_identities"])
    lost_at_zero = len(at_zero["lost_identities"])
    lost_at_ninety = len(at_ninety["lost_identities"])
    passed = (
        new_count >= gate.minimum_new_fully_correct_sections
        and lost_at_zero <= gate.maximum_lost_fully_correct_sections
        and lost_at_ninety <= gate.maximum_lost_fully_correct_sections
        and float(metrics["correct_addition_rate"])
        >= gate.minimum_correct_addition_rate
        and float(metrics["recovery_rate"]) >= gate.minimum_recovery_rate
    )
    if validation:
        if gate.minimum_contact_f1_change_at_10_frames is None:
            raise ValueError("validation contact F1 gate is missing")
        timing = _mapping(metrics["contact_timing"], "contact timing")
        at_ten = _mapping(timing["10"], "contact timing at ten frames")
        passed = (
            passed
            and float(at_ten["f1_change"])
            >= gate.minimum_contact_f1_change_at_10_frames
        )
        if gate.allow_per_video_fully_correct_loss is False:
            passed = passed and not bool(at_zero["lost_by_fixture"])
    return passed


def model_choice_key(
    model_id: str,
    metrics: Mapping[str, Any],
) -> tuple[int, int, float, int]:
    """Return the exact tie order for passing training choices."""
    fully_correct = _mapping(metrics["fully_correct_at_10_frames"], "fully correct")
    at_zero = _mapping(fully_correct["0.0"], "fully correct at zero")
    return (
        int(at_zero["alternative_count"]),
        -int(metrics["newly_added_contacts"]),
        float(metrics["correct_addition_rate"]),
        int(model_id == "logistic_regression"),
    )
