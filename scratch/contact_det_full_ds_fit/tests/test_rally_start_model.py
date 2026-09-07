from __future__ import annotations

from pathlib import Path

import pytest

from scratch.contact_det.scripts.score_contact_rallies import RallyReference
from scratch.contact_det_full_ds_fit.scripts import rally_start_model as model
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    ResultGate,
    load_rally_start_model_config,
)

CONFIG_PATH = Path(__file__).parents[1] / "records/rally_start_model_runs.json"


def test_wider_candidate_pool_requires_explicit_opt_in() -> None:
    video = _saved_video()
    original = model.build_candidate_rows([video], default_group="V")
    candidates = video["candidate_lists"][0]["candidates"]
    candidates.extend({
        "frame": frame, "contact_score": 0.5, "is_fixed_contact": False,
        "kept": False, "predicted_side": "Top",
    } for frame in (100, 110))
    with pytest.raises(ValueError, match="candidate-list size"):
        model.build_candidate_rows([video], default_group="V")
    expanded = model.build_candidate_rows([video], default_group="V", max_earlier_candidates=4)
    assert len(expanded) == 4
    assert expanded[:2] == original


def _saved_video(fixture: str = "sset_01", group: str = "A") -> dict[str, object]:
    return {
        "group": group,
        "video": {
            "fixture": fixture,
            "video_id": int(fixture.removeprefix("sset_")),
            "fps": 30.0,
            "frame_count": 220,
        },
        "spans": [{"span_id": 0, "start_frame": 140, "end_frame": 200}],
        "kept_contacts": [
            {
                "frame": 150,
                "interval_id": 0,
                "contact_score": 0.95,
                "span_id": 0,
                "predicted_side": "Bot",
            },
            {
                "frame": 160,
                "interval_id": 0,
                "contact_score": 0.96,
                "span_id": 0,
                "predicted_side": "Top",
            },
        ],
        "candidate_lists": [
            {
                "fixture": fixture,
                "span_id": 0,
                "section_start_frame": 140,
                "section_end_frame": 200,
                "interval_id": 0,
                "prefix_start_frame": 100,
                "fixed_contact_frame": 150,
                "duplicate_distance_frames": 6,
                "candidates": [
                    {
                        "frame": 150,
                        "contact_score": 0.95,
                        "is_fixed_contact": True,
                        "kept": True,
                        "predicted_side": "Bot",
                    },
                    {
                        "frame": 120,
                        "contact_score": 0.60,
                        "is_fixed_contact": False,
                        "kept": False,
                        "predicted_side": "Top",
                    },
                    {
                        "frame": 130,
                        "contact_score": 0.70,
                        "is_fixed_contact": False,
                        "kept": False,
                        "predicted_side": "Bot",
                    },
                ],
            }
        ],
    }


def _labels(fixtures: tuple[str, ...] = ("sset_01",)) -> model.HumanLabels:
    rallies = {
        fixture: (RallyReference(fixture, 0, f"{fixture}:rally_1", (120, 150, 160)),)
        for fixture in fixtures
    }
    sides = {
        (fixture, frame): side
        for fixture in fixtures
        for frame, side in ((120, "Top"), (150, "Bot"), (160, "Top"))
    }
    return model.HumanLabels(rallies, sides)


def test_candidate_rows_use_only_the_nine_fixed_inputs() -> None:
    rows = model.build_candidate_rows([_saved_video()], default_group="V")

    assert len(rows) == 2
    assert rows[0].identity == ("sset_01", 0, 120)
    assert rows[0].features == (
        0.60,
        0.95,
        30.0,
        -20.0,
        60.0,
        0.0,
        1.0,
        1.0,
        0.0,
    )
    assert rows[1].features[-1] == 1.0


def test_candidate_search_window_may_start_inside_a_long_section() -> None:
    video = _saved_video()
    video["spans"][0]["start_frame"] = 100
    candidate_list = video["candidate_lists"][0]
    candidate_list["section_start_frame"] = 100
    candidate_list["prefix_start_frame"] = 110

    rows = model.build_candidate_rows([video], default_group="V")

    assert [row.frame for row in rows] == [120, 130]
    assert rows[0].features[3] == 20.0


def test_training_answer_requires_timing_and_player_side() -> None:
    video = _saved_video()
    rows = model.build_candidate_rows([video], default_group="V")
    targets = model.assign_candidate_targets(
        rows,
        [video],
        _labels(),
        default_group="V",
    )

    first = targets.by_candidate[("sset_01", 0, 120)]
    second = targets.by_candidate[("sset_01", 0, 130)]
    assert first.positive is True
    assert first.timing_match is True
    assert first.side_match is True
    assert second.positive is False
    assert second.timing_match is True
    assert second.side_match is False
    assert targets.recoverable_sections == {("sset_01", 0)}


def test_no_rally_and_already_matched_sections_have_no_positive() -> None:
    video = _saved_video()
    rows = model.build_candidate_rows([video], default_group="V")
    no_rally = model.HumanLabels({"sset_01": ()}, {})

    no_rally_targets = model.assign_candidate_targets(
        rows,
        [video],
        no_rally,
        default_group="V",
    )

    assert set(no_rally_targets.section_statuses.values()) == {"no_labelled_rally"}
    assert all(
        not target.positive
        for target in no_rally_targets.by_candidate.values()
    )

    matched_labels = model.HumanLabels(
        {
            "sset_01": (
                RallyReference("sset_01", 0, "set1:1", (150, 160)),
            )
        },
        {("sset_01", 150): "Bot", ("sset_01", 160): "Top"},
    )
    matched_targets = model.assign_candidate_targets(
        rows,
        [video],
        matched_labels,
        default_group="V",
    )

    assert set(matched_targets.section_statuses.values()) == {
        "first_contact_already_matched"
    }
    assert all(
        not target.positive
        for target in matched_targets.by_candidate.values()
    )


def test_equal_timing_error_uses_score_and_requires_a_side() -> None:
    video = _saved_video()
    candidates = video["candidate_lists"][0]["candidates"]
    candidates[1]["predicted_side"] = None
    candidates[2]["predicted_side"] = "Top"
    rows = model.build_candidate_rows([video], default_group="V")
    labels = model.HumanLabels(
        {
            "sset_01": (
                RallyReference("sset_01", 0, "set1:1", (125, 150, 160)),
            )
        },
        {
            ("sset_01", 125): "Top",
            ("sset_01", 150): "Bot",
            ("sset_01", 160): "Top",
        },
    )

    targets = model.assign_candidate_targets(
        rows,
        [video],
        labels,
        default_group="V",
    )

    missing_side = targets.by_candidate[("sset_01", 0, 120)]
    higher_score = targets.by_candidate[("sset_01", 0, 130)]
    assert missing_side.timing_match is True
    assert missing_side.side_match is False
    assert missing_side.positive is False
    assert higher_score.positive is True


def test_one_rally_touching_two_sections_is_left_out_of_training() -> None:
    video = _saved_video()
    video["spans"] = [
        {"span_id": 0, "start_frame": 140, "end_frame": 155},
        {"span_id": 1, "start_frame": 155, "end_frame": 200},
    ]
    video["candidate_lists"][0]["section_end_frame"] = 155
    rows = model.build_candidate_rows([video], default_group="V")
    targets = model.assign_candidate_targets(
        rows,
        [video],
        _labels(),
        default_group="V",
    )

    assert all(
        not target.included_in_training for target in targets.by_candidate.values()
    )
    assert set(targets.section_statuses.values()) == {
        "labelled_rally_touches_more_than_one_section"
    }


def test_four_group_models_score_every_held_out_candidate() -> None:
    groups = ("A", "B", "C", "D")
    fixtures = tuple(f"sset_{index:02d}" for index in range(1, 5))
    videos = [
        _saved_video(fixture, group)
        for fixture, group in zip(fixtures, groups, strict=True)
    ]
    rows = model.build_candidate_rows(videos, default_group="V")
    targets = model.assign_candidate_targets(
        rows,
        videos,
        _labels(fixtures),
        default_group="V",
    )
    config = load_rally_start_model_config(CONFIG_PATH)

    scores = model.held_out_candidate_scores(rows, targets, config)

    expected_identities = {row.identity for row in rows}
    assert set(scores) == {"logistic_regression", "shallow_hgb"}
    assert all(
        set(model_scores) == expected_identities for model_scores in scores.values()
    )


def test_four_group_models_never_fit_on_the_held_out_group(monkeypatch) -> None:
    groups = ("A", "B", "C", "D")
    fixtures = tuple(f"sset_{index:02d}" for index in range(1, 5))
    videos = [
        _saved_video(fixture, group)
        for fixture, group in zip(fixtures, groups, strict=True)
    ]
    rows = model.build_candidate_rows(videos, default_group="V")
    targets = model.assign_candidate_targets(
        rows,
        videos,
        _labels(fixtures),
        default_group="V",
    )
    config = load_rally_start_model_config(CONFIG_PATH)
    fitted_groups: list[frozenset[str]] = []

    def fake_fit(_spec, training_rows, _targets):
        fitted_groups.append(frozenset(row.group for row in training_rows))
        return object()

    monkeypatch.setattr(model, "_fit_model", fake_fit)
    monkeypatch.setattr(
        model,
        "predict_candidate_scores",
        lambda _fitted, held_out_rows: {
            row.identity: 0.5 for row in held_out_rows
        },
    )

    model.held_out_candidate_scores(rows, targets, config)

    assert fitted_groups == [
        frozenset(groups) - {held_out}
        for held_out in groups
        for _model_spec in config.models
    ]


def test_selected_candidate_moves_the_start_and_repairs_the_rally() -> None:
    video = _saved_video()
    rows = model.build_candidate_rows([video], default_group="V")
    targets = model.assign_candidate_targets(
        rows,
        [video],
        _labels(),
        default_group="V",
    )
    scores = {row.identity: 0.95 if row.frame == 120 else 0.10 for row in rows}
    selections = model.select_candidates(rows, scores, 0.9)

    baseline = model.apply_selected_candidates([video], {}, default_group="V")
    alternative = model.apply_selected_candidates(
        [video],
        selections,
        default_group="V",
    )
    metrics = model.score_candidate_choice(
        baseline,
        alternative,
        _labels(),
        {"sset_01": 30.0},
        targets,
        selections,
    )

    assert alternative.spans[0].start_frame == 120
    assert [event.frame for event in alternative.spans[0].events] == [120, 150, 160]
    assert metrics["correct_addition_rate"] == 1.0
    assert metrics["recovery_rate"] == 1.0
    assert len(metrics["fully_correct_at_10_frames"]["0.0"]["new_identities"]) == 1
    assert metrics["contact_timing"]["10"]["f1_change"] > 0.0
    assert model.passes_result_gate(
        metrics,
        ResultGate(1, 0, 0.8, 0.2, 0.0, False),
        validation=True,
    )
