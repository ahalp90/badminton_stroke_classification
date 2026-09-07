"""An action cancellation preserves all other decisions and uses actual keep scores."""

from scratch.contact_det_closing_pass.scripts.replay_simple_replacements import (
    replay_choices,
    replay_rows,
)


def option(span_id: int, kind: str, frame: int | None = None) -> dict:
    return {"fixture": "video", "span_id": span_id, "kind": kind, "candidate_frame": frame,
            "deleted_frame": 200 if kind == "replace_delete" else None}


def test_cancel_simple_replace_uses_keep_score_without_reselection() -> None:
    keep = option(1, "keep")
    replace = option(1, "replace", 90)
    alternative = option(1, "add", 90)
    combined = option(2, "replace_delete", 290)
    result = replay_choices([keep, replace, alternative, combined], [0.2, 0.99, 0.95, 0.98],
                            [replace, combined], True)
    assert result[0] == {**keep, "score": 0.2, "cancelled_simple_replace": True}
    assert result[1] == {**combined, "score": 0.98, "cancelled_simple_replace": False}


def test_replay_restores_baseline_bounds_and_leaves_other_rows_intact() -> None:
    baseline = [{"fixture": "video", "span_id": 1, "start_frame": 100},
                {"fixture": "video", "span_id": 2, "start_frame": 300}]
    edited = [{"fixture": "video", "span_id": 1, "start_frame": 90},
              {"fixture": "video", "span_id": 2, "start_frame": 290}]
    choices = [{**option(1, "keep"), "cancelled_simple_replace": True},
               {**option(2, "replace_delete", 290), "cancelled_simple_replace": False}]
    assert replay_rows(baseline, edited, choices) == [baseline[0], edited[1]]
