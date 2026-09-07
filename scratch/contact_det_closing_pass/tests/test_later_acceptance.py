"""Check the fixed acceptance coverage curves and policy fallback."""

from types import SimpleNamespace

from scratch.contact_det_closing_pass.scripts.run_later_acceptance import (
    _policy,
    _subset_reference,
    _tail_curve,
)


def _row(index: int, score: float, outcome10: str, outcome5: str | None = None) -> dict:
    outcome5 = outcome10 if outcome5 is None else outcome5
    return {
        "fixture": f"video-{index % 2}",
        "group": "ABCD"[index % 4],
        "raw_selected_score": score,
        "acceptance_selected_score": score,
        "acceptance_all_evidence_score": score,
        "judgements": {
            "10": {"outcome": outcome10},
            "5": {"outcome": outcome5},
        },
    }


def test_five_tails_include_ties_and_report_both_tolerances() -> None:
    rows = [_row(index, 1.0 - index / 100.0, "correct" if index % 2 == 0 else "wrong")
            for index in range(40)]
    rows[1]["raw_selected_score"] = rows[0]["raw_selected_score"]
    rows[1]["acceptance_selected_score"] = rows[0]["acceptance_selected_score"]
    rows[1]["acceptance_all_evidence_score"] = rows[0]["acceptance_all_evidence_score"]
    rows[2]["raw_selected_score"] = rows[0]["raw_selected_score"]
    rows[2]["acceptance_selected_score"] = rows[0]["acceptance_selected_score"]
    rows[2]["acceptance_all_evidence_score"] = rows[0]["acceptance_all_evidence_score"]
    rows[2]["judgements"]["10"] = {"outcome": "unjudgeable"}
    rows[3]["judgements"]["10"] = {"outcome": "correct"}

    curve = _tail_curve(rows, "raw_selected_score")

    assert [record["tail"] for record in curve] == [
        "top32", "top5pct", "top10pct", "top20pct", "top40pct",
    ]
    top5 = curve[1]
    assert top5["requested_count"] == 2
    assert top5["accepted_count"] == 3
    assert top5["by_tolerance"]["10"]["counts"] == {
        "correct": 1, "wrong": 1, "unjudgeable": 1,
    }
    assert top5["by_tolerance"]["10"]["verified_correct_share_allaccepted"] == 1 / 3
    assert top5["by_tolerance"]["10"]["rejected_correct"] == 19
    assert set(top5["by_tolerance"]) == {"10", "5"}
    assert set(top5["by_group"]) == {"A", "B", "C", "D"}
    assert set(top5["by_video"]) == {"video-0", "video-1"}


def test_95_and_99_rules_keep_nonempty_diagnostic_fallback_when_unmet() -> None:
    rows = [_row(index, 1.0 - index / 100.0, "correct" if index % 2 == 0 else "wrong")
            for index in range(40)]
    policy = _policy("raw_selected_score", _tail_curve(rows, "raw_selected_score"))

    assert policy["target_rules"]["0.95"]["target_status"] == "unmet"
    assert policy["target_rules"]["0.95"]["selected_rule"] is None
    assert policy["target_rules"]["0.95"]["nonempty_fallback"] is not None
    assert policy["target_rules"]["0.99"]["target_status"] == "unmet"
    assert policy["target_rules"]["0.99"]["selected_rule"] is None
    assert policy["target_rules"]["0.99"]["nonempty_fallback"] is not None
    assert policy["target_rules"]["0.99"]["nonempty_fallback"]["by_tolerance"]["10"]["judged_count"] >= 32


def test_reference_subset_is_scoped_to_the_current_training_group() -> None:
    option_a = SimpleNamespace(base=SimpleNamespace(identity=("video-a", 1)))
    option_b = SimpleNamespace(base=SimpleNamespace(identity=("video-b", 2)))
    reference = {option_a.base.identity: "reference-a", option_b.base.identity: "reference-b"}

    assert _subset_reference([option_a], reference) == {option_a.base.identity: "reference-a"}
