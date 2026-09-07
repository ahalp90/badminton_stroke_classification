"""Choose score-only acceptance rules on the saved development predictions."""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.replay_simple_replacements import (
    replay_choices,
    replay_rows,
)
from scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison import (
    ROOT,
    prepare_population,
)
from scratch.contact_det_closing_pass.scripts.score_acceptance import (
    ACCEPTANCE_THRESHOLDS,
    build_acceptance_rows,
    select_acceptance_rules,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import GROUPS
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

VARIANT = "opening_sides_and_physics"


def run(output_root: Path) -> None:
    original = prediction_io.read_json(ROOT / "results/whole_rally_result.json.gz")
    policy = prediction_io.read_json(ROOT / "results/broader_action_policy.json.gz")
    reference = original["development_descriptive"][VARIANT]
    choices = replay_choices(original["development_options"], reference["scores"], reference["selected_actions"],
                             policy["cancel_simple_replace"])
    sections = {}
    for tolerance in ("10", "5"):
        sections[tolerance] = replay_rows(
            original["development_baseline"][tolerance]["fixed_side"]["sections"],
            reference["evaluation"][tolerance]["edited_fixed_side"]["sections"], choices,
        )
    pack = prediction_io.load_development_predictions()
    development = prepare_population(pack, frozenset(GROUPS), start._candidate_videos())
    labels = load_human_labels(start.LABEL_PATH, development.videos)
    rows = build_acceptance_rows(sections, choices, labels, {})
    ranked_scores = sorted((row["score"] for row in rows), reverse=True)
    tail_counts = (32, ceil(0.05 * len(rows)), ceil(0.10 * len(rows)))
    tail_thresholds = {str(count): ranked_scores[count - 1] for count in tail_counts}
    thresholds = sorted(set(ACCEPTANCE_THRESHOLDS) | set(tail_thresholds.values()))
    rules = select_acceptance_rules(rows, thresholds)
    record = {"schema": "contact-closing-acceptance-policy/1", "status": "frozen",
              "selection_data": "32 development videos, grouped chooser predictions",
              "validation_used_for_selection": False, "broader_labels_read": False,
              "action_policy": policy, "development_rank_tail_thresholds": tail_thresholds, **rules}
    write_json(output_root / "broader_acceptance_policy.json.gz", record)
    write_json(output_root / "broader_acceptance_development.json.gz", {
        "status": "complete", "rows": rows, "frozen_policy": record,
    })
    print("Development acceptance", rules["target_status"], flush=True)
    for rule in rules["curve"]:
        summary = rule.get("summary", rule)
        print(summary["threshold"], summary["accepted_count"], summary["by_tolerance"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    run(parser.parse_args().output_root)


if __name__ == "__main__":
    main()
