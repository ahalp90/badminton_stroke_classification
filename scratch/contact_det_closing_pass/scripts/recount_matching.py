"""Compare historical and corrected scoring of frozen contact predictions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_contacts,
    score_sections,
    test_labels,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_followup.scripts.prediction_io import (
    load_development_predictions,
    load_frozen_test_predictions,
)
from scratch.contact_det_followup.scripts.score_start_model import LABEL_PATH
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    _match_contacts,
)


def historical_matches(expected: list[int], predicted: list[int], tolerance: int) -> list[tuple[int, int, int]]:
    return _match_contacts(np.asarray(expected), np.asarray(predicted), tolerance)


def recount(dataset: str) -> dict:
    started = time.monotonic()
    if dataset == "shuttleset22":
        pack = load_frozen_test_predictions()
        labels = test_labels()
        fps = {str(video["fixture"]): float(video["fps"]) for video in pack.videos}
    else:
        pack = load_development_predictions()
        labels = load_human_labels(LABEL_PATH, pack.videos)
        fps = {video.fixture: video.fps for video in pack.videos}
    output = {"schema": "contact-closing-matching/1", "dataset": dataset, "tolerances": {}}
    for tolerance in (10, 5):
        results = {}
        for name, matcher in (("historical", historical_matches), ("corrected", match_contacts)):
            contacts = score_contacts(pack.events_by_fixture, labels, fps, tolerance, matcher)
            sections = score_sections(pack.spans, labels, fps, tolerance, matcher)
            results[name] = {
                "contacts": contacts, "sections": sections,
                "correct_sections": sum(row["fully_correct"] for row in sections),
                "correct_sections_with_side_rule": sum(row["side_rule_fully_correct"] for row in sections),
            }
            print(dataset, tolerance, name, contacts["total"],
                  results[name]["correct_sections_with_side_rule"], flush=True)
        changed = []
        for old, new in zip(results["historical"]["contacts"]["by_video"],
                            results["corrected"]["contacts"]["by_video"], strict=True):
            old_pairs = {tuple(row) for row in old["pairs"]}
            new_pairs = {tuple(row) for row in new["pairs"]}
            if old_pairs != new_pairs:
                changed.append({"fixture": old["fixture"], "removed": sorted(old_pairs - new_pairs),
                                "added": sorted(new_pairs - old_pairs)})
        changed_sections = []
        for old, new in zip(results["historical"]["sections"], results["corrected"]["sections"], strict=True):
            if sorted(old["matches"]) != sorted(new["matches"]) or old["fully_correct"] != new["fully_correct"]:
                changed_sections.append({"before": old, "after": new})
        results["changed_contact_pairings"] = changed
        results["changed_sections"] = changed_sections
        output["tolerances"][str(tolerance)] = results
    if dataset == "shuttleset22":
        for tolerance, expected in ((10, (32603, 524, 995)), (5, (32243, 483, 901))):
            old = output["tolerances"][str(tolerance)]["historical"]
            actual = (old["contacts"]["total"]["matched"], old["correct_sections"],
                      old["correct_sections_with_side_rule"])
            if actual != expected or len(old["sections"]) != 3982:
                raise AssertionError(f"Historical recount differs: {tolerance}: {actual} != {expected}")
    output["elapsed_seconds"] = time.monotonic() - started
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("shuttleset22", "development"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, recount(args.dataset))


if __name__ == "__main__":
    main()
