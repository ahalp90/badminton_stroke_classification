"""Refit the existing local chooser using answers scored after fixed padding."""

from __future__ import annotations

import argparse
import lzma
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.evaluation import (
    paired_sections,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.later_options import (
    MIN_EDIT_ADVANTAGE,
    apply_options,
    option_record,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.run_boundary_broader import _score_stream
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_insertion_followup import (
    ROOT,
    fit_fold,
    prepare,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    section_views,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    GROUPS,
    WHOLE_MODEL_SETTINGS,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)


def compare_finished(
    before: ContactStreams, after: ContactStreams, labels: HumanLabels,
    fps: dict[str, float], groups: dict[str, str],
) -> dict[str, Any]:
    result = {}
    for tolerance in (10, 5):
        old = section_views(before.spans, labels, fps, groups, tolerance)["fixed_side"]["sections"]
        new = section_views(after.spans, labels, fps, groups, tolerance)["fixed_side"]["sections"]
        by_group = {}
        for group in GROUPS:
            by_group[group] = paired_sections([row for row in old if row["group"] == group],
                                              [row for row in new if row["group"] == group])
        by_video = {}
        for fixture in fps:
            by_video[fixture] = paired_sections([row for row in old if row["fixture"] == fixture],
                                                [row for row in new if row["fixture"] == fixture])
        result[str(tolerance)] = {"paired": paired_sections(old, new), "by_group": by_group,
                                  "by_video": by_video, "before_rows": old, "after_rows": new}
    return result


def run(census: Path, output_root: Path, jobs: int) -> None:
    started = perf_counter()
    census_report = prediction_io.read_json(census / "census.json.gz")
    if census_report["status"] != "complete":
        raise ValueError("A complete target census is required before fitting")
    expected_reference = "scratch/contact_det_closing_pass/results/later/later_margin_predictions.json.gz"
    assert census_report["input_identifiers"]["reference"] == expected_reference
    output_root.mkdir(parents=True, exist_ok=False)
    data = prepare("local")
    population, options = data["population"], data["options"]
    with lzma.open(census / "targets.npy.xz", "rb") as handle:
        targets = np.load(handle)
    assert targets.shape == data["targets"].shape
    assert np.array_equal(targets == -1, data["targets"] == -1)
    expected = data["targets"].copy()
    for change in census_report["changed_rows"]:
        index = change["index"]
        assert option_record(options[index]) == change["option_record"]
        assert expected[index] == change["old_target"]
        expected[index] = change["new_target"]
    assert np.array_equal(targets, expected), "Census target changes do not match the prepared alternatives"
    data["targets"] = targets
    reference_payload = prediction_io.read_json(ROOT / "results/later/later_margin_predictions.json.gz")
    current_payload = prediction_io.read_json(ROOT / "raw/followups/development_predictions/local_predictions.json.gz")
    reference = restore_choices(options, reference_payload["selected_actions"])
    current = restore_choices(options, current_payload["selected_actions"])
    with joblib.parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=6):
        fitted = joblib.Parallel()(joblib.delayed(fit_fold)(data, group) for group in GROUPS)
    scores = np.full(len(options), np.nan)
    for indices, values, _model, _record in fitted:
        scores[indices] = values
    assert np.isfinite(scores).all()
    selected = select_with_reference(options, scores, reference, MIN_EDIT_ADVANTAGE)
    current_stream = apply_options(population.spans, population.events, current)
    selected_stream = apply_options(population.spans, population.events, selected)
    before = pad_contact_boundaries(current_stream.spans, current_stream.events_by_fixture,
                                    population.fps, preserve_membership=True)
    after = pad_contact_boundaries(selected_stream.spans, selected_stream.events_by_fixture,
                                   population.fps, preserve_membership=True)
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    comparison = compare_finished(before, after, labels, population.fps, population.groups)
    for tolerance, row in comparison.items():
        paired = row["paired"]
        print(tolerance, paired["correct_before"], paired["correct_after"],
              "repairs", len(paired["repaired"]), "losses", len(paired["lost"]), flush=True)
    with lzma.open(output_root / "option_scores.npy.xz", "wb", format=lzma.FORMAT_XZ, preset=9) as handle:
        np.save(handle, scores)
    write_json(output_root / "development_predictions.json.gz", {
        "schema": "contact-padded-target-predictions/1", "status": "complete",
        "prediction_selection_uses_labels": False,
        "upstream_detector_scores_retain_cross_group_dependence": True,
        "selected_actions": [option_record(option) for option in selected.values()],
        "unpadded_output": stream_records(selected_stream), "output": stream_records(after),
    })
    result = {
        "schema": "contact-padded-target-fit/1", "status": "development_complete",
        "target_tolerance_base30": 10, "boundary_mode": "fixed_membership",
        "minimum_edit_advantage": MIN_EDIT_ADVANTAGE, "model_settings": WHOLE_MODEL_SETTINGS,
        "features_and_opening_local_models_unchanged": True,
        "comparison": comparison,
        "contacts_before": _score_stream(before, labels, population.fps),
        "contacts_after": _score_stream(after, labels, population.fps),
        "fits": [entry[3] for entry in fitted], "seconds": perf_counter() - started,
    }
    write_json(output_root / "development_result.json.gz", result)
    _indices, _scores, final, final_record = fit_fold(data, None)
    frozen = joblib.load(ROOT / "raw/later_run/models.joblib")
    models = {**frozen, "whole": final, "variant": "local", "local": data["local"]["final"],
              "local_feature_names": data["local"]["feature_names"]}
    joblib.dump(models, output_root / "models.joblib", compress=3)
    result.update(status="complete", final_fit=final_record, seconds=perf_counter() - started)
    write_json(output_root / "development_result.json.gz", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    run(args.census, args.output_root, args.jobs)


if __name__ == "__main__":
    main()
