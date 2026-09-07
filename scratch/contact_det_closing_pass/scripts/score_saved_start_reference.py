"""Score the historical frozen first-contact chooser with corrected matching."""

from __future__ import annotations

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _evaluate_streams,
    _harm_metrics,
    _subset_development,
)
from scratch.contact_det_closing_pass.scripts.targets import assign_targets
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as old
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    build_candidate_rows,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)


def main() -> None:
    pack = prediction_io.load_development_predictions()
    videos, spans, events = _subset_development(pack, frozenset({"V"}))
    raw = prediction_io.read_json(prediction_io.VALIDATION_PREDICTIONS)["videos"]
    rows = old.build_action_rows(build_candidate_rows(raw, default_group="V"))
    source = prediction_io.REPO_ROOT / "scratch/contact_det_followup/results/start_model_validation_predictions.json"
    saved = prediction_io.read_json(source)
    scores = {}
    for row in saved["scores"]:
        identity = (row["fixture"], row["span_id"], row["frame"], row["action"])
        scores[identity] = float(row["score"])
    selections = old.select_actions(rows, scores, float(saved["cutoff"]))
    selected_ids = {row.identity for row in selections.values()}
    if selected_ids != {tuple(identity) for identity in saved["selected_action_identities"]}:
        raise ValueError("Saved historical chooser selections differ from its scores")
    baseline = old.apply_selected_actions(spans, events, {})
    edited = old.apply_selected_actions(spans, events, selections)
    labels = load_human_labels(old.LABEL_PATH, videos)
    fps = {video.fixture: video.fps for video in videos}
    groups = {video.fixture: "V" for video in videos}
    evaluation = _evaluate_streams(baseline, edited, labels, fps, groups)
    harm = {}
    for tolerance in (10, 5):
        targets, _ = assign_targets(rows, spans, events, labels, fps, tolerance)
        harm[str(tolerance)] = _harm_metrics(
            selections, targets, "opening", evaluation[str(tolerance)]["baseline_fixed_side"]["sections"],
        )
        print(tolerance, evaluation[str(tolerance)]["paired_fixed_side"], flush=True)
    write_json(
        prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass/results/historical_start_reference.json.gz",
        {"schema": "contact-closing-historical-start/1", "model_id": saved["model_id"],
         "cutoff": saved["cutoff"], "source": str(source.relative_to(prediction_io.REPO_ROOT)),
         "model_refit": False, "selected_actions": sorted(selected_ids), "evaluation": evaluation, "harm": harm},
    )


if __name__ == "__main__":
    main()
