"""Measure acceptance on the recommended local-insertion detector with guarded edges."""

from __future__ import annotations

import argparse
import lzma
from time import perf_counter

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.chosen_acceptance import (
    build_nested_local_scores,
    guarded_group_block,
)
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_gap_acceptance import fit_acceptor
from scratch.contact_det_closing_pass.scripts.run_later_acceptance import (
    _outcome_value,
    _tail_curve,
)
from scratch.contact_det_closing_pass.scripts.run_serve_followups import OUTPUT, ROOT
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.serve_metrics import accepted_serves
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import GROUPS
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

RAW = ROOT / "raw/serve_followups"


def run(jobs: int = 4, prepare_only: bool = False) -> None:
    started = perf_counter()
    prepared = joblib.load(ROOT / "raw/later_run/prepared.joblib")
    population, options = prepared["base_population"], prepared["options"]
    nested = build_nested_local_scores(prepared, RAW, jobs)
    if prepare_only:
        print("Nested chosen-detector fits ready", perf_counter() - started, flush=True)
        return
    local = nested["local"]
    previous = prediction_io.read_json(ROOT / "results/later/later_margin_predictions.json.gz")
    reference = restore_choices(options, previous["selected_actions"])
    saved = prediction_io.read_json(RAW / "reference_development_predictions/local_predictions.json.gz")
    if "options" in saved:
        global_scores = np.asarray([row["score"] for row in saved.pop("options")], dtype=np.float64)
    else:
        with lzma.open(ROOT / "raw/followups" / saved["option_scores_file"], "rb") as handle:
            global_scores = np.load(handle, allow_pickle=False)
    saved_current = restore_choices(options, saved["selected_actions"])
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    serve_result = prediction_io.read_json(OUTPUT / "development_serves.json.gz")["variants"]["recommended"]
    blocks = {}

    def block(group: str, allowed: frozenset[str], outer: bool) -> dict:
        key = (group, allowed)
        if key in blocks:
            return blocks[key]
        path = RAW / f"acceptance_block_{group}_{''.join(sorted(allowed))}.joblib"
        if path.exists():
            result = joblib.load(path)
        else:
            scores = global_scores if outer else nested["pair_scores"][allowed]
            previous_reference = reference if outer else nested["pair_references"][allowed]
            result = guarded_group_block(
                group, options, scores, previous_reference, population, prepared["measurements"],
                prepared["later_candidates"], labels, local["models"][allowed],
            )
            if outer and any(
                choice.span != saved_current[identity].span
                for identity, choice in result["raw_selected"].items()
            ):
                raise ValueError("Acceptance changed a saved recommended detector choice")
            result["training_groups"] = sorted(allowed)
            result["scored_group"] = group
            result["outputs"] = stream_records(result["stream"])
            # Feature matrices, actual streams and labels suffice to reproduce acceptance.
            result.pop("raw_selected")
            result.pop("guarded_selected")
            result.pop("stream")
            joblib.dump(result, path, compress=3)
        blocks[key] = result
        return result

    output_rows, matrices, records = [], [], []
    for held in GROUPS:
        allowed = frozenset(GROUPS) - {held}
        training = [block(group, allowed - {group}, False) for group in GROUPS if group != held]
        answers = np.asarray([_outcome_value(row, "10") for item in training for row in item["rows"]], dtype=np.int8)
        fit_started = perf_counter()
        models = {
            variant: fit_acceptor(np.vstack([item[f"{variant}_features"] for item in training]), answers)
            for variant in ("base", "gap")
        }
        current = block(held, allowed, True)
        values = {variant: _positive_scores(model, current[f"{variant}_features"]) for variant, model in models.items()}
        for index, row in enumerate(current["rows"]):
            output_rows.append({**row, "group": held, "base_score": float(values["base"][index]),
                                "gap_score": float(values["gap"][index])})
        matrices.append(current["gap_features"])
        records.append({
            "held_group": held,
            "training_groups": sorted(allowed),
            "fit_seconds": perf_counter() - fit_started,
        })
        print("Chosen acceptance", held, "complete", flush=True)
    curves = {variant: _tail_curve(output_rows, f"{variant}_score") for variant in ("base", "gap")}
    policies = {}
    for variant, curve in curves.items():
        policies[variant] = {"comparison": next(row for row in curve if row["tail"] == "top20pct")}
        for target in (0.95, 0.99):
            candidates = [
                row for row in curve
                if (row["by_tolerance"]["10"]["verified_correct_share_allaccepted"] or 0) >= target
            ]
            policies[variant][str(target)] = max(candidates, key=lambda row: row["accepted_count"], default=None)
    matrix = np.vstack(matrices)
    width = len(current["base_feature_names"])
    answers = np.asarray([_outcome_value(row, "10") for row in output_rows], dtype=np.int8)
    final = {"base": fit_acceptor(matrix[:, :width], answers), "gap": fit_acceptor(matrix, answers)}
    frozen = {
        "models": final, "base_feature_names": current["base_feature_names"],
        "gap_feature_names": current["gap_feature_names"], "local": local["final"], "policies": policies,
        "detector": "local_insertion_with_guarded_edges", "target_tolerance_base30": 10,
    }
    joblib.dump(frozen, RAW / "chosen_acceptance_models.joblib", compress=3)
    accepted = {}
    for variant, policy_set in policies.items():
        accepted[variant] = {}
        for name, policy in policy_set.items():
            if policy is None:
                continue
            identities = {
                (row["fixture"], row["span_id"])
                for row in output_rows
                if row[f"{variant}_score"] >= policy["threshold"]
            }
            accepted[variant][name] = {
                tolerance: accepted_serves(result, identities)
                for tolerance, result in serve_result.items()
            }
    write_json(OUTPUT / "chosen_acceptance_development.json.gz", {
        "schema": "contact-chosen-acceptance/1", "status": "complete", "detector": frozen["detector"],
        "rows": output_rows, "curves": curves, "policies": policies, "accepted_serves": accepted,
        "feature_names": current["gap_feature_names"], "fits": records,
        "upstream_fit_records": nested["pair_fit_records"], "seconds": perf_counter() - started,
        "new_fits_exclude_scored_group": True, "old_cached_detector_scores_retain_cross_group_dependence": True,
        "policy_choice": (
            "Keep the existing top-20% comparison; also freeze largest reported tail "
            "reaching 95% or 99% verified-correct share at ±10, without a minimum-count gate"
        ),
    })
    for variant, curve in curves.items():
        print(variant, [(row["tail"], row["by_tolerance"]["10"]["counts"]) for row in curve], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.prepare_only)
