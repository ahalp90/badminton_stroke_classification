"""Compare gap evidence on identical saved detector outputs with nested fits."""

from time import perf_counter
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.gap_evidence import gap_evidence
from scratch.contact_det_closing_pass.scripts.run_later_acceptance import (
    ACCEPTANCE_SETTINGS,
    _group_features,
    _group_indices,
    _outcome_value,
    _tail_curve,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    ROOT,
    _reference_selection,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import GROUPS
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

RAW = ROOT / "raw/followups"
RESULTS = ROOT / "results/followups"


def fit_acceptor(features: np.ndarray, targets: np.ndarray) -> Any:
    known = targets >= 0
    if set(targets[known].tolist()) != {0, 1}:
        raise ValueError("Acceptance training requires both known outcomes")
    return HistGradientBoostingClassifier(**ACCEPTANCE_SETTINGS).fit(features[known], targets[known])


def run() -> None:
    started = perf_counter()
    prepared = joblib.load(ROOT / "raw/later_run/prepared.joblib")
    population, options = prepared["base_population"], prepared["options"]
    local = joblib.load(RAW / "local_local_cache.joblib")
    pairs = joblib.load(ROOT / "raw/later_acceptance/nested_pair_fits.joblib")
    saved = prediction_io.read_json(ROOT / "results/later/later_predictions.json.gz")
    global_scores = np.asarray([row["score"] for row in saved["options"]])
    current = prediction_io.read_json(ROOT / "results/later/later_margin_predictions.json.gz")
    current_selection = restore_choices(options, current["selected_actions"])
    global_reference = _reference_selection(population.options, frozenset(population.fps))
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    by_group = _group_indices(options, population)

    def block(group: str, allowed: frozenset[str], outer: bool) -> tuple:
        key = "".join(group for group in GROUPS if group in allowed)
        scores = global_scores if outer else pairs["pair_scores"][key]
        reference = global_reference if outer else pairs["pair_references"][key]
        features, names, rows, selected = _group_features(
            group, by_group[group], options, scores, prepared["later_candidates"],
            population, prepared["measurements"], labels, reference,
        )
        if outer and any(option.span != current_selection[identity].span for identity, option in selected.items()):
            raise ValueError("Acceptance comparison changed a saved detector output")
        gap, gap_names, identities = gap_evidence(
            selected, prepared["later_candidates"], population.fps,
            prepared["measurements"], local["models"][allowed],
        )
        indices = {identity: index for index, identity in enumerate(identities)}
        ordered_gap = gap[[indices[(row["fixture"], row["span_id"])] for row in rows]]
        return features, np.column_stack((features, ordered_gap)), names, gap_names, rows

    output_rows = []
    output_features = []
    records = []
    for held in GROUPS:
        allowed = frozenset(GROUPS) - {held}
        train_base, train_gap, train_targets = [], [], []
        for group in GROUPS:
            if group == held:
                continue
            base, gap, _names, _gap_names, rows = block(group, allowed - {group}, False)
            train_base.append(base)
            train_gap.append(gap)
            train_targets.extend(_outcome_value(row, "10") for row in rows)
        answers = np.asarray(train_targets, dtype=np.int8)
        fit_started = perf_counter()
        models = {"control": fit_acceptor(np.vstack(train_base), answers),
                  "gap": fit_acceptor(np.vstack(train_gap), answers)}
        fit_seconds = perf_counter() - fit_started
        base, gap, names, gap_names, rows = block(held, allowed, True)
        control_scores = _positive_scores(models["control"], base)
        gap_scores = _positive_scores(models["gap"], gap)
        output_features.append(gap)
        for index, row in enumerate(rows):
            output_rows.append({**row, "group": held, "control_score": float(control_scores[index]),
                                "gap_score": float(gap_scores[index])})
        records.append({"held_out": held, "training_groups": sorted(allowed), "fit_seconds": fit_seconds})
        print("Gap acceptance group", held, "complete", flush=True)
    curves = {key: _tail_curve(output_rows, f"{key}_score") for key in ("control", "gap")}
    policies = {key: next(row for row in curve if row["tail"] == "top20pct") for key, curve in curves.items()}
    matrix = np.vstack(output_features)
    answers = np.asarray([_outcome_value(row, "10") for row in output_rows], dtype=np.int8)
    width = len(names)
    final = {"control": fit_acceptor(matrix[:, :width], answers), "gap": fit_acceptor(matrix, answers)}
    joblib.dump({"models": final, "base_feature_names": names, "gap_feature_names": gap_names,
                 "local": local["final"], "policies": policies, "detector": "session_start"},
                RAW / "gap_acceptance_models.joblib", compress=3)
    write_json(RESULTS / "gap_acceptance_result.json.gz", {
        "schema": "contact-gap-acceptance/1", "status": "complete", "detector": "session_start",
        "identical_detector_outputs": True, "held_groups_excluded_from_all_new_fits": True,
        "upstream_detector_scores_retain_cross_group_dependence": True,
        "rows": output_rows, "curves": curves, "policies": policies,
        "policy_choice": "20 percent development coverage fixed before this comparison",
        "feature_names": [*names, *gap_names], "fits": records, "seconds": perf_counter() - started,
    })
    for key, policy in policies.items():
        print(key, policy["by_tolerance"], flush=True)


if __name__ == "__main__":
    run()
