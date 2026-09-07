"""Replay the fixed score-advantage rule without refitting the expanded chooser."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.later_evaluation import compare_outputs
from scratch.contact_det_closing_pass.scripts.later_options import (
    MIN_EDIT_ADVANTAGE,
    apply_options,
    option_record,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESULT_ROOT,
    PREDICTION_NAME,
    PREPARED_NAME,
    _reference_selection,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)


def run(prepared_path: Path, result_root: Path) -> dict[str, Any]:
    started = perf_counter()
    prepared = joblib.load(prepared_path)
    population = prepared['base_population']
    options = prepared['options']
    raw = prediction_io.read_json(result_root / PREDICTION_NAME)
    scores = np.asarray([row['score'] for row in raw['options']], dtype=np.float64)
    reference = _reference_selection(population.options, frozenset(population.fps))
    selected = select_with_reference(options, scores, reference)
    stream = apply_options(population.spans, population.events, selected)
    choices = [option_record(option) for option in selected.values()]
    policy = {
        'minimum_edit_advantage': MIN_EDIT_ADVANTAGE,
        'reference': 'preserved opening_sides_and_physics combined chooser',
        'selection_data': 'development groups A-D only',
        'broader_labels_used': False,
        'scores_are_calibrated_probabilities': False,
    }
    write_json(result_root / 'later_margin_predictions.json.gz', {
        'schema': 'contact-closing-later-margin-predictions/1', 'status': 'complete',
        'policy': policy, 'option_scores_file': PREDICTION_NAME,
        'selected_actions': choices, 'outputs': stream_records(stream),
        'prediction_selection_uses_labels': False,
        'upstream_detector_scores_retain_cross_group_dependence': True,
    })
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    comparison = compare_outputs(
        tuple(option.span for option in reference.values()), selected, labels, population.fps, population.groups,
    )
    counts = Counter(option.base.kind for option in selected.values())
    result = {
        'schema': 'contact-closing-later-margin-comparison/1', 'status': 'complete',
        'policy': policy, 'sections': len(selected), 'action_counts': dict(counts),
        'selected_insertions': sum(option.inserted is not None for option in selected.values()),
        'comparison_to_frozen_combined': comparison,
        'total_seconds': perf_counter() - started,
    }
    write_json(result_root / 'later_margin_result.json.gz', result)
    write_json(result_root / 'later_detector_policy.json.gz', policy)
    for tolerance, data in comparison.items():
        pair = data['paired']
        print(tolerance, 'correct', pair['correct_before'], pair['correct_after'],
              'repairs', len(pair['repaired']), 'losses', len(pair['lost']), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=DEFAULT_OUTPUT_ROOT / PREPARED_NAME)
    parser.add_argument('--result-root', type=Path, default=DEFAULT_RESULT_ROOT)
    arguments = parser.parse_args()
    run(arguments.prepared, arguments.result_root)


if __name__ == '__main__':
    main()
