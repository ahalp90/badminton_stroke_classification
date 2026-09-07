"""Replay the current chooser and count real one- and two-insertion opportunity."""

from collections import Counter
from time import perf_counter

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.evaluation import (
    overlapping_rallies,
    section_result,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.later_options import (
    apply_options,
    build_later_options,
    option_record,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    ROOT,
    _reference_selection,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    scale_base30_frames,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

OUTPUT = ROOT / "results/followups"
RAW = ROOT / "raw/followups"


def run() -> None:
    started = perf_counter()
    prepared = joblib.load(ROOT / "raw/later_run/prepared.joblib")
    population = prepared["base_population"]
    singles = prepared["options"]
    saved = prediction_io.read_json(ROOT / "results/later/later_margin_predictions.json.gz")
    reference = restore_choices(singles, saved["selected_actions"])
    if stream_records(apply_options(population.spans, population.events, reference)) != saved["outputs"]:
        raise ValueError("Session-start full output differs from the saved contact stream")
    scores = prediction_io.read_json(ROOT / "results/later/later_predictions.json.gz")
    replay = select_with_reference(
        singles, np.asarray([row["score"] for row in scores["options"]]),
        _reference_selection(population.options, frozenset(population.fps)),
    )
    if any(replay[identity].span != option.span for identity, option in reference.items()):
        raise ValueError("Session-start chooser replay differs from saved selections")
    RAW.mkdir(parents=True, exist_ok=True)
    if (RAW / "pair_options.joblib").exists():
        pairs = joblib.load(RAW / "pair_options.joblib")
    else:
        expanded = build_later_options(population.options, prepared["later_candidates"], population.fps, max_insertions=2)
        pairs = tuple(option for option in expanded if option.second_inserted is not None)
        joblib.dump(pairs, RAW / "pair_options.joblib", compress=3)
        write_json(OUTPUT / "pair_candidates.json.gz", {
            "labels_read": False, "options": [option_record(option) for option in pairs],
        })
    print(f"Replayed {len(reference)} saved sections; {len(singles)} old and {len(pairs)} pair alternatives", flush=True)
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    grouped = {}
    for option in (*singles, *pairs):
        grouped.setdefault(option.base.identity, []).append(option)
    results = {}
    for tolerance in (10, 5):
        rows = []
        totals = Counter()
        groups = {}
        for identity, options in grouped.items():
            current = reference[identity]
            scaled = scale_base30_frames(tolerance, population.fps[identity[0]])
            before = section_result(current.span, labels, scaled)
            ceilings = {"any_base": [False] * 3, "same_base": [False] * 3}
            overlap_cache = {}
            correctness_cache = {}
            for option in options:
                edges = (option.span.start_frame, option.span.end_frame)
                if edges not in overlap_cache:
                    overlap_cache[edges] = overlapping_rallies(option.span, labels)
                rallies = overlap_cache[edges]
                if len(rallies) != 1 or len(option.span.events) != len(rallies[0].frames):
                    continue
                if option.span not in correctness_cache:
                    correctness_cache[option.span] = section_result(option.span, labels, scaled)["side_rule_fully_correct"]
                if correctness_cache[option.span]:
                    count = len(option.inserted_events)
                    ceilings["any_base"][count] = True
                    if option.base == current.base:
                        ceilings["same_base"][count] = True
            row = {"fixture": identity[0], "span_id": identity[1], "reference_correct": before["side_rule_fully_correct"]}
            for context, possible in ceilings.items():
                row[f"{context}_zero"] = possible[0]
                row[f"{context}_one"] = any(possible[:2])
                row[f"{context}_two"] = any(possible)
                row[f"{context}_pair_only"] = possible[2] and not any(possible[:2])
            rows.append(row)
            counts = {name: int(value) for name, value in row.items() if isinstance(value, bool)}
            totals.update(counts)
            groups.setdefault(population.groups[identity[0]], Counter()).update(counts)
        results[str(tolerance)] = {"counts": dict(totals), "by_group": groups, "sections": rows}
        print(tolerance, dict(totals), flush=True)
    write_json(OUTPUT / "initial_diagnostics.json.gz", {
        "schema": "contact-followup-diagnostics/1", "status": "complete", "reference_replay": "passed",
        "sections": len(reference), "single_options": len(singles), "pair_options": len(pairs),
        "opportunity": results, "seconds": perf_counter() - started,
    })


if __name__ == "__main__":
    run()
