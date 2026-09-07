"""Locate selected extra and missed events using the saved within-clip matches."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "contact_det_closing_pass/results"


def run() -> None:
    with gzip.open(BASELINE / "metric_summary.json.gz", "rt") as source:
        summary = json.load(source)
    with gzip.open(BASELINE / "followups/local_boundary_broader_predictions_fixed_membership.json.gz", "rt") as source:
        predictions = json.load(source)
    spans = {(int(video["fixture"]), span["span_id"]): span
             for video in predictions["videos"] for span in video["output"]["spans"]}
    contacts = pd.read_csv(ROOT / "results/contacts.csv.gz")
    primary = contacts[(contacts.population == "retained") & (contacts.tolerance_base30 == 10)]
    labels = {key: group.sort_values("label_index") for key, group in primary.groupby(["fixture", "rally_id"])}
    records = []
    for row in summary["selected_rows"]["retained"]["10"]:
        if row["outcome"] != "wrong":
            continue
        key = int(row["fixture"]), row["span_id"]
        frames = spans[key]["frames"]
        truth = labels[key[0], row["rally_id"]]
        first_frame, last_frame = truth.source_frame.min(), truth.source_frame.max()
        matched_predictions = {pred for _, pred, _ in row["matches"]}
        matched_labels = {label for label, _, _ in row["matches"]}
        common = {"fixture": key[0], "span_id": key[1], "rally_id": row["rally_id"]}
        for index, frame in enumerate(frames):
            if index in matched_predictions:
                continue
            position = "within labelled span"
            if frame < first_frame:
                position = "before first label"
            elif frame > last_frame:
                position = "after last label"
            records.append({**common, "kind": "extra", "frame": frame, "position": position})
        for contact in truth.itertuples(index=False):
            if contact.label_index not in matched_labels:
                records.append({**common, "kind": "missed", "frame": contact.source_frame,
                                "position": contact.position})
    table = pd.DataFrame(records)
    proposals = pd.read_csv(ROOT / "results/proposals.csv.gz")
    selected = proposals[(proposals.population == "retained") & (proposals.tolerance_base30 == 10)
                         & proposals.selected & (proposals.outcome == "wrong")]
    assert sum(table.kind == "extra") == selected.extra.sum()
    assert sum(table.kind == "missed") == selected.missing.sum()
    table.to_csv(ROOT / "results/selected_event_errors.csv.gz", index=False)
    print(table.groupby(["kind", "position"]).size().to_string())


if __name__ == "__main__":
    run()
