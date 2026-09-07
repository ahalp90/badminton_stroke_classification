"""Attach original game and score rows to the saved visual sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def run(sample: Path, annotations: Path, output: Path) -> None:
    requests = pd.read_csv(sample)
    matches = pd.read_csv(annotations / "set/match.csv").set_index("id")
    tables = {}
    records = []
    for row in requests.itertuples(index=False):
        set_id, rally = row.rally_id.split(":")
        key = (row.fixture, set_id)
        if key not in tables:
            name = matches.loc[row.fixture, "video"]
            tables[key] = pd.read_csv(annotations / "set" / name / f"{set_id}.csv")
        source = tables[key]
        group = source[source.rally == int(rally)].sort_values(["ball_round", "frame_num"])
        assert len(group) == row.labelled_contacts, (row.sample_id, len(group), row.labelled_contacts)
        label = group.iloc[row.label_index]
        assert int(label.frame_num) == row.source_frame, (row.sample_id, label.frame_num, row.source_frame)
        record = row._asdict()
        record.update(label_game=int(set_id.removeprefix("set")),
                      label_score_a=int(label.roundscore_A), label_score_b=int(label.roundscore_B),
                      label_player=label.player, label_ball_round=int(label.ball_round))
        records.append(record)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"Verified and copied {len(records)} exact source rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.sample, args.annotations, args.output)
