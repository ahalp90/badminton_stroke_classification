"""Choose reproducible source-frame checks for weak videos and missed labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260907
LOW_VIDEOS = (12, 20, 21, 24, 39, 17, 38)


def run() -> None:
    contacts = pd.read_csv(ROOT / "results/contacts.csv.gz")
    contacts = contacts[(contacts.population == "retained") & (contacts.tolerance_base30 == 10)]
    context = pd.read_csv(ROOT / "results/contexts.csv.gz")
    contacts = contacts.merge(context[["fixture", "source_frame", "fps", "court_present"]],
                              on=["fixture", "source_frame", "fps"], validate="many_to_one")
    missed = contacts[~contacts.matched].sort_values(["fixture", "source_frame", "rally_id", "label_index"])
    assert len(missed) == 4502
    records = []
    random_misses = missed.sample(n=24, random_state=SEED)
    for row in random_misses.itertuples(index=False):
        records.append({**row._asdict(), "sampling_group": "random missed label", "selection": "uniform over 4502 misses"})
    for fixture in LOW_VIDEOS:
        group = missed[missed.fixture == fixture].sort_values("source_frame")
        for third, indices in enumerate(np.array_split(np.arange(len(group)), 3), start=1):
            row = group.iloc[indices[len(indices) // 2]]
            records.append({**row.to_dict(), "sampling_group": "weak video", "selection": f"median miss in time third {third}"})
    group = missed[missed.fixture == 53].sort_values("source_frame")
    for third, indices in enumerate(np.array_split(np.arange(len(group)), 3), start=1):
        row = group.iloc[indices[len(indices) // 2]]
        records.append({**row.to_dict(), "sampling_group": "video 53", "selection": f"median miss in time third {third}"})
    rallies = pd.read_csv(ROOT / "results/rallies.csv.gz")
    correct = rallies[(rallies.population == "retained") & (rallies.tolerance_base30 == 10) & rallies.fully_correct]
    for fixture, count in ((53, 2), (41, 1), (33, 1), (16, 1)):
        group = correct[correct.fixture == fixture].sort_values("first_frame")
        chosen = group.iloc[np.linspace(0, len(group) - 1, count, dtype=int)]
        for rally in chosen.itertuples(index=False):
            hits = contacts[(contacts.fixture == fixture) & (contacts.rally_id == rally.rally_id)]
            row = hits.iloc[len(hits) // 2]
            records.append({**row.to_dict(), "sampling_group": "correct rally control", "selection": "middle labelled contact"})
    sample = pd.DataFrame(records)
    sample.insert(0, "sample_id", [f"A{index:02d}" for index in range(1, len(sample) + 1)])
    columns = ["sample_id", "fixture", "fps", "source_frame", "rally_id", "label_index", "labelled_contacts",
               "sampling_group", "selection", "matched", "player_correct", "court_present"]
    sample[columns].to_csv(ROOT / "results/alignment_sample.csv.gz", index=False)
    print(sample[columns].to_string(index=False))
    print(f"{len(sample)} windows; {sample.fixture.nunique()} videos; seed {SEED}")


if __name__ == "__main__":
    run()
