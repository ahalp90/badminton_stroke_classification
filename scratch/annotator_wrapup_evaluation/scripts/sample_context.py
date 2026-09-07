"""Choose a small, recorded visual sample with successful same-video controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260906


def run() -> None:
    contacts = pd.read_csv(ROOT / "results/contacts.csv.gz")
    contacts = contacts[(contacts.population == "retained") & (contacts.tolerance_base30 == 10)
                        & (contacts.fixture != 15)]
    context = pd.read_csv(ROOT / "results/contexts.csv.gz")
    data = contacts.merge(context, on=["fixture", "source_frame", "fps"], validate="many_to_one")
    rallies = pd.read_csv(ROOT / "results/rallies.csv.gz")
    rallies = rallies[(rallies.population == "retained") & (rallies.tolerance_base30 == 10)]
    data = data.merge(rallies[["fixture", "rally_id", "fully_correct"]],
                      on=["fixture", "rally_id"], validate="many_to_one")
    random = np.random.default_rng(SEED)
    ordinary = data[(data.position == "middle") & (data.distance_to_nearest_cut_seconds > 2)]
    missed = ordinary[~ordinary.matched]
    both_picked = (missed["pose_valid_top_t+0"] == 1) & (missed["pose_valid_bot_t+0"] == 1)
    strata = [(f"court rejected; video {fixture}", missed[(missed.fixture == fixture) & ~missed.court_present])
              for fixture in (53, 12, 20, 21)]
    strata += [("court accepted; a player pick missing", missed[missed.court_present & ~both_picked]),
               ("court accepted; both players picked", missed[missed.court_present & both_picked])]
    records = []
    for stratum, group in strata:
        count = 1 if stratum.startswith("court rejected") else 2
        candidates = group.iloc[random.permutation(len(group))]
        chosen = 0
        for error in candidates.itertuples(index=False):
            controls = ordinary[(ordinary.fixture == error.fixture) & ordinary.matched & ordinary.fully_correct]
            controls = controls[controls.rally_id != error.rally_id].copy()
            if controls.empty:
                continue
            controls["length_difference"] = abs(controls.labelled_contacts - error.labelled_contacts)
            controls["time_difference"] = abs(controls.source_frame - error.source_frame)
            control = controls.sort_values(["length_difference", "time_difference", "source_frame"]).iloc[0]
            pair_id = len(records) // 2 + 1
            for role, row in (("missed", error._asdict()), ("matched in correct rally", control.to_dict())):
                records.append({"pair_id": pair_id, "role": role, "stratum": stratum,
                                "fixture": row["fixture"], "fps": row["fps"], "source_frame": row["source_frame"],
                                "rally_id": row["rally_id"], "label_index": row["label_index"],
                                "labelled_contacts": row["labelled_contacts"]})
            chosen += 1
            if chosen == count:
                break
        assert chosen == count, stratum
    table = pd.DataFrame(records).iloc[random.permutation(len(records))].reset_index(drop=True)
    table.insert(0, "sample_id", [f"V{index + 1:02d}" for index in range(len(table))])
    assert len(table) == 16 and not table.duplicated(["fixture", "source_frame"]).any()
    table.to_csv(ROOT / "results/visual_sample.csv.gz", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    run()
