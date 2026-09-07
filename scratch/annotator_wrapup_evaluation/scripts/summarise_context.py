"""Compare saved upstream evidence at missed and matched labelled contacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATES = ("Court rejected; tracking skipped", "Court accepted; a player pick missing",
          "Court accepted; both players picked")
COLOURS = ("#D55E00", "#7651A8", "#0072B2")


def join_context() -> pd.DataFrame:
    contacts = pd.read_csv(ROOT / "results/contacts.csv.gz")
    context = pd.read_csv(ROOT / "results/contexts.csv.gz")
    data = contacts.merge(context, on=["fixture", "source_frame", "fps"], validate="many_to_one", indicator=True)
    assert (data._merge == "both").all()
    assert not contacts.duplicated(["population", "tolerance_base30", "fixture", "rally_id", "label_index"]).any()
    assert (data.court_present == data.tracker_covered).all()
    assert (data.excluded == ~data.court_present).all()
    covered = data.court_present
    assert data.loc[covered, "feature_exact_frame"].notna().all()
    both = (data["pose_valid_top_t+0"] == 1) & (data["pose_valid_bot_t+0"] == 1)
    data["input_state"] = np.select([~covered, covered & ~both], STATES[:2], default=STATES[2])
    return data


def plot_states(data: pd.DataFrame) -> None:
    selected = data[(data.population == "retained") & (data.tolerance_base30 == 10) & (data.fixture != 15)]
    figure, axis = plt.subplots(figsize=(10, 4.7))
    positions = [1, 0]
    left = np.zeros(2)
    totals = np.array([sum(~selected.matched), sum(selected.matched)])
    for state, colour in zip(STATES, COLOURS, strict=True):
        counts = np.array([sum((selected.input_state == state) & (selected.matched == matched))
                           for matched in (False, True)])
        width = 100 * counts / totals
        axis.barh(positions, width, left=left, height=0.55, color=colour, label=state)
        for position, start, percent, count in zip(positions, left, width, counts, strict=True):
            if percent > 8:
                axis.text(start + percent / 2, position, f"{percent:.1f}%\n{count:,} contacts",
                          color="white", ha="center", va="center", fontsize=12)
        left += width
    axis.set_yticks(positions, [f"Missed\n{totals[0]:,} contacts", f"Matched\n{totals[1]:,} contacts"])
    axis.set(xlim=(0, 100), xlabel="Share of labelled contacts in each timing outcome (%)",
             ylabel="Final timing result")
    axis.set_title("Most misses fall in scenes that the court stage rejected", loc="left", weight="bold", pad=48)
    figure.text(0.125, 0.92, "46 previously examined videos · trusted labels · ±10 frames at 30 fps\n"
                "Video 15 omitted here because its labels and footage disagree. Inputs measured at each labelled frame.",
                fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.27), frameon=False, fontsize=10)
    figure.savefig(ROOT / "figures/upstream_context.png", dpi=170, bbox_inches="tight", facecolor="white")
    figure.savefig(ROOT / "figures/upstream_context.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run() -> None:
    data = join_context()
    records, positions = [], []
    for population in ("retained", "all_gt"):
        for tolerance in (10, 5):
            base = data[(data.population == population) & (data.tolerance_base30 == tolerance)]
            for excluded_videos in ((), (15,), (15, 53)):
                group = base[~base.fixture.isin(excluded_videos)]
                for matched, contacts in group.groupby("matched"):
                    common = {"population": population, "tolerance_base30": tolerance,
                              "omitted_videos": ",".join(map(str, excluded_videos)), "matched": matched}
                    records.append({**common, "contacts": len(contacts),
                                    "court_rejected": int(sum(~contacts.court_present)),
                                    "accepted_missing_pick": int(sum(contacts.input_state == STATES[1])),
                                    "accepted_both_picked": int(sum(contacts.input_state == STATES[2])),
                                    "no_nearby_scored_row": int(sum(contacts.nearby_saved_rows == 0)),
                                    "no_filled_shuttle": int(sum(~contacts.filled_shuttle_visible)),
                                    "within_one_second_saved_cut": int(sum(contacts.distance_to_nearest_cut_seconds <= 1))})
                for position, contacts in group[group.court_present].groupby("position"):
                    positions.append({"population": population, "tolerance_base30": tolerance,
                                      "omitted_videos": ",".join(map(str, excluded_videos)), "position": position,
                                      "labelled": len(contacts), "missed": int(sum(~contacts.matched))})
    pd.DataFrame(records).to_csv(ROOT / "results/upstream_summary.csv.gz", index=False)
    pd.DataFrame(positions).to_csv(ROOT / "results/contact_position_court_accepted.csv.gz", index=False)
    plot_states(data)
    print("Joined", len(data), "label/tolerance rows; all joins and coverage checks passed")


if __name__ == "__main__":
    run()
