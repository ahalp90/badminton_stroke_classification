"""Build readable figures and compact tables from the frozen-output recount."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#999999"
PURPLE = "#7651A8"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(ROOT / "figures" / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    figure.savefig(ROOT / "figures" / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_selection(proposals: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.4))
    left = np.zeros(2)
    for outcome, colour in (("correct", BLUE), ("wrong", ORANGE), ("unknown", GREY)):
        counts = np.array([sum((proposals.selected == selected) & (proposals.outcome == outcome))
                           for selected in (True, False)])
        axis.barh([1, 0], counts, left=left, color=colour, label=outcome.capitalize())
        for position, count, start in zip([1, 0], counts, left, strict=True):
            if count > 150:
                axis.text(start + count / 2, position, f"{count:,}", ha="center", va="center", color="white")
        left += counts
    axis.set_yticks([1, 0], ["Selected\n784 clips", "Rejected\n3,198 clips"])
    axis.set_xlabel("Number of proposed rally clips")
    axis.set_ylabel("Fixed selection decision")
    axis.set_title("Selection keeps 616 correct clips and leaves 1,147 behind", loc="left", weight="bold", pad=42)
    figure.text(0.125, 0.91, "47 previously examined videos · trusted labels · ±10 frames at 30 fps", fontsize=11)
    axis.text(800, 1, "616 correct · 124 wrong · 44 unknown", va="center", fontsize=11)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    axis.set_xlim(0, 3350)
    save(figure, "selection")


def plot_errors(proposals: pd.DataFrame) -> None:
    wrong = proposals[proposals.selected & (proposals.outcome == "wrong")]
    assert len(wrong) == 124 and (wrong.overlapping_rallies == 1).all()
    names = {"missing": "Missed contacts", "extra": "Extra contacts",
             "wrong_player": "Wrong player", "boundary_error": "Clip cuts off rally"}
    counts: Counter[str] = Counter()
    for row in wrong.itertuples(index=False):
        label = " + ".join(name for key, name in names.items() if getattr(row, key) > 0)
        counts[label] += 1
    ordered = counts.most_common()
    labels = [label.replace(" + ", "\n+ ") for label, _ in ordered]
    values = [value for _, value in ordered]
    figure, axis = plt.subplots(figsize=(11.5, 7.5))
    positions = np.arange(len(values))
    axis.barh(positions, values, color=BLUE)
    axis.set_yticks(positions, labels, fontsize=10)
    axis.invert_yaxis()
    for position, value in zip(positions, values, strict=True):
        axis.text(value + 0.5, position, str(value), va="center")
    axis.set_xlim(0, max(values) + 5)
    axis.set_xlabel("Number of known wrong selected clips (124 in total)")
    axis.set_ylabel("Errors occurring together in one clip")
    axis.set_title("Selected clips usually fail because contacts are extra or missing", loc="left", weight="bold", pad=62)
    figure.text(0.125, 0.915, "47 previously examined videos · trusted labels · ±10 frames at 30 fps\n"
                "Each clip appears once. Player-only errors: 0.", fontsize=11)
    save(figure, "selected_errors")
    pd.DataFrame(ordered, columns=["errors_together", "clips"]).to_csv(
        ROOT / "results/error_combinations.csv.gz", index=False,
    )


def plot_contacts(contacts: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.3))
    labels = ["Serve", "Middle contact", "Last contact"]
    positions = np.arange(3)
    records = []
    for shift, tolerance, colour in ((-0.2, 10, BLUE), (0.2, 5, ORANGE)):
        data = contacts[contacts.tolerance_base30 == tolerance]
        groups = data.groupby("position").matched.agg(["size", "sum"]).loc[["serve", "middle", "last"]]
        missed = groups["size"] - groups["sum"]
        rates = 100 * missed / groups["size"]
        axis.bar(positions + shift, rates, width=0.36, color=colour, label=f"±{tolerance} frames")
        for position, (kind, row), count, rate in zip(positions + shift, groups.iterrows(), missed, rates, strict=True):
            axis.text(position, rate + 0.7, f"{rate:.1f}%", ha="center", fontsize=11)
            records.append({"position": kind, "tolerance_base30": tolerance,
                            "labelled": int(row["size"]), "missed": int(count)})
    axis.set_xticks(positions, [f"{label}\n{count:,} labelled contacts" for label, count in
                              zip(labels, [3422, 31415, 3381], strict=True)])
    axis.set_ylim(0, 36)
    axis.set_xlabel("Position within the labelled rally")
    axis.set_ylabel("Labelled contacts without a timing match (%)")
    axis.set_title("Serves and final contacts are missed more often", loc="left", weight="bold", pad=43)
    figure.text(0.125, 0.91, "47 previously examined videos · 38,218 trusted contact labels\n"
                "Same saved predictions; one-to-one matching across each full video. Frame clock: 30 fps.", fontsize=11)
    axis.legend(frameon=False)
    figure.text(0.125, -0.07, "The 41 one-contact rallies count as serves only. A timing match may still name the wrong player.", fontsize=10)
    save(figure, "contact_position")
    pd.DataFrame(records).to_csv(ROOT / "results/contact_position.csv.gz", index=False)


def plot_videos(rallies: pd.DataFrame, contacts: pd.DataFrame) -> None:
    per_video = rallies.groupby("fixture").agg(labelled_rallies=("fully_correct", "size"),
                                                correct_rallies=("fully_correct", "sum"))
    timing = contacts[contacts.tolerance_base30 == 10].groupby("fixture").matched.agg(["size", "sum"])
    per_video["labelled_contacts"] = timing["size"]
    per_video["matched_contacts"] = timing["sum"]
    per_video["rally_percent"] = 100 * per_video.correct_rallies / per_video.labelled_rallies
    per_video["contact_percent"] = 100 * per_video.matched_contacts / per_video.labelled_contacts
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(per_video.contact_percent, per_video.rally_percent, color=BLUE, alpha=0.75, s=50)
    for fixture in (15, 53):
        row = per_video.loc[fixture]
        axis.annotate(f"Video {fixture}\n{row.correct_rallies:.0f}/{row.labelled_rallies:.0f} rallies",
                      (row.contact_percent, row.rally_percent), xytext=(10, 10), textcoords="offset points", fontsize=11)
    axis.set(xlim=(0, 100), ylim=(-4, 100), xlabel="Labelled contacts with a timing match (%)",
             ylabel="Labelled rallies with a fully correct clip (%)")
    axis.set_title("Most videos cluster together; two need closer inspection", loc="left", weight="bold", pad=38)
    figure.text(0.125, 0.91, "One point per video · all 47 previously examined videos · trusted labels · ±10 frames at 30 fps", fontsize=10)
    axis.grid(alpha=0.15)
    save(figure, "video_variation")
    per_video.to_csv(ROOT / "results/per_video.csv.gz")


def run() -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    proposals = pd.read_csv(ROOT / "results/proposals.csv.gz")
    contacts = pd.read_csv(ROOT / "results/contacts.csv.gz")
    rallies = pd.read_csv(ROOT / "results/rallies.csv.gz")
    trusted_proposals = proposals[(proposals.population == "retained") & (proposals.tolerance_base30 == 10)]
    trusted_contacts = contacts[contacts.population == "retained"]
    trusted_rallies = rallies[(rallies.population == "retained") & (rallies.tolerance_base30 == 10)]
    plot_selection(trusted_proposals)
    plot_errors(trusted_proposals)
    plot_contacts(trusted_contacts)
    plot_videos(trusted_rallies, trusted_contacts)


if __name__ == "__main__":
    run()
