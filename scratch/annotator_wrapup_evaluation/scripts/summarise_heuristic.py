"""Compare ordinary heuristic output with the fixed learned detector on identical labels."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["population", "tolerance_base30", "fixture", "rally_id", "label_index"]


def run() -> None:
    heuristic = pd.read_csv(ROOT / "results/heuristic_contacts.csv.gz")
    learned = pd.read_csv(ROOT / "results/contacts.csv.gz")
    joined = heuristic.merge(learned[KEY + ["source_frame", "matched", "player_correct"]],
                             on=KEY, validate="one_to_one", suffixes=("_heuristic", "_learned"), indicator=True)
    assert len(joined) == len(learned) and (joined._merge == "both").all()
    assert (joined.source_frame_heuristic == joined.source_frame_learned).all()
    del joined["_merge"]
    context = pd.read_csv(ROOT / "results/contexts.csv.gz")
    joined = joined.rename(columns={"source_frame_heuristic": "source_frame"})
    joined = joined.merge(context[["fixture", "source_frame", "court_present", "pose_valid_top_t+0",
                                   "pose_valid_bot_t+0", "nearby_saved_rows"]],
                           on=["fixture", "source_frame"], validate="many_to_one", indicator=True)
    assert (joined._merge == "both").all()
    records, input_rows, positions = [], [], []
    for population in ("retained", "all_gt"):
        for tolerance in (10, 5):
            base = joined[(joined.population == population) & (joined.tolerance_base30 == tolerance)]
            for omitted in ((), (15,)):
                group = base[~base.fixture.isin(omitted)]
                common = {"population": population, "tolerance_base30": tolerance,
                          "omitted_videos": ",".join(map(str, omitted))}
                for (before, after), rows in group.groupby(["matched_heuristic", "matched_learned"]):
                    records.append({**common, "heuristic_matched": before, "learned_matched": after,
                                    "contacts": len(rows)})
                for model in ("heuristic", "learned"):
                    for matched, rows in group.groupby(f"matched_{model}"):
                        both = (rows["pose_valid_top_t+0"] == 1) & (rows["pose_valid_bot_t+0"] == 1)
                        input_rows.append({**common, "model": model, "matched": matched, "contacts": len(rows),
                                           "court_rejected": int(sum(~rows.court_present)),
                                           "accepted_missing_pick": int(sum(rows.court_present & ~both)),
                                           "accepted_both_picked": int(sum(rows.court_present & both))})
                    for position, rows in group.groupby("position"):
                        positions.append({**common, "model": model, "position": position, "labelled": len(rows),
                                          "missed": int(sum(~rows[f"matched_{model}"]))})
    pd.DataFrame(records).to_csv(ROOT / "results/heuristic_paired_contacts.csv.gz", index=False)
    pd.DataFrame(input_rows).to_csv(ROOT / "results/heuristic_upstream.csv.gz", index=False)
    position_table = pd.DataFrame(positions)
    position_table.to_csv(ROOT / "results/heuristic_position.csv.gz", index=False)
    plot_positions(position_table)
    primary = joined[(joined.population == "retained") & (joined.tolerance_base30 == 10)]
    filtering = primary.groupby(["raw_matched", "matched_heuristic"]).size().rename("contacts").reset_index()
    filtering.to_csv(ROOT / "results/heuristic_filtering_matches.csv.gz", index=False)
    per_video = primary.groupby("fixture").agg(labelled=("matched_heuristic", "size"),
                                               heuristic_matched=("matched_heuristic", "sum"),
                                               learned_matched=("matched_learned", "sum"))
    hr = pd.read_csv(ROOT / "results/heuristic_rallies.csv.gz")
    lr = pd.read_csv(ROOT / "results/rallies.csv.gz")
    rally_keys = KEY[:-1]
    paired = hr.merge(lr[rally_keys + ["fully_correct"]], on=rally_keys, validate="one_to_one",
                      suffixes=("_heuristic", "_learned"), indicator=True)
    assert len(paired) == len(lr) and (paired._merge == "both").all()
    primary_rallies = paired[(paired.population == "retained") & (paired.tolerance_base30 == 10)]
    per_video = per_video.join(primary_rallies.groupby("fixture").agg(
        labelled_rallies=("fully_correct_heuristic", "size"),
        heuristic_correct_rallies=("fully_correct_heuristic", "sum"),
        learned_correct_rallies=("fully_correct_learned", "sum")))
    per_video.to_csv(ROOT / "results/heuristic_per_video.csv.gz")
    paired.groupby(["population", "tolerance_base30", "fully_correct_heuristic", "fully_correct_learned"]).size().rename(
        "rallies").reset_index().to_csv(ROOT / "results/heuristic_paired_rallies.csv.gz", index=False)
    proposals = pd.read_csv(ROOT / "results/heuristic_proposals.csv.gz")
    proposals = proposals[(proposals.population == "retained") & (proposals.tolerance_base30 == 10)]
    wrong = proposals[proposals.outcome == "wrong"]
    combinations = []
    for row in wrong.itertuples(index=False):
        problems = []
        for key, label in (("missing", "Missing contacts"), ("extra", "Extra contacts"),
                           ("wrong_player", "Wrong matched player")):
            if getattr(row, key) > 0:
                problems.append(label)
        if row.boundary_error:
            problems.append("Clip cuts off rally")
        if row.overlapping_rallies > 1:
            problems.append("Multiple labelled rallies")
        assert problems, (row.fixture, row.span_id)
        combinations.append(" + ".join(problems))
    pd.Series(combinations).value_counts().rename_axis("problems").reset_index(name="clips").to_csv(
        ROOT / "results/heuristic_error_combinations.csv.gz", index=False)
    with gzip.open(ROOT / "results/heuristic_summary.json.gz", "rt") as source:
        summary = json.load(source)
    plot_contacts(summary)
    print("All", len(joined), "contact rows and", len(paired), "rally rows join exactly.")
    print(pd.DataFrame(summary["summary"]).to_string(index=False))
    print(pd.DataFrame(input_rows).query("population == 'retained' and tolerance_base30 == 10").to_string(index=False))
    print(filtering.to_string(index=False))
    print("Wrong proposals", len(wrong), "missing", sum(wrong.missing > 0), "extra", sum(wrong.extra > 0),
          "player", sum(wrong.wrong_player > 0), "boundary", sum(wrong.boundary_error),
          "multiple", sum(wrong.overlapping_rallies > 1))


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(ROOT / "figures" / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_contacts(summary: dict) -> None:
    row = next(row for row in summary["summary"] if row["population"] == "retained" and row["tolerance_base30"] == 10)
    names = ["Heuristic candidates\nbefore filtering", "Ordinary heuristic\noutput", "Final learned\ndetector"]
    emitted = np.array([row["raw"], row["filtered"], 41605])
    matched = np.array([row["raw_matched"], row["matched"], 33716])
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.barh(range(3), matched, color="#0072B2", label="Matches a cleaned label")
    axis.barh(range(3), emitted - matched, left=matched, color="#D55E00", label="No match in cleaned labels")
    for index, (count, total) in enumerate(zip(matched, emitted, strict=True)):
        axis.text(count / 2, index, f"{count:,}", ha="center", va="center", color="white", fontsize=12)
        axis.text(count + (total - count) / 2, index, f"{total - count:,}", ha="center", va="center", color="white", fontsize=12)
    axis.invert_yaxis()
    axis.set(yticks=range(3), yticklabels=names, xlabel="Number of emitted events", ylabel="Contact output stage")
    axis.set_title("The learned detector finds more labelled contacts with fewer unmatched events", loc="left", weight="bold", pad=45)
    figure.text(0.125, 0.93, "47 previously examined videos · 38,218 cleaned labels · ±10 frames at 30 fps\n"
                "Unmatched does not mean physically false: labels omit some play and video 15 is misaligned.", fontsize=10)
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "heuristic_contact_comparison")


def plot_positions(table: pd.DataFrame) -> None:
    data = table[(table.population == "retained") & (table.tolerance_base30 == 10) & (table.omitted_videos == "15")]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    order = ["serve", "middle", "last"]
    for index, (model, colour, label) in enumerate((("heuristic", "#D55E00", "Ordinary heuristic"),
                                                   ("learned", "#0072B2", "Final learned detector"))):
        rows = data[data.model == model].set_index("position").loc[order]
        rates = 100 * rows.missed / rows.labelled
        bars = axis.bar(np.arange(3) + (index - 0.5) * 0.36, rates, width=0.34, color=colour, label=label)
        axis.bar_label(bars, labels=[f"{value:.1f}%" for value in rates], padding=4)
    axis.set(xticks=range(3), xticklabels=["Serve", "Middle contact", "Final contact"],
             xlabel="Contact position within the labelled rally", ylabel="Labelled contacts missed (%)")
    axis.set_ylim(0, max(axis.get_ylim()) * 1.15)
    axis.set_title("Compare misses at the start, middle and end of a rally", loc="left", weight="bold", pad=42)
    figure.text(0.125, 0.93, "46 previously examined videos · cleaned labels · ±10 frames at 30 fps\n"
                "Video 15 omitted. Both outputs use the same court and tracking inputs; no new model was fitted.", fontsize=10)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "heuristic_contact_position")


if __name__ == "__main__":
    run()
