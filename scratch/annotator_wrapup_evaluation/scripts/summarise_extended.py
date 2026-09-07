"""Answer follow-up questions using the completed evaluation's row-level tables."""

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


def primary(table: pd.DataFrame) -> pd.DataFrame:
    return table[(table.population == "retained") & (table.tolerance_base30 == 10)].copy()


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(ROOT / "figures" / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def timing_summary(contacts: pd.DataFrame) -> list[dict]:
    records = []
    for omitted in ((), (15,), (15, 53)):
        group = contacts[~contacts.fixture.isin(omitted)]
        matched = group[group.matched]
        offsets = matched.offset_base30
        records.append({
            "omitted_videos": list(omitted), "labelled": len(group), "matched": len(matched),
            "missed": int(sum(~group.matched)), "exact_frame": int(sum(offsets == 0)),
            "within_two_frames": int(sum(offsets.abs() <= 2)),
            "within_five_frames": int(sum(offsets.abs() <= 5)),
            "median_offset": float(offsets.median()), "mean_offset": float(offsets.mean()),
            "side_correct": int(sum(matched.player_correct)),
            "matched_unknown_target": int(matched.target_side.isna().sum()),
            "matched_missing_prediction_side": int(matched.predicted_side.isna().sum()),
        })
    group = contacts[contacts.matched & (contacts.fixture != 15)]
    counts = group.groupby("offset_base30").size().reindex(range(-10, 11), fill_value=0)
    figure, axis = plt.subplots(figsize=(10, 4.7))
    axis.bar(counts.index, counts.values, width=0.82, color="#0072B2")
    axis.set(xlabel="Prediction frame minus label frame (30 fps; negative means early)",
             ylabel="Number of matched labelled contacts", xticks=range(-10, 11, 2))
    axis.set_title("Most timing matches are close to the labelled frame", loc="left", weight="bold", pad=42)
    figure.text(0.125, 0.92, f"46 previously examined videos · cleaned labels · ±10-frame matching\n"
                f"Video 15 omitted. Only the {len(group):,} timing matches appear; missing contacts are outside this plot.",
                fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, "timing_offsets")
    return records


def rally_summary(rallies: pd.DataFrame) -> list[dict]:
    records = []
    for omitted in ((), (15,), (15, 53)):
        group = rallies[~rallies.fixture.isin(omitted)].copy()
        group["state"] = np.select([
            group.fully_correct, group.contained, group.overlapping_proposals > 0,
        ], ["Fully correct clip", "Whole rally fits in a clip; errors remain", "Only partial rally coverage"],
            default="No clip reaches a labelled contact")
        for state, rows in group.groupby("state", sort=False):
            records.append({"omitted_videos": list(omitted), "state": state, "rallies": len(rows),
                            "total_rallies": len(group)})
    shown = pd.DataFrame([record for record in records if not record["omitted_videos"]])
    order = ["Fully correct clip", "Whole rally fits in a clip; errors remain", "Only partial rally coverage",
             "No clip reaches a labelled contact"]
    counts = shown.set_index("state").loc[order, "rallies"]
    figure, axis = plt.subplots(figsize=(10, 4.6))
    bars = axis.barh(range(4), counts, color=["#0072B2", "#7651A8", "#D55E00", "#777777"])
    axis.invert_yaxis()
    axis.set(yticks=range(4), yticklabels=order, xlabel="Number of cleaned labelled rallies (3,422 total)",
             ylabel="Best available clip coverage")
    axis.bar_label(bars, padding=5, fontsize=11)
    axis.set_xlim(0, counts.max() * 1.14)
    axis.set_title("Some rallies lack a complete clip before contact details can be judged", loc="left", weight="bold", pad=40)
    figure.text(0.125, 0.92, "All 47 previously examined videos · ±10 frames at 30 fps · before selection\n"
                "Each labelled rally appears once. These groups describe output; they are not independent causes.", fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, "rally_coverage")
    return records



def video15_contributions(contacts: pd.DataFrame, proposals: pd.DataFrame, rallies: pd.DataFrame) -> None:
    predictions = primary(pd.read_csv(ROOT / "results/predictions.csv.gz"))
    events = pd.read_csv(ROOT / "results/selected_event_errors.csv.gz")
    groups = [
        ("Missed labelled contacts; full video", contacts[~contacts.matched]),
        ("Emitted events without a label match", predictions[~predictions.matched]),
        ("Labelled rallies without a fully correct clip", rallies[~rallies.fully_correct]),
        ("Known wrong selected clips", proposals[proposals.selected & (proposals.outcome == "wrong")]),
        ("Extra events in wrong selected clips", events[events.kind == "extra"]),
        ("Missed events in wrong selected clips", events[events.kind == "missed"]),
        ("Unknown selected clips", proposals[proposals.selected & (proposals.outcome == "unknown")]),
    ]
    rows = []
    for outcome, group in groups:
        count = int(sum(group.fixture == 15))
        rows.append({"outcome": outcome, "video15": count, "all47": len(group),
                     "video15_share_percent": 100 * count / len(group),
                     "interpretation": "occurs in video 15; not a causal attribution"})
    pd.DataFrame(rows).to_csv(ROOT / "results/video15_error_contribution.csv.gz", index=False)


def run() -> None:
    all_contacts = pd.read_csv(ROOT / "results/contacts.csv.gz")
    all_proposals = pd.read_csv(ROOT / "results/proposals.csv.gz")
    contacts, proposals = primary(all_contacts), primary(all_proposals)
    rallies = primary(pd.read_csv(ROOT / "results/rallies.csv.gz"))
    assert len(contacts) == 38218 and len(proposals) == 3982 and len(rallies) == 3422
    video15_contributions(contacts, proposals, rallies)
    report = {"timing": timing_summary(contacts), "rally_coverage": rally_summary(rallies)}
    side_rows = contacts[contacts.matched].copy()
    side_rows["target_side"] = side_rows.target_side.fillna("Unknown")
    side_rows["predicted_side"] = side_rows.predicted_side.fillna("Unassigned")
    confusion = side_rows.groupby(["target_side", "predicted_side"]).size().rename("contacts").reset_index()
    confusion.to_csv(ROOT / "results/player_confusion.csv.gz", index=False)
    assert confusion.contacts.sum() == contacts.matched.sum()
    broader = all_proposals[(all_proposals.population == "all_gt") & (all_proposals.tolerance_base30 == 10)]
    joined = proposals.merge(broader[["fixture", "span_id", "outcome"]], on=["fixture", "span_id"],
                             validate="one_to_one", suffixes=("_cleaned", "_all_source"))
    transitions = joined.groupby(["selected", "outcome_cleaned", "outcome_all_source"]).size()
    transitions.rename("clips").reset_index().to_csv(ROOT / "results/label_judgement_changes.csv.gz", index=False)
    video = proposals.groupby("fixture").outcome.value_counts().unstack(fill_value=0)
    chosen = proposals[proposals.selected].groupby("fixture").outcome.value_counts().unstack(fill_value=0)
    video = video.join(chosen.add_prefix("selected_"), how="left").fillna(0).astype(int)
    video = video.join(rallies.groupby("fixture").size().rename("labelled_rallies"))
    video["correct_retained_percent"] = 100 * video.selected_correct / video.correct.replace(0, np.nan)
    video.to_csv(ROOT / "results/selection_per_video.csv.gz")
    severity = proposals[proposals.selected & (proposals.outcome == "wrong")]
    severity.groupby(["missing", "extra", "wrong_player", "boundary_error"]).size().rename("clips").reset_index().to_csv(
        ROOT / "results/selected_error_severity.csv.gz", index=False,
    )
    context = pd.read_csv(ROOT / "results/contexts.csv.gz")
    data = contacts.merge(context, on=["fixture", "source_frame", "fps"], validate="many_to_one", indicator=True)
    assert (data._merge == "both").all()
    data = data[data.fixture != 15].copy()
    both_picked = (data["pose_valid_top_t+0"] == 1) & (data["pose_valid_bot_t+0"] == 1)
    data["state"] = np.select([~data.court_present, data.court_present & ~both_picked],
                              ["Court rejected", "Accepted; player pick missing"], default="Accepted; both picked")
    rates = data.groupby("state").matched.agg(labelled="size", matched="sum")
    rates["missed"] = rates.labelled - rates.matched
    rates["missed_percent"] = 100 * rates.missed / rates.labelled
    rates.to_csv(ROOT / "results/input_conditional_rates.csv.gz")
    per_video = pd.read_csv(ROOT / "results/per_video.csv.gz")
    report["per_video"] = {"median_rally_percent": float(per_video.rally_percent.median()),
                           "median_contact_percent": float(per_video.contact_percent.median()),
                           "best": per_video.sort_values("rally_percent", ascending=False).head(5).to_dict("records"),
                           "worst": per_video.sort_values("rally_percent").head(5).to_dict("records")}
    report["one_label_rally"] = {"proposals": int(sum(proposals.overlapping_rallies == 1)),
                                 "multiple_rally_proposals": int(sum(proposals.overlapping_rallies > 1)),
                                 "no_label_proposals": int(sum(proposals.overlapping_rallies == 0))}
    with gzip.open(ROOT / "results/extended_summary.json.gz", "wt") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(json.dumps(report, indent=2))
    print(confusion.to_string(index=False)); print(transitions.to_string()); print(rates.to_string())


if __name__ == "__main__":
    run()
