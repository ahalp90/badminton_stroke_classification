"""Summarise the saved outputs with videos 15 and 53 omitted in turn.

The tables are descriptive recounts of the saved learned and ordinary-heuristic
outputs.  They do not treat an unlabelled video frame as a negative contact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

POPULATIONS = ("retained", "all_gt")
TOLERANCES = (10, 5)
MODELS = ("learned", "heuristic")
OMISSIONS: Mapping[str, tuple[int, ...]] = {
    "all47": (),
    "omit15": (15,),
    "omit15_53": (15, 53),
}
OMISSION_LABELS = {name: "none" if not videos else ",".join(map(str, videos))
                   for name, videos in OMISSIONS.items()}

CONTACT_KEY = ["population", "tolerance_base30", "fixture", "rally_id", "label_index"]
RALLY_KEY = ["population", "tolerance_base30", "fixture", "rally_id"]
PROPOSAL_KEY = ["population", "tolerance_base30", "fixture", "span_id"]
CONTEXT_KEY = ["fixture", "source_frame", "fps"]

BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#7651A8"
GREY = "#777777"
COLOURS = {"matchedcorrect": BLUE, "matchednotcorrect": ORANGE, "missed": GREY}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv.gz")


def load_tables() -> dict[str, pd.DataFrame]:
    """Read the saved recount tables used by this aggregation."""
    return {
        "contacts": _read("contacts"),
        "heuristic_contacts": _read("heuristic_contacts"),
        "proposals": _read("proposals"),
        "heuristic_proposals": _read("heuristic_proposals"),
        "rallies": _read("rallies"),
        "heuristic_rallies": _read("heuristic_rallies"),
        "predictions": _read("predictions"),
        "contexts": _read("contexts"),
        "heuristic_receipts": _read("heuristic_receipts"),
    }


def check_unique(table: pd.DataFrame, key: list[str], name: str) -> None:
    """Fail loudly when a saved identity table is duplicated."""
    if table.duplicated(key).any():
        raise AssertionError(f"{name} has duplicate identities for {key}")


def check_cross_model_contacts(tables: dict[str, pd.DataFrame]) -> None:
    """Check that learned and heuristic scores use identical labelled rows."""
    learned = tables["contacts"]
    heuristic = tables["heuristic_contacts"]
    check_unique(learned, CONTACT_KEY, "contacts")
    check_unique(heuristic, CONTACT_KEY, "heuristic_contacts")
    other = learned[CONTACT_KEY + ["source_frame", "fps"]]
    joined = heuristic.merge(other, on=CONTACT_KEY, validate="one_to_one", indicator=True,
                             suffixes=("_heuristic", "_learned"))
    if len(joined) != len(learned) or not (joined["_merge"] == "both").all():
        raise AssertionError("learned and heuristic contact identities do not join exactly")
    if not (joined["source_frame_heuristic"] == joined["source_frame_learned"]).all():
        raise AssertionError("learned and heuristic source frames differ")
    if not np.allclose(joined["fps_heuristic"], joined["fps_learned"]):
        raise AssertionError("learned and heuristic frame rates differ")


def add_rally_timing(rallies: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    """Add exact-sequence timing from a containing single-rally proposal.

    ``timing_complete`` in the proposal scorer requires every emitted event and
    every labelled contact to match inside a contained, single-rally clip.  It
    does not inspect player side correctness, so it is the requested timing-only
    rally measure.
    """
    check_unique(rallies, RALLY_KEY, "rallies")
    check_unique(proposals, PROPOSAL_KEY, "proposals")
    complete = proposals[
        proposals["overlapping_rallies"].eq(1)
        & proposals["whole_rally_contained"]
        & proposals["timing_complete"]
        & proposals["rally_id"].notna()
    ][RALLY_KEY].drop_duplicates()
    complete["timing_complete_rally"] = True
    enriched = rallies.merge(complete, on=RALLY_KEY, how="left", validate="one_to_one")
    enriched["timing_complete_rally"] = enriched["timing_complete_rally"].eq(True)
    if "timing_complete" in enriched and not (
        enriched["timing_complete"] == enriched["timing_complete_rally"]
    ).all():
        raise AssertionError("saved rally timing flag differs from proposal timing semantics")
    return enriched


def add_context(contacts: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    """Attach saved upstream state to each labelled contact row."""
    context_columns = CONTEXT_KEY + ["court_present", "pose_valid_top_t+0", "pose_valid_bot_t+0"]
    joined = contacts.merge(contexts[context_columns], on=CONTEXT_KEY, how="left", validate="many_to_one",
                            indicator=True)
    if not (joined["_merge"] == "both").all():
        raise AssertionError("a labelled contact has no saved upstream context")
    joined = joined.drop(columns="_merge")
    both_picked = (joined["pose_valid_top_t+0"] == 1) & (joined["pose_valid_bot_t+0"] == 1)
    joined["upstream_state"] = np.select(
        [~joined["court_present"], joined["court_present"] & ~both_picked],
        ["court_rejected", "accepted_missing_pick"],
        default="accepted_both_picked",
    )
    return joined


def contact_categories(contacts: pd.DataFrame) -> pd.DataFrame:
    """Classify labelled contacts without collapsing unknown target sides."""
    rows = contacts.copy()
    matched = rows["matched"].astype(bool)
    target_known = rows["target_side"].notna()
    prediction_assigned = rows["predicted_side"].notna()
    rows["matched_correct"] = matched & target_known & prediction_assigned & (
        rows["target_side"] == rows["predicted_side"]
    )
    rows["matched_wrong_side"] = matched & target_known & prediction_assigned & (
        rows["target_side"] != rows["predicted_side"]
    )
    rows["matched_missing_predicted_side"] = matched & target_known & ~prediction_assigned
    rows["matched_unknown_target_side"] = matched & ~target_known
    rows["unmatched_labelled_hit"] = ~matched
    expected = rows["matched_correct"]
    if not (rows["player_correct"].astype(bool) == expected).all():
        raise AssertionError("saved player_correct flag differs from timing-and-player attribution")
    return rows


def _rate(count: int, denominator: int) -> float:
    return 100.0 * count / denominator if denominator else np.nan


def _event_counts(
    model: str,
    fixture: int | None,
    population: str,
    tolerance: int,
    tables: dict[str, pd.DataFrame],
) -> tuple[int, int]:
    """Return emitted events and emitted events without a label timing match."""
    if model == "learned":
        events = tables["predictions"]
        events = events[(events.population == population) & (events.tolerance_base30 == tolerance)]
        if fixture is not None:
            events = events[events.fixture == fixture]
        emitted = len(events)
        unmatched = int((~events.matched).sum())
        return emitted, unmatched
    receipts = tables["heuristic_receipts"]
    if fixture is not None:
        receipts = receipts[receipts.fixture == fixture]
    if len(receipts) != (1 if fixture is not None else receipts.fixture.nunique()):
        raise AssertionError("heuristic receipt identity is not unique")
    emitted = int(receipts.filtered_contacts.sum())
    contacts = tables["heuristic_contacts"]
    contacts = contacts[(contacts.population == population) & (contacts.tolerance_base30 == tolerance)]
    if fixture is not None:
        contacts = contacts[contacts.fixture == fixture]
    return emitted, emitted - int(contacts.matched.sum())


def _contact_counts(contacts: pd.DataFrame) -> dict[str, int]:
    return {
        "labelled_contacts": len(contacts),
        "matched_contacts": int(contacts.matched.sum()),
        "confirmed_contacts": int(contacts.matched_correct.sum()),
        "matched_wrong_side": int(contacts.matched_wrong_side.sum()),
        "matched_missing_predicted_side": int(contacts.matched_missing_predicted_side.sum()),
        "matched_unknown_target_side": int(contacts.matched_unknown_target_side.sum()),
        "unmatched_labelled_hits": int(contacts.unmatched_labelled_hit.sum()),
    }


def _serve_counts(contacts: pd.DataFrame) -> dict[str, int]:
    return {
        "labelled_serves": int((contacts.position == "serve").sum()),
        "matched_serves": int(((contacts.position == "serve") & contacts.matched).sum()),
        "confirmed_serves": int(((contacts.position == "serve") & contacts.matched_correct).sum()),
    }


def _rally_counts(rallies: pd.DataFrame, model: str) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "labelled_rallies": len(rallies),
        "contained_rallies": int(rallies.contained.sum()),
        "timing_complete_rallies": int(rallies.timing_complete_rally.sum()),
        "fully_correct_rallies": int(rallies.fully_correct.sum()),
        "selected_correct_rallies": int(rallies.selected_correct.sum()) if model == "learned" else None,
    }
    return result


def _proposal_counts(proposals: pd.DataFrame, model: str) -> dict[str, int | None]:
    outcomes = proposals.outcome.value_counts()
    result: dict[str, int | None] = {
        "proposals": len(proposals),
        "proposal_correct": int(outcomes.get("correct", 0)),
        "proposal_wrong": int(outcomes.get("wrong", 0)),
        "proposal_unknown": int(outcomes.get("unknown", 0)),
        "selected_proposals": int(proposals.selected.sum()) if model == "learned" else None,
        "selected_correct": int((proposals.selected & proposals.outcome.eq("correct")).sum())
        if model == "learned" else None,
        "selected_wrong": int((proposals.selected & proposals.outcome.eq("wrong")).sum())
        if model == "learned" else None,
        "selected_unknown": int((proposals.selected & proposals.outcome.eq("unknown")).sum())
        if model == "learned" else None,
    }
    return result


def _upstream_counts(contacts: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for state in ("court_rejected", "accepted_missing_pick", "accepted_both_picked"):
        state_rows = contacts[contacts.upstream_state == state]
        result[f"labelled_{state}"] = len(state_rows)
        result[f"matched_{state}"] = int(state_rows.matched.sum())
        result[f"missed_{state}"] = int((~state_rows.matched).sum())
    return result


def _aggregate_row(
    model: str,
    population: str,
    tolerance: int,
    omission: str,
    contacts: pd.DataFrame,
    proposals: pd.DataFrame,
    rallies: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    fixture: int | None,
) -> dict[str, object]:
    contact = contact_categories(contacts)
    counts = _contact_counts(contact)
    serve = _serve_counts(contact)
    rally = _rally_counts(rallies, model)
    proposal = _proposal_counts(proposals, model)
    upstream = _upstream_counts(contact)
    if fixture is None:
        emitted = 0
        emitted_unmatched = 0
        for included_fixture in contacts.fixture.unique():
            events, unmatched = _event_counts(model, int(included_fixture), population, tolerance, tables)
            emitted += events
            emitted_unmatched += unmatched
    else:
        emitted, emitted_unmatched = _event_counts(model, fixture, population, tolerance, tables)
    denominator = counts["labelled_contacts"]
    serve_denominator = serve["labelled_serves"]
    rally_denominator = rally["labelled_rallies"]
    row: dict[str, object] = {
        "model": model,
        "population": population,
        "tolerance_base30": tolerance,
        "omission": omission,
        "omitted_videos": OMISSION_LABELS[omission],
        "included_videos": contacts.fixture.nunique(),
        **counts,
        "matched_correct_contacts": counts["confirmed_contacts"],
        "matched_not_correct_contacts": counts["matched_contacts"] - counts["confirmed_contacts"],
        "timing_recall_percent": _rate(counts["matched_contacts"], denominator),
        "attribution_confirmed_percent": _rate(counts["confirmed_contacts"], denominator),
        "matched_side_answered_percent": _rate(
            counts["confirmed_contacts"] + counts["matched_wrong_side"], counts["matched_contacts"]
        ),
        **serve,
        "serve_timing_recall_percent": _rate(serve["matched_serves"], serve_denominator),
        "serve_attribution_confirmed_percent": _rate(serve["confirmed_serves"], serve_denominator),
        **rally,
        "containment_rate_percent": _rate(rally["contained_rallies"], rally_denominator),
        "timing_complete_rate_percent": _rate(rally["timing_complete_rallies"], rally_denominator),
        "fully_correct_rate_percent": _rate(rally["fully_correct_rallies"], rally_denominator),
        "selected_correct_rate_percent": _rate(rally["selected_correct_rallies"], rally_denominator)
        if rally["selected_correct_rallies"] is not None else np.nan,
        **proposal,
        "selected_correct_rate_of_proposals_percent": _rate(
            proposal["selected_correct"], proposal["selected_proposals"]
        ) if proposal["selected_proposals"] is not None else np.nan,
        "emitted_events": emitted,
        "emitted_unmatched_events": emitted_unmatched,
        **upstream,
    }
    if fixture is not None:
        row["fixture"] = fixture
    return row


def build_tables(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aggregate, per-video outcome and side-confusion tables."""
    check_cross_model_contacts(tables)
    contexts = tables["contexts"]
    learned_contacts = add_context(tables["contacts"], contexts)
    heuristic_contacts = add_context(tables["heuristic_contacts"], contexts)
    contacts_by_model = {"learned": learned_contacts, "heuristic": heuristic_contacts}
    proposals_by_model = {"learned": tables["proposals"], "heuristic": tables["heuristic_proposals"]}
    rallies_by_model = {
        "learned": add_rally_timing(tables["rallies"], tables["proposals"]),
        "heuristic": add_rally_timing(tables["heuristic_rallies"], tables["heuristic_proposals"]),
    }
    for model, table in contacts_by_model.items():
        check_unique(table, CONTACT_KEY, f"{model} contacts with context")

    aggregate_rows: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    all_fixtures = sorted(tables["contacts"].fixture.unique())
    for model in MODELS:
        for population in POPULATIONS:
            for tolerance in TOLERANCES:
                contacts = contacts_by_model[model]
                contacts = contacts[(contacts.population == population) & (contacts.tolerance_base30 == tolerance)]
                proposals = proposals_by_model[model]
                proposals = proposals[(proposals.population == population) & (proposals.tolerance_base30 == tolerance)]
                rallies = rallies_by_model[model]
                rallies = rallies[(rallies.population == population) & (rallies.tolerance_base30 == tolerance)]
                for omission, omitted in OMISSIONS.items():
                    included = [fixture for fixture in all_fixtures if fixture not in omitted]
                    group_contacts = contacts[~contacts.fixture.isin(omitted)]
                    group_proposals = proposals[~proposals.fixture.isin(omitted)]
                    group_rallies = rallies[~rallies.fixture.isin(omitted)]
                    aggregate_rows.append(_aggregate_row(
                        model, population, tolerance, omission, group_contacts, group_proposals,
                        group_rallies, tables, fixture=None,
                    ))
                    if omission != "all47":
                        continue
                    for fixture in included:
                        video_contacts = contacts[contacts.fixture == fixture]
                        video_proposals = proposals[proposals.fixture == fixture]
                        video_rallies = rallies[rallies.fixture == fixture]
                        video_rows.append(_aggregate_row(
                            model, population, tolerance, omission, video_contacts, video_proposals,
                            video_rallies, tables, fixture=fixture,
                        ))
                        confusion_rows.extend(_confusion_rows(
                            model, population, tolerance, omission, fixture, video_contacts,
                        ))
    metrics = pd.DataFrame(aggregate_rows)
    outcomes = pd.DataFrame(video_rows)
    confusion = pd.DataFrame(confusion_rows)
    metrics = metrics.sort_values(["model", "population", "tolerance_base30", "omission"]).reset_index(drop=True)
    outcomes = outcomes.sort_values(
        ["model", "population", "tolerance_base30", "omission", "fixture"],
    ).reset_index(drop=True)
    confusion = confusion.sort_values(
        ["model", "population", "tolerance_base30", "omission", "fixture", "target_player", "predicted_player"],
    ).reset_index(drop=True)
    return metrics, outcomes, confusion


def _confusion_rows(
    model: str,
    population: str,
    tolerance: int,
    omission: str,
    fixture: int,
    contacts: pd.DataFrame,
) -> list[dict[str, object]]:
    """Return a complete far/near confusion grid for one video."""
    target_values = ("Far", "Near", "Unknown")
    prediction_values = ("Far", "Near", "Unassigned", "Missed prediction")
    target_side = contacts.target_side.map({"Top": "Far", "Bot": "Near"}).fillna("Unknown")
    predicted_side = contacts.predicted_side.map({"Top": "Far", "Bot": "Near"})
    predicted_side = predicted_side.where(contacts.matched, "Missed prediction").fillna("Unassigned")
    cells = pd.DataFrame({"target_player": target_side, "predicted_player": predicted_side})
    counts = cells.value_counts().to_dict()
    rows = []
    for target in target_values:
        for prediction in prediction_values:
            count = int(counts.get((target, prediction), 0))
            rows.append({
                "model": model,
                "population": population,
                "tolerance_base30": tolerance,
                "omission": omission,
                "omitted_videos": OMISSION_LABELS[omission],
                "fixture": fixture,
                "target_player": target,
                "predicted_player": prediction,
                "target_side": {"Far": "Top", "Near": "Bot", "Unknown": "Unknown"}[target],
                "predicted_side": {"Far": "Top", "Near": "Bot", "Unassigned": "Unassigned",
                                   "Missed prediction": "Missed prediction"}[prediction],
                "unassigned_prediction": prediction == "Unassigned",
                "missed_prediction": prediction == "Missed prediction",
                "contacts": count,
            })
    return rows


def _primary(metrics: pd.DataFrame, model: str, omission: str = "all47") -> pd.DataFrame:
    return metrics[
        (metrics.model == model)
        & (metrics.population == "retained")
        & (metrics.tolerance_base30 == 10)
        & (metrics.omission == omission)
    ].iloc[0]


def _plot_horizontal_rates(
    metrics: pd.DataFrame,
    filename: str,
    title: str,
    subtitle: str,
    fields: tuple[tuple[str, str, str, str], ...],
    denominator_field: str,
) -> None:
    groups = [(model, omission) for model in MODELS for omission in OMISSIONS]
    figure_height = 4.0 if len(fields) == 1 else 4.8
    figure, axis = plt.subplots(figsize=(10, figure_height))
    group_positions = np.arange(len(groups), dtype=float)
    bar_height = 0.27
    offsets = (-(len(fields) - 1) * bar_height / 1.15
               + np.arange(len(fields)) * bar_height * 1.75)
    for field_index, (field, count_field, label, colour) in enumerate(fields):
        for group_position, (model, omission) in zip(group_positions, groups, strict=True):
            row = metrics[
                (metrics.model == model)
                & (metrics.population == "retained")
                & (metrics.tolerance_base30 == 10)
                & (metrics.omission == omission)
            ].iloc[0]
            value = float(row[field])
            position = group_position + offsets[field_index]
            axis.barh(position, value, height=bar_height, color=colour, label=label if group_position == 0 else None)
            annotation = f"{int(row[count_field]):,}/{int(row[denominator_field]):,} ({value:.1f}%)"
            axis.text(value + 1.0, position, annotation, va="center", fontsize=8, clip_on=False)
    labels = [
        f"{'Final learned' if model == 'learned' else 'Ordinary heuristic'} · "
        f"{'All 47' if omission == 'all47' else 'Omit ' + OMISSION_LABELS[omission].replace(',', ' + ')}"
        for model, omission in groups
    ]
    axis.set_yticks(group_positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of labelled items meeting the condition (%)")
    axis.grid(axis="x", alpha=0.15)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=len(fields))
    figure.suptitle(title, x=0.08, y=0.995, ha="left", weight="bold", fontsize=13)
    figure.text(0.08, 0.88, subtitle, fontsize=9)
    figure.subplots_adjust(left=0.22, right=0.80, top=0.78, bottom=0.16)
    figure.savefig(FIGURES / filename, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_rallies(metrics: pd.DataFrame) -> None:
    _plot_horizontal_rates(
        metrics, "excluding_videos_rallies.png", "Whole-rally exact timing",
        "Cleaned labels · ±10 frames at 30 fps · every bar shows count/labelled-rally denominator and rate\n"
        "A result requires one containing single-rally clip with every contact timed; extra events prevent exactness. Player side is ignored.",
        (("timing_complete_rate_percent", "timing_complete_rallies", "Exact contact sequence", PURPLE),),
        "labelled_rallies",
    )


def plot_fully_correct(metrics: pd.DataFrame) -> None:
    _plot_horizontal_rates(
        metrics, "excluding_videos_fully_correct.png", "Fully correct labelled rallies",
        "Cleaned labels · ±10 frames at 30 fps · every bar shows count/labelled-rally denominator and rate\n"
        "A fully correct result requires the exact contact sequence and the correct near/far player at every contact.",
        (("fully_correct_rate_percent", "fully_correct_rallies", "Fully correct rally", BLUE),),
        "labelled_rallies",
    )


def plot_contacts(metrics: pd.DataFrame) -> None:
    _plot_horizontal_rates(
        metrics, "excluding_videos_contacts.png", "Labelled contact timing and player attribution",
        "Cleaned labels · ±10 frames at 30 fps · every bar shows count/labelled-contact denominator and rate\n"
        "A confirmed attribution requires both a timing match and the correct near/far player.",
        (
            ("timing_recall_percent", "matched_contacts", "Timing match", BLUE),
            ("attribution_confirmed_percent", "confirmed_contacts", "Timing + correct player", ORANGE),
        ),
        "labelled_contacts",
    )


def plot_serves(metrics: pd.DataFrame) -> None:
    _plot_horizontal_rates(
        metrics, "excluding_videos_serves.png", "Labelled serve timing and player attribution",
        "Cleaned labels · serve is the first labelled contact in each rally · ±10 frames at 30 fps\n"
        "Every bar shows count/labelled-serve denominator and rate. These are descriptive output counts.",
        (
            ("serve_timing_recall_percent", "matched_serves", "Timing match", BLUE),
            ("serve_attribution_confirmed_percent", "confirmed_serves", "Timing + correct player", ORANGE),
        ),
        "labelled_serves",
    )


def plot_video_pages(outcomes: pd.DataFrame) -> None:
    """Plot all 47 primary videos as two readable horizontal stacked pages."""
    data = outcomes[
        (outcomes.model == "learned")
        & (outcomes.population == "retained")
        & (outcomes.tolerance_base30 == 10)
        & (outcomes.omission == "all47")
    ].copy()
    data["matched_not_correct_contacts"] = data.matched_contacts - data.confirmed_contacts
    data["fully_correct_rate"] = data.fully_correct_rallies / data.labelled_rallies
    data = data.sort_values(["fully_correct_rate", "fixture"], ascending=[False, True]).reset_index(drop=True)
    if len(data) != 47:
        raise AssertionError(f"expected 47 primary learned videos, got {len(data)}")
    for page, page_rows in enumerate((data.iloc[:24], data.iloc[24:]), start=1):
        figure, axis = plt.subplots(figsize=(10, 9.2 if page == 1 else 8.9))
        positions = np.arange(len(page_rows))
        left = np.zeros(len(page_rows))
        for field, label in (("confirmed_contacts", "Matched + correct player"),
                             ("matched_not_correct_contacts", "Matched, player not confirmed"),
                             ("unmatched_labelled_hits", "Missed labelled contact")):
            counts = page_rows[field].to_numpy(dtype=float)
            denominator = page_rows.labelled_contacts.to_numpy(dtype=float)
            widths = 100 * counts / denominator
            axis.barh(positions, widths, left=left, color=COLOURS[
                "matchedcorrect" if field == "confirmed_contacts" else
                "matchednotcorrect" if field == "matched_not_correct_contacts" else "missed"
            ], label=label, height=0.74)
            for position, start, width_value, count in zip(positions, left, widths, counts, strict=True):
                if width_value >= 9:
                    axis.text(start + width_value / 2, position, f"{int(count):,}",
                              ha="center", va="center", color="white", fontsize=7.2)
            left += widths
        labels = [f"Video {int(row.fixture)}" for row in page_rows.itertuples(index=False)]
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 100)
        axis.set_xlabel("Share of labelled contacts (%) · each bar sums to 100%")
        axis.set_ylabel("Video ID")
        axis.grid(axis="x", alpha=0.12)
        for position, row in zip(positions, page_rows.itertuples(index=False), strict=True):
            axis.text(101.5, position, f"{int(row.fully_correct_rallies)}/{int(row.labelled_rallies)} rallies "
                      f"({row.fully_correct_rate * 100:.1f}%)", va="center", fontsize=8, clip_on=False)
        figure.suptitle(f"Per-video final learned contact outcomes ({page}/2)", x=0.08, y=0.995,
                        ha="left", weight="bold", fontsize=13)
        figure.text(0.08, 0.91,
                    "47 previously examined videos · cleaned labels · ±10 frames at 30 fps · sorted by fully correct rally rate\n"
                    "Matched + correct player means timing and attribution are both correct; unknown target sides remain in the middle segment.",
                    fontsize=9)
        axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=3)
        figure.subplots_adjust(left=0.13, right=0.82, top=0.86, bottom=0.12)
        figure.savefig(FIGURES / f"video_outcome_breakdown_{page}.png", dpi=170,
                       bbox_inches="tight", facecolor="white")
        plt.close(figure)


def verify_primary_totals(metrics: pd.DataFrame) -> None:
    learned = _primary(metrics, "learned")
    heuristic = _primary(metrics, "heuristic")
    expected_learned = {
        "labelled_contacts": 38218, "matched_contacts": 33716, "confirmed_contacts": 32667,
        "fully_correct_rallies": 1763,
    }
    expected_heuristic = {
        "labelled_contacts": 38218, "matched_contacts": 29206, "confirmed_contacts": 20204,
        "fully_correct_rallies": 4,
    }
    for row, expected in ((learned, expected_learned), (heuristic, expected_heuristic)):
        for field, value in expected.items():
            if int(row[field]) != value:
                raise AssertionError(f"primary {field}: got {row[field]}, expected {value}")


def write_outputs(metrics: pd.DataFrame, outcomes: pd.DataFrame, confusion: pd.DataFrame) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS / "exclusion_metrics.csv.gz", index=False)
    outcomes.to_csv(RESULTS / "video_outcome_breakdown.csv.gz", index=False)
    confusion.to_csv(RESULTS / "video_player_confusion.csv.gz", index=False)


def run() -> None:
    tables = load_tables()
    metrics, outcomes, confusion = build_tables(tables)
    verify_primary_totals(metrics)
    write_outputs(metrics, outcomes, confusion)
    plot_rallies(metrics)
    plot_fully_correct(metrics)
    plot_contacts(metrics)
    plot_serves(metrics)
    plot_video_pages(outcomes)
    print(metrics[
        ["model", "population", "tolerance_base30", "omission", "labelled_contacts", "matched_contacts",
         "confirmed_contacts", "fully_correct_rallies"]
    ].to_string(index=False))
    print(f"Wrote {len(metrics)} aggregate rows, {len(outcomes)} per-video rows, {len(confusion)} confusion rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
