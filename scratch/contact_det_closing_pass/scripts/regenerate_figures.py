"""Rebuild the readable report figures from saved experiment results."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


def save(fig: Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    svg = out / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def progression(values: list[float], population: str, stem: str, out: Path) -> None:
    stages = [
        "Previous\nmodel",
        "Serve\nrepair",
        "Score possible\nsequences",
        "Add 1 missed contact\n(≥0.05 rally gain)",
        "Evaluate added\ncontact independently",
        "Correct rally\nstart/end",
    ]
    fig, ax = plt.subplots(figsize=(10.2, 4.7))
    bars = ax.bar(np.arange(len(stages)), values)
    ax.set_ylim(0, 60)
    ax.set_ylabel("Fully-correct rally recall (%)")
    ax.set_title(f"Fully-correct rally recall through the closing-pass refinements\n{population}")
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages, fontsize=8.6)
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.text(
        0.5,
        0.015,
        "Stages are cumulative. The ≥0.05 guard compares the whole edited rally; "
        "the next stage additionally evaluates the proposed added contact on its own.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, out, stem)


def high_confidence_selection(out: Path, exact: list[float], whole: list[float], counts: Mapping[str, Any]) -> None:
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    b1 = ax.bar(
        x - width / 2,
        exact,
        width,
        label="Every contact + player correct",
    )
    b2 = ax.bar(
        x + width / 2,
        whole,
        width,
        label="Contains exactly one whole rally",
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent")
    ax.set_title("High-confidence rally selection — trusted GT")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.text(
        0.5,
        0.01,
        f"A fixed ranking threshold keeps {counts['selected']:,} of {counts['proposals']:,} proposed clips; "
        f"{counts['judgeable']:,} are judgeable against trusted GT.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, out, "high_confidence_selection")


def whole_sequence_comparison(out: Path, values: list[int], proposals: int) -> None:
    labels = ["Previous model", "Score possible\nfinished sequences"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 250)
    ax.set_ylabel("Fully-correct proposals at ±10")
    ax.set_title(f"Eight-video comparison — {proposals:,} proposed rallies")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            f"{value} ({value / proposals * 100:.1f}%)",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    save(fig, out, "whole_sequence_comparison")


def broader_gain(out: Path, values: list[float]) -> None:
    labels = ["Previous model", "Serve repair", "Score possible\nfinished sequences"]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Fully-correct rally recall (%)")
    ax.set_title("First 47-video comparison — trusted GT")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    save(fig, out, "broader_gain")


def later_by_length(out: Path, before: list[int], after: list[int]) -> None:
    labels = ["1–5", "6–10", "11–20", "21+"]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.bar(x - width / 2, before, width, label="Before later-contact repair")
    ax.bar(x + width / 2, after, width, label="After later-contact repair")
    ax.set_ylabel("Fully-correct rallies")
    ax.set_title("Later-contact repair helps longer rallies most — trusted GT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Labelled contacts in rally")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    fig.tight_layout()
    save(fig, out, "later_by_length")


def final_followup(out: Path, values: list[float]) -> None:
    labels = [
        "One missed contact\n+ ≥0.05 rally guard",
        "Also evaluate added\ncontact independently",
        "Rally start/end\ncorrection only",
        "Recommended:\nboth",
        "Wider serve\nshortlist",
    ]
    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    bars = ax.bar(labels, values)
    ax.set_ylim(40, 54)
    ax.set_ylabel("Fully-correct rally recall (%)")
    ax.set_title("Final detector refinements — trusted GT")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    save(fig, out, "final_followup")


def contact_prf(out: Path, timing: list[float], timing_player: list[float]) -> None:
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    b1 = ax.bar(x - width / 2, timing, width, label="Timing")
    b2 = ax.bar(x + width / 2, timing_player, width, label="Timing + correct player")
    ax.set_ylim(70, 92)
    ax.set_ylabel("Percent")
    ax.set_title("Final contact detection — trusted GT, ±10 frames")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.25,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    save(fig, out, "contact_prf")


def contact_recovery(out: Path, values: list[float]) -> None:
    labels = [
        "Non-serve\ntiming",
        "Non-serve\n+ player",
        "Serve\ntiming",
        "Serve\n+ player",
    ]

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    bars = ax.bar(labels, values)
    ax.set_ylim(70, 92)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Serve versus later-contact recovery — trusted GT, ±10 frames")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    save(fig, out, "contact_recovery")


def contact_progression(out: Path, timing_f1: list[float], player_f1: list[float]) -> None:
    stages = [
        "Score possible\nsequences",
        "+ one missed\nlater contact",
        "Recommended\ndetector",
    ]
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.plot(x, timing_f1, marker="o", label="Timing F1")
    ax.plot(x, player_f1, marker="o", label="Timing + player F1")
    ax.set_ylim(75, 86)
    ax.set_ylabel("F1 (%)")
    ax.set_title("Contact-level progress — trusted GT")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    for xx, value in zip(x, timing_f1, strict=True):
        ax.text(xx, value + 0.2, f"{value:.1f}%", ha="center", fontsize=8.5)
    for xx, value in zip(x, player_f1, strict=True):
        ax.text(xx, value - 0.55, f"{value:.1f}%", ha="center", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "contact_progression")


def near_miss_errors(out: Path, values: list[int]) -> None:
    labels = [
        "Exact annotation\ncorrect",
        "Whole rally,\nlocal contact errors",
        "Fundamental\nrally problem",
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Judgeable high-confidence clips")
    ax.set_title(f"Structure of the {sum(values):,} high-confidence clips with trusted GT")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 8,
            str(value),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    save(fig, out, "near_miss_errors")


def selected_errors(out: Path, values: list[int], wrong: int) -> None:
    labels = [
        "Extra contact(s)",
        "Misses serve",
        "Misses later contact",
        "Wrong/missing player",
        "Not one whole rally",
    ]

    fig, ax = plt.subplots(figsize=(8.7, 5.0))
    bars = ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_xlabel("High-confidence proposals")
    ax.set_title(f"Why {wrong:,} high-confidence clips fail exact annotation\n(categories overlap)")
    ax.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    save(fig, out, "selected_errors")


def closing_checks(out: Path, edge_gain: int, before: int, after: int, repairable: int, wrong: int) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Closing checks after the detector recommendation", fontsize=18, pad=10)

    cards = [
        (
            0.3,
            "Independent edge padding",
            f"{edge_gain:,} fully-correct-rally gain",
            "Keep existing boundary rule",
        ),
        (
            4.2,
            "Corrected chooser targets",
            f"{before:,} → {after:,} on 47 videos",
            "Reject refit",
        ),
        (
            8.1,
            "High-confidence\nsmall-edit census",
            f"{repairable:,} / {wrong:,} wrong clips\nhave repair headroom",
            "Opportunity, not achieved gain",
        ),
    ]
    for x, title, middle, bottom in cards:
        ax.add_patch(Rectangle((x, 0.45), 3.3, 2.75, fill=False, linewidth=1.5))
        ax.text(
            x + 1.65,
            2.45,
            title,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.text(x + 1.65, 1.65, middle, ha="center", va="center", fontsize=10.5)
        ax.text(x + 1.65, 0.95, bottom, ha="center", va="center", fontsize=9.8)
    fig.tight_layout()
    save(fig, out, "closing_checks")


def promising_opportunities(out: Path, values: list[int]) -> None:
    labels = [
        "Right-rally\ncontact cleanup",
        "Good serve candidate\nalready present",
        "Missing later contacts\nwith no candidate row",
        "High-confidence clips\nwithout exact GT",
    ]

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Diagnostic count (different denominators)")
    ax.set_title("Open questions after the contact-detector closing pass")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 18,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.text(
        0.5,
        0.01,
        "Counts describe different diagnostics and should not be compared as rates.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "promising_opportunities")


def document_map(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 7.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("How to read the closing-pass documentation", fontsize=19, pad=12)

    boxes = {
        "README.md": (
            0.4,
            4.8,
            3.2,
            1.1,
            "Main result, system summary,\nconfidence/recall trade-off",
        ),
        "serve_tables.md": (
            4.4,
            4.8,
            3.2,
            1.1,
            "Compact source of truth\n+ reproduction",
        ),
        "serve_and_acceptance.md": (
            8.4,
            4.8,
            3.2,
            1.1,
            "Deployment: contacts, serves,\nhigh-confidence rally selection",
        ),
        "contact_performance.md": (
            0.4,
            2.9,
            3.2,
            1.1,
            "Contact + serve\nperformance details",
        ),
        "experiment_lineage.md": (
            4.4,
            2.9,
            3.2,
            1.1,
            "Canonical experiment map\n+ internal code names",
        ),
        "promising_leads.md": (
            8.4,
            2.9,
            3.2,
            1.1,
            "Live research backlog only:\nwhat still deserves investigation",
        ),
        "stage reports": (
            2.2,
            1.0,
            3.2,
            1.1,
            "serve → sequences → later contact\n→ final refinements",
        ),
        "last_followups.md": (
            6.6,
            1.0,
            3.2,
            1.1,
            "Completed closing checks,\ndead ends + residual headroom",
        ),
    }

    for name, (x, y, w, h, subtitle) in boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.5))
        ax.text(
            x + w / 2,
            y + h * 0.64,
            name,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            x + w / 2,
            y + h * 0.31,
            subtitle,
            ha="center",
            va="center",
            fontsize=10,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "lw": 1.25},
        )

    arrow(3.6, 5.35, 4.4, 5.35)
    arrow(7.6, 5.35, 8.4, 5.35)
    arrow(2.0, 4.8, 2.0, 4.0)
    arrow(6.0, 4.8, 6.0, 4.0)
    arrow(10.0, 4.8, 10.0, 4.0)
    arrow(6.0, 2.9, 3.8, 2.1)
    arrow(6.0, 2.9, 8.2, 2.1)

    fig.tight_layout()
    save(fig, out, "document_map")


def experiment_lineage(
    out: Path, stages: Mapping[str, int], counts: Mapping[str, Any], changes: Mapping[str, int]
) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 9.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Contact-detector closing-pass lineage", fontsize=18, pad=12)

    steps = [
        (8.65, "Previous model", f"{stages['original']:,} fully correct rallies"),
        (7.55, "Serve repair", f"{stages['original']:,} → {stages['opening']:,}"),
        (6.45, "Score possible finished sequences", f"{stages['opening']:,} → {stages['combined']:,}"),
        (
            5.35,
            "Add one missed later contact",
            f"{stages['combined']:,} → {stages['later']:,}; keep ≥0.05 whole-rally guard",
        ),
        (
            4.25,
            "Evaluate added contact independently",
            f"{stages['later']:,} → {stages['local']:,} by itself",
        ),
        (
            3.15,
            "Correct rally start/end",
            f"recommended combination → {stages['recommended']:,}",
        ),
        (
            2.05,
            "Confidence ranking on final detector",
            f"{counts['selected']:,} / {counts['proposals']:,} kept at fixed threshold",
        ),
        (
            0.95,
            "Closing checks",
            "no cheap follow-up displaced the recommendation",
        ),
    ]
    for y, title, subtitle in steps:
        ax.add_patch(Rectangle((0.8, y - 0.38), 7.7, 0.76, fill=False, linewidth=1.4))
        ax.text(
            4.65,
            y + 0.09,
            title,
            ha="center",
            va="center",
            fontsize=11.2,
            fontweight="bold",
        )
        ax.text(4.65, y - 0.18, subtitle, ha="center", va="center", fontsize=9.3)

    for index in range(len(steps) - 1):
        ax.annotate(
            "",
            xy=(4.65, steps[index + 1][0] + 0.39),
            xytext=(4.65, steps[index][0] - 0.39),
            arrowprops={"arrowstyle": "->", "lw": 1.1},
        )

    ax.add_patch(Rectangle((9.0, 2.72), 2.35, 0.9, fill=False, linewidth=1.2))
    ax.text(
        10.175,
        3.34,
        "Saved alternative",
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
    )
    ax.text(
        10.175,
        2.98,
        f"Wider serve shortlist\n{stages['early']:,}; {changes['repaired']} repairs / {changes['lost']} losses",
        ha="center",
        va="center",
        fontsize=8.2,
    )
    ax.annotate(
        "",
        xy=(9.0, 3.15),
        xytext=(8.5, 3.15),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    fig.tight_layout()
    save(fig, out, "experiment_lineage")


def read_result(results: Path, name: str) -> Any:
    with gzip.open(results / name, "rt") as source:
        return json.load(source)


def stage_recall(result: Mapping[str, Any], stages: Sequence[str], population: str = "retained") -> list[float]:
    values = []
    for stage in stages:
        counts = result["stages"][stage][population]["10"]
        values.append(100 * counts["unique_complete"] / counts["labelled_rallies"])
    return values


def prf_values(correct: int, predicted: int, labelled: int) -> list[float]:
    return [100 * correct / predicted, 100 * correct / labelled, 200 * correct / (predicted + labelled)]


def selection_metrics(result: Mapping[str, Any], key: str, population: str = "retained") -> list[float]:
    selected = result["selected"][population]["10"]
    predicted = selected["proposals"] - (selected["unknown"] if population == "retained" else 0)
    return prf_values(selected[key], predicted, selected["labelled_rallies"])


def historical_figures(result: Mapping[str, Any], results: Path, out: Path, counts: Mapping[str, Any]) -> None:
    whole = read_result(results, "whole_rally_result.json.gz")
    comparison = whole["validation"]["opening_sides_and_physics"]["evaluation"]["10"]["paired_fixed_side"]
    whole_sequence_comparison(
        out, [comparison["correct_before"], comparison["correct_after"]], comparison["sections_after"]
    )

    later = read_result(results, "later/later_broader_result.json.gz")
    lengths = later["comparison_to_frozen_combined"]["10"]["by_labelled_length_after_prediction"]
    length_order = ("1–5", "6–10", "11–20", "21+")
    later_by_length(
        out,
        [lengths[length]["correct_before"] for length in length_order],
        [lengths[length]["correct_after"] for length in length_order],
    )
    broader = read_result(results, "broader_result.json.gz")
    contact_stages = [
        broader["systems"]["combined"]["contacts"]["10"]["fixed_side"]["total"],
        later["contacts"]["10"]["total"],
        result["contacts"]["retained"]["10"],
    ]
    timing_f1, player_f1 = [], []
    for stage in contact_stages:
        denominator = stage["predicted"] + stage["labelled"]
        timing_f1.append(200 * stage["matched"] / denominator)
        player_f1.append(200 * stage["side_correct"] / denominator)
    contact_progression(out, timing_f1, player_f1)

    breakdown = read_result(results, "serve_followups/acceptance_breakdown.json.gz")
    policy = breakdown["broader"]["variants"]["gap"]["policies"]["comparison"]
    errors = policy["by_tolerance"]["10"]["accepted_wrong_error_categories"]
    selected_errors(
        out,
        [errors[key] for key in ("extras", "missed_serve", "later_miss", "side_errors", "section_problem")],
        errors["wrong_rows"],
    )

    diagnosis = read_result(results, "serve_followups/development_diagnosis.json.gz")
    census = read_result(results, "missed_candidate_census.json.gz")
    absent_later = sum(
        not row["is_first"] and row["category"] == "no_nearby_frozen_row"
        for row in census["tolerances"]["10"]["missed"]
    )
    selected = result["selected"]["retained"]["10"]
    near_misses = selected["unique_contained"] - selected["unique_complete"]
    promising_opportunities(
        out,
        [
            near_misses,
            diagnosis["by_tolerance"]["10"]["missed_serves"]["counts"]["shortlisted_not_chosen"],
            absent_later,
            selected["unknown"],
        ],
    )

    edge = read_result(results, "last_followups/edge_padding.json.gz")["populations"]["retained"]["10"]["paired"]
    padded = read_result(results, "last_followups/padded_fit_broader.json.gz")["populations"]["retained"]["10"][
        "paired"
    ]
    repairs = read_result(results, "last_followups/selected_repairs.json.gz")["summary"]
    closing_checks(
        out,
        edge["correct_after"] - edge["correct_before"],
        padded["correct_before"],
        padded["correct_after"],
        repairs["wrong_repairable"],
        repairs["baseline"]["wrong"],
    )
    early = read_result(results, "followups/early_boundary_broader_result_fixed_membership.json.gz")
    paired = early["comparison_to_local_boundary"]["10"]["paired"]
    changes = {"repaired": len(paired["repaired"]), "lost": len(paired["lost"])}
    stage_counts = {
        stage: populations["retained"]["10"]["unique_complete"] for stage, populations in result["stages"].items()
    }
    experiment_lineage(out, stage_counts, counts, changes)
    document_map(out)


def regenerate_metric_figures(
    result: Mapping[str, Any],
    out: Path = OUT,
    results: Path = ROOT / "results",
) -> None:
    """Render the complete report bundle, using the supplied summary for final metrics."""
    out.mkdir(parents=True, exist_ok=True)
    stages = ["original", "opening", "combined", "later", "local", "recommended"]
    for population, title, stem in (
        ("retained", "Trusted GT", "system_progression_trusted"),
        ("all_gt", "All source labels", "system_progression_all_gt"),
    ):
        labelled = result["stages"]["recommended"][population]["10"]["labelled_rallies"]
        progression(stage_recall(result, stages, population), f"{title} — {labelled:,} rallies", stem, out)
    broader_gain(out, stage_recall(result, ["original", "opening", "combined"]))
    final_followup(out, stage_recall(result, ["later", "local", "boundaries", "recommended", "early"]))

    selected = result["selected"]["retained"]["10"]
    counts = {
        "selected": selected["proposals"],
        "judgeable": selected["proposals"] - selected["unknown"],
        "proposals": result["stages"]["recommended"]["retained"]["10"]["proposals"],
    }
    high_confidence_selection(
        out, selection_metrics(result, "unique_complete"), selection_metrics(result, "unique_contained"), counts
    )
    near_miss_errors(
        out,
        [
            selected["unique_complete"],
            selected["unique_contained"] - selected["unique_complete"],
            counts["judgeable"] - selected["unique_contained"],
        ],
    )

    contacts = result["contacts"]["retained"]["10"]
    contact_prf(
        out,
        prf_values(contacts["matched"], contacts["predicted"], contacts["labelled"]),
        prf_values(contacts["side_correct"], contacts["predicted"], contacts["labelled"]),
    )
    nonserve_labels = contacts["labelled"] - contacts["labelled_serves"]
    contact_recovery(
        out,
        [
            100 * (contacts["matched"] - contacts["serve_matched"]) / nonserve_labels,
            100 * (contacts["side_correct"] - contacts["serve_side_correct"]) / nonserve_labels,
            100 * contacts["serve_matched"] / contacts["labelled_serves"],
            100 * contacts["serve_side_correct"] / contacts["labelled_serves"],
        ],
    )
    historical_figures(result, results, out, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    regenerate_metric_figures(read_result(args.results, "metric_summary.json.gz"), args.output_dir, args.results)


if __name__ == "__main__":
    main()
