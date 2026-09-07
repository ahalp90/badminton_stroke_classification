#!/usr/bin/env python3
"""Draw the report figures from the saved evaluation counts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

OUT = Path(__file__).resolve().parent
EVALUATION_ROOT = next(parent for parent in OUT.parents if parent.name == "annotator_wrapup_evaluation")

POPULATIONS = [
    "All 47\nhistorical",
    "Without video 15",
    "Without videos 15 + 53\nsensitivity",
]


def load_results() -> tuple[pd.DataFrame, pd.Series]:
    results = pd.read_csv(EVALUATION_ROOT / "results/exclusion_metrics.csv.gz")
    main_comparison = (results.population == "retained") & (results.tolerance_base30 == 10)
    results = results.loc[main_comparison]
    learned = results.loc[results.model == "learned"].set_index("omission")
    learned = learned.loc[["all47", "omit15", "omit15_53"]]
    heuristic_current = results.loc[(results.model == "heuristic") & (results.omission == "omit15")].iloc[0]
    return learned, heuristic_current


def percent(num: ArrayLike, den: ArrayLike) -> NDArray[np.float64]:
    return 100.0 * np.asarray(num, dtype=float) / np.asarray(den, dtype=float)


def annotate_bars(ax: Axes, bars: BarContainer, values: NDArray[np.float64], suffix: str = "%") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rally_correctness(learned: pd.DataFrame) -> None:
    exact = percent(learned.timing_complete_rallies.to_numpy(), learned.labelled_rallies.to_numpy())
    full = percent(learned.fully_correct_rallies.to_numpy(), learned.labelled_rallies.to_numpy())
    x = np.arange(len(POPULATIONS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.6))
    b1 = ax.bar(x - width / 2, exact, width, label="Exact contact sequence")
    b2 = ax.bar(x + width / 2, full, width, label="Exact sequence + correct players")
    ax.set_title(
        "Whole-rally correctness\n"
        "Cumulative exclusions: remove video 15, then video 53 only for sensitivity"
    )
    ax.set_ylabel("Share of labelled rallies (%)")
    ax.set_xticks(x, POPULATIONS)
    ax.set_ylim(0, 65)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, b1, exact)
    annotate_bars(ax, b2, full)
    save(fig, "rally_correctness.png")


def plot_contact_correctness(learned: pd.DataFrame) -> None:
    timing = percent(learned.matched_contacts.to_numpy(), learned.labelled_contacts.to_numpy())
    player = percent(learned.confirmed_contacts.to_numpy(), learned.labelled_contacts.to_numpy())
    x = np.arange(len(POPULATIONS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.6))
    b1 = ax.bar(x - width / 2, timing, width, label="Timing match")
    b2 = ax.bar(x + width / 2, player, width, label="Timing + correct player")
    ax.set_title(
        "Labelled contacts recovered\n"
        "Cumulative exclusions: remove video 15, then video 53 only for sensitivity"
    )
    ax.set_ylabel("Recall of labelled contacts (%)")
    ax.set_xticks(x, POPULATIONS)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, b1, timing)
    annotate_bars(ax, b2, player)
    save(fig, "contact_correctness.png")


def plot_review_queue(learned: pd.DataFrame) -> None:
    selected_correct = learned.selected_correct.to_numpy(dtype=int)
    selected_wrong = learned.selected_wrong.to_numpy(dtype=int)
    selected_unknown = learned.selected_unknown.to_numpy(dtype=int)
    x = np.arange(len(POPULATIONS))
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.bar(x, selected_correct, label="Known correct")
    ax.bar(x, selected_wrong, bottom=selected_correct, label="Known wrong")
    ax.bar(
        x,
        selected_unknown,
        bottom=selected_correct + selected_wrong,
        label="Labels cannot judge",
    )
    ax.set_title(
        "Clips kept by the unchanged review-queue rule\n"
        "Counts after cumulative video exclusions"
    )
    ax.set_ylabel("Selected clips")
    ax.set_xticks(x, POPULATIONS)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    totals = selected_correct + selected_wrong + selected_unknown
    for i, total in enumerate(totals):
        ax.text(i, total + 8, f"{total} total", ha="center", va="bottom", fontsize=9)
        ax.text(i, selected_correct[i] / 2, f"{selected_correct[i]} correct",
                ha="center", va="center", fontsize=9)
        ax.text(i, selected_correct[i] + selected_wrong[i] / 2,
                f"{selected_wrong[i]} wrong", ha="center", va="center", fontsize=9)
        ax.text(i, selected_correct[i] + selected_wrong[i] + selected_unknown[i] / 2,
                f"{selected_unknown[i]} unknown", ha="center", va="center", fontsize=8)
    save(fig, "review_queue.png")


def plot_misses_by_input_state(current: pd.Series) -> None:
    labels = ["Court rejected\nscene", "Court accepted;\nplayer missing", "Court accepted;\nboth players available"]
    values = [
        int(current.missed_court_rejected),
        int(current.missed_accepted_missing_pick),
        int(current.missed_accepted_both_picked),
    ]
    total = sum(values)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    bars = ax.bar(labels, values)
    ax.set_title(
        f"Where the {total:,} missed contacts occur\n"
        "Without video 15, cleaned labels, ±10 frames"
    )
    ax.set_ylabel("Missed labelled contacts")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        share = 100 * value / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 35,
            f"{value:,}\n({share:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(values) * 1.18)
    save(fig, "misses_by_input_state.png")


def plot_heuristic_vs_learned(current: pd.Series, heuristic: pd.Series) -> None:
    counts = np.array([int(heuristic.fully_correct_rallies), int(current.fully_correct_rallies)])
    rallies = int(current.labelled_rallies)
    values = percent(counts, rallies)
    labels = ["Ordinary heuristic", "Final learned output"]
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    bars = ax.bar(labels, values)
    ax.set_title(
        "Fully correct rallies after removing video 15\n"
        "Every labelled hit must match once and have the correct player"
    )
    ax.set_ylabel(f"Share of {rallies:,} labelled rallies (%)")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", alpha=0.25)
    for bar, value, count in zip(bars, values, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{count:,} rallies\n({value:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save(fig, "heuristic_vs_learned.png")


def main() -> None:
    learned, heuristic = load_results()
    current = learned.loc["omit15"]
    plot_rally_correctness(learned)
    plot_contact_correctness(learned)
    plot_review_queue(learned)
    plot_misses_by_input_state(current)
    plot_heuristic_vs_learned(current, heuristic)


if __name__ == "__main__":
    main()
