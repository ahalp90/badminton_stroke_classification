"""Plot paired complete-rally repairs and losses from the saved comparison."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("summary_whole", "summary_opening", "physical_whole", "physical_opening")
LABELS = ("Summary / whole rally", "Summary / opening", "Physical / whole rally", "Physical / opening")


def main() -> None:
    with gzip.open(ROOT / "results/start_comparison_result.json.gz", "rt") as source:
        result = json.load(source)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    populations = (
        ("development_oof_descriptive", "32 development videos\nDescriptive grouped predictions; 2,850 sections", 802),
        ("validation_frozen_choices", "8 reused validation videos\nExcluded from fitting; 677 sections", 182),
    )
    positions = np.arange(len(VARIANTS))
    for axis, (population, title, baseline) in zip(axes, populations, strict=True):
        variants = result[population]["variants"]
        pairs = [variants[name]["evaluation"]["10"]["paired_fixed_side"] for name in VARIANTS]
        repaired = [len(pair["repaired"]) for pair in pairs]
        lost = [len(pair["lost"]) for pair in pairs]
        repair_bars = axis.barh(positions - 0.18, repaired, height=0.34, color="#0072B2", label="Rallies repaired")
        loss_bars = axis.barh(positions + 0.18, lost, height=0.34, color="#E69F00", hatch="//", label="Previously correct rallies lost")
        axis.bar_label(repair_bars, padding=3, fontsize=10)
        axis.bar_label(loss_bars, padding=3, fontsize=10)
        axis.set_title(title, fontsize=11, pad=12)
        axis.set_xlabel(f"Rally count (baseline: {baseline} correct)")
        axis.set_xlim(0, max(repaired) * 1.2)
        axis.set_yticks(positions, LABELS)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    figure.suptitle("Opening targets recover more rallies, with development losses\nFixed side vote; ±10 frames on a 30 fps clock", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=2, frameon=False)
    figure.tight_layout(rect=(0, 0.09, 1, 0.88))
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    figure.savefig(directory / "paired_repairs.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
