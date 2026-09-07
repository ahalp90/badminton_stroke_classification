"""Plot coverage and judged correctness for the frozen score cutoffs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.contact_det_followup.scripts.prediction_io import read_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    development = read_json(ROOT / "results/broader_acceptance_policy.json.gz")
    broader = read_json(ROOT / "results/broader_result.json.gz")
    curves = (
        [row["summary"] for row in development["curve"]],
        broader["systems"]["combined"]["acceptance"]["curve"],
    )
    titles = ("Development: 32 videos, 2,850 sections", "Broader comparison: 47 videos, 3,982 sections")
    threshold = development["fallback_rule"]["threshold"]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4), sharex=True, sharey=True, layout="constrained")
    for axis, curve, title in zip(axes, curves, titles, strict=True):
        for tolerance, colour, marker in (("10", "#0072B2", "o"), ("5", "#D55E00", "s")):
            judged = sorted(
                (row for row in curve if row["by_tolerance"][tolerance]["judged_count"]),
                key=lambda row: row["accepted_count"],
            )
            coverage = [100 * row["coverage"] for row in judged]
            precision = [100 * row["by_tolerance"][tolerance]["judged_precision"] for row in judged]
            axis.plot(coverage, precision, marker=marker, color=colour, markersize=4, label=f"±{tolerance} frames")
            chosen = next(row for row in curve if row["threshold"] == threshold)
            value = chosen["by_tolerance"][tolerance]["judged_precision"]
            if value is not None:
                axis.scatter(100 * chosen["coverage"], 100 * value, s=100, marker=marker,
                             facecolors="none", edgecolors=colour, linewidths=1.6, zorder=3)
        axis.axhline(95, color="#555555", linestyle="--", linewidth=1, label="95% target")
        axis.set(title=title, xlabel="All sections accepted (%)", xlim=(0, 100), ylim=(0, 102))
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Correct among judgeable accepted sections (%)")
    axes[0].legend(loc="lower left", frameon=False)
    output = ROOT / "figures/broader_acceptance.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
