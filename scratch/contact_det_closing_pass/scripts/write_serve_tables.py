"""Write readable tables and a timing plot from the saved serve recount."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.contact_det_closing_pass.scripts.run_serve_followups import (
    OUTPUT,
    ROOT,
    VARIANTS,
)
from scratch.contact_det_followup.scripts.prediction_io import read_json

NAMES = {
    "original": "Original contacts", "preceding": "Preceding detector", "guarded_only": "Guarded edges only",
    "recommended": "Local insertion + guarded edges", "wider_early": "Wider early shortlist + edges",
}


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator:,} / {denominator:,} ({100 * numerator / denominator:.1f}%)" if denominator else f"{numerator} / 0 (—)"


def table(headers: tuple[str, ...], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_population(population: str, payload: dict) -> str:
    timing, attribution, starts, start_sides = [], [], [], []
    for variant in VARIANTS:
        for tolerance in ("10", "5"):
            counts = payload["variants"][variant][tolerance]["total"]
            prefix = [NAMES[variant], f"±{tolerance}"]
            timing.append(prefix + [
                ratio(counts["matched"], counts["firsts"]),
                ratio(counts["raw_joint_correct"], counts["known_side_firsts"]),
                ratio(counts["joint_correct"], counts["known_side_firsts"]),
            ])
            for side in ("raw", "final"):
                correct, wrong = counts[f"{side}_correct"], counts[f"{side}_wrong"]
                missing, unknown = counts[f"{side}_missing_prediction"], counts[f"{side}_missing_label"]
                attribution.append(prefix + [side, f"{correct:,} / {wrong:,} / {missing:,} / {unknown:,}",
                                             ratio(correct, correct + wrong), ratio(correct + wrong, counts["matched"])])
            starts.append(prefix + [
                ratio(counts["timing_correct_starts"], counts["all_starts"]),
                ratio(counts["timing_correct_starts"], counts["judgeable_timing_starts"]),
                str(counts["later_hit"]), str(counts["extra_leading"]), str(counts["unknown"]), str(counts["empty"]),
            ])
            start_sides.append(prefix + [
                ratio(counts["raw_joint_correct_starts"], counts["all_starts"]),
                ratio(counts["joint_correct_starts"], counts["all_starts"]),
                ratio(counts["joint_correct_starts"], counts["judgeable_side_starts"]),
            ])
    text = f"## {population.capitalize()}\n\n"
    text += "### Find the serve anywhere in the full stream\n\n"
    text += table(("Detector", "Tolerance", "Serve timing / all retained starts", "Timing + raw side / known-side starts",
                   "Timing + final side / known-side starts"), timing)
    text += "\n### Identify the server among timing-matched serves\n\n"
    text += table(("Detector", "Tolerance", "Side answer", "Correct / wrong / missing prediction / missing label",
                   "Correct / answered", "Answered / matched serves"), attribution)
    text += "\n### Start the proposed output at the serve\n\n"
    text += table(("Detector", "Tolerance", "Correct / all nonempty starts", "Correct / judgeable starts",
                   "Later hit", "Extra leading", "Unknown", "Empty sections"), starts)
    text += "\n### Start at the serve and identify its server\n\n"
    text += table(("Detector", "Tolerance", "Raw timing + side / all starts", "Final timing + side / all starts",
                   "Final timing + side / judgeable starts"), start_sides)
    return text


def save_csv(payloads: dict) -> None:
    rows = []
    for population, payload in payloads.items():
        for variant, tolerances in payload["variants"].items():
            for tolerance, result in tolerances.items():
                for video in result["by_video"]:
                    rows.append({"population": population, "detector": variant, "tolerance_base30": tolerance, **video})
    fields = sorted({key for row in rows for key in row})
    with gzip.open(OUTPUT / "serve_per_video.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timing_plot(payloads: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, (population, payload) in zip(axes, payloads.items(), strict=True):
        result = payload["variants"]["recommended"]["10"]
        offsets = [row["timing_error_base30"] for row in result["serve_rows"] if row["timing_matched"]]
        counts = result["total"]
        axis.hist(offsets, bins=np.arange(-10.5, 11.5), color="#3174a8", edgecolor="white")
        axis.axvline(-5, color="#b46923", linestyle="--")
        axis.axvline(5, color="#b46923", linestyle="--")
        axis.set(title=f"{population.capitalize()}: {counts['matched']:,} matched, {counts['firsts'] - counts['matched']:,} missed",
                 xlabel="Predicted − labelled serve time (base-30 frames)", ylabel="Matched serves")
        axis.set_xlim(-11, 11)
    figure.suptitle("Recommended detector: serve timing at ±10; dashed lines mark ±5")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run() -> None:
    payloads = {population: read_json(OUTPUT / f"{population}_serves.json.gz") for population in ("development", "broader")}
    text = "# Serve discovery and server attribution\n\n"
    headline = payloads["broader"]["variants"]["recommended"]["10"]["total"]
    text += (
        f"**The recommended detector finds {ratio(headline['matched'], headline['firsts'])} labelled serves at ±10.** "
        f"It both finds the serve and identifies the server in {ratio(headline['joint_correct'], headline['known_side_firsts'])} "
        f"retained rallies. Among nonempty proposals, {ratio(headline['timing_correct_starts'], headline['all_starts'])} "
        f"start at the serve, and {ratio(headline['joint_correct_starts'], headline['all_starts'])} also name the right server. "
        f"The other {headline['empty']} proposed sections are empty.\n\n"
    )
    text += (
        "These tables recount saved predictions. The serve is the first labelled contact of each retained rally. "
        "The proposed start is the first event of a nonempty section. Matching uses the full video contact stream "
        "once at each tolerance; it does not match against serve labels alone. Tolerances use a 30 fps clock and "
        "scale once to source fps. Raw sides are wrist/net guesses; final sides use the existing alternating-sequence vote.\n\n"
        "Development contains 32 grouped videos. The broader comparison contains 47 previously examined videos. "
        "Old cached detector scores retain cross-group dependence; these are not fresh independent test estimates.\n\n"
        "Unmatched starts inside an unambiguous retained rally's contact envelope count as extra leading events. "
        "Unmatched starts outside that support remain unknown. Unknowns stay in the all-start denominator. "
        "Empty sections are listed separately. Missing predicted sides are failures; missing label sides are unknown.\n\n"
        "- [Development](#development)\n- [Broader comparison](#broader)\n- [Serve timing](#serve-timing)\n\n"
    )
    for population, payload in payloads.items():
        text += write_population(population, payload) + "\n"
    text += "## Serve timing\n\n![Serve timing errors and missed serves for the recommended detector.](figures/serve_timing.png)\n\n"
    text += "[Per-video counts](results/serve_followups/serve_per_video.csv.gz) accompany the full saved rows and identity comparisons.\n"
    (ROOT / "serve_tables.md").write_text(text, encoding="utf-8")
    save_csv(payloads)
    timing_plot(payloads, ROOT / "figures/serve_timing.png")


if __name__ == "__main__":
    run()
