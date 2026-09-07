"""Write saved chosen-acceptance summaries, per-video rows and one figure."""

from __future__ import annotations

import csv
import gzip
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.contact_det_closing_pass.scripts.acceptance_breakdown import summarise
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_followup.scripts.prediction_io import read_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/serve_followups"
SOURCE_MANIFEST = ROOT.parents[1] / "configs/shuttleset22/sources.toml"
TOLERANCES = ("10", "5")
VARIANTS = ("base", "gap")
COLOURS = {"base": "#0072B2", "gap": "#D55E00"}
ACCEPTED_SERVE_FIELDS = (
    "accepted_starts", "timing_correct_starts", "raw_joint_correct_starts", "joint_correct_starts",
    "judgeable_timing_starts", "judgeable_side_starts", "accepted_later_hit", "accepted_extra_leading",
    "accepted_unknown", "accepted_empty", "raw_correct", "raw_wrong", "raw_missing_prediction",
    "raw_missing_label", "final_correct", "final_wrong", "final_missing_prediction", "final_missing_label",
)
CSV_FIELDS = (
    "population", "fixture", "source", "variant", "policy", "tolerance",
    "all_correct", "all_wrong", "all_unknown", "accepted_correct", "accepted_wrong",
    "accepted_unknown", "rejected_correct", "accepted_count", "population_count",
    "accepted_serve_timing_matched", "accepted_serve_timing_denominator",
    "accepted_serve_joint_correct", "accepted_serve_joint_denominator",
    *(f"serve_{field}" for field in ACCEPTED_SERVE_FIELDS),
)


def _load_population(population: str) -> tuple[dict[str, Any], dict[str, Any]]:
    acceptance = read_json(OUTPUT / f"chosen_acceptance_{population}.json.gz")
    serve = read_json(OUTPUT / f"{population}_serves.json.gz")
    recommended = serve["variants"]["recommended"]
    sections = {tolerance: recommended[tolerance]["sections"] for tolerance in TOLERANCES}
    policies = acceptance["policies"] if population == "development" else acceptance["frozen_policies"]
    breakdown = summarise(acceptance["rows"], sections, policies)
    return acceptance, breakdown


def _source_names() -> dict[str, str]:
    with SOURCE_MANIFEST.open("rb") as handle:
        payload = tomllib.load(handle)
    return {str(row["id"]): str(row["video"]) for row in payload["videos"]}


def _outcome_counts(summary: Mapping[str, Any], partition: str, tolerance: str) -> dict[str, int]:
    outcomes = summary["by_tolerance"][tolerance]["outcomes"][partition]
    return {
        "correct": int(outcomes["correct"]),
        "wrong": int(outcomes["wrong"]),
        "unknown": int(outcomes["unjudgeable"]),
    }


def _serve_counts(accepted: Mapping[str, Any], variant: str, policy: str, tolerance: str) -> dict[str, dict[str, Any]]:
    rows = accepted["accepted_serves"][variant][policy][tolerance]["by_video"]
    return {str(row["fixture"]): row for row in rows}


def _csv_rows(
    populations: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    source_names = _source_names()
    output: list[dict[str, Any]] = []
    for population, (acceptance, breakdown) in populations.items():
        for variant, variant_summary in breakdown["variants"].items():
            for policy, policy_summary in variant_summary["policies"].items():
                serve_by_tolerance = {
                    tolerance: _serve_counts(acceptance, variant, policy, tolerance)
                    for tolerance in TOLERANCES
                }
                for fixture, video_summary in policy_summary["by_video"].items():
                    source = fixture if population == "development" else source_names[fixture]
                    for tolerance in TOLERANCES:
                        all_counts = _outcome_counts(video_summary, "all", tolerance)
                        accepted_counts = _outcome_counts(video_summary, "accepted", tolerance)
                        rejected_counts = _outcome_counts(video_summary, "rejected", tolerance)
                        serve = serve_by_tolerance[tolerance][fixture]
                        output.append({
                            "population": population, "fixture": fixture, "source": source,
                            "variant": variant, "policy": policy, "tolerance": tolerance,
                            "all_correct": all_counts["correct"], "all_wrong": all_counts["wrong"],
                            "all_unknown": all_counts["unknown"],
                            "accepted_correct": accepted_counts["correct"],
                            "accepted_wrong": accepted_counts["wrong"],
                            "accepted_unknown": accepted_counts["unknown"],
                            "rejected_correct": rejected_counts["correct"],
                            "accepted_count": int(video_summary["accepted_count"]),
                            "population_count": int(video_summary["population_count"]),
                            "accepted_serve_timing_matched": int(serve["accepted_matched"]),
                            "accepted_serve_timing_denominator": int(serve["firsts"]),
                            "accepted_serve_joint_correct": int(serve["joint_correct"]),
                            "accepted_serve_joint_denominator": int(serve["all_known_gt"]),
                            **{f"serve_{field}": int(serve[field]) for field in ACCEPTED_SERVE_FIELDS},
                        })
    return output


def _write_csv(rows: list[dict[str, Any]]) -> None:
    path = OUTPUT / "acceptance_per_video.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curve(axis: Any, curve: list[Mapping[str, Any]], label: str, colour: str) -> None:
    points = [row for row in curve if row["by_tolerance"]["10"]["verified_correct_share_allaccepted"] is not None]
    points.sort(key=lambda row: float(row["coverage"]))
    axis.plot(
        [100 * float(row["coverage"]) for row in points],
        [100 * float(row["by_tolerance"]["10"]["verified_correct_share_allaccepted"]) for row in points],
        marker="o", markersize=4, color=colour, label=label,
    )


def _plot(populations: Mapping[str, tuple[dict[str, Any], dict[str, Any]]]) -> None:
    development = populations["development"][0]
    broader = populations["broader"][0]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True, layout="constrained")
    for variant in VARIANTS:
        _plot_curve(
            axes[0],
            development["curves"][variant],
            "Base features" if variant == "base" else "Base + gap evidence",
            COLOURS[variant],
        )
    axes[0].set_title("Development: 32 grouped videos")
    for variant in VARIANTS:
        metrics = broader["accepted_metrics"].get(variant, {})
        points = []
        for policy, summary in metrics.items():
            value = summary["by_tolerance"]["10"]["verified_correct_share_allaccepted"]
            if value is not None:
                points.append((float(summary["coverage"]), float(value), policy))
        axes[1].scatter(
            [100 * point[0] for point in points], [100 * point[1] for point in points],
            color=COLOURS[variant], label=f"{variant} frozen policies", s=28,
        )
        for coverage, value, policy in points:
            summary = metrics[policy]
            correct = summary["by_tolerance"]["10"]["counts"]["correct"]
            label = "Base" if variant == "base" else "With gap evidence"
            offset = (-12, -35) if variant == "base" else (12, 20)
            axes[1].annotate(
                f"{label}: {correct}/{summary['accepted_count']}\n{100 * value:.1f}% verified correct",
                (100 * coverage, 100 * value), xytext=offset, textcoords="offset points",
                ha="right" if variant == "base" else "left", fontsize=9, color=COLOURS[variant],
                arrowprops={"arrowstyle": "-", "color": COLOURS[variant], "lw": 0.8},
            )
    axes[1].set_title("Broader: 47 previously examined videos")
    for axis in axes:
        axis.axhline(95, color="#555555", linestyle="--", linewidth=1, label="95% reference")
        axis.set(xlim=(0, 45), ylim=(0, 102), xlabel="Accepted / all proposed sections (%)")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Verified correct / all accepted (%) at ±10 base-30 frames")
    axes[0].legend(loc="lower left", frameon=False)
    figure.suptitle("Acceptance on local insertion + guarded edges; broader points use frozen thresholds")
    path = ROOT / "figures/chosen_acceptance.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run() -> None:
    populations = {population: _load_population(population) for population in ("development", "broader")}
    write_json(OUTPUT / "acceptance_breakdown.json.gz", {
        population: breakdown for population, (_acceptance, breakdown) in populations.items()
    })
    _write_csv(_csv_rows(populations))
    _plot(populations)


if __name__ == "__main__":
    run()
