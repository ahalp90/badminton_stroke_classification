"""Show how complete-rally repairs and losses are distributed across videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_followup.scripts.prediction_io import read_json

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#0072B2"
ORANGE = "#D55E00"


def plot(result_path: Path, output_root: Path, detector_name: str) -> None:
    result = read_json(result_path)
    comparison = result["comparison_to_session_start"]["10"]
    videos = comparison["by_video"]
    fixtures = sorted(videos, key=int)
    repairs = np.asarray([len(videos[fixture]["repaired"]) for fixture in fixtures])
    losses = np.asarray([len(videos[fixture]["lost"]) for fixture in fixtures])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "followup_video_changes.json", {
        "source_result": result_path.name,
        "detector": detector_name,
        "reference": "Session-start later-contact detector",
        "population": "47 previously examined ShuttleSet22 videos; 3,982 sections",
        "tolerance_base30": 10,
        "videos": [
            {"video": fixture, "repaired": int(repaired), "lost": int(lost)}
            for fixture, repaired, lost in zip(fixtures, repairs, losses, strict=True)
        ],
    })
    positions = np.arange(len(fixtures))
    figure, axis = plt.subplots(figsize=(14, 4.2))
    axis.bar(positions, repairs, color=BLUE, label="Repaired")
    axis.bar(positions, -losses, color=ORANGE, label="Previously correct, now lost")
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(positions, fixtures, fontsize=8)
    axis.set_xlabel("Video ID — 47 previously examined ShuttleSet22 videos, 3,982 sections")
    axis.set_ylabel("Complete rallies repaired (+) or lost (−)")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_title(f"{detector_name} versus session start, ±10 base-30 frames")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(output_root / f"followup_video_changes.{extension}", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--detector-name", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "figures")
    arguments = parser.parse_args()
    plot(arguments.result, arguments.output_root, arguments.detector_name)
