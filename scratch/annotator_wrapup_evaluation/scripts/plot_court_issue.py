"""Draw numbered court corners before and after the two failed corrections."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#0072B2"
ORANGE = "#C65D00"
PURPLE = "#7651A8"
NUMBER_KEY = "Where each corner belongs: 0 top-left · 1 top-right · 2 bottom-right · 3 bottom-left"


def draw_outline(axis: Axes, pixels: np.ndarray, corners: np.ndarray, colours: list[str], title: str) -> None:
    axis.imshow(pixels)
    polygon = np.vstack([corners, corners[0]])
    axis.plot(polygon[:, 0], polygon[:, 1], color=colours[0], linewidth=2.5)
    for number, (position, colour) in enumerate(zip(corners, colours, strict=True)):
        axis.text(*position, str(number), ha="center", va="center", fontsize=15, weight="bold",
                  color="white", bbox={"boxstyle": "circle,pad=0.22", "facecolor": colour,
                                       "edgecolor": "white", "linewidth": 1.4}, zorder=5)
    axis.set_title(title, fontsize=16, loc="left", pad=14)
    axis.set(xlim=(0, 1920), ylim=(1140, 0), xlabel="Image x (pixels)", ylabel="Image y (pixels)")
    axis.set_xticks([0, 640, 1280, 1920])
    axis.set_yticks([0, 360, 720, 1080])
    axis.tick_params(labelsize=11)


def run() -> None:
    with gzip.open(ROOT / "results/video53_nn_replay.json.gz", "rt") as source:
        replay = json.load(source)
    with gzip.open(ROOT / "results/visual_geometry.json.gz", "rt") as source:
        geometry = {record["sample_id"]: record for record in json.load(source)}
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    pixels = plt.imread(ROOT / "raw/control_sheets/V14_centre.jpg")
    before = np.asarray(replay["nn_median_corners_px"])
    after = np.asarray(replay["saved_scene"]["raw_corners_px"])
    np.testing.assert_array_equal(after, geometry["V14"]["scene"]["raw_corners_px"])
    draw_outline(axes[0], pixels, before, [BLUE] * 4, "Before: neural-net corner positions")
    draw_outline(axes[1], pixels, after, [BLUE, ORANGE, BLUE, BLUE], "After: OpenCV replaces corner 1")
    axes[0].annotate("Corner 1: low confidence,\nbut near the right place", before[1], xytext=(930, 120),
                     fontsize=12, color="white", bbox={"facecolor": "#202020", "alpha": 0.9, "pad": 5},
                     arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 1.5})
    axes[1].annotate("Corner 1 moved down here", after[1], xytext=(780, 780),
                     fontsize=12, color="white", bbox={"facecolor": "#202020", "alpha": 0.9, "pad": 5},
                     arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 1.5})
    figure.suptitle("Video 53: OpenCV moves corner 1 to the bottom of the picture",
                     fontsize=20, weight="bold", y=0.98)
    figure.text(0.07, 0.88, NUMBER_KEY + "\nBlue: neural net · Orange: OpenCV replacement", fontsize=13)
    figure.text(0.07, 0.085, "Same image: frame 83,084 · Scene 334 · The fallback outline caused the scene to be rejected.",
                fontsize=13)
    figure.text(0.07, 0.035, "Before: median of the 10 sampled NN outputs, reconstructed with the original weights.\n"
                "Corner 1 confidence: 0.015; required: 0.020. After: saved fallback output. Replay agrees within 0.02 pixels.",
                fontsize=11)
    figure.subplots_adjust(top=0.77, bottom=0.20, left=0.07, right=0.97, wspace=0.22)
    figure.savefig(ROOT / "figures/video53_nn_to_opencv.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    pixels = plt.imread(ROOT / "raw/control_sheets/V04_centre.jpg")
    scene = geometry["V04"]["scene"]
    assert scene["raw_source"] == "model" and scene["consensus_flag"]
    before = np.asarray(scene["raw_corners_px"])
    after = np.asarray(scene["active_corners_native_px"])
    draw_outline(axes[0], pixels, before, [BLUE] * 4, "Before: neural net fits this camera view")
    draw_outline(axes[1], pixels, after, [PURPLE] * 4, "After: shared outline is too small")
    for axis in axes:
        axis.annotate("Visible far player", (1117, 190), xytext=(80, 140), fontsize=12, color="white",
                      bbox={"facecolor": "#202020", "alpha": 0.9, "pad": 5},
                      arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 1.5})
    figure.suptitle("Video 17: the shared outline replaces a better scene outline", fontsize=20, weight="bold", y=0.98)
    figure.text(0.07, 0.88, NUMBER_KEY + "\nBlue: neural net · Purple: shared outline used by the player picker", fontsize=13)
    figure.text(0.07, 0.085, "Same image: frame 47,276 · Scene 0 · All four NN corners were confident, so OpenCV was not used here.",
                fontsize=12)
    figure.text(0.07, 0.035, "Both outlines come from the saved run. The player picker lost the far player with the shared outline.\n"
                "Changing only the outline back restored the player pick in an isolated check.", fontsize=12)
    figure.subplots_adjust(top=0.77, bottom=0.20, left=0.07, right=0.97, wspace=0.22)
    figure.savefig(ROOT / "figures/video17_nn_to_shared.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    run()
