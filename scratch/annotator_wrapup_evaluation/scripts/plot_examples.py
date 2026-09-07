"""Show saved court outlines and visible label disagreements on source footage."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot_court(geometry: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    examples = (("V14", "Missed contact: scene rejected", "0/319 frames pass the people check"),
                ("V10", "Matched contact: scene accepted", "331/331 frames pass the people check"))
    for axis, (sample, title, vote) in zip(axes, examples, strict=True):
        record = geometry[sample]
        pixels = plt.imread(ROOT / "raw/control_sheets" / f"{sample}_centre.jpg")
        axis.imshow(pixels)
        corners = np.asarray(record["scene"]["raw_corners_px"])
        polygon = np.vstack([corners, corners[0]])
        axis.plot(polygon[:, 0], polygon[:, 1], color="#FFB000", linewidth=2.8, marker="o", markersize=5)
        axis.set_title(f"{title}\n{vote}", fontsize=11, pad=12)
        axis.set_xticks([0, 640, 1280, 1920])
        axis.set_yticks([0, 360, 720, 1080])
        axis.set(xlim=(0, 1920), ylim=(1080, 0), xlabel="Image x (source pixels)", ylabel="Image y (source pixels)")
        axis.text(0, -0.25, f"Video {record['fixture']} · scene {record['scene']['scene_index']} · "
                  f"frame {record['source_frame']} ({record['source_frame'] / 30:.2f} s)",
                  transform=axis.transAxes, fontsize=9)
    misplaced = np.asarray(geometry["V14"]["scene"]["raw_corners_px"])[1]
    axes[0].annotate("Top-right corner placed\nnear the bottom of the image", misplaced,
                     xytext=(1010, 780), color="white", fontsize=9,
                     bbox={"facecolor": "black", "alpha": 0.8, "pad": 4},
                     arrowprops={"arrowstyle": "->", "color": "white"})
    figure.suptitle("Usable footage can be rejected because the court outline is wrong", fontsize=15, weight="bold", y=1.02)
    figure.text(0.125, -0.01, "Orange lines and dots are the saved raw court estimate. "
                "Both frames come from video 53; neither image is a detector simulation.", fontsize=10)
    figure.subplots_adjust(wspace=0.25, top=0.82, bottom=0.2)
    figure.savefig(ROOT / "figures/court_example.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_alignment() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    opening = plt.imread(ROOT / "raw/pilot_sheets/P01_centre.jpg")
    later = plt.imread(ROOT / "raw/pilot_sheets/P07_centre.jpg")
    axes[0].imshow(opening)
    axes[0].set_title("Label: first serve\nVideo: opening graphics", fontsize=12, pad=14)
    axes[1].imshow(later)
    axes[1].set_xlim(90, 380)
    axes[1].set_ylim(402, 270)
    axes[1].set_title("Label: third-game rally 23, score 12–11\nVideo: third-game score 0–0", fontsize=12, pad=14)
    for axis, note in zip(axes, ("Frame 186 · 6.20 seconds", "Frame 97,748 · 3,258.27 seconds · scoreboard enlarged"), strict=True):
        axis.axis("off")
        axis.text(0.5, -0.13, note, ha="center", transform=axis.transAxes, fontsize=10)
    figure.suptitle("Video 15's source labels refer to different footage", fontsize=15, weight="bold", y=1.04)
    figure.text(0.125, -0.06, "An Se Young–Akane Yamaguchi, Uber Cup 2022 semi-final · source frame clock: 30 fps\n"
                "The later frame is inside the labelled rally's 97,395–97,776 frame span. No label offsets were changed.",
                fontsize=10)
    figure.subplots_adjust(wspace=0.12, top=0.78, bottom=0.12)
    figure.savefig(ROOT / "figures/label_alignment.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_player_geometry(geometry: dict) -> None:
    record = geometry["V04"]
    pixels = plt.imread(ROOT / "raw/control_sheets/V04_centre.jpg")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    examples = (
        ("active_corners_native_px", "Outline used by the tracker\nFar player not picked", "#0072B2"),
        ("raw_corners_px", "Original outline for this scene\nFar player picked in isolated check", "#D55E00"),
    )
    for axis, (key, title, colour) in zip(axes, examples, strict=True):
        axis.imshow(pixels)
        corners = np.asarray(record["scene"][key])
        polygon = np.vstack([corners, corners[0]])
        axis.plot(polygon[:, 0], polygon[:, 1], color=colour, linewidth=3, marker="o", markersize=4)
        axis.annotate("Visible far player", (1117, 190), xytext=(180, 150), color="white", fontsize=10,
                      bbox={"facecolor": "black", "alpha": 0.8, "pad": 3},
                      arrowprops={"arrowstyle": "->", "color": "white"})
        axis.set_title(title, fontsize=12, pad=12)
        axis.set(xlim=(0, 1920), ylim=(1080, 0), xlabel="Image x (source pixels)", ylabel="Image y (source pixels)")
        axis.set_xticks([0, 640, 1280, 1920])
        axis.set_yticks([0, 360, 720, 1080])
    figure.suptitle("A shared court outline can make the tracker lose a visible player", fontsize=15, weight="bold", y=1.04)
    figure.text(0.125, -0.05, "Same source image: video 17, frame 47,276 (1,575.87 s). Only the court outline changes.\n"
                "The check keeps the original incoming tracker state. It does not rerun contacts or change the detector.", fontsize=10)
    figure.subplots_adjust(wspace=0.25, top=0.8, bottom=0.17)
    figure.savefig(ROOT / "figures/player_geometry.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    with gzip.open(ROOT / "results/visual_geometry.json.gz", "rt") as source:
        saved_geometry = {record["sample_id"]: record for record in json.load(source)}
    plot_court(saved_geometry)
    plot_alignment()
    plot_player_geometry(saved_geometry)
