"""Show why video-15 timing matches do not establish the right rally identity."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def run() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    examples = (
        ("G01", "6 of 6 labels have a timing match", "Label: game 2, score 21–18\nVideo: game 2, score 9–8", 70261),
        ("G02", "2 of 2 labels have a timing match", "Label: game 2, score 5–2\nVideo: game 1, score 13–5", 36550),
    )
    for axis, (sample, title, explanation, frame) in zip(axes, examples, strict=True):
        axis.imshow(plt.imread(ROOT / "raw/video15_followup" / f"{sample}_centre.jpg"))
        axis.set_xlim(230, 770)
        axis.set_ylim(165, 45)
        axis.axis("off")
        axis.set_title(title, fontsize=12, pad=14)
        axis.text(0.5, -0.35, explanation, transform=axis.transAxes, ha="center", fontsize=12)
        axis.text(0.5, -0.88, f"Video 15 · frame {frame:,} · {frame / 30:.2f} seconds",
                  transform=axis.transAxes, ha="center", fontsize=10)
    figure.suptitle("Even complete timing matches can refer to the wrong rally", fontsize=15, weight="bold", y=1.04)
    figure.text(0.125, -0.17, "Both scoreboards are enlarged source images. The check compares game and score, not just visible play.\n"
                "Timing uses the fixed detector and cleaned labels, ±10 frames at 30 fps. No labels were shifted.", fontsize=10)
    figure.subplots_adjust(top=0.77, bottom=0.38, wspace=0.16)
    figure.savefig(ROOT / "figures/video15_best_matches.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    run()
