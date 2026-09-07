"""Compare independently read scoreboards with the sampled source-label rows."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def classify(row: pd.Series) -> str:
    if pd.isna(row.visible_game) or pd.isna(row.visible_score_top) or pd.isna(row.visible_score_bottom):
        return "Game or score unreadable"
    source = (row.label_score_a, row.label_score_b)
    visible = (row.visible_score_top, row.visible_score_bottom)
    same_score = any(sum(abs(label - score) for label, score in zip(source, order, strict=True)) <= 1
                     for order in (visible, visible[::-1]))
    if row.label_game == row.visible_game and same_score:
        return "Game and score consistent"
    return "Different game or score"


def plot_examples() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 13))
    cases = [("A47", "Missed contact in video 53: the full court and both players are visible",
              "Game 2 · video score 2–4 · source score 3–4 after reversing player order"),
             ("A46", "Another missed contact in video 53: the camera shows one player from the side",
              "Game 1 · video score 7–6 · source score 8–6 after reversing player order")]
    for axis, (sample, title, detail) in zip(axes, cases, strict=True):
        axis.imshow(plt.imread(ROOT / "raw/alignment_checks" / f"{sample}_centre.jpg"))
        axis.set_title(title, fontsize=13, loc="left", pad=12)
        axis.set_axis_off()
        axis.text(0, -0.055, detail, transform=axis.transAxes, fontsize=11)
    figure.suptitle("Video 53's missed contacts can still refer to the right rally", x=0.06, ha="left",
                     fontsize=17, weight="bold", y=0.99)
    figure.text(0.06, 0.945, "Two targeted source frames · both court-rejected · labels unchanged\n"
                "Scores allow the point to finish. Matching game/score does not certify every contact time.", fontsize=11)
    figure.subplots_adjust(top=0.9, bottom=0.045, hspace=0.25, left=0.04, right=0.98)
    figure.savefig(ROOT / "figures/video53_alignment_checks.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run() -> None:
    labels = pd.read_csv(ROOT / "results/alignment_labels.csv.gz")
    observations = pd.read_csv(ROOT / "results/alignment_observations.csv.gz")
    rows = labels.merge(observations, on="sample_id", validate="one_to_one", indicator=True)
    assert len(rows) == 53 and (rows._merge == "both").all()
    rows = rows.drop(columns="_merge")
    rows["alignment_check"] = rows.apply(classify, axis=1)
    rows.to_csv(ROOT / "results/alignment_review.csv.gz", index=False)
    summary = rows.groupby(["sampling_group", "alignment_check", "court_present"]).size()
    summary.rename("windows").reset_index().to_csv(ROOT / "results/alignment_summary.csv.gz", index=False)
    video = rows.groupby(["fixture", "alignment_check"]).size().rename("windows").reset_index()
    video.to_csv(ROOT / "results/alignment_per_video.csv.gz", index=False)
    print(summary.to_string())
    print(video.to_string(index=False))
    plot_examples()


if __name__ == "__main__":
    run()
