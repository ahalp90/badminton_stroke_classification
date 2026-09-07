"""Plot broader rally gains and the frozen acceptance tradeoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scratch.contact_det_followup.scripts.prediction_io import read_json

ROOT = Path(__file__).resolve().parents[1]
BLUE = '#0072B2'
ORANGE = '#D55E00'
GREY = '#999999'


def chosen_acceptance(record: dict[str, Any]) -> dict[str, Any]:
    policy = record['frozen_development_policy']
    rule = policy['selected_rule'] or policy['nonempty_fallback']
    return next(point for point in record['curve'] if point['threshold'] == rule['threshold'])


def plot_gains(result: dict[str, Any], output: Path) -> None:
    videos = result['comparison_to_frozen_combined']['10']['by_video']
    fixtures = sorted(videos, key=int)
    gains = [videos[fixture]['correct_after'] - videos[fixture]['correct_before'] for fixture in fixtures]
    positions = np.arange(len(fixtures))
    figure, axis = plt.subplots(figsize=(14, 4))
    axis.bar(positions, gains, color=[BLUE if gain >= 0 else ORANGE for gain in gains])
    axis.axhline(0, color='black', linewidth=.8)
    axis.set_xticks(positions, fixtures, fontsize=8)
    axis.set_xlabel('Video ID (the same 47 previously examined ShuttleSet22 videos)')
    axis.set_ylabel('Additional complete rallies')
    axis.set_title('Later-contact chooser versus the preserved combined detector, ±10 frames')
    axis.spines[['top', 'right']].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_acceptance(result: dict[str, Any], output: Path) -> None:
    labels = ['Prior score cutoff', 'Current output score', 'Score learner', 'Added evidence']
    previous = read_json(ROOT / 'results/broader_result.json.gz')['systems']['combined']['acceptance']
    policy = previous['frozen_rules']
    rule = policy['selected_rule'] or policy['fallback_rule']
    point = next(point for point in previous['curve'] if point['threshold'] == rule['threshold'])
    prior_counts = point['by_tolerance']['10']
    counts = [[prior_counts['correct'], prior_counts['wrong'], prior_counts['unjudgeable']]]
    for name in ('raw_selected_score', 'selected_score', 'all_evidence'):
        point = chosen_acceptance(result['acceptance'][name])
        outcomes = point['by_tolerance']['10']['counts']
        counts.append([outcomes['correct'], outcomes['wrong'], outcomes['unjudgeable']])
    values = np.asarray(counts)
    figure, axis = plt.subplots(figsize=(9, 4.2))
    left = np.zeros(len(labels), dtype=int)
    for index, (name, colour) in enumerate(zip(('Correct', 'Wrong', 'Cannot judge'), (BLUE, ORANGE, GREY), strict=True)):
        axis.barh(labels, values[:, index], left=left, color=colour, label=name)
        left += values[:, index]
    for index, total in enumerate(left):
        axis.text(total + max(left) * .015, index, str(total), va='center')
    axis.invert_yaxis()
    axis.set_xlim(0, max(left) * 1.15)
    axis.set_xlabel('Accepted sections out of all 3,982 proposed sections')
    axis.set_title('What the frozen development acceptance rules admit at ±10')
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3, frameon=False)
    axis.spines[['top', 'right']].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    result = read_json(ROOT / 'results/later/later_broader_result.json.gz')
    output = ROOT / 'figures'
    output.mkdir(exist_ok=True)
    plot_gains(result, output / 'later_video_gains.png')
    plot_acceptance(result, output / 'later_acceptance.png')


if __name__ == '__main__':
    main()
