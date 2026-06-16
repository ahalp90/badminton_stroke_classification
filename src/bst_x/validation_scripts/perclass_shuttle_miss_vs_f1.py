"""Per-class shuttle-miss vs per-class F1 correlation.

For each completed run, joins:

- the per-stroke shuttle-miss-rate table from a
  ``zeroed_frames_analysis_outputs/analysis_*.txt`` file (the
  "Per-stroke shuttle miss rate within +-10 frames of hit" section),
- against the per-class F1 scores in a run's ``manifest.yaml``
  (median across all 5 serials per class).

Then computes Spearman + Pearson correlations and emits a sorted table
plus a scatter plot. **Hypothesis**: classes with high shuttle-miss
rate should have lower median F1 if the shuttle stream is genuinely
load-bearing for those classes. Weak correlation suggests the model
copes fine without shuttle and the mask-channel arm is less likely to
help.

Label matching: handles taxonomies with Top_/Bottom_ prefixes by
collapsing pairs (median across the two sides). For ``une_merge_v1_nosides``
the labels already match the validation table directly with no
collapse needed; pass ``--no-collapse-sides`` to skip averaging.

**Metric note**: per-class precision is not currently saved in the
manifest schema. F1 is what we have, and is arguably the better
signal for the "is the model handling this class well overall"
question. To switch to precision, extend ``Task.test()`` in
``bst_x_train.py`` to also compute and store ``per_class_precision``
(via ``torcheval.metrics.functional.multiclass_precision`` with
``average=None``), then re-run this with ``--metric precision``.

Usage::

    python -m validation_scripts.perclass_shuttle_miss_vs_f1 \\
        --analysis-txt validation_scripts/zeroed_frames_analysis_outputs/analysis_unemergev1nosides_v2_20260429_1904.txt \\
        --manifest src/bst_x/stroke_classification/main_on_shuttleset/experiments/run_20260430_XXXXXX/manifest.yaml
"""
from __future__ import annotations

import argparse
import re
import socket
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import yaml

DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent / "zeroed_frames_analysis_outputs"
)


def parse_per_stroke_miss_rates(txt_path: Path) -> dict[str, float]:
    """Pull the per-stroke shuttle-miss-rate table out of an analysis txt.

    Looks for the section starting with ``--- Per-stroke shuttle miss
    rate within +-10 frames of hit`` (the +-10 hyphens are read as a
    plain dash here). Returns ``{stroke_type: miss_rate_pct}``.
    """
    text = txt_path.read_text()
    lines = text.splitlines()
    in_section = False
    miss: dict[str, float] = {}
    # Header looks like:
    #   Stroke Type                    Missed / Total    Miss Rate
    # Body lines are 2-space-indented:
    #   long_service                      791 / 7,528       10.51%
    row_re = re.compile(
        r"^\s+(?P<stroke>[a-z_]+)\s+[\d,]+\s*/\s*[\d,]+\s+(?P<rate>[\d.]+)%\s*$"
    )
    for line in lines:
        if line.startswith("--- Per-stroke shuttle miss rate"):
            in_section = True
            continue
        if in_section:
            if line.startswith("---") or line.startswith("==="):
                break
            if not line.strip() or line.lstrip().startswith("Stroke Type"):
                continue
            m = row_re.match(line)
            if m:
                miss[m.group("stroke")] = float(m.group("rate"))
    return miss


def collect_per_class_f1(
    manifest_path: Path, metric_key: str = "per_class_f1",
) -> dict[str, list[float]]:
    """Pull per-class metric values from each serial in a manifest.

    Returns ``{class_label: [serial1_value, serial2_value, ...]}``.
    """
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    out: dict[str, list[float]] = {}
    for serial in manifest.get("serials", []):
        per_class = serial.get("metrics", {}).get(metric_key, {})
        for cls, v in per_class.items():
            out.setdefault(cls, []).append(float(v))
    return out


def _strip_side(label: str) -> tuple[str, str | None]:
    """Return ``(stroke_type, side)`` where side is 'Top'/'Bottom'/None."""
    if label.startswith("Top_"):
        return label[4:], "Top"
    if label.startswith("Bottom_"):
        return label[7:], "Bottom"
    return label, None


def join_metrics(
    miss: dict[str, float],
    per_class: dict[str, list[float]],
    collapse_sides: bool,
) -> list[tuple[str, float, float, int]]:
    """Return ``[(stroke, miss_rate, median_metric, n_serials), ...]``.

    With ``collapse_sides=True``, Top_/Bottom_ pairs are pooled before
    the median. ``n_serials`` is the count of underlying values used
    (5 per side; 10 if collapsed and both sides present).
    """
    pooled: dict[str, list[float]] = {}
    for label, vals in per_class.items():
        stroke, side = _strip_side(label)
        if side is None:
            pooled.setdefault(stroke, []).extend(vals)
        elif collapse_sides:
            pooled.setdefault(stroke, []).extend(vals)
        else:
            pooled.setdefault(label, []).extend(vals)

    rows: list[tuple[str, float, float, int]] = []
    for stroke, vals in pooled.items():
        # When sides aren't collapsed, also try to look up a side-stripped
        # match in the miss table because the validation analysis usually
        # only emits the stroke-level miss rate.
        miss_key = stroke if stroke in miss else _strip_side(stroke)[0]
        if miss_key not in miss:
            continue
        rows.append(
            (stroke, miss[miss_key], float(np.median(vals)), len(vals))
        )
    rows.sort(key=lambda r: -r[1])
    return rows


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Stdlib Spearman: rank-correlation of two arrays."""
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--analysis-txt",
        type=Path,
        required=True,
        help=(
            "Path to a validate_zeroed_frames analysis .txt with a "
            "per-stroke shuttle miss rate table."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a run manifest.yaml.",
    )
    parser.add_argument(
        "--metric",
        choices=["f1", "precision", "recall"],
        default="f1",
        help=(
            "Which per-class metric to read from manifest. f1 is what "
            "the manifest currently stores; precision/recall require "
            "a schema extension to bst_x_train.py to produce."
        ),
    )
    parser.add_argument(
        "--no-collapse-sides",
        action="store_true",
        help=(
            "Don't average Top_/Bottom_ pairs (defaults to collapsing). "
            "Use for nosides taxonomies where labels already lack the "
            "side prefix."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.analysis_txt.is_file():
        print(f"ERROR: analysis txt not found: {args.analysis_txt}")
        return 1
    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}")
        return 1

    metric_key = f"per_class_{args.metric}"

    miss = parse_per_stroke_miss_rates(args.analysis_txt)
    if not miss:
        print(
            f"ERROR: could not parse per-stroke shuttle miss rates from "
            f"{args.analysis_txt}. Expected a section starting with "
            f"'--- Per-stroke shuttle miss rate within +-10 frames of hit'."
        )
        return 1

    per_class = collect_per_class_f1(args.manifest, metric_key=metric_key)
    if not per_class:
        if args.metric != "f1":
            print(
                f"ERROR: no '{metric_key}' field found in {args.manifest}. "
                f"This metric is not in the current manifest schema; "
                f"extend bst_x_train.py Task.test() to populate it, or "
                f"re-run with --metric f1."
            )
        else:
            print(f"ERROR: no per-class F1 found in {args.manifest}.")
        return 1

    collapse = not args.no_collapse_sides
    rows = join_metrics(miss, per_class, collapse_sides=collapse)
    if not rows:
        print(
            "ERROR: no overlap between manifest classes and analysis "
            "stroke types. Check the taxonomy / label naming."
        )
        return 1

    miss_arr = np.array([r[1] for r in rows], dtype=np.float64)
    metric_arr = np.array([r[2] for r in rows], dtype=np.float64)
    pearson = float(np.corrcoef(miss_arr, metric_arr)[0, 1])
    spearman = _spearman(miss_arr, metric_arr)

    run_id = args.manifest.parent.name

    print()
    print("=" * 78)
    print(f" Per-class shuttle-miss vs per-class median {args.metric}")
    print("=" * 78)
    print(f"  manifest:     {args.manifest}")
    print(f"  run_id:       {run_id}")
    print(f"  analysis txt: {args.analysis_txt}")
    print(f"  metric:       {args.metric} (median across serials)")
    print(f"  collapse Top/Bottom sides: {collapse}")
    print()
    print(
        f"  classes joined: {len(rows)}  "
        f"(of {len(miss)} stroke types in the analysis)"
    )
    print(f"  Pearson  r: {pearson:+.3f}")
    print(f"  Spearman r: {spearman:+.3f}")
    print()
    print(
        f"  Negative correlation = high shuttle miss matches low {args.metric}"
        f" (the predicted direction)."
    )
    print()
    print("--- Per-class table (sorted by miss rate, descending) ---")
    print(
        f"  {'stroke':30s}  {'miss%':>7s}  "
        f"{'median ' + args.metric:>13s}  {'n_serials':>10s}"
    )
    for stroke, miss_pct, metric_val, n in rows:
        print(
            f"  {stroke:30s}  {miss_pct:7.2f}  "
            f"{metric_val:13.4f}  {n:10d}"
        )

    if args.no_output:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    host = socket.gethostname().split(".")[0]
    stem = (
        f"perclass_shuttle_miss_vs_{args.metric}_{run_id}_{host}_{timestamp}"
    )

    # Scatter: x = miss rate %, y = median metric. Annotated with stroke names.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(miss_arr, metric_arr, s=60, color="#4477aa", edgecolor="black")
    for stroke, miss_pct, metric_val, _ in rows:
        ax.annotate(
            stroke,
            (miss_pct, metric_val),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=8,
        )
    ax.set_xlabel("shuttle miss rate near hit (%)")
    ax.set_ylabel(f"median per-class {args.metric}")
    ax.set_title(
        f"Per-class shuttle-miss vs median {args.metric} "
        f"(run {run_id})\n"
        f"Pearson r = {pearson:+.3f}, Spearman r = {spearman:+.3f}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = args.out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"Saved scatter:   {png_path}")

    md = [
        f"# Per-class shuttle-miss vs per-class median {args.metric}",
        "",
        f"- run_id: `{run_id}`",
        f"- manifest: `{args.manifest}`",
        f"- analysis txt: `{args.analysis_txt}`",
        f"- metric: `{args.metric}` (median across serials)",
        f"- collapse Top/Bottom sides: {collapse}",
        f"- host: `{host}`",
        f"- timestamp: {timestamp}",
        "",
        f"- classes joined: {len(rows)}",
        f"- Pearson r: **{pearson:+.3f}**",
        f"- Spearman r: **{spearman:+.3f}**",
        "",
        "Negative correlation = high shuttle miss matches low "
        f"{args.metric} (the predicted direction).",
        "",
        "## Per-class table (sorted by miss rate, descending)",
        "",
        f"| stroke | miss % | median {args.metric} | n_serials |",
        "|---|---|---|---|",
    ]
    for stroke, miss_pct, metric_val, n in rows:
        md.append(
            f"| {stroke} | {miss_pct:.2f} | {metric_val:.4f} | {n} |"
        )
    md_path = args.out_dir / f"{stem}.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"Saved report:    {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
