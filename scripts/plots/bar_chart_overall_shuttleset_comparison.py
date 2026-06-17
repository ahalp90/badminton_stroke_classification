"""Overall macro F1 / min F1 / top-2 accuracy across TemPose, BST, BST-X, BRIC.

Sweep-asset style: blue / sand / pale-blue bars per entry for macro / min /
top-2, black tick = serial mean (only on multi-seed internal runs). Per-entry
readouts sit directly under each cell name; shared section header below them
names the taxonomy block. Y-axis floored at 0.40.

Internal bests pulled live from manifests:
  25-class: run_20260530_210600_435552 S1 (BST-25 / BST-baseline split)
  14-class: run_20260602_143618_156220 S2 (the cell used in the 14-class plot)
BRIC: deployed registry entry rgb_shuttle (tcn) outgoing_only, seed 42.
TemPose-TF and Chang's BST: literal values (paper / README).
"""
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP = REPO_ROOT / "experiments/bst_x/shuttleset"
BRIC_DEPLOYED = REPO_ROOT / "runtime/deployed/bric"
OUT_PATH = REPO_ROOT / "docs/images/bar_chart_overall_shuttleset_comparison.png"

FLOOR = 0.40
Y_TOP = 1.00

# Tol-ish, protan-safe. Macro and min carry the headline contrast (cool blue
# vs warm sand); top-2 is a paler blue so it reads as context.
C_MAC = "#4477AA"   # macro F1
C_MIN = "#DDCC77"   # min F1
C_TOP2 = "#BBDDEE"  # top-2 accuracy (palest)
MEAN_HALO = [pe.withStroke(linewidth=3.0, foreground="white")]


def load_bst_x_run(run_dir: str) -> dict:
    """Read manifest, return best-serial values + per-serial means.

    Best serial = manifest's best_serials[0] when present; otherwise the
    serial whose weights file survives on disk (the deletion pass keeps only
    the best).
    """
    rd = EXP / run_dir
    manifest = yaml.safe_load((rd / "manifest.yaml").read_text())
    serials = manifest["serials"]
    if manifest.get("best_serials"):
        best_sn = manifest["best_serials"][0]
    else:
        surviving = {p.name for p in (rd / "weights").glob("*.pt")}
        best_sn = next(s["serial_no"] for s in serials
                       if Path(s["weights_path"]).name in surviving)
    best = next(s for s in serials if s["serial_no"] == best_sn)["metrics"]
    macros = np.array([s["metrics"]["macro_f1"] for s in serials])
    mins = np.array([s["metrics"]["min_f1"] for s in serials])
    top2s = np.array([s["metrics"]["top2_accuracy"] for s in serials])
    return {
        "macro_best": float(best["macro_f1"]),
        "min_best": float(best["min_f1"]),
        "top2_best": float(best["top2_accuracy"]),
        "macro_mean": float(macros.mean()),
        "min_mean": float(mins.mean()),
        "top2_mean": float(top2s.mean()),
    }


def load_bric(run_dir: str) -> dict:
    """Read BRIC's test summary; single-seed so no mean."""
    summary = json.loads((BRIC_DEPLOYED / run_dir / "eval/test_summary.json").read_text())
    m = summary["metrics"]
    return {
        "macro_best": float(m["macro_f1"]),
        "min_best": float(m["min_f1"]),
        "top2_best": float(m["top2_accuracy"]),
        "macro_mean": None,
        "min_mean": None,
        "top2_mean": None,
    }


def published(macro: float, min_f1: float, top2: float) -> dict:
    """Literal numbers from a paper / README header — no mean line."""
    return {
        "macro_best": macro,
        "min_best": min_f1,
        "top2_best": top2,
        "macro_mean": None,
        "min_mean": None,
        "top2_mean": None,
    }


def main():
    bric = load_bric("20260518_013238_rgb_shuttle-tcn-outgoing_only_une_merge_v1_nosides_42")
    bst_x_14 = load_bst_x_run("run_20260602_143618_156220")
    bst_x_25 = load_bst_x_run("run_20260530_210600_435552")
    chang_bst = published(0.810, 0.576, 0.959)   # README, BST paper variable-length
    tempose = published(0.803, 0.542, 0.957)     # TemPose-TF, BST windowing

    # 25-class first (TemPose -> BST -> BST-X chronological), then 14-class
    # (BRIC -> BST-X). Group labels are shared section headers below the bars.
    entries = [
        {"group": "25c", "label": "TemPose-TF\n(BST windowing)", **tempose},
        {"group": "25c", "label": "BST\n(Chang)",                **chang_bst},
        {"group": "25c", "label": "BST-X",                       **bst_x_25},
        {"group": "14c", "label": "BRIC",                        **bric},
        {"group": "14c", "label": "BST-X",                       **bst_x_14},
    ]

    xs = np.arange(len(entries), dtype=float)
    for i, e in enumerate(entries):
        if i > 0 and e["group"] != entries[i - 1]["group"]:
            xs[i:] += 0.7

    bw = 0.25
    offs = (-bw, 0.0, +bw)

    fig, ax = plt.subplots(figsize=(14, 8.0))

    # Only shade the 14-class block so the visual contrast names a taxonomy
    # rather than alternating decoration.
    gx_14c = [xs[i] for i, e in enumerate(entries) if e["group"] == "14c"]
    ax.axvspan(min(gx_14c) - 0.55, max(gx_14c) + 0.55,
               color="#000000", alpha=0.05, zorder=0)

    for xi, e in zip(xs, entries):
        for off, key_b, key_m, colour in (
            (offs[0], "macro_best", "macro_mean", C_MAC),
            (offs[1], "min_best",   "min_mean",   C_MIN),
            (offs[2], "top2_best",  "top2_mean",  C_TOP2),
        ):
            value = e[key_b]
            mean = e[key_m]
            xpos = xi + off
            if value <= FLOOR:
                ax.text(xpos, FLOOR * 1.005, f"{value:.2f}\noff-scale",
                        ha="center", va="bottom", fontsize=6, color="#555555", rotation=90)
                continue
            ax.bar(xpos, value - FLOOR, bw, bottom=FLOOR, color=colour,
                   edgecolor="#444444" if colour == C_TOP2 else "none",
                   linewidth=0.4 if colour == C_TOP2 else 0,
                   zorder=3)
            if mean is not None and mean > FLOOR:
                ax.plot([xpos - bw / 2, xpos + bw / 2], [mean, mean], color="black",
                        lw=1.5, solid_capstyle="butt", path_effects=MEAN_HALO, zorder=5)

        # Per-entry readout directly under the cell name, horizontal so the
        # bar chart keeps its vertical granularity.
        if e["macro_mean"] is None:
            txt = f"best  {e['macro_best']:.3f} / {e['min_best']:.3f} / {e['top2_best']:.3f}"
        else:
            txt = (f"best  {e['macro_best']:.3f} / {e['min_best']:.3f} / {e['top2_best']:.3f}\n"
                   f"mean  {e['macro_mean']:.3f} / {e['min_mean']:.3f} / {e['top2_mean']:.3f}")
        ax.text(xi, -0.06, txt, transform=ax.get_xaxis_transform(),
                ha="center", va="top", rotation=10, fontsize=8,
                color="#222222", linespacing=1.25, family="monospace",
                clip_on=False)

    ax.set_xticks(xs)
    ax.set_xticklabels([e["label"] for e in entries], fontsize=9)
    ax.set_ylim(FLOOR, Y_TOP)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([f"{v:.1f}" for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)])
    ax.minorticks_off()
    ax.set_ylabel("F1 (Macro/Min), Top-2 Acc. (0.4 floor)")
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)

    # Shared section headers below the per-entry readouts.
    section_headers = [
        ("25c", "25-class (BST paper taxon)"),
        ("14c", "14-class (custom taxon: top/bottom combined, reveals confusion pairs)"),
    ]
    for group, label in section_headers:
        gx = [xs[i] for i, e in enumerate(entries) if e["group"] == group]
        center = (min(gx) + max(gx)) / 2
        ax.text(center, -0.16, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10.5, fontweight="bold",
                color="#222222", clip_on=False)

    handles = [
        Patch(facecolor=C_MAC,  label="macro F1 (best serial)"),
        Patch(facecolor=C_MIN,  label="min F1 (best serial)"),
        Patch(facecolor=C_TOP2, edgecolor="#444444", linewidth=0.4,
              label="top-2 accuracy (best serial)"),
        Line2D([0], [0], color="black", lw=1.5, label="serial mean (where available)"),
    ]
    fig.suptitle(
        "TemPose, BST, BST-X and BRIC Performance on ShuttleSet",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=len(handles), frameon=True, fontsize=8)
    fig.tight_layout(rect=(0, 0.0, 1, 0.90))
    fig.subplots_adjust(top=0.90, bottom=0.15)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
