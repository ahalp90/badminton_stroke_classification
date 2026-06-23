"""Macro-F1 / min-F1 bar charts across the 2026-05-30..06-01 taxon_pinned_w_preds runs.

Two figures:
  1. standard vs focal_alpha_revert_overallocated across six taxonomy/split cells,
     with the bst_25/split_bst_baseline cell led by the published BST-CG-AP bar.
  2. the AdamW weight-decay sweep on une_v1_14 / split_v2, anchored by the no-wd
     standard run and the focal_alpha_revert run.

Per run: a blue bar (macro F1) and a sand bar (min F1) at the best serial (the one
whose weights survived; serial 1 has no numeric suffix, 2-5 carry _{n}.pt). A black
tick marks the 5-serial mean for each metric. Best and mean figures are printed at
30 degrees above each group. Log y, floored at 0.40, so the high cluster spreads out.

Figures are read live from each run's manifest.yaml; only the BST-CG-AP point is
literal (bst_25: best macro 0.821 / min 0.611, serial mean 0.8097 / 0.5762).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import yaml

BASE = Path(
    "/home/ariel/Documents/COSC594/badminton_stroke_classification/"
    "experiments/bst_x/shuttleset"
)
OUT_DIR = Path(
    "/home/ariel/Documents/COSC594/badminton_stroke_classification/local_scratch/presentation_prep"
)

FLOOR = 0.40
Y_TOP = 0.92

# Tol-ish, protanopia-safe: clear blue vs sand sit on the blue-yellow axis.
C_MAC = "#4477AA"  # macro F1
C_MIN = "#DDCC77"  # min F1
MEAN_HALO = [pe.withStroke(linewidth=3.0, foreground="white")]

# BST-CG-AP, bst_25: best serial + serial mean (Chang's BST reproduction).
PUBLISHED = {"mac_best": 0.821, "min_best": 0.611, "mac_mean": 0.8097, "min_mean": 0.5762}


def _baseline_config_text() -> str:
    """Compact recipe for the top-left panel (labels aligned, monospace)."""
    rows = [
        ("adaptive_focal:", "tau 1.0 · gamma 1.0 · momentum 0.9 · warm_up_epochs 5 · f1_floor 0.0"),
        ("augmentation:", "p_flip 0.5 · p_jitter 0.3 · cap_y 0.05 · cap_x 0.1 · eps 0.15"),
        ("optimiser:", "AdamW · wd 1e-2 · all layers decay"),
        ("schedule:", "n_epochs 80 · batch_size 128 · lr 5e-4 · cosine decay 0.5 · cg_ap 15 epochs"),
        ("detection:", "sticky anchor player detection"),
    ]
    w = max(len(key) for key, _ in rows) + 2
    return "\n".join(f"{key:<{w}}{val}" for key, val in rows)


BASELINE_CONFIG = _baseline_config_text()


def load_metrics(run_id: str) -> dict:
    """Pull best-serial and 5-serial-mean macro/min F1 from a run's manifest.

    Best serial = the one whose weights file still exists on disk.

    :param run_id: experiment run directory name
    :return: dict with best_mac, best_min, mean_mac, mean_min, best_no, cfg
    """
    rd = BASE / run_id
    manifest = yaml.safe_load((rd / "manifest.yaml").read_text())
    serials = manifest["serials"]
    surviving = {p.name for p in (rd / "weights").glob("*.pt")}
    best = next(
        (s for s in serials if Path(s["weights_path"]).name in surviving),
        serials[0],
    )
    macros = np.array([s["metrics"]["macro_f1"] for s in serials])
    mins = np.array([s["metrics"]["min_f1"] for s in serials])
    return {
        "cfg": manifest["config"],
        "best_mac": float(best["metrics"]["macro_f1"]),
        "best_min": float(best["metrics"]["min_f1"]),
        "mean_mac": float(macros.mean()),
        "mean_min": float(mins.mean()),
        "best_no": int(best["serial_no"]),
    }


def entry(run_id: str, group: str, line1: str, line2: str) -> dict:
    """Build a plot entry (one x-position, two bars) from a run."""
    m = load_metrics(run_id)
    return {
        "group": group,
        "line1": line1,
        "line2": line2,
        "mac_best": m["best_mac"],
        "min_best": m["best_min"],
        "mac_mean": m["mean_mac"],
        "min_mean": m["mean_min"],
        "published": False,
    }


def published_entry(group: str, line1: str, line2: str) -> dict:
    """The external BST-CG-AP reference: best-serial bars + serial-mean tick."""
    return {
        "group": group,
        "line1": line1,
        "line2": line2,
        "mac_best": PUBLISHED["mac_best"],
        "min_best": PUBLISHED["min_best"],
        "mac_mean": PUBLISHED["mac_mean"],
        "min_mean": PUBLISHED["min_mean"],
        "published": True,
    }


def layout_x(entries: list[dict], gap: float = 0.85) -> np.ndarray:
    """Unit spacing within a group, an extra gap between groups."""
    xs = []
    x = 0.0
    prev = None
    for e in entries:
        if prev is not None and e["group"] != prev:
            x += gap
        xs.append(x)
        x += 1.0
        prev = e["group"]
    return np.array(xs)


def draw_figure(entries: list[dict], outpath: Path, suptitle: str, subtitle: str,
                shade_groups: set[str], band: tuple[str, str] | None = None,
                figsize: tuple[float, float] = (17, 8), config_panel: bool = True) -> None:
    """Render one grouped bar chart.

    :param entries: ordered plot entries, one per x-position
    :param shade_groups: group ids to draw a faint background band behind
    :param band: optional (group_id, text) to label a region (e.g. the wd sweep)
    """
    xs = layout_x(entries)
    bw = 0.40
    fig, ax = plt.subplots(figsize=figsize)

    # Faint alternating background per group so std/focal (and the trio) read together.
    seen_groups = []
    for g in dict.fromkeys(e["group"] for e in entries):
        idx = [i for i, e in enumerate(entries) if e["group"] == g]
        lo, hi = xs[idx[0]] - 0.5, xs[idx[-1]] + 0.5
        if g in shade_groups:
            ax.axvspan(lo, hi, color="#000000", alpha=0.045, zorder=0)
        seen_groups.append((g, lo, hi))
        if band is not None and g == band[0]:
            ax.text((lo + hi) / 2, Y_TOP * 0.93, band[1], ha="center", va="top",
                    fontsize=10, style="italic", color="#333333")

    for xi, e in zip(xs, entries):
        for value, mean, colour, off in (
            (e["mac_best"], e["mac_mean"], C_MAC, -bw / 2),
            (e["min_best"], e["min_mean"], C_MIN, +bw / 2),
        ):
            xpos = xi + off
            if value <= FLOOR:
                # Off the log floor (e.g. shuttleset_18 min F1 = 0). Mark it, don't fake a bar.
                ax.text(xpos, FLOOR * 1.004, "0.00\noff-scale", ha="center", va="bottom",
                        fontsize=6, color="#555555", rotation=90)
                continue
            ax.bar(xpos, value - FLOOR, bw, bottom=FLOOR, color=colour,
                   edgecolor="black" if e["published"] else "none",
                   hatch="//" if e["published"] else None,
                   linewidth=0.7, zorder=3)
            if mean is not None and mean > FLOOR:
                ax.plot([xpos - bw / 2, xpos + bw / 2], [mean, mean], color="black",
                        lw=1.5, solid_capstyle="butt", path_effects=MEAN_HALO, zorder=5)

        # 30-degree best/mean readout hovering above the taller bar.
        top = max(e["mac_best"], e["min_best"], FLOOR)
        if e["mac_mean"] is None:
            txt = f"{e['mac_best']:.3f} / {e['min_best']:.3f}"
        else:
            txt = (f"best {e['mac_best']:.3f} / {e['min_best']:.3f}\n"
                   f"mean {e['mac_mean']:.3f} / {e['min_mean']:.3f}")
        ax.text(xi - 0.15, top * 1.01, txt, rotation=30, ha="left", va="bottom",
                fontsize=7, color="#222222", linespacing=1.25)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{e['line1']}\n{e['line2']}" for e in entries], fontsize=7.5)
    ax.set_ylim(FLOOR, Y_TOP)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels([f"{v:.1f}" for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)])
    ax.minorticks_off()
    ax.set_ylabel("F1 (linear, floored at 0.40)")
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)

    handles = [
        Patch(facecolor=C_MAC, label="macro F1 (best serial)"),
        Patch(facecolor=C_MIN, label="min F1 (best serial)"),
        Line2D([0], [0], color="black", lw=1.5, label="5-serial mean"),
    ]
    if any(e["published"] for e in entries):
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="//",
                             label="BST-CG-AP published"))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=9, color="#333333")
    legend_y = 0.900
    if config_panel:
        # Config callout on top, colour legend tucked directly beneath it.
        fig.text(0.012, 0.910, "baseline config", ha="left", va="top",
                 fontsize=7.5, fontweight="bold", color="#333333")
        fig.text(0.012, 0.884, BASELINE_CONFIG, ha="left", va="top",
                 family="monospace", fontsize=6.5, linespacing=1.3, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7",
                           edgecolor="#b0b0b0", linewidth=0.7))
        legend_y = 0.792
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, legend_y),
               ncol=len(handles), frameon=True, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.765))
    fig.subplots_adjust(top=0.765)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved: {outpath}")


def build_graph1() -> None:
    entries = [
        published_entry("A", "bst_25 / bst_base", "BST-CG-AP pub"),
        entry("run_20260530_210600_435552", "A", "bst_25 / bst_base", "standard"),
        entry("run_20260531_225619_826430", "A", "bst_25 / bst_base", "focal-revert"),
        entry("run_20260530_225714_593038", "B", "bst_24 / bst_base", "standard"),
        entry("run_20260601_005010_962006", "B", "bst_24 / bst_base", "focal-revert"),
        entry("run_20260530_192738_970644", "C", "bst_12 / v2", "standard"),
        entry("run_20260531_211838_567072", "C", "bst_12 / v2", "focal-revert"),
        entry("run_20260530_174818_410060", "D", "bst_24 / v2", "standard"),
        entry("run_20260531_193021_308927", "D", "bst_24 / v2", "focal-revert"),
        entry("run_20260531_005535_005154", "E", "une_v1_14 / v2", "standard"),
        entry("run_20260601_023543_278210", "E", "une_v1_14 / v2", "focal-revert"),
        entry("run_20260530_161525_131279", "F", "shuttleset_18 / v2", "standard"),
        entry("run_20260531_163906_107348", "F", "shuttleset_18 / v2", "focal-revert"),
    ]
    draw_figure(
        entries,
        OUT_DIR / "f1_standard_vs_focal_revert.png",
        suptitle="taxon_pinned_w_preds: standard vs focal_alpha_revert_overallocated",
        subtitle=("Bars = best serial; black tick = 5-serial mean. "
                  "bst_25/bst_baseline led by BST-CG-AP published; other cells by macro F1 desc."),
        shade_groups={"A", "C", "E"},
        figsize=(18, 9),
    )


def build_graph2() -> None:
    entries = [
        entry("run_20260531_005535_005154", "anchor", "1e-2*", "baseline"),
        entry("run_20260601_023543_278210", "anchor", "1e-2*", "focal-revert"),
        entry("run_20260531_201350_026614", "sweep", "1e-2", ""),
        entry("run_20260531_214009_170864", "sweep", "5e-2", ""),
        entry("run_20260531_231403_971803", "sweep", "1e-1", ""),
        entry("run_20260601_003918_078077", "sweep", "2e-1", ""),
        entry("run_20260601_021234_940276", "sweep", "4e-1", ""),
    ]
    draw_figure(
        entries,
        OUT_DIR / "f1_une_v1_14_wd_sweep.png",
        suptitle=("AdamW wd regularisation on une_v1_14, split_v2\n"
                  "norms / bias / embeddings excluded from wd across the sweep"),
        subtitle=("Bars = best serial; black tick = 5-serial mean. "
                  "* = wd also applied to norms / bias / embeddings (not excluded); baseline & focal-revert are 1e-2*."),
        shade_groups={"sweep"},
        band=("sweep", "AdamW wd sweep"),
        figsize=(13, 8),
    )


def build_series_g() -> None:
    """Series G baseline batch: six taxonomy/split cells, paper-baseline cell anchored by BST-CG-AP."""
    entries = [
        published_entry("A", "bst_25 / bst_base", "BST-CG-AP pub"),
        entry("run_20260530_210600_435552", "A", "bst_25 / bst_base", "Series G"),
        entry("run_20260530_225714_593038", "B", "bst_24 / bst_base", "Series G"),
        entry("run_20260530_192738_970644", "C", "bst_12 / v2", "Series G"),
        entry("run_20260530_174818_410060", "D", "bst_24 / v2", "Series G"),
        entry("run_20260531_005535_005154", "E", "une_v1_14 / v2", "Series G"),
        entry("run_20260530_161525_131279", "F", "shuttleset_18 / v2", "Series G"),
    ]
    draw_figure(
        entries,
        OUT_DIR / "f1_series_g_baseline.png",
        suptitle="Taxon & split comparison, frozen hp baseline",
        subtitle=("Bars = best serial; black tick = 5-serial mean. "
                  "bst_25/bst_baseline anchored by BST-CG-AP published; remaining cells by macro F1 desc."),
        shade_groups={"A", "C", "E"},
        figsize=(15, 8),
    )


if __name__ == "__main__":
    build_graph1()
    build_graph2()
    build_series_g()
