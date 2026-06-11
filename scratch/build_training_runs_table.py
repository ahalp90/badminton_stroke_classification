"""Regenerate scratch/bst_x_training_runs.md from the per-run manifest.yaml files.

Self-contained: parses every experiments/*/manifest.yaml, computes
mean-across-serials and best-serial test metrics, resolves canonical taxonomy
names (legacy name bracketed, lossy aliases flagged), sorts by started_at to
assign the global run number, then writes the global / per-taxon / per-series
tables. Re-run after new runs land; update DESC / SERIES for the new run
numbers (they shift, since the index is chronological).

    ~/.venvs/badminton-cicd/bin/python scratch/build_training_runs_table.py
"""
import glob
import statistics as stats
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "src/bst_x/stroke_classification/main_on_shuttleset/experiments"
OUT = ROOT / "scratch/bst_x_training_runs.md"

# Legacy -> canonical taxonomy (pipeline/config.py TAXONOMY_ALIASES).
TAX_ALIAS = {
    "une_merge_v1_nosides": "une_v1_14",
    "une_merge_v1": "une_v1_15",
    "merged_25": "bst_25",
    "raw_35": "bst_25",
}
DEFAULT_SPLIT = "split_bst_baseline"  # pipeline/data_access.py DEFAULT_SPLIT_COLUMN

# --------------------------------------------------------------------------- #
# Parse manifests
# --------------------------------------------------------------------------- #
def best_serial_no(d: dict, mdir: Path) -> int | None:
    """Recorded best_serials, else the serial whose weights survived in weights/."""
    bs = d.get("best_serials")
    if bs:
        return bs[0]
    pts = sorted((mdir / "weights").glob("*.pt")) if (mdir / "weights").is_dir() else []
    if len(pts) == 1:
        for s in d.get("serials", []) or []:
            if Path(s.get("weights_path", "")).name == pts[0].name:
                return s["serial_no"]
    return None


rows = []
for p in sorted(glob.glob(str(EXP / "*/manifest.yaml"))):
    d = yaml.safe_load(open(p))
    cfg = d.get("config", {}) or {}
    serials = d.get("serials", []) or []

    def colmean(key):
        vals = [s["metrics"][key] for s in serials if s.get("metrics") and key in s["metrics"]]
        return round(stats.mean(vals), 4) if vals else None

    bs_no = best_serial_no(d, Path(p).parent)
    bsm = next((s["metrics"] for s in serials if s["serial_no"] == bs_no), {})
    raw_tax = cfg.get("taxonomy", "?")
    rows.append({
        "run_id": d["run_id"],
        "started_at": d["started_at"],
        "raw_tax": raw_tax,
        "canon_tax": TAX_ALIAS.get(raw_tax, raw_tax),
        "split": cfg.get("split_column", DEFAULT_SPLIT),
        "drop_unknown": cfg.get("drop_unknown"),
        "mean": {k: colmean(k) for k in ("macro_f1", "min_f1", "accuracy", "top2_accuracy")},
        "best": {k: (round(bsm[k], 4) if bsm else None)
                 for k in ("macro_f1", "min_f1", "accuracy", "top2_accuracy")},
    })

rows.sort(key=lambda r: r["started_at"])
for idx, r in enumerate(rows, 1):
    r["run_num"] = idx
by_num = {r["run_num"]: r for r in rows}

# --------------------------------------------------------------------------- #
# Canonical taxonomy display (legacy bracketed; lossy aliases flagged)
# --------------------------------------------------------------------------- #
def tax_display(r) -> tuple[str, str]:
    legacy, canon = r["raw_tax"], r["canon_tax"]
    if legacy == canon:  # native canonical name, no bracket
        return canon, ""
    if legacy == "merged_25":
        return "bst_25", "merged_25"
    if legacy == "une_merge_v1":  # historical sided 28-class run; no clean canonical
        return "une_merge_v1", ""
    if legacy == "une_merge_v1_nosides":
        return "une_v1_14", "une_merge_v1_nosides"
    return canon, legacy


def tax_cell(n: int) -> str:
    canon, br = tax_display(by_num[n])
    return f"{canon} [{br}]" if br else canon


TAX_FAMILY = {n: tax_display(by_num[n])[0] for n in by_num}

# Runs that trained with the unknown ghost channel (head dim > active classes);
# flagged with * on the run number across every table. See the appendix.
GHOST = {7, 8, 9, 10, 12, 13, 14, 15, 16}


def num_cell(n: int) -> str:
    """Run number, with a trailing * for ghost-channel runs."""
    return f"{n}*" if n in GHOST else str(n)


def run_id_cell(rid: str) -> str:
    """Run ID broken onto two lines for a narrower column: after run_<date>, or
    (for the one legacy name) between the descriptive part and its date tail.
    Each half is its own code span so the underscores stay literal and the id
    keeps monospace; the <br> sits outside the spans so it renders as a break."""
    if rid == "bst_cg_ap_base_17_04_2026":
        head, tail = "bst_cg_ap_base", "_17_04_2026"
    elif rid.startswith("run_") and rid[4:12].isdigit():
        head, tail = rid[:12], rid[12:]  # run_YYYYMMDD | _HHMMSS[_micros]
    else:
        return f"`{rid}`"
    return f"`{head}`<br>`{tail}`"

# --------------------------------------------------------------------------- #
# Hand-authored succinct descriptions (from manifest notes), keyed by run #
# --------------------------------------------------------------------------- #
DESC = {
 1: "BST paper hparams verbatim, 1600ep. TrackNetV3 *with* inpaint (vs paper's without) beats the paper. Baseline reference.",
 2: "Retuned LR: 120ep, cosine cycles 0.25→0.5 so LR reaches 0 in-window, warmup 400→100, patience 300→40.",
 3: "First CG/AP aux-loss anneal, gentle (cosine fade to 0 by ep60); most seeds picked val before the fade bit.",
 4: "Aggressive CG/AP anneal (fade to 0 by ep15, then pure backbone); first 5-serial run. Top macro of the schedule arms.",
 5: "CG/AP ablation, always-on arm (aux held 1.0 all 80ep).",
 6: "CG/AP ablation, control arm: the auxiliary loss pinned to 0 from ep1 (CG/AP fully off).",
 7: "First une_merge_v1 run (sided, 28-class) and first CSV flat-pipeline run; drop-unknown, BST split. Schedule mirrors #4.",
 8: "Same as #7 (sided, 28-class), split swapped to split_v2 (player-overlap-minimised).",
 9: "Sanity check that the re-extraction worked: re-extracted and sticky-anchored only the 1,716 worst (hit-zone-busted) clips, symlinked the rest. Sided 28-class.",
 10: "Nosides collapse of #9 (28→14 cls by dropping Top_/Bottom_); collapse rescues wrist_smash min-F1.",
 11: "Phase-2 sanity 1/3: re-run of the bst_25 baseline combo (drop-unknown) on the unified 32,203-stem sticky-anchor clean dir.",
 12: "Phase-2 sanity 2/3: une sided (28-class) + v2 on the unified clean dir.",
 13: "Phase-2 sanity 3/3: une_v1_14 + v2 on the unified clean dir. LS=0.1 (default of the era).",
 14: "Label-smoothing ablation: LS=0.0 vs #13's LS=0.1.",
 15: "Label-smoothing LS=0.15.",
 16: "Class-weighting smoke test: LS=0.15 + class_weights{wrist_smash:2.0, smash:2.0}.",
 17: "First CDB-F1 run: adaptive focal (tau=1, gamma=1, momentum=0.9, warm_up=5), LS dropped to 0.",
 18: "CDB-F1 follow-up: gamma 1→0, the per-sample focusing term (per-class alpha shape unchanged).",
 19: "CDB-F1 follow-up: tau 1→0.5, softening the per-class alpha weighting (range narrows ~0.48–1.44).",
 20: "CDB-F1 + alpha pair-cap forcing alpha[smash] ≥ 0.7×alpha[wrist_smash] each epoch.",
 21: "CDB-F1 gamma 1→2 (Lin et al. focal default); didn't lift the floor.",
 22: "Capacity bump: MLP head hidden 400→1200 (encoder untouched). CDB-F1 parity test.",
 23: "Shuttle-unzeroing (wipe_drop): stop zeroing shuttle on keypoint-fail (~14k frames, 0.84%). Project best at the time; smash & wrist_smash lift together.",
 24: "Shuttle-mask (mask_wiring): wipe_drop + a post-TCN shuttle_missing channel fused via mask_proj + shuttle_fuse.",
 25: "Jitter-off ablation (RandomTranslation prob=0) vs the wipe_drop best; min-F1 −4.4.",
 26: "Augmentation framework v1: coupled centreline flip + pos/shuttle constrained jitter (p_flip0.5, p_jitter0.2). Replaces the broken joints-only jitter.",
 27: "Aug v1, p_jitter 0.2→0.3. The current-best aug config; sweep reference.",
 28: "Aug sweep cell `p_flip_25`: p_flip 0.5→0.25 (recover cross_court_net_shot?).",
 29: "Aug sweep cell `cap_bump`: cap_y0.05→0.075, cap_x0.10→0.15. Killed at S4 on the wrapper's macro tolerance.",
 30: "Aug sweep cell `p_jitter_40`: p_jitter 0.3→0.4.",
 31: "Aug sweep cell `p_flip_25 × p_jitter_30`: override matched base, so config collided with #28 (same aug, different seeds).",
 32: "Multi-taxon batch: shuttleset_18 / v2 (finest 18-class cut). min-F1=0 is driven_flight, a single-test-clip dice-roll with no real signal.",
 33: "Multi-taxon batch: bst_24 / v2. v2 val overshoots test ~3.9%.",
 34: "Multi-taxon batch: bst_12 / v2 (fewest classes; top macro of the batch).",
 35: "Multi-taxon batch: bst_25 / baseline, keep-unknown (paper-faithful).",
 36: "Multi-taxon batch: bst_24 / baseline, drop-unknown. Top macro overall; dropping unknown lifts the 24 real strokes.",
 37: "Multi-taxon batch: une_v1_14 / v2. Splitting wrist_smash off smash and passive_drop off drop is expensive on the parents.",
 38: "Gate+focal-revert arm: shuttleset_18 / v2. min=0 driven_flight dice-roll again.",
 39: "Gate+focal-revert arm: bst_24 / v2.",
 40: "WD sweep: wd 1e-2 *excluding* norms/bias/embeddings from decay. New une_v1_14 high (mean macro/min +0.003/+0.014).",
 41: "Gate+focal-revert arm: bst_12 / v2.",
 42: "WD sweep: wd 5e-2.",
 43: "Gate+focal-revert arm: bst_25 / baseline, keep-unknown.",
 44: "WD sweep: wd 1e-1.",
 45: "WD sweep: wd 2e-1.",
 46: "Gate+focal-revert arm: bst_24 / baseline, drop-unknown.",
 47: "WD sweep: wd 4e-1. Best mean min-F1 of the wd magnitudes (0.498), clears 0.5 on wrist_smash for the best serial.",
 48: "Gate+focal-revert arm: une_v1_14 / v2, the taxonomy the alpha-revert was built to target.",
 49: "wd 1e-2, gate off: shuttleset_18 / v2. Floor is driven_flight (n/a); macro tracks the standard.",
 50: "wd 4e-1, gate off: shuttleset_18 / v2. Small macro nudge over the standard; floor n/a.",
 51: "wd 1e-2, gate off: bst_24 / v2. Exclusion lifts the floor over the all-layers standard; 4e-1 lifts it more.",
 52: "wd 4e-1, gate off: bst_24 / v2. Best floor here, +5.4% mean min over the standard; the keeper, new bst_24/v2 best.",
 53: "wd 1e-2, gate off: bst_12 / v2. Exclusion grabs the floor; the bst_12 keeper (4e-1 adds nothing).",
 54: "wd 4e-1, gate off: bst_12 / v2. No gain over 1e-2, slightly over-regularises.",
 55: "wd 1e-2, gate off: bst_25 / baseline. Keep-unknown: exclusion drops the floor below the standard.",
 56: "wd 4e-1, gate off: bst_25 / baseline. Recovers most of 1e-2's drop, still under the standard; bst_25 stays on it.",
 57: "wd 1e-2, gate off: bst_24 / baseline. Big floor lift over the standard; 4e-1 edges it further.",
 58: "wd 4e-1, gate off: bst_24 / baseline. Best floor, +6.3% mean min; the keeper, new bst_24/baseline best.",
 59: "wd 1e-2, gate on (focal_alpha_revert): shuttleset_18 / v2. No gain over gate-off; floor n/a.",
 60: "wd 4e-1, gate on (focal_alpha_revert): shuttleset_18 / v2. No gain over gate-off; floor n/a.",
 61: "wd 1e-2, gate on (focal_alpha_revert): bst_25 / baseline. Recovers the bare-1e-2 floor drop, still under the standard.",
 62: "wd 4e-1, gate on (focal_alpha_revert): bst_25 / baseline. No better than gate-off 4e-1; under the standard.",
 63: "wd 1e-2, gate on (focal_alpha_revert): une_v1_14 / v2. Floor below the plain wd 4e-1 (#47).",
 64: "wd 4e-1, gate on (focal_alpha_revert): une_v1_14 / v2. Top une mean macro (0.748, 4 serials) but floor trails #47; alpha-revert retired.",
}

# --------------------------------------------------------------------------- #
# Chronological sweep-series partition (each run in exactly one)
# --------------------------------------------------------------------------- #
SERIES = [
 ("A", "BST paper baseline & LR / CG-AP schedule",
  "Apr 17–18. Reproduce the BST paper on bst_25 (keep-unknown, baseline split), retune the cosine LR so decay bites in-window, then ablate the CG/AP auxiliary-loss schedule (gentle / aggressive anneal, always-on, null).",
  [1, 2, 3, 4, 5, 6]),
 ("B", "New-taxonomy migration (une_merge_v1, split_v2, sticky-anchor)",
  "Apr 20–25. Switch from BST's 25-class merge to the UNE merge, move to the CSV flat pipeline, trial split_v2 over the BST split, fold in sticky-anchor pose cleaning, and test the no-sides collapse.",
  [7, 8, 9, 10]),
 ("C", "Phase-2 unified-data sanity (3 combos)",
  "Apr 29–30. Re-run three representative combos on the unified 32,203-stem sticky-anchor clean directory to confirm the full-extract data matches the Phase-1 baselines.",
  [11, 12, 13]),
 ("D", "Regularisation & loss sweep on une_v1_14",
  "Apr 30 – May 3. Hold the une_v1_14 / v2 baseline and sweep the loss: label smoothing (0.0 / 0.1 / 0.15), class weighting, the adaptive-focal CDB-F1 family (tau / gamma / alpha pair-cap), and an MLP-head capacity bump.",
  [14, 15, 16, 17, 18, 19, 20, 21, 22]),
 ("E", "Data-quality: shuttle-unzeroing & mask-wiring",
  "May 3–4. Stop zeroing the shuttle track on keypoint-fail (wipe_drop), add a post-TCN shuttle-missing channel (mask_wiring), and check the augmentation jitter actually helps (jitter-off).",
  [23, 24, 25]),
 ("F", "Augmentation framework v1 sweep",
  "May 5–6. Coupled centreline-flip + pos/shuttle jitter on the wipe_drop substrate, then a round-1 hparam sweep over p_flip / p_jitter / jitter caps.",
  [26, 27, 28, 29, 30, 31]),
 ("G", "Multi-taxonomy baseline batch (taxon-pinned)",
  "May 30–31. One clean cell per taxonomy on the taxon_pinned_w_preds collation: shuttleset_18, bst_24, bst_12, bst_25, une_v1_14, across v2 and baseline splits. Establishes the per-taxonomy reference on the final data.",
  [32, 33, 34, 35, 36, 37]),
 ("H", "Val-improvability-gate + focal-alpha-revert across taxa",
  "May 31 – Jun 1 (bourbaki). The val-gate + focal_alpha_revert_overallocated arm run across five taxonomies/splits. Ran interleaved with series I as one batch.",
  [38, 39, 41, 43, 46, 48]),
 ("I", "Weight-decay sweep (gate off, decay exclusion)",
  "May 31 – Jun 2 (carmack). AdamW weight decay with norms / bias / embeddings held out of decay, val-gate off. The une_v1_14 / v2 magnitude sweep (1e-2 through 4e-1) plus the two endpoints (1e-2, 4e-1) across the other five taxon / split cells. wd 4e-1 lifts the floor where it started lowest (bst_24 both splits, une); flat-to-down elsewhere. Default optimiser setting going forward: wd 4e-1 with the exclusion.",
  [40, 42, 44, 45, 47, 51, 52, 57, 58, 53, 54, 55, 56, 49, 50]),
 ("J", "Weight-decay endpoints x focal-alpha-revert (gate on)",
  "Jun 2 (carmack). The val-improvability gate + focal_alpha_revert_overallocated arm crossed with the two wd endpoints (1e-2, 4e-1), on bst_25 / baseline, une_v1_14 / v2 and shuttleset_18 / v2. Never the best config for any of the three; second batch after series H to show alpha-revert earns nothing, so it can be retired.",
  [61, 62, 63, 64, 59, 60]),
]
_seen = [n for _, _, _, nums in SERIES for n in nums]
assert sorted(_seen) == sorted(by_num), "series partition must cover every run exactly once"

# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
METRIC_HEAD = "Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean)"
METRIC_SEP = "---|---|---|---"


def metric_cells(r, flags=frozenset()) -> str:
    """Four `mean / best` cells. Macro/min figures get **bold** where the row
    holds the scope max for that figure (flags name which); acc/top-2 never bold."""
    def fmt(v):
        return f"{v:.4f}" if v is not None else "—"
    def cell(key, mean_flag, best_flag):
        mv, bv = r["mean"][key], r["best"][key]
        m = f"**{fmt(mv)}**" if mean_flag in flags else fmt(mv)
        b = f"**{fmt(bv)}**" if best_flag in flags else fmt(bv)
        return f"{b}<br>{m}"  # best stacked over mean: 2-line cell, keeps the column narrow
    return " | ".join((
        cell("macro_f1", "macro_mean", "macro_best"),
        cell("min_f1", "min_mean", "min_best"),
        cell("accuracy", None, None),
        cell("top2_accuracy", None, None),
    ))


# Which of macro-mean / macro-best / min-mean / min-best each row holds the max
# of, within a scope. Ties bold every row at the max.
_MAX_FIELDS = (
    ("macro_mean", "mean", "macro_f1"),
    ("macro_best", "best", "macro_f1"),
    ("min_mean", "mean", "min_f1"),
    ("min_best", "best", "min_f1"),
)


def winners(scope_rows) -> dict:
    """Return {run_num: set(flags)} naming the argmax row(s) of each figure.
    A figure whose scope max is 0 (an all-floor min-F1, e.g. shuttleset_18's
    driven_flight) is skipped, so a 0.0 never gets bolded or ticked."""
    flags: dict[int, set] = {}
    for flag, agg, key in _MAX_FIELDS:
        scored = [(r[agg][key], r["run_num"]) for r in scope_rows if r[agg][key] is not None]
        if not scored:
            continue
        mx = max(v for v, _ in scored)
        if mx <= 0:
            continue
        for v, rn in scored:
            if v == mx:
                flags.setdefault(rn, set()).add(flag)
    return flags


def tick_mean(flags) -> str:  # a mean-serial figure (macro or min) was the scope max
    return "✓" if flags & {"macro_mean", "min_mean"} else ""


def tick_best(flags) -> str:  # a best-serial figure (macro or min) was the scope max
    return "✓" if flags & {"macro_best", "min_best"} else ""


# Global table bolds the argmax per (canonical taxon, split) combo, for these
# six comparable cells only; legacy names fold in via TAX_FAMILY.
COMBOS = {
    ("shuttleset_18", "split_v2"),
    ("bst_24", "split_v2"),
    ("bst_12", "split_v2"),
    ("bst_25", "split_bst_baseline"),
    ("bst_24", "split_bst_baseline"),
    ("une_v1_14", "split_v2"),
}


def date_cell(r) -> str:
    return r["started_at"][:10]  # YYYY-MM-DD, date only; the time lives in the run_id


out = []
W = out.append

W("# BST-X Architecture-1 Training Runs\n")
W("All 64 recorded training runs of the BST-X (Architecture-1) model on ShuttleSet, built directly "
  "from the per-run `manifest.yaml` files under "
  "`src/bst_x/stroke_classification/main_on_shuttleset/experiments/`. Generated 2026-06-02.\n")
W("**Metrics from held-out test set.** Shows `best-serial / mean-across-serials`, to 4 dp. The "
  "best serial comes from the manifest's `best_serials` field where it's filled in (#1–31 and #49–64); "
  "otherwise, matched to the only `weights/` .pt retained.\n")
W("## Reading the tables\n")
W("- **# (run number)** is the global chronological index by `started_at`, 1–48. All tables match "
  "the global list.\n")
W("- **Taxonomy** is shown by its current canonical name (`pipeline/config.py` registry), with the "
  "legacy name the manifest stored in `[brackets]`. The one exception is legacy `une_merge_v1` "
  "(sided, 28-class): it has no canonical equivalent, so it's kept as-is. The names used "
  "here:\n")
W("  | Shown as | Legacy name in manifests | Classes | Sides | Unknown |\n  |---|---|---|---|---|\n"
  "  | `bst_25` | `merged_25` | 25 | yes | kept |\n"
  "  | `bst_24` | (native) | 24 | yes | dropped |\n"
  "  | `bst_12` | (native) | 12 | no | dropped |\n"
  "  | `une_v1_14` | `une_merge_v1_nosides` | 14 | no | dropped |\n"
  "  | `une_merge_v1` | (kept as-is) | 28 | yes | dropped |\n"
  "  | `shuttleset_18` | (native) | 18 | no | dropped |\n")
W("- **Split**: `split_v2` is the player-overlap-minimised split; `split_bst_baseline` is the "
  "BST paper's original partition. Six early runs stored no split column and default to "
  "`split_bst_baseline`.\n")
W("- **Test population** is set by split + unknown handling, so min-F1 is only comparable within the "
  "same population: baseline + keep-unknown = 3486 strokes, baseline + drop-unknown = 3335, "
  "split_v2 = 4202.\n")
W("- **Unknown ghost channel (`*`).** Nine early drop-unknown runs (#7–10, #12–16) trained with a "
  "dead `unknown` output slot that took softmax space but saw no samples and was never reported "
  "(drop-unknown hack), a bug patched at #17 and removed completely at the pinned collations (#32). "
  "They're marked `*` on the run number; full explanation: "
  "[appendix](#appendix-the-unknown-ghost-channel-era).\n")
W("- **shuttleset_18 min-F1 = 0.000** (#32, #38) is `driven_flight`: a single test clip, so its F1 "
  "is 0 if that clip is missed and ~1 if hit.\n")

# (the unknown ghost-channel write-up is built as an appendix at the end)

# 1. Global — bold the argmax per listed (taxon, split) combo; runs outside the
# six combos (the sided une_merge_v1 ones) get nothing.
W("\n---\n\n## 1. Global: all runs (chronological)\n")
W("Macro / min argmax is bolded **per (taxonomy, split) combo** here, not table-wide: "
  "the six comparable combos are shuttleset_18·v2, bst_24·v2, bst_12·v2, bst_25·baseline, "
  "bst_24·baseline, une_v1_14·v2 (legacy names fold in). The four sided `une_merge_v1` runs sit "
  "outside those, so they're left unbolded.\n")
W(f"Date | # | Run ID | Taxonomy [legacy] | Split | {METRIC_HEAD} | best | mean | Description")
W(f"---|---|---|---|---|{METRIC_SEP}|---|---|---")
_combo_groups: dict = {}
for r in rows:
    _combo_groups.setdefault((TAX_FAMILY[r["run_num"]], r["split"]), []).append(r)
global_flags: dict = {}
for _key, _grp in _combo_groups.items():
    if _key in COMBOS:
        global_flags.update(winners(_grp))
for r in rows:
    fl = global_flags.get(r["run_num"], frozenset())
    W(f"{date_cell(r)} | {num_cell(r['run_num'])} | {run_id_cell(r['run_id'])} | {tax_cell(r['run_num'])} | "
      f"{r['split']} | {metric_cells(r, fl)} | {tick_best(fl)} | {tick_mean(fl)} | {DESC[r['run_num']]}")

# 2. Per taxonomy
W("\n---\n\n## 2. Per taxonomy\n")
W("Grouped by taxonomy; split shown as a bold separator row within each. Run numbers stay global "
  "(`*` = unknown ghost channel; see appendix).\n")
TAXON_ORDER = ["bst_25", "bst_24", "bst_12", "une_v1_14", "une_merge_v1", "shuttleset_18"]
TAXON_HEAD = {
 "bst_25": "bst_25 (25-class, sided, keep-unknown), incl. legacy `merged_25`",
 "bst_24": "bst_24 (24-class, sided, drop-unknown)",
 "bst_12": "bst_12 (12-class, no-sides, drop-unknown)",
 "une_v1_14": "une_v1_14 (14-class, no-sides, drop-unknown), incl. legacy `une_merge_v1_nosides`",
 "une_merge_v1": "une_merge_v1 (early sided 28-class drop-unknown runs; no clean canonical name)",
 "shuttleset_18": "shuttleset_18 (18-class, no-sides, drop-unknown)",
}
SPLIT_ORDER = ["split_bst_baseline", "split_v2"]
for tax in TAXON_ORDER:
    trows = [r for r in rows if TAX_FAMILY[r["run_num"]] == tax]
    tflags = winners(trows)  # whole-table argmax (may span splits)
    W(f"\n### {TAXON_HEAD[tax]}\n")
    W(f"Date | # | Run ID | {METRIC_HEAD} | best | mean | Description")
    W(f"---|---|---|{METRIC_SEP}|---|---|---")
    for split in SPLIT_ORDER:
        srows = [r for r in trows if r["split"] == split]
        if not srows:
            continue
        W(f"**{split}** | | | | | | | | |")
        for r in srows:
            fl = tflags.get(r["run_num"], frozenset())
            W(f"{date_cell(r)} | {num_cell(r['run_num'])} | {run_id_cell(r['run_id'])} | "
              f"{metric_cells(r, fl)} | {tick_best(fl)} | {tick_mean(fl)} | {DESC[r['run_num']]}")

# 3. Per sweep / ablation series
W("\n---\n\n## 3. Per sweep / ablation series\n")
W("Ten chronological series, derived from the manifests' `ablation_id`, config knobs, dates and "
  "hosts. Each run belongs to exactly one. Series that mix taxonomies/splits carry both columns.\n")
for code, title, blurb, nums in SERIES:
    srows = [by_num[n] for n in nums]
    sflags = winners(srows)  # whole-series argmax
    W(f"\n### Series {code}: {title}\n")
    W(f"{blurb}\n")
    W(f"Date | # | Run ID | Taxonomy [legacy] | Split | {METRIC_HEAD} | best | mean | Description")
    W(f"---|---|---|---|---|{METRIC_SEP}|---|---|---")
    for n in nums:
        r = by_num[n]
        fl = sflags.get(n, frozenset())
        W(f"{date_cell(r)} | {num_cell(n)} | {run_id_cell(r['run_id'])} | {tax_cell(n)} | {r['split']} | "
          f"{metric_cells(r, fl)} | {tick_best(fl)} | {tick_mean(fl)} | {DESC[n]}")

# Appendix: the unknown ghost-channel write-up (prose, not a table)
W("\n---\n\n## Appendix: the unknown ghost-channel era\n")
W("From late April to 1 May, every drop-unknown run on a taxonomy that carried an `unknown` class "
  "trained with a dead `unknown` output slot. Worth knowing which runs, because it puts that batch in "
  "a slightly different architecture from everything after.\n")
W("**What was happening.** The model head was always sized to `taxonomy.n_classes`, and every "
  "taxonomy in the registry lists `unknown`. `drop_unknown=True` only told the collator to drop "
  "`raw_type_en == 'unknown'` rows; it never shrank the head. So a dropunk run on an unknown-bearing "
  "taxonomy got an `unknown` output channel that saw no positive samples, still ate a softmax slot, "
  "still took a label-smoothed target on every sample, and (in the one class-weighted run) pushed the "
  "class-weight renorm onto an n+1 basis. Never populated, never in per-class F1: a ghost. It turned "
  "up in the class-weighted run (#16, `run_20260501_110525`) on the morning of 1 May, where the live "
  "loss printout listed `unknown weight=0.882` for a class that never trained.\n")
W("**Which runs carried the ghost** (head dim read straight off the saved "
  "`mlp_head.mlp.mlp.3.weight`):\n"
  "- #7, #8, #9, #12 (sided `une_merge_v1`): 29-channel head, 28 real classes.\n"
  "- #10, #13, #14, #15, #16 (nosides `une_merge_v1_nosides`): 15-channel head, 14 real classes.\n\n"
  "All drop-unknown, all on or before 1 May morning. The fix doc names #13–16; the weights show the "
  "same ghost back to #7.\n")
W("**The one that looks like a ghost but isn't.** #11 (`run_20260429_202144`, merged_25 dropunk) has "
  "a 25-channel head reporting 24, but index-0 is fed by the 52 `driven_flight` rows (the merge map "
  "sends `driven_flight` to `unknown`), so that channel is trained, not dead. It reports 24 only "
  "because the baseline test split happens to carry no `driven_flight` clip. Its numbers stand. The "
  "genuine keep-unknown runs (#1–6 merged_25, #35/#43 bst_25) likewise train `unknown` as a real "
  "25th class and are fine.\n")
W("**Fixed in effect (1 May, #17 on).** The head started being derived from the classes actually "
  "present in the train labels, so dropunk une collapsed from 15 to 14. #17 (`run_20260501_164658`) "
  "is the first 14-channel run, the same afternoon as the morning catch; #17–31 are all clean.\n")
W("**Fully fixed (pinned collations, #32 on, 30 May).** The whole band-aid came out with the "
  "canonical taxonomy registry, where each taxon spells out its class list and excluded types "
  "directly, so the ghost can't recur by construction rather than by a runtime derivation.\n")
W("**Does it move the numbers?** The dead logit rarely wins an argmax, so the practical hit to the "
  "affected runs' macro / min / acc is small, but #7–10 and #12–16 dropunk do sit in a different "
  "architecture era from #17 on: not a like-for-like head against the clean runs. The weights are "
  "what they are, so there's nothing to redo.\n")
W("Full diagnosis and fix design: "
  "[`architecture_notes/unknown_channel_fix_review.md`]"
  "(architecture_notes/unknown_channel_fix_review.md).\n")

OUT.write_text("\n".join(out) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}  ({len(out)} lines, {len(rows)} runs)")
