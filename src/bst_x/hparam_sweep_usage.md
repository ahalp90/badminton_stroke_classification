# hparam_sweep usage guide

How to run a hparam search using the wrapper around `bst_x_train.py`.

The wrapper drives a list of "cells" (each one a hparam variant you want
to test) through the existing training pipeline, applies kill rules
between serials so dead-end cells don't burn full 5-serial wall-clock,
and writes a per-session search log you can come back to.

A full 5-cell search runs in ~10 hours on engelbart. Designed for
launch-and-walk-away.

For the design rationale and decision trail, see
`docs/architecture_notes/hparam_search_wrapper.md`. This file is
the how-to.

## The workflow

1. Make a new session dir with a template config.
2. Edit the config to define the cells you want to run.
3. Launch the wrapper.
4. Come back later, read the search log, pick the best cell.
5. If anything goes wrong mid-run, fix it and re-launch (resumes
   automatically).

## Step 1: make a session

```bash
PYTHONPATH=src/bst_x \
    python -m hparam_sweep --new-session aug_v1_round_1
```

This drops a session dir under
`experiments/bst_x/shuttleset/aug_hparam_sweep/`
with a template `config.yaml`. The dir name has a timestamp so multiple
sessions don't collide. Open the printed path in your editor.

## Step 2: write the config

The template gives you the shape. The fields you'll edit:

```yaml
session_name: aug_v1_round_1

reference:
  current_best_run: run_20260505_154907   # what the cells must beat
  wipe_drop_best_run: run_20260503_172922 # absolute floor reference

base_config:
  augmentation:
    p_flip:   0.5
    p_jitter: 0.3
    cap_y:    0.05
    cap_x:    0.10
    eps:      0.15

cells:
  - name: p_flip_25
    augmentation:
      p_flip: 0.25

  - name: cap_bump
    augmentation:
      cap_y: 0.075
      cap_x: 0.15

  - name: p_flip_25_x_cap_bump
    requires: "p_flip_25 != LOSE and cap_bump == WIN"
    augmentation:
      p_flip: 0.25
      cap_y: 0.075
      cap_x: 0.15
```

Cells run in YAML order. Each cell starts from `base_config.augmentation`
and overrides whatever keys it lists. Anything not listed inherits from
base. So `p_flip_25` runs with p_flip=0.25 and the rest from base.

`requires:` is optional. When set, it gates whether the cell runs based
on earlier cells' verdicts. Operators allowed: `==`, `!=`, `and`, `or`,
parentheses. Verdict values are `WIN` / `TIE` / `LOSE` / `SKIPPED` /
`FAILED`. If a `requires` clause evaluates False, the cell is skipped
and recorded with the reason.

Common patterns:
- `requires: "parent != LOSE"` — child runs unless parent was killed or
  scored worse than tolerated. Use this when the child only makes sense
  if the parent's knob isn't actively hurting things.
- `requires: "parent == WIN"` — strict; child only runs if parent
  genuinely improved. Use this for interaction tests where you want to
  combine two confirmed winners.
- `requires: "a == WIN and b == WIN"` — multi-parent gate.

## Step 3: launch

```bash
PYTHONPATH=src/bst_x \
    python -m hparam_sweep <session_dir>
```

That's it. The wrapper picks up `config.yaml`, validates it, and starts
the first cell. It calls `bst_x_train.py` per serial, watches the manifest
for completion, decides whether to kill or continue, and moves to the
next cell when done.

You'll see a header like this at the start of each cell:

```
======================================================================
Cell: cap_bump
Augmentation: {'p_flip': 0.5, 'p_jitter': 0.3, 'cap_y': 0.075, ...}
Progress: 1 done, 0 killed, 0 skipped, 2 pending after this cell

Reference (cell-start snapshot):
  Current best: run_20260505_154907
    Mean 0.7447 / 0.4779 / 0.7635 / 0.9394
  Wipe_drop best: run_20260503_172922
    Mean 0.7481 / 0.4742 / 0.7653 / 0.9353

Time estimate (no kills): ~6h00m, expected complete 04:30 06-May
======================================================================
```

And a footer at the end of each cell with the verdict and top class
movers. bst_x_train's own training logs print between, exactly as they
would on a manual run.

## Step 4: read the results

Two artefacts to look at:

**`<session_dir>/manifest.md`** is the human-readable search log. Top of
the file is a summary table: cell name, status, 5-serial mean (macro /
min), best serial, verdict. Below the table is one section per cell
with per-serial F1 lines, the picked best serial, deltas vs the
cell-start reference and vs wipe_drop, the verdict, and the top-3 class
movers.

**`<session_dir>/state.json`** is the orchestration state: which cells
have completed, their verdicts, the cell-start refs, etc. You don't
normally need to read this, but it's there if you want to grep across
the search.

The per-cell run dirs (`experiments/bst_x/shuttleset/run_<timestamp>_<microseconds>/`)
have the standard manifest.yaml + best_model_id.txt + TB events as
usual. Same format as a manual run, just driven by the wrapper.

## How the wrapper decides things

**Within a cell** (kill rules, applied after each serial completes):

- S1: never killed on macro. Min F1 floor at 0.38; below kills.
- S2: cumulative mean macro must stay within 2.5% of the cell-start
  ref. Min F1 floor at 0.40 from S2 onward.
- S3: macro tolerance tightens to 1.5%.
- S4: tightens further to 0.7%.
- S5: just record. No kill.

The thresholds are calibrated against the per-seed macro stdev observed
across past 5-serial runs (~0.006), so they're roughly 4-6 sigma at each
step. Generous enough to not false-kill bad-luck seeds, tight enough to
catch genuine dead ends.

**Between cells** (verdict, computed from the 5-serial mean):

- WIN: macro >= ref + 0.5% AND min >= ref - 0.5%.
- TIE: |macro - ref| < 0.5% AND min >= ref - 1.0%.
- LOSE: anything else, including any cell that got killed.

The +0.5% macro threshold is roughly 1.3 sigma above the
two-5-serial-mean comparison stderr, which is the "more likely real than
not" line. Tighter thresholds let noise in; wider miss small lifts.

**Reference promotion**: if a cell completes 5 serials with a mean
macro above the current best, it becomes the new ref for cells that
follow. Killed and failed cells never promote, even if their partial
mean was high. The cell-start ref each cell uses is snapshotted when
the cell starts and stays fixed for that cell's lifetime, so cells are
graded against a single consistent target.

**Top movers**: at the end of each cell, the wrapper prints the three
classes with the largest absolute change in mean F1 vs the
cell-start ref. Quick way to see what shifted, without having to scroll
through every per-class number.

## Step 5: things going wrong

**Wrapper killed mid-cell** (tmux death, network blip, Ctrl-C): just
re-run with the same session_dir. The wrapper reads
`<run_id>/manifest.yaml` to figure out how many serials actually
completed and picks up at the next one. The killed serial's metrics
(if it died mid-training) get re-rolled because bst_x_train doesn't pin
seeds across invocations: this is a small noise contribution but the
wrapper just records what it gets.

**bst_x_train returns non-zero** (CUDA OOM, transient driver thing,
filesystem hiccup): the cell is marked `failed`, verdict LOSE, and the
wrapper advances to the next cell rather than nuking the queue. The
search log makes failed cells obvious. If you want to investigate or
re-run that cell specifically, delete its run dir and its entry in
state.json's cells block, then re-launch.

**state.json is malformed** (you shouldn't see this, but): the wrapper
fails loudly at start with instructions. Run `git checkout --
state.json` from the session_dir to restore the previous good state, or
delete the file to start the session over from cell 1.

**You edit config.yaml mid-session**: the wrapper detects added or
removed cells on resume and refuses, with the cell names listed.
Either restore the original config or start a new session. Editing
augmentation values on an already-completed cell is invisible to the
wrapper (it doesn't re-validate cell config against state, only cell
names), but you'd be silently building a search log with mismatched
configs vs results, so don't.

**Two wrappers pointed at the same session**: a `.lock` file with the
PID of the running wrapper sits in the session dir. A second launch
checks the lock, sees the live PID, and refuses. If the original
wrapper crashed without releasing the lock, the next launch sees the
stale PID and clears it automatically.

## Tunables you might override

The `pruning` and `verdict` blocks at the top of config.yaml accept
overrides if you want to widen kill tolerance, tighten verdict
thresholds, or move the min F1 floor. Defaults match what the design
doc calibrated for the aug search; for an x3ds search where per-seed
variance might be different, you'd override here:

```yaml
pruning:
  macro_tolerance:
    s2: 0.025
    s3: 0.015
    s4: 0.007
  min_f1_floor:
    s1: 0.38
    s2_onward: 0.40
verdict:
  win_macro_delta: 0.005
  tie_macro_delta: 0.005
  win_min_delta: -0.005
  tie_min_delta: -0.010
high_variance_warn_stdev: 0.010
```

## Sanity-check before launch

```bash
PYTHONPATH=src/bst_x \
    python -m hparam_sweep --dry-run <session_dir>
```

Validates the config and prints the cell queue with their requires
clauses. Cheap way to catch typos in YAML before committing 10 hours of
GPU time. If the config has an issue (unknown ref run, malformed
requires, duplicate cell name) you'll see it here.

## What the wrapper does NOT do

- It doesn't change bst_x_train's training loop, loss, or any training
  knob beyond what you put in `augmentation:`. Whatever else is in the
  module-level Hyp namedtuple stays as-is for the run.
- It doesn't add seeds to bst_x_train. Re-running the same cell
  configuration produces a slightly different mean each time. That's
  the existing behaviour; the wrapper just exposes it.
- It doesn't run cells in parallel. Single GPU, single host, single
  cell at a time. Cells run sequentially in YAML order.
- It doesn't average across cells or combine results in any way beyond
  the per-cell 5-serial mean. Each cell is its own data point.
- It doesn't touch existing per-run outputs (manifest.yaml,
  best_model_id.txt, test logs, TB). Those stay in the format the rest
  of the project expects, unchanged.

## Where to find what

- Design rationale, kill rule calibration, decisions log:
  `docs/architecture_notes/hparam_search_wrapper.md`.
- Wrapper code: `hparam_sweep.py` (alongside this file).
- bst_x_train CLI changes: `bst_x_train.py` (the `__main__` block at the
  bottom).
- Tests: `tests/test_hparam_sweep.py` at the repo root.
- Search log of a running/completed session:
  `<session_dir>/manifest.md`.
- Orchestration state: `<session_dir>/state.json`.
- Per-cell run output (one per cell):
  `experiments/bst_x/shuttleset/run_<timestamp>_<microseconds>/`.
