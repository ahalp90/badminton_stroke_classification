# Hparam search wrapper: planning doc

A small orchestration layer around `bst_train.py` for sequencing
multi-cell hparam searches with within-cell pruning, between-cell
verdict-conditional skipping, and resumption from abrupt termination.

First use case: the augmentation-framework hparam sweep (p_flip,
p_jitter, cap_y, cap_x, eps). Designed so the same wrapper carries
into the x3ds hparam search later with cell-config edits, not a
rebuild.

## Scope

In scope:
- Sequence cells from a YAML config in priority order.
- Drive `bst_train.py` per cell, capturing per-serial test metrics.
- Apply within-cell early-kill rules on macro F1 and a min F1 floor.
- Apply between-cell verdict logic: only run conditional cells if
  parents earned the right verdict.
- Maintain resumable state across abrupt termination.
- Write a per-session search log that's human-readable end-to-end.

Out of scope:
- A surrogate model or response-surface fitting (Optuna and friends).
  This is a hand-sequenced search, not Bayesian optimisation.
- Distributed orchestration. Single-host, single-GPU on engelbart.
- Modifications to `bst_train.py` beyond what's strictly needed for
  per-serial invocation (see open question 1).
- A web dashboard or live UI. The search log markdown file is the UI.

## Architecture

The wrapper is one Python script (`src/bst_refactor/stroke_classification/main_on_shuttleset/hparam_sweep.py`) that:
1. Loads a cell-config YAML (the search plan).
2. For each cell in queue order: sets the augmentation hparams,
   invokes bst_train one serial at a time, parses the per-serial test
   output as it lands, applies kill rules, writes per-serial lines to
   the search log, finally writes the cell verdict.
3. Updates `state.json` after every state transition for resumption.
4. Skips cells whose `requires:` clause is unsatisfied by parent
   verdicts.

It does NOT:
- Re-implement training. It calls the existing entry point.
- Re-implement test-metric collection. It reads from the per-run
  manifest.yaml that bst_train already writes.
- Touch the existing per-run output format (manifest.yaml,
  best_model_id.txt, test logs, TB events). Those are written by the
  existing pipeline; the wrapper just reads them.

The wrapper lives at
`src/bst_refactor/stroke_classification/main_on_shuttleset/hparam_sweep.py`,
beside `bst_train.py`. Despite being generic in design, it directly
invokes bst_train.py and shares its experiments/ output tree, so
sitting alongside bst_train.py is the natural home rather than the
top-level `scripts/`. Search-session outputs land in
`src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/aug_hparam_sweep/`.

## Cell config schema

YAML, one file per search session, copied into the session subdir on
launch so the run is reproducible from the session dir alone.

```yaml
session_name: aug_v1_search_round_1   # used in dir name
reference:
  current_best_run: run_20260505_154907
  wipe_drop_best_run: run_20260503_172922
base_config:
  # The locked baseline that single-knob cells diff from.
  augmentation:
    p_flip: 0.5
    p_jitter: 0.3
    cap_y: 0.05
    cap_x: 0.10
    eps: 0.15
cells:
  - name: p_flip_25
    augmentation:
      p_flip: 0.25
    # Other aug fields inherit from base_config.

  - name: cap_bump
    augmentation:
      cap_y: 0.075
      cap_x: 0.15

  - name: p_jitter_40
    augmentation:
      p_jitter: 0.4

  - name: p_flip_25_x_p_jitter_30
    requires: "p_flip_25 != LOSE"
    augmentation:
      p_flip: 0.25
      p_jitter: 0.3   # explicit even though it matches base

  - name: p_flip_25_x_cap_bump
    requires: "p_flip_25 != LOSE and cap_bump == WIN"
    augmentation:
      p_flip: 0.25
      cap_y: 0.075
      cap_x: 0.15
```

Cells are run in YAML order. `requires:` is a tiny boolean expression
over parent verdicts (`==`, `!=`, `and`, `or`). Cell names are unique
and used as keys in state.json.

## Pruning rules

### Within-cell: macro tolerance schedule

After each serial completes, compute the cumulative mean macro across
serials run so far. Compare to `reference.current_best_run`'s mean
macro (read from its manifest.yaml). Kill the cell if the cumulative
mean falls below `ref - tolerance(k)`:

| k completed | tolerance | rationale |
|-------------|-----------|-----------|
| 1           | never kill | one serial too noisy |
| 2           | 2.5%      | ~6 sigma below ref; only kills clear losers |
| 3           | 1.5%      | ~4 sigma; catches dead ends with confidence |
| 4           | 0.7%      | tight; only spares cells that can plausibly recover |
| 5           | n/a, record | done either way |

Sigma values are based on the macro stdev observed across the existing
5-serial runs (~0.006), giving partial-mean stderr of stdev/√k.

### Within-cell: min F1 floor

Empirical floor across all winning/tying runs: lowest single-serial
min F1 stayed >= 0.435. Jitter-off ablation hit 0.39, 0.36 single
serials in a clear-loser run.

Rule (tiered to absorb S1 noise, since min F1 is ~2x noisier than
macro per-serial):
- **S1**: kill if single-serial min F1 < 0.38.
- **S2 onward**: kill if single-serial min F1 < 0.40.

The S1 widening gives one bad-luck serial of absorption. The S2+
threshold sits 3.5pp below the empirical worst single-serial min F1
across winning runs, which is a real margin.

### Between-cell: verdicts

After all surviving serials of a cell complete (5 if not killed), the
cell gets a verdict relative to the current best:

- **WIN**: mean macro >= ref + 0.5% AND mean min F1 >= ref - 0.5%.
  +0.5% is roughly 1.3 sigma on the 5-serial-mean comparison
  (combined stderr ~0.0038, see Verdict thresholds calibration);
  above that, the lift is more likely real than not.
- **TIE**: |mean macro - ref| < 0.5% AND mean min F1 >= ref - 1.0%.
  Within ~1 sigma on macro and floor doesn't materially drop.
- **LOSE**: anything else (mean macro below ref - 0.5%, or floor
  collapsed beyond TIE tolerance, or the cell was killed).

Verdicts are recorded in state.json. Conditional cells read parent
verdicts via the `requires:` clause and skip if unsatisfied.

### Reference promotion and snapshotting

Two distinct concerns:

**Promotion**: which cell becomes `current_best_run` for *future*
cells.
- Promotion requires `serials_done == 5`. Killed cells (k < 5) never
  promote, regardless of partial mean. A 2-serial cumulative mean
  has different stderr than a 5-serial mean and isn't comparable.
- Among completed cells (5 serials), the highest-mean-macro cell
  promotes. Verdict (WIN/TIE/LOSE) does not gate promotion: a TIE
  with macro 0.7449 vs prior best 0.7447 still promotes, because the
  kill comparisons should be against the empirical max we have.

**Snapshotting**: the comparison reference each cell uses is *frozen
at cell start*. When cell N starts, it captures the current value
of `current_best_run`'s mean as its `kill_ref` and `verdict_ref`.
Even if a previous cell promoted moments ago, the new cell uses
the post-promotion ref for the entire cell's lifetime. The ref does
not float during the cell's run.

Why snapshot: without it, S2 of cell N+1 might be compared against
ref X, while S4 of the same cell uses ref X+0.005 (because cell N-1
finished and promoted between S2 and S4). The kill schedule then
applies inconsistent thresholds within a single cell. Snapshotting
locks the cell's comparison frame.

**Top movers** (the per-cell empirical readout): also computed
against the cell-start snapshot — same ref the verdict was made
against. Keeps the search log internally consistent.

The wipe_drop reference stays static across the entire search.

See "Verdict thresholds" calibration discussion below.

## Reference display

At cell start and cell end, the wrapper prints (and appends to the
search log):

```
=== Reference ===
Current best: run_20260505_154907 (p_jitter=0.3)
  Mean 0.7447 / 0.4779 / 0.7635 / 0.9394
  S5 picked: macro 0.7479, min 0.5147, smash 0.515 / ws 0.519

Wipe_drop best: run_20260503_172922
  Mean 0.7481 / 0.4742 / 0.7653 / 0.9353
```

At cell end, immediately after, with explicit delta lines:

```
=== Cell complete: cap_bump ===
Mean 0.7468 / 0.4810 / 0.7651 / 0.9412
Best serial: S4 (macro 0.7501, min 0.4925)
Vs current best: macro +0.2, min +0.3, acc +0.2, top-2 +0.2 — TIE
Vs wipe_drop:    macro -0.1, min +0.7, acc ~0,   top-2 +0.6
Verdict: TIE. Children: 1 cell pruned (cap-pair conditional needs WIN).
```

Best-serial pick rule, mechanical: highest min F1, ties broken on
macro. Mirrors the manual rule we've been using.

## Status updates at cell start

Printed once per cell, before the first serial:

```
Cell 3/5: cap_bump (cap_y=0.075, cap_x=0.15)
Queue ahead: 2 cells (p_jitter_40, then conditional pairs)
Estimated time remaining: ~6hr (no kills assumed)
Started at 14:32, expected complete ~16:35
```

Queue length = total not-yet-pruned cells in the session config. Pruned
cells aren't counted. Time estimate uses the empirical 25min/serial
on engelbart (~2hr per full 5-serial cell): `25min ×
serials_remaining_in_current_cell + 2hr ×
not_yet_pruned_cells_ahead`. Refreshed on each serial completion.

## Search log format

`experiments/aug_hparam_sweep/sweep_<start_timestamp>/manifest.md`,
markdown, appended-to as the session progresses. Top of file is a
running summary table; below that, one section per cell.

```markdown
# Aug hparam search: sweep_<timestamp>

Reference at start:
- Current best: run_20260505_154907 (mean 0.7447 / 0.4779 / 0.7635 / 0.9394)
- Wipe_drop best: run_20260503_172922 (mean 0.7481 / 0.4742 / 0.7653 / 0.9353)

## Summary

| Cell | Status | 5-serial mean (macro / min) | Best S | Verdict |
|------|--------|------------------------------|--------|---------|
| p_flip_25 | complete | 0.7421 / 0.4823 | S2 | TIE |
| cap_bump | running (S3 done) | 0.7468 / 0.4810 partial | — | — |
| p_jitter_40 | pending | — | — | — |
| p_flip_25_x_p_jitter_30 | pending (gated) | — | — | — |
| p_flip_25_x_cap_bump | pending (gated) | — | — | — |

## Cell: p_flip_25
Run id: run_20260506_HHMMSS
Config diff vs base: p_flip 0.5 → 0.25
Started: 2026-05-06 09:00. Completed: 2026-05-06 21:30.

- S1: macro 0.7402, min 0.4720 — cumulative mean 0.7402 / 0.4720
- S2: macro 0.7355, min 0.4901 — cumulative mean 0.7378 / 0.4811
- S3: macro 0.7448, min 0.4791 — cumulative mean 0.7402 / 0.4804
- S4: macro 0.7470, min 0.4920 — cumulative mean 0.7419 / 0.4833
- S5: macro 0.7430, min 0.4783 — cumulative mean 0.7421 / 0.4823

PICK: S2 (min 0.4901, macro 0.7355).
Vs current best (run_20260505_154907 mean): macro -0.3, min +0.4. TIE.
Vs wipe_drop: macro -0.6, min +0.8.
Verdict: TIE. Children: p_flip_25_x_p_jitter_30 stays in queue.

Top movers vs current best: cross_court_net_shot +3.2, smash -1.4, drive +1.1.

## Cell: cap_bump
... (in progress)
```

The "Top movers" line is appended automatically at cell completion:
the three classes with the largest absolute delta in 5-serial mean
F1 vs the current best run's per-class means. No YAML setup, no
hypothesis-specific config. Cheap to compute, useful for any cell.

The search log is the canonical human-readable artefact for the
search. The per-run manifest.yaml and best_model_id.txt continue
to be written for each cell's run, untouched by the wrapper.

## State management and resumption

`state.json` in the session dir, atomic-write (write to .tmp then
rename). Updated after every transition. If a write or rename fails
(rare; ENOSPC, permission, readonly mount), the wrapper halts
immediately rather than continuing with stale persisted state. The
in-memory state at that point cannot be trusted to match disk and
running on would risk a divergence that resume can't recover from.

```json
{
  "session_name": "aug_v1_round_1",
  "session_dir": "/.../experiments/aug_hparam_sweep/sweep_20260506_090000_aug_v1_round_1",
  "current_best_run": "run_20260505_154907",
  "current_best_mean": {
    "macro_f1": 0.7447, "min_f1": 0.4779,
    "accuracy": 0.7635, "top2_accuracy": 0.9394
  },
  "wipe_drop_best_run": "run_20260503_172922",
  "wipe_drop_best_mean": {"macro_f1": 0.7481, "min_f1": 0.4742, "...": "..."},
  "cells": {
    "p_flip_25": {
      "status": "complete",
      "run_id": "run_20260506_090015_123456",
      "verdict": "TIE",
      "serials_done": 5,
      "mean": {"macro_f1": 0.7421, "...": "..."},
      "kill_ref_macro": 0.7447,
      "verdict_ref_macro": 0.7447,
      "verdict_ref_min": 0.4779,
      "verdict_ref_per_class": {"smash": 0.5815, "...": "..."},
      "top_movers": [["smash", -0.018], ["...", "..."]],
      "best_serial": 5,
      "macro_stdev": 0.0072
    },
    "cap_bump": {
      "status": "running", "run_id": "run_20260506_213000_098765",
      "verdict": null, "serials_done": 3,
      "cumulative_mean": {"macro_f1": 0.7468, "...": "..."},
      "kill_ref_macro": 0.7449
    },
    "p_jitter_40": {"status": "pending"}
  }
}
```

Means are stored as dicts (not arrays) so search-log writers and tests
can read by metric name without tracking position. Cell entries also
carry `failed_reason` / `failed_at_serial` (when bst_train returns
non-zero), `killed_reason` / `killed_at_serial` (kill rule trip), and
`skipped_reason` (requires not satisfied). Statuses cover `pending`,
`running`, `complete`, `killed`, `failed`, `skipped`.

### Resumption flow

Re-run the wrapper against the same session_dir; resumption is implicit
(no `--resume` flag, the wrapper checks for an existing `state.json`
and picks up where it left off):

1. Read state.json. Identify the cell with status="running".
2. Inspect that cell's run dir: read manifest.yaml's `serials:` list
   to confirm the actual count of completed serials.
3. Reconcile state.json's `serials_done` to manifest.yaml's count.
   Manifest is authoritative — if state.json says 4 but manifest only
   has 3 entries, the wrapper died mid-S4. Resume from S4.
4. Re-run from the next missing serial. Note: bst_train.py does not
   currently set explicit seeds (no `torch.manual_seed` etc.), so
   each Python invocation pulls a fresh OS-random seed at process
   start. A re-run S4 will produce *different* metrics than the
   original S4 would have. The wrapper records what it gets — it
   doesn't depend on determinism — but the cell's 5-serial mean is
   then over a slightly different mix of serials than originally
   planned. Small noise contribution. If bit-exact reproducibility
   becomes important, add `torch.manual_seed(serial_no)` etc. in
   bst_train's serial setup; that's a separate, deliberate change.
5. Continue through remaining serials and remaining cells per normal.

Failure modes handled:
- **Wrapper killed mid-training**: re-run that serial fresh on resume.
- **Wrapper killed between cells**: state.json has the verdict and
  the cell's run dir is complete; resume picks up at the next cell.
- **bst_train exits non-zero (CUDA OOM, transient driver hiccup,
  filesystem blip)**: the wrapper marks the cell `failed`, records the
  exit code in `failed_reason`, sets verdict LOSE, and advances to the
  next cell. The session keeps running rather than nuking the queue
  for one bad serial. Failed cells appear with status `failed` in the
  search log so they're visible on return.
- **state.json corrupted/malformed**: wrapper fails loudly on resume
  with a clear error message pointing to git restore. There's no
  auto-rebuild from per-cell manifests because the session dir is
  expected to be tracked in git, so `git checkout -- state.json`
  is the recovery path. Cells already completed have their record
  intact in state.json's prior commit.
- **Config drifted between sessions**: the wrapper detects added or
  removed cell names on resume and refuses with a clear message.
  Cell-name keying is strict; mid-session config edits would silently
  KeyError or skip cells the user expected to run.
- **Two wrapper processes pointed at the same session_dir** (typically
  a tmux pane rebound to the same job): the wrapper writes a `.lock`
  file with its PID at session start and refuses to launch if the
  lock is held by a still-alive process. Stale locks (PID gone) are
  cleared silently.

`requires:` clauses can reference any earlier cell's outcome by name.
Verdicts populate the namespace with `WIN` / `TIE` / `LOSE`; cells
that were skipped (parent didn't satisfy their requires) appear as
`SKIPPED`; cells where bst_train returned non-zero appear as
`FAILED`. Use `parent != LOSE` if you want to gate on "not killed";
`parent == WIN` if you want strict promotion. SKIPPED is treated as
neither WIN nor LOSE — write the clause defensively if you care.

The dual sources of truth — state.json for orchestration, manifest.yaml
for per-cell ground truth — are intentional. State.json captures
queue and verdict; manifest.yaml captures run output. Each is
authoritative for what it owns.

## Per-serial invocation

The existing `bst_train.py` runs all 5 serials in one Python process
inside `train_network`. For the wrapper to apply per-serial pruning
and resume from a partial cell, training has to be invokable
serial-by-serial. Two approaches:

**Option A**: add a `--serial-no` CLI flag that runs only the named
serial (1-5) and exits. Wrapper calls bst_train 5 times per cell
sequentially. Cleanest. Modifies bst_train's entry point but not
its core training logic.

**Option B**: leave bst_train's serial loop intact, have the wrapper
poll the cell's manifest.yaml for new entries while training runs in
background. Pruning is best-effort: kills a running training process
mid-serial if the previous serial's metrics tripped a kill rule.
Messier (PID management, signal handling) and you can't kill a serial
that's already past its midpoint cleanly.

Resolved: Option A (see Decisions log below).

## CLI

```
python -m main_on_shuttleset.hparam_sweep --new-session <name>
python -m main_on_shuttleset.hparam_sweep <session_dir>
python -m main_on_shuttleset.hparam_sweep --dry-run <session_dir>
```

`--dry-run` prints the queue, the parsed `requires:` graph, and the
estimated time, then exits. Useful for catching YAML errors before
launching a 20+hr run.

## Reuse for x3ds

The wrapper has no aug-specific assumptions in its core logic. The
cell config is a generic "patch this hparam dict, run, parse metrics".
For x3ds, the changes would be:

- `base_config` carries the x3ds defaults instead of aug defaults.
- Cells diff x3ds-relevant fields (likely a different sub-dict than
  `augmentation`).
- Reference reads pull from x3ds's existing best run.

Things that should stay generic, not aug-specific:
- Tolerance schedule values (could be overridden per-config if x3ds
  variance is different).
- Verdict thresholds (likewise, configurable).
- Min F1 floor (configurable; 0.40 is aug-specific empirical).

These should be top-level keys in the cell config so future searches
can override without touching wrapper code:

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

## What touches existing code

- `src/bst_refactor/stroke_classification/main_on_shuttleset/hparam_sweep.py`: new file.
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py`:
  add argparse with `--serial-no`, `--run-id`, `--log-path`, and the
  five augmentation overrides (`--p-flip`, `--p-jitter`, `--cap-y`,
  `--cap-x`, `--eps`). Gate the serial loop on `--serial-no` when
  set. Flip test log open mode to `'a'` when serial_no > 1. Re-gate
  `_validate_and_record_arch` on "manifest doesn't already have an
  arch block" so per-serial invocations don't double-validate or
  miss validation on resumed cells. ~80-120 lines total.
- Existing per-run output (manifest.yaml, best_model_id.txt, test
  logs, TB events): format unchanged. The wrapper reads, doesn't
  write.
- `tests/test_hparam_sweep.py`: new test file.

## Estimated build size

Wrapper (`hparam_sweep.py`): ~800 lines (over the initial ~300-400
estimate; the per-cell rendering, console blocks, and lockfile / drift
detection added more than expected), covering config parsing,
kill rules, verdict logic, search-log writing, state.json IO, resume
flow, and CLI.

bst_train.py changes: ~80-120 lines, larger than initially estimated.
Reason: bst_train.py currently has zero argparse — the `Hyp` namedtuple
is constructed inline at the bottom of the file and `resume_from` is
hardcoded `None`. Adding `--serial-no` is one flag, but we also need:
- `--run-id` to resume into an existing run dir
- `--log-path` to pin the test log file across per-serial invocations
- `--p-flip`, `--p-jitter`, `--cap-y`, `--cap-x`, `--eps` to inject
  the cell's augmentation config from the wrapper

Either as individual flags (cleaner, more lines) or a single
`--config-overrides path/to/yaml` (uglier, smaller diff). I lean
individual flags — explicit and greppable.

Tests: ~400-600 lines spanning unit + integration + smoke. The mock
`bst_train` shim is ~50 lines.

Not an infra project. Sensible orchestration over the existing pipeline.

## Decisions log

All previously open questions resolved. Recording the decisions
here so the design rationale is preserved.

**Per-serial invocation: Option A confirmed.** Add `--serial-no N`
flag to bst_train. Wrapper invokes per-serial. The existing
`resume_from` parameter, `track_run`/`track_serial` infrastructure,
and the serial loop at bst_train.py:1124 already support resuming
into an existing run dir; the change is small (~30-50 lines added,
no core training-logic changes):

1. CLI flag `--serial-no N` parsed alongside the existing args.
2. The serial loop at bst_train.py:1124 gates on the flag: when set,
   run only that serial and exit.
3. Test log open mode (bst_train.py:1122 `'w'`) flips to `'a'` when
   `--serial-no > 1`. Otherwise S2-S5 invocations would clobber the
   S1 block.
4. `_validate_and_record_arch` is currently `serial_no == 1`-gated.
   Re-gate on "manifest doesn't already have an arch block" so re-runs
   of S1 (or per-serial invocations starting at S2+ in a resumed
   cell) don't double-validate or skip-validate incorrectly.

**Wrapper location: stroke_classification/main_on_shuttleset/.**
Sits beside bst_train.py rather than top-level `scripts/`. It's
a per-project orchestration tool, not a generic utility — the
location reflects its dependency on bst_train.py.

**Cell-config YAML: session-dir SSOT.**
- Config lives at `<session_dir>/config.yaml`, never duplicated.
- New session: `python hparam_sweep.py --new-session <name>` creates
  `experiments/aug_hparam_sweep/sweep_<timestamp>_<name>/`, drops a
  template config.yaml in, exits. User edits, then runs
  `python hparam_sweep.py <session_dir>`.

**Verdict thresholds: ±0.5% on macro, ±1.0% on min.**
Reasoning: per-seed macro stdev observed ~0.006, so each 5-serial
mean's stderr is 0.006/√5 ≈ 0.00268. Comparing two such means, the
combined stderr is √2 × 0.00268 ≈ 0.0038. +0.5% macro ≈ 1.3 sigma
— "more likely than not real". Tighter (+0.2-0.3%) declares noise
as wins; wider (+0.7-1.0%) misses small-but-real lifts. Min F1 has
~2x the noise of macro (single-serial range 0.435-0.515), so its
guard rail is wider at ±1.0%.

Caveat: these thresholds assume the cell's per-seed variance is
similar to the baseline runs we calibrated against (~0.006 macro
stdev). A cell with substantially higher variance (say S1-S5 stdev
> 0.010) would have wider stderr; the wrapper logs a warning when
it sees per-seed stdev that high and the verdict should be treated
as advisory. Doesn't change the math; just flags when the
calibration may not hold.

**Reference promotion: highest-mean-macro cell becomes
current_best regardless of WIN/TIE.**
The verdict tree governs whether children run; the reference
governs the kill threshold. Promote unconditionally on higher
mean so kill comparisons are against the empirical max.

**Focus class: dropped. Top-3 movers always shown.**
Always-on, computed mechanically as the three classes with the
largest absolute delta in 5-serial mean F1 vs current best's
per-class means. No per-cell config field. Hypothesis-driven cells
and exploratory cells both get the same empirical readout.

**Time estimates: 25min/serial empirical constant.**
From manifest `recorded_at` deltas across the recent runs:
~25min/serial on engelbart, ~2hr per 5-serial cell. Full 5-cell
search ~10hr without kills. Wrapper formula:
`25min × serials_remaining_in_current_cell + 2hr ×
not_yet_pruned_cells_ahead`. Refresh displayed estimate at every
serial completion.

**Git tracking: full session dir.**
Track everything in `experiments/aug_hparam_sweep/sweep_*/`:
config.yaml, state.json, manifest.md. Total ~30KB per session;
state.json being tracked lets you diagnose where a search died
later. Per-cell run dirs continue to use the existing weights
gitignore rules.

## Test suite

The wrapper has enough state machinery (kill rules, verdict logic,
resume reconciliation) to warrant unit + integration tests. Same
location as the augmentation tests: `tests/test_hparam_sweep.py`.

Mock approach: a fake `bst_train` shim that, when invoked with
`--serial-no N` and a `--run-dir`, simulates training by writing a
manifest.yaml entry with pre-canned metrics and a (small) fake
weight file. Tests scriptable via a `metrics_fixture.yaml` defining
per-cell-per-serial fake outputs. No actual training in the test
suite; the wrapper's logic is what we're testing.

### Unit tests

- **Config parsing**:
  - Valid YAML with all fields → loads cleanly.
  - Missing `base_config` → rejects with clear error.
  - Cell with unknown field in `augmentation:` → rejects.
  - `requires:` referencing unknown cell name → rejects.
  - `requires:` parser: simple equality, `!=`, `and`, `or`,
    parenthesised expressions, malformed.
- **Kill rules: macro tolerance**:
  - S1 cumulative below ref by any amount → never kill (S1 exempt).
  - S2 cumulative at ref - 0.025 → kill triggered.
  - S2 cumulative at ref - 0.024 → no kill.
  - S3 cumulative at ref - 0.015 → kill triggered.
  - S3 cumulative at ref - 0.014 → no kill.
  - S4 cumulative at ref - 0.007 → kill triggered.
  - S4 cumulative at ref - 0.006 → no kill.
  - S5 → never kill, just record.
- **Kill rules: min F1 floor**:
  - First serial min F1 = 0.40 → no kill (boundary, exclusive).
  - First serial min F1 = 0.399 → kill.
  - First serial min F1 = 0.50 → no kill.
- **Verdict computation**:
  - Mean macro = ref + 0.005, min = ref → WIN.
  - Mean macro = ref + 0.0049, min = ref → TIE.
  - Mean macro = ref + 0.005, min = ref - 0.0051 → LOSE (min guard
    rail tripped).
  - Mean macro = ref - 0.005, min = ref → TIE.
  - Mean macro = ref - 0.0051, min = ref → LOSE.
  - Cell killed mid-run → LOSE regardless of partial mean.
- **Top-3 movers**:
  - Computes correctly when current_best has 14 classes and cell
    has same.
  - Ties broken by class order (deterministic).
- **Reference promotion**:
  - Cell with TIE verdict but mean > current_best mean → promotes.
  - Cell with WIN verdict but mean < current_best (impossible by
    definition, but defensive test) → no promotion.
  - LOSE cell → no promotion regardless of mean.
- **Search-log writing**:
  - Per-serial line format matches spec.
  - Top-of-file summary table updates correctly across multiple
    cells.
  - PICK selection: highest min F1, ties broken on macro.

### Integration tests

- **State.json round-trip**: serialise full state, write atomically,
  read back, deep-compare.
- **Resume from clean state.json**: state matches manifest.yaml on
  disk → wrapper resumes seamlessly.
- **Resume from divergent state.json**: state.json claims 4 serials
  done, manifest.yaml has 3 → wrapper trusts manifest, runs S4.
- **Resume after mid-cell kill**: state has cell killed at S3,
  resume → wrapper skips that cell, advances to next.
- **Resume from corrupted state.json**: state.json missing/malformed
  → wrapper rebuilds from session dir's per-cell manifest.yamls.
- **Conditional cell skipping**: parent verdict LOSE → child cell
  marked skipped, doesn't run.
- **Conditional cell skipping with multi-parent requires**: one
  parent LOSE one parent WIN, `and` clause → skipped; `or` clause
  → runs.
- **End-to-end happy path**: 2-cell config, mock bst_train returning
  pre-set metrics, run to completion, assert search log is fully
  populated and final state.json correct.
- **End-to-end with kill**: cell 1 metrics trip S3 macro tolerance,
  wrapper kills, advances to cell 2. Assert cell 1's state is
  killed-at-S3 and cell 2 ran fresh.
- **Concurrent re-launch protection**: try to launch a wrapper
  pointed at a session dir whose state.json says a cell is currently
  running → wrapper detects and refuses (lock file or status check).

### Smoke tests

A single end-to-end test that uses the real bst_train.py with a
2-epoch + 1-class subset of data, 2 serials, 1 cell. Verifies
the wrapper's invocation pattern actually drives bst_train as
intended. Slow (~30s) but worth one.
