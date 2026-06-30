# Simplification pass: history for the repo

The durable record of the simplification pass. The two review docs were the input
findings; the summary + worklog are the per-commit landed-on-main trail. The split
function-invariant docs that gated the structural items live alongside at
`../../function_invariants/`.

Merged to `main` 2026-06-30 at `18e5c2c` ("Merge simplification + cleanup pass").
17 commits on `refactor/simplification-pass` landed under a single `--no-ff` merge.
Working tree clean throughout, pytest baseline held (456 / 2 known-red / 19 skipped
on the laptop; 473 / 2 / 2 on HPC).

## What's in this dir

- `refactor_summary.md` -- plain-language summary. What each batch did, the main
  wins, what stayed put
- `refactor_worklog.md` -- per-commit log: files / change / gate / SHA / verifier
  verdict. Audit trail for the diff against `main`
- `simplification_review.md` -- the review pass that surfaced the dead-code +
  structural items the pass acted on. Findings table + per-module notes + what
  was declined and why
- `comment_density_review.md` -- the parallel review for `bst.py` and the rest
  of the comment-density work. Same shape: findings + declined items

Sibling dir at `../../function_invariants/` carries the three split-time invariant
maps (`collate_npy.md`, `detect_players.md`, `train_network.md`) that gated B4,
B5, and B7. The invariants are durable reference for anyone re-entering those
functions.

## What's not in this dir (stays archived)

The full planning trail (scripts assessment, placement recommendation, both
runbooks, the agentic-pass launch protocol, the gitignore audit, the
adversarial-review docs, harness goldens at ~37 MB) stays in
`code_simplification_and_streamline/` as the working archive. See that dir's
`README.md` for the index.

## Six-month skim

The two BST-X review docs flagged ~150 lines of dead code, ~31% comment lines in
`bst.py` that were TensorFlow-analogue learning notes (written while picking up
PyTorch from a TF background), and seven structural cleanups (duplicated court
helpers, a one-of-three test forward pass, a 250-line `collate_npy`, a 420-line
`train_network` setup).

All of those landed across 17 commits. Each batch ran through the captured
goldens, the HPC bit-exact where applicable, and an independent review agent
before the next started.

The maths is untouched. Model graph is bit-identical (5 variants × 464 weight
tensors, all outputs equal). Pipeline outputs are per-clip identical across ~66k
clips on two taxonomies + splits. Training trajectory is identical under a fixed
seed (weights .pt SHA-equal, prediction npzs value-equal, manifest metrics
identical).
