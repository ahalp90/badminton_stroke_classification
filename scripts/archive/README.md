# scripts/archive/

One-shot tooling kept for reference, not part of the active build. Nothing in here is invoked by the current pipeline, training loop, or CI. Scripts move in here once their job is done; they stay here so the project history is recoverable.

## Conventions

- Move scripts in via `git mv` so blame and history follow.
- Add a one-line entry below describing what the script did, when, and why it is archived rather than deleted.

## Archived scripts

- `flatten_copy.sh` — one-shot bash helper that copied the pre-flatten nested clip tree into the post-Phase-2 flat layout. Phase-2 directory flatten work, 2026 March/April; the live pipeline now writes flat from the start.
- `verify_flatten.py` — companion verifier for `flatten_copy.sh`. Confirms per-clip stem counts and sample SHAs match between nested and flat dirs. Same era; not needed once the flatten was committed.
- `symlink_merge_phase1.py` — built the Phase-1 mixed-retrain merged dir by symlinking sticky_anchor outputs over the committed extract for the 1,716 hit-zone busted stems. Used during the sticky_anchor ablation; not on the active build path now that the ablation is settled.
- `verify_v1_collate.py` — sanity-checked a une_merge_v1 collated dir produced by `prepare_train_on_shuttleset` matched expected stem counts and shapes. Phase-2 ablation kickoff; the byte-identity gate covers this work today.
- `test_clip_index.py` — early dev-time sanity check for `pipeline/clip_index.py`. Superseded by the proper pytest suite in `tests/`.
