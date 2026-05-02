# Nosides runs: hparams + means + best serials

| Run | Loss config | Mean (macro / min / acc / top-2) | Best | Best (macro / min / acc / top-2) |
|---|---|---|---|---|
| run_20260425_185421 (P1) | LS 0.1\* | 0.7427 / 0.3969 / 0.7658 / 0.9376 | S1 | 0.7462 / 0.4238 / 0.7651 / 0.9396 |
| **-- Phase 2 below --** | | | | |
| run_20260430_170325 | LS 0.1\* | 0.7419 / 0.3751 / 0.7673 / 0.9378 | S4 | 0.7466 / 0.4027 / 0.7703 / 0.9398 |
| run_20260430_213933 | LS 0.0 | 0.7433 / 0.3591 / 0.7681 / **0.9392** | S2 | 0.7403 / 0.4044 / 0.7661 / **0.9405** |
| run_20260501_073430 | LS 0.15 | 0.7471 / 0.4168 / 0.7686 / 0.9378 | S3 | 0.7525 / 0.4482 / **0.7730** / 0.9396 |
| run_20260501_110525 | LS 0.15 + cw{ws,sm}=2.0 | **0.7478** / 0.4221 / **0.7695** / 0.9360 | S2 | 0.7508 / **0.5179** / 0.7656 / 0.9357 |
| run_20260501_164658 | LS 0 + CDB τ1 γ1 | 0.7432 / **0.4621** / 0.7617 / 0.9351 | S2 | 0.7530 / 0.4863 / 0.7692 / 0.9403 |
| run_20260501_192113 | LS 0 + CDB τ1 γ0 | 0.7401 / 0.4342 / 0.7585 / 0.9354 | S1 | 0.7425 / 0.4938 / 0.7582 / 0.9357 |
| run_20260501_192519 | LS 0 + CDB τ0.5 γ1 | 0.7452 / 0.4119 / 0.7665 / 0.9389 | S2 | **0.7533** / 0.3670 / 0.7720 / 0.9403 |
| run_20260501_230252 | LS 0 + CDB τ1 γ1 + cap sm/ws=0.7 | 0.7402 / 0.4105 / 0.7604 / 0.9344 | S4 | 0.7480 / 0.4181 / 0.7675 / 0.9331 |
| run_20260502_075808 | LS 0 + CDB τ1 γ2 | 0.7359 / 0.4207 / 0.7559 / 0.9346 | S3 | 0.7330 / 0.4873 / 0.7494 / 0.9350 |

Bold marks the column max in each of mean and best blocks.

## Shared across all runs

Combo A nosides (`une_merge_v1_nosides` + `split_v2` + `dropunk`), 80 epochs, lr=5e-4, batch=128, aux_fade_end=15, 5 serials each. CDB runs: momentum=0.9, warm_up=5, f1_floor=0.

## Notes

- `LS 0.1*` = BST paper default inherited (no explicit override in config); every other run sets LS explicitly.
- `(P1)` = Phase 1 mixed sticky_anchor data. Everything else is Phase 2 unified clean dir over 32,203 stems. Not directly comparable to Phase 2 means.

## Quick read

Mean macro range across Phase 2 is 0.7359-0.7478, a 1.2% spread. Mean min/ws spans 0.359-0.462, a 10% spread; the floor is the only metric loss-side has actually moved. Class weights set the project ws ceiling at S2 0.518 but bimodal across seeds; CDB-F1 (τ1 γ1) lifted the mean instead. The τ/γ sweep, the pair-cap, and γ=2 all landed inside or below the τ1 γ1 mean. Every untouched CDB knob has now been tested.

## 2dp truncated view (macro + min only)

| Run | Loss config | Mean (macro / min) | Best | Best (macro / min) |
|---|---|---|---|---|
| run_20260425_185421 (P1) | LS 0.1\* | 0.74 / 0.39 | S1 | 0.74 / 0.42 |
| **-- Phase 2 below --** | | | | |
| run_20260430_170325 | LS 0.1\* | 0.74 / 0.37 | S4 | 0.74 / 0.40 |
| run_20260430_213933 | LS 0.0 | 0.74 / 0.35 | S2 | 0.74 / 0.40 |
| run_20260501_073430 | LS 0.15 | 0.74 / 0.41 | S3 | 0.75 / 0.44 |
| run_20260501_110525 | LS 0.15 + cw{ws,sm}=2.0 | **0.74** / 0.42 | S2 | 0.75 / **0.51** |
| run_20260501_164658 | LS 0 + CDB τ1 γ1 | 0.74 / **0.46** | S2 | 0.75 / 0.48 |
| run_20260501_192113 | LS 0 + CDB τ1 γ0 | 0.74 / 0.43 | S1 | 0.74 / 0.49 |
| run_20260501_192519 | LS 0 + CDB τ0.5 γ1 | 0.74 / 0.41 | S2 | **0.75** / 0.36 |
| run_20260501_230252 | LS 0 + CDB τ1 γ1 + cap sm/ws=0.7 | 0.74 / 0.41 | S4 | 0.74 / 0.41 |
| run_20260502_075808 | LS 0 + CDB τ1 γ2 | 0.73 / 0.42 | S3 | 0.73 / 0.48 |

Bolds carried over from the full-precision table (they mark the row that owned the column max before truncation). Acc and top-2 dropped from this view since they don't move much: acc clusters at 0.75-0.77 across both mean and best, top-2 at 0.93-0.94. The only column with real spread at 2dp is min/ws.
