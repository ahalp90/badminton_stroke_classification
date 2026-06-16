# Hparam search: aug_v1_round_1

Started: 2026-05-05T21:30:08

## Reference

- Current best: run_20260505_154907
  - Mean 0.7447 / 0.4778 / 0.7635 / 0.9394
- Wipe_drop best: run_20260503_172922
  - Mean 0.7481 / 0.4741 / 0.7653 / 0.9353

## Summary

| Cell | Status | Mean (macro / min) | Best S | Verdict |
|------|--------|--------------------|--------|---------|
| p_flip_25 | complete | 0.7402 / 0.4783 | 2 | TIE |
| cap_bump | killed | 0.7339 / 0.4587 (partial, 4/5) | 3 | LOSE |
| p_jitter_40 | complete | 0.7426 / 0.4822 | 3 | TIE |
| p_flip_25_x_p_jitter_30 | complete | 0.7389 / 0.4569 | 1 | LOSE |
| p_flip_25_x_cap_bump | skipped | — | — | — |

## Cell: p_flip_25

- Run id: `run_20260505_213008_504674`
- Augmentation: `{'p_flip': 0.25, 'p_jitter': 0.3, 'cap_y': 0.05, 'cap_x': 0.1, 'eps': 0.15}`
- Cell-start ref: macro 0.7447, min 0.4778

- S1: macro 0.7450, min 0.4718, acc 0.7656, top-2 0.9407 — cumulative 0.7450 / 0.4718
- S2: macro 0.7346, min 0.5097, acc 0.7556, top-2 0.9405 — cumulative 0.7398 / 0.4908
- S3: macro 0.7427, min 0.4785, acc 0.7615, top-2 0.9391 — cumulative 0.7408 / 0.4867
- S4: macro 0.7359, min 0.4675, acc 0.7568, top-2 0.9336 — cumulative 0.7396 / 0.4819
- S5: macro 0.7426, min 0.4641, acc 0.7613, top-2 0.9376 — cumulative 0.7402 / 0.4783

PICK: S2
Mean 0.7402 / 0.4783 / 0.7602 / 0.9383
Vs cell-start ref: macro -0.5, min +0.0
Vs wipe_drop:      macro -0.8, min +0.4, acc -0.5, top-2 +0.3
Verdict: TIE
Top movers vs cell-start ref: smash +5.3, push -4.1, rush -2.4

## Cell: cap_bump

- Run id: `run_20260505_233645_734631`
- Augmentation: `{'p_flip': 0.5, 'p_jitter': 0.3, 'cap_y': 0.075, 'cap_x': 0.15, 'eps': 0.15}`
- Cell-start ref: macro 0.7447, min 0.4778

- S1: macro 0.7303, min 0.4648, acc 0.7530, top-2 0.9346 — cumulative 0.7303 / 0.4648
- S2: macro 0.7326, min 0.4426, acc 0.7501, top-2 0.9307 — cumulative 0.7314 / 0.4537
- S3: macro 0.7349, min 0.4803, acc 0.7558, top-2 0.9386 — cumulative 0.7326 / 0.4626
- S4: macro 0.7379, min 0.4471, acc 0.7568, top-2 0.9336 — cumulative 0.7339 / 0.4587

**Killed at S4**: macro tolerance: cumulative mean macro after S4 is 0.7339, ref 0.7447, deficit 1.08% exceeds tolerance 0.7%.
Verdict: LOSE.

## Cell: p_jitter_40

- Run id: `run_20260506_011851_522295`
- Augmentation: `{'p_flip': 0.5, 'p_jitter': 0.4, 'cap_y': 0.05, 'cap_x': 0.1, 'eps': 0.15}`
- Cell-start ref: macro 0.7447, min 0.4778

- S1: macro 0.7506, min 0.4760, acc 0.7684, top-2 0.9455 — cumulative 0.7506 / 0.4760
- S2: macro 0.7327, min 0.4729, acc 0.7501, top-2 0.9381 — cumulative 0.7417 / 0.4744
- S3: macro 0.7397, min 0.5104, acc 0.7594, top-2 0.9396 — cumulative 0.7410 / 0.4864
- S4: macro 0.7423, min 0.4871, acc 0.7630, top-2 0.9417 — cumulative 0.7413 / 0.4866
- S5: macro 0.7476, min 0.4648, acc 0.7642, top-2 0.9396 — cumulative 0.7426 / 0.4822

PICK: S3
Mean 0.7426 / 0.4822 / 0.7610 / 0.9409
Vs cell-start ref: macro -0.2, min +0.4
Vs wipe_drop:      macro -0.6, min +0.8, acc -0.4, top-2 +0.6
Verdict: TIE
Top movers vs cell-start ref: cross_court_net_shot -2.9, smash +2.8, rush -1.4

## Cell: p_flip_25_x_p_jitter_30

- Run id: `run_20260506_032632_652587`
- Augmentation: `{'p_flip': 0.25, 'p_jitter': 0.3, 'cap_y': 0.05, 'cap_x': 0.1, 'eps': 0.15}`
- Cell-start ref: macro 0.7447, min 0.4778

- S1: macro 0.7447, min 0.5231, acc 0.7625, top-2 0.9391 — cumulative 0.7447 / 0.5231
- S2: macro 0.7374, min 0.4262, acc 0.7582, top-2 0.9362 — cumulative 0.7411 / 0.4746
- S3: macro 0.7400, min 0.4556, acc 0.7630, top-2 0.9400 — cumulative 0.7407 / 0.4683
- S4: macro 0.7337, min 0.4483, acc 0.7532, top-2 0.9381 — cumulative 0.7390 / 0.4633
- S5: macro 0.7387, min 0.4311, acc 0.7573, top-2 0.9388 — cumulative 0.7389 / 0.4569

PICK: S1
Mean 0.7389 / 0.4569 / 0.7588 / 0.9385
Vs cell-start ref: macro -0.6, min -2.1
Vs wipe_drop:      macro -0.9, min -1.7, acc -0.7, top-2 +0.3
Verdict: LOSE
Top movers vs cell-start ref: smash +3.4, rush -2.8, cross_court_net_shot -2.5

## Cell: p_flip_25_x_cap_bump

- Run id: `—`
- Augmentation: `—`

**Skipped**: requires not satisfied: p_flip_25 != LOSE and cap_bump == WIN

