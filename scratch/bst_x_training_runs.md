# BST-X Architecture-1 Training Runs

All 64 recorded training runs of the BST-X (Architecture-1) model on ShuttleSet, built directly from the per-run `manifest.yaml` files under `experiments/bst_x/shuttleset/`. Generated 2026-06-02.

**Metrics from held-out test set.** Shows `best-serial / mean-across-serials`, to 4 dp. The best serial comes from the manifest's `best_serials` field where it's filled in (#1–31 and #49–64); otherwise, matched to the only `weights/` .pt retained.

## Reading the tables

- **# (run number)** is the global chronological index by `started_at`, 1–48. All tables match the global list.

- **Taxonomy** is shown by its current canonical name (`pipeline/config.py` registry), with the legacy name the manifest stored in `[brackets]`. The one exception is legacy `une_merge_v1` (sided, 28-class): it has no canonical equivalent, so it's kept as-is. The names used here:

  | Shown as | Legacy name in manifests | Classes | Sides | Unknown |
  |---|---|---|---|---|
  | `bst_25` | `merged_25` | 25 | yes | kept |
  | `bst_24` | (native) | 24 | yes | dropped |
  | `bst_12` | (native) | 12 | no | dropped |
  | `une_v1_14` | `une_merge_v1_nosides` | 14 | no | dropped |
  | `une_merge_v1` | (kept as-is) | 28 | yes | dropped |
  | `shuttleset_18` | (native) | 18 | no | dropped |

- **Split**: `split_v2` is the player-overlap-minimised split; `split_bst_baseline` is the BST paper's original partition. Six early runs stored no split column and default to `split_bst_baseline`.

- **Test population** is set by split + unknown handling, so min-F1 is only comparable within the same population: baseline + keep-unknown = 3486 strokes, baseline + drop-unknown = 3335, split_v2 = 4202.

- **Unknown ghost channel (`*`).** Nine early drop-unknown runs (#7–10, #12–16) trained with a dead `unknown` output slot that took softmax space but saw no samples and was never reported (drop-unknown hack), a bug patched at #17 and removed completely at the pinned collations (#32). They're marked `*` on the run number; full explanation: [appendix](#appendix-the-unknown-ghost-channel-era).

- **shuttleset_18 min-F1 = 0.000** (#32, #38) is `driven_flight`: a single test clip, so its F1 is 0 if that clip is missed and ~1 if hit.


---

## 1. Global: all runs (chronological)

Macro / min argmax is bolded **per (taxonomy, split) combo** here, not table-wide: the six comparable combos are shuttleset_18·v2, bst_24·v2, bst_12·v2, bst_25·baseline, bst_24·baseline, une_v1_14·v2 (legacy names fold in). The four sided `une_merge_v1` runs sit outside those, so they're left unbolded.

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-04-17 | 1 | `bst_cg_ap_base`<br>`_17_04_2026` | bst_25 [merged_25] | split_bst_baseline | 0.8230<br>0.8177 | 0.5850<br>0.5670 | 0.8410<br>0.8373 | 0.9630<br>0.9607 |  |  | BST paper hparams verbatim, 1600ep. TrackNetV3 *with* inpaint (vs paper's without) beats the paper. Baseline reference.
2026-04-17 | 2 | `run_20260417`<br>`_191851` | bst_25 [merged_25] | split_bst_baseline | 0.8300<br>0.8263 | 0.6270<br>0.6073 | 0.8440<br>0.8420 | 0.9640<br>0.9633 |  |  | Retuned LR: 120ep, cosine cycles 0.25→0.5 so LR reaches 0 in-window, warmup 400→100, patience 300→40.
2026-04-18 | 3 | `run_20260418`<br>`_104152` | bst_25 [merged_25] | split_bst_baseline | 0.8300<br>0.8243 | 0.6260<br>0.5860 | 0.8460<br>0.8433 | 0.9640<br>0.9633 |  |  | First CG/AP aux-loss anneal, gentle (cosine fade to 0 by ep60); most seeds picked val before the fade bit.
2026-04-18 | 4 | `run_20260418`<br>`_151139` | bst_25 [merged_25] | split_bst_baseline | **0.8310**<br>0.8288 | 0.6000<br>0.5996 | 0.8500<br>0.8442 | 0.9680<br>0.9642 | ✓ |  | Aggressive CG/AP anneal (fade to 0 by ep15, then pure backbone); first 5-serial run. Top macro of the schedule arms.
2026-04-18 | 5 | `run_20260418`<br>`_174238` | bst_25 [merged_25] | split_bst_baseline | 0.8280<br>0.8262 | 0.6030<br>0.5768 | 0.8440<br>0.8448 | 0.9630<br>0.9630 |  |  | CG/AP ablation, always-on arm (aux held 1.0 all 80ep).
2026-04-18 | 6 | `run_20260418`<br>`_234822` | bst_25 [merged_25] | split_bst_baseline | 0.8300<br>0.8221 | 0.5860<br>0.5777 | 0.8419<br>0.8398 | 0.9633<br>0.9609 |  |  | CG/AP ablation, control arm: the auxiliary loss pinned to 0 from ep1 (CG/AP fully off).
2026-04-20 | 7* | `run_20260420`<br>`_141629` | une_merge_v1 | split_bst_baseline | 0.7715<br>0.7664 | 0.3810<br>0.3489 | 0.7907<br>0.7873 | 0.9451<br>0.9410 |  |  | First une_merge_v1 run (sided, 28-class) and first CSV flat-pipeline run; drop-unknown, BST split. Schedule mirrors #4.
2026-04-20 | 8* | `run_20260420`<br>`_171101` | une_merge_v1 | split_v2 | 0.7428<br>0.7407 | 0.4315<br>0.3894 | 0.7656<br>0.7663 | 0.9367<br>0.9363 |  |  | Same as #7 (sided, 28-class), split swapped to split_v2 (player-overlap-minimised).
2026-04-25 | 9* | `run_20260425`<br>`_150548` | une_merge_v1 | split_v2 | 0.7550<br>0.7476 | 0.3518<br>0.3326 | 0.7801<br>0.7737 | 0.9426<br>0.9418 |  |  | Sanity check that the re-extraction worked: re-extracted and sticky-anchored only the 1,716 worst (hit-zone-busted) clips, symlinked the rest. Sided 28-class.
2026-04-25 | 10* | `run_20260425`<br>`_185421` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7462<br>0.7427 | 0.4238<br>0.3969 | 0.7651<br>0.7658 | 0.9396<br>0.9376 |  |  | Nosides collapse of #9 (28→14 cls by dropping Top_/Bottom_); collapse rescues wrist_smash min-F1.
2026-04-29 | 11 | `run_20260429`<br>`_202144` | bst_25 [merged_25] | split_bst_baseline | 0.8307<br>**0.8314** | 0.5769<br>0.5775 | 0.8495<br>0.8477 | 0.9676<br>0.9687 |  | ✓ | Phase-2 sanity 1/3: re-run of the bst_25 baseline combo (drop-unknown) on the unified 32,203-stem sticky-anchor clean dir.
2026-04-30 | 12* | `run_20260430`<br>`_110101` | une_merge_v1 | split_v2 | 0.7431<br>0.7393 | 0.2447<br>0.3172 | 0.7703<br>0.7664 | 0.9438<br>0.9384 |  |  | Phase-2 sanity 2/3: une sided (28-class) + v2 on the unified clean dir.
2026-04-30 | 13* | `run_20260430`<br>`_170325` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7466<br>0.7419 | 0.4027<br>0.3751 | 0.7703<br>0.7673 | 0.9398<br>0.9378 |  |  | Phase-2 sanity 3/3: une_v1_14 + v2 on the unified clean dir. LS=0.1 (default of the era).
2026-04-30 | 14* | `run_20260430`<br>`_213933` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7403<br>0.7433 | 0.4044<br>0.3591 | 0.7661<br>0.7681 | 0.9405<br>0.9392 |  |  | Label-smoothing ablation: LS=0.0 vs #13's LS=0.1.
2026-05-01 | 15* | `run_20260501`<br>`_073430` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7525<br>0.7471 | 0.4482<br>0.4168 | 0.7730<br>0.7686 | 0.9396<br>0.9378 |  |  | Label-smoothing LS=0.15.
2026-05-01 | 16* | `run_20260501`<br>`_110525` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7508<br>0.7478 | 0.5179<br>0.4221 | 0.7656<br>0.7695 | 0.9357<br>0.9360 |  |  | Class-weighting smoke test: LS=0.15 + class_weights{wrist_smash:2.0, smash:2.0}.
2026-05-01 | 17 | `run_20260501`<br>`_164658` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7530<br>0.7432 | 0.4863<br>0.4621 | 0.7692<br>0.7617 | 0.9403<br>0.9351 |  |  | First CDB-F1 run: adaptive focal (tau=1, gamma=1, momentum=0.9, warm_up=5), LS dropped to 0.
2026-05-01 | 18 | `run_20260501`<br>`_192113` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7425<br>0.7401 | 0.4938<br>0.4342 | 0.7582<br>0.7585 | 0.9357<br>0.9354 |  |  | CDB-F1 follow-up: gamma 1→0, the per-sample focusing term (per-class alpha shape unchanged).
2026-05-01 | 19 | `run_20260501`<br>`_192519` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7533<br>0.7452 | 0.3670<br>0.4119 | 0.7720<br>0.7665 | 0.9403<br>0.9389 |  |  | CDB-F1 follow-up: tau 1→0.5, softening the per-class alpha weighting (range narrows ~0.48–1.44).
2026-05-01 | 20 | `run_20260501`<br>`_230252` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7480<br>0.7402 | 0.4181<br>0.4105 | 0.7675<br>0.7604 | 0.9331<br>0.9344 |  |  | CDB-F1 + alpha pair-cap forcing alpha[smash] ≥ 0.7×alpha[wrist_smash] each epoch.
2026-05-02 | 21 | `run_20260502`<br>`_075808` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7330<br>0.7359 | 0.4873<br>0.4207 | 0.7494<br>0.7559 | 0.9350<br>0.9346 |  |  | CDB-F1 gamma 1→2 (Lin et al. focal default); didn't lift the floor.
2026-05-03 | 22 | `run_20260503`<br>`_104300` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7434<br>0.7414 | 0.4449<br>0.4138 | 0.7570<br>0.7604 | 0.9365<br>0.9319 |  |  | Capacity bump: MLP head hidden 400→1200 (encoder untouched). CDB-F1 parity test.
2026-05-03 | 23 | `run_20260503`<br>`_172922` | une_v1_14 [une_merge_v1_nosides] | split_v2 | **0.7559**<br>0.7481 | 0.4935<br>0.4741 | 0.7684<br>0.7653 | 0.9334<br>0.9353 | ✓ |  | Shuttle-unzeroing (wipe_drop): stop zeroing shuttle on keypoint-fail (~14k frames, 0.84%). Project best at the time; smash & wrist_smash lift together.
2026-05-03 | 24 | `run_20260503`<br>`_192718` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7456<br>0.7440 | 0.4899<br>0.4568 | 0.7646<br>0.7630 | 0.9391<br>0.9365 |  |  | Shuttle-mask (mask_wiring): wipe_drop + a post-TCN shuttle_missing channel fused via mask_proj + shuttle_fuse.
2026-05-04 | 25 | `run_20260504`<br>`_152529` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7398<br>0.7401 | 0.4848<br>0.4301 | 0.7584<br>0.7586 | 0.9348<br>0.9365 |  |  | Jitter-off ablation (RandomTranslation prob=0) vs the wipe_drop best; min-F1 −4.4.
2026-05-05 | 26 | `run_20260505`<br>`_111211` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7341<br>0.7388 | 0.5034<br>0.4750 | 0.7573<br>0.7591 | 0.9405<br>0.9402 |  |  | Augmentation framework v1: coupled centreline flip + pos/shuttle constrained jitter (p_flip0.5, p_jitter0.2). Replaces the broken joints-only jitter.
2026-05-05 | 27 | `run_20260505`<br>`_154907` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7479<br>0.7447 | 0.5147<br>0.4778 | 0.7675<br>0.7635 | 0.9407<br>0.9394 |  |  | Aug v1, p_jitter 0.2→0.3. The current-best aug config; sweep reference.
2026-05-05 | 28 | `run_20260505`<br>`_213008_504674` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7427<br>0.7402 | 0.4785<br>0.4783 | 0.7615<br>0.7602 | 0.9391<br>0.9383 |  |  | Aug sweep cell `p_flip_25`: p_flip 0.5→0.25 (recover cross_court_net_shot?).
2026-05-05 | 29 | `run_20260505`<br>`_233645_734631` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7349<br>0.7339 | 0.4803<br>0.4587 | 0.7558<br>0.7539 | 0.9386<br>0.9344 |  |  | Aug sweep cell `cap_bump`: cap_y0.05→0.075, cap_x0.10→0.15. Killed at S4 on the wrapper's macro tolerance.
2026-05-06 | 30 | `run_20260506`<br>`_011851_522295` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7506<br>0.7426 | 0.4760<br>0.4822 | 0.7684<br>0.7610 | 0.9455<br>0.9409 |  |  | Aug sweep cell `p_jitter_40`: p_jitter 0.3→0.4.
2026-05-06 | 31 | `run_20260506`<br>`_032632_652587` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7447<br>0.7389 | 0.5231<br>0.4569 | 0.7625<br>0.7588 | 0.9391<br>0.9385 |  |  | Aug sweep cell `p_flip_25 × p_jitter_30`: override matched base, so config collided with #28 (same aug, different seeds).
2026-05-30 | 32 | `run_20260530`<br>`_161525_131279` | shuttleset_18 | split_v2 | 0.6652<br>0.6544 | 0.0000<br>0.0000 | 0.7656<br>0.7535 | 0.9372<br>0.9290 |  |  | Multi-taxon batch: shuttleset_18 / v2 (finest 18-class cut). min-F1=0 is driven_flight, a single-test-clip dice-roll with no real signal.
2026-05-30 | 33 | `run_20260530`<br>`_174818_410060` | bst_24 | split_v2 | 0.8156<br>0.8093 | 0.5714<br>0.5336 | 0.8277<br>0.8221 | 0.9603<br>0.9575 |  |  | Multi-taxon batch: bst_24 / v2. v2 val overshoots test ~3.9%.
2026-05-30 | 34 | `run_20260530`<br>`_192738_970644` | bst_12 | split_v2 | **0.8210**<br>0.8116 | 0.6314<br>0.5990 | 0.8284<br>0.8221 | 0.9591<br>0.9577 | ✓ |  | Multi-taxon batch: bst_12 / v2 (fewest classes; top macro of the batch).
2026-05-30 | 35 | `run_20260530`<br>`_210600_435552` | bst_25 | split_bst_baseline | 0.8303<br>0.8242 | **0.6562**<br>**0.6203** | 0.8431<br>0.8408 | 0.9647<br>0.9654 | ✓ | ✓ | Multi-taxon batch: bst_25 / baseline, keep-unknown (paper-faithful).
2026-05-30 | 36 | `run_20260530`<br>`_225714_593038` | bst_24 | split_bst_baseline | **0.8421**<br>0.8265 | 0.6122<br>0.5683 | 0.8534<br>0.8422 | 0.9640<br>0.9661 | ✓ |  | Multi-taxon batch: bst_24 / baseline, drop-unknown. Top macro overall; dropping unknown lifts the 24 real strokes.
2026-05-31 | 37 | `run_20260531`<br>`_005535_005154` | une_v1_14 | split_v2 | 0.7514<br>0.7423 | 0.4851<br>0.4714 | 0.7703<br>0.7619 | 0.9405<br>0.9386 |  |  | Multi-taxon batch: une_v1_14 / v2. Splitting wrist_smash off smash and passive_drop off drop is expensive on the parents.
2026-05-31 | 38 | `run_20260531`<br>`_163906_107348` | shuttleset_18 | split_v2 | 0.6670<br>0.6538 | 0.0000<br>0.0000 | 0.7634<br>0.7535 | 0.9341<br>0.9290 |  |  | Gate+focal-revert arm: shuttleset_18 / v2. min=0 driven_flight dice-roll again.
2026-05-31 | 39 | `run_20260531`<br>`_193021_308927` | bst_24 | split_v2 | 0.8108<br>0.8065 | 0.5636<br>0.5294 | 0.8258<br>0.8206 | 0.9612<br>0.9573 |  |  | Gate+focal-revert arm: bst_24 / v2.
2026-05-31 | 40 | `run_20260531`<br>`_201350_026614` | une_v1_14 | split_v2 | 0.7525<br>0.7447 | 0.5009<br>0.4848 | 0.7663<br>0.7636 | 0.9403<br>0.9377 |  |  | WD sweep: wd 1e-2 *excluding* norms/bias/embeddings from decay. New une_v1_14 high (mean macro/min +0.003/+0.014).
2026-05-31 | 41 | `run_20260531`<br>`_211838_567072` | bst_12 | split_v2 | 0.8136<br>0.8093 | 0.6008<br>0.5870 | 0.8282<br>0.8222 | 0.9584<br>0.9578 |  |  | Gate+focal-revert arm: bst_12 / v2.
2026-05-31 | 42 | `run_20260531`<br>`_214009_170864` | une_v1_14 | split_v2 | 0.7451<br>0.7399 | 0.4873<br>0.4753 | 0.7680<br>0.7612 | 0.9410<br>0.9396 |  |  | WD sweep: wd 5e-2.
2026-05-31 | 43 | `run_20260531`<br>`_225619_826430` | bst_25 | split_bst_baseline | 0.8261<br>0.8193 | 0.6522<br>0.6148 | 0.8396<br>0.8382 | 0.9676<br>0.9666 |  |  | Gate+focal-revert arm: bst_25 / baseline, keep-unknown.
2026-05-31 | 44 | `run_20260531`<br>`_231403_971803` | une_v1_14 | split_v2 | 0.7335<br>0.7380 | 0.4923<br>0.4816 | 0.7520<br>0.7575 | 0.9336<br>0.9366 |  |  | WD sweep: wd 1e-1.
2026-06-01 | 45 | `run_20260601`<br>`_003918_078077` | une_v1_14 | split_v2 | 0.7487<br>0.7437 | 0.4830<br>0.4752 | 0.7646<br>0.7600 | 0.9438<br>0.9408 |  |  | WD sweep: wd 2e-1.
2026-06-01 | 46 | `run_20260601`<br>`_005010_962006` | bst_24 | split_bst_baseline | 0.8336<br>0.8286 | 0.5652<br>0.5726 | 0.8480<br>0.8451 | 0.9673<br>0.9669 |  |  | Gate+focal-revert arm: bst_24 / baseline, drop-unknown.
2026-06-01 | 47 | `run_20260601`<br>`_021234_940276` | une_v1_14 | split_v2 | 0.7479<br>0.7463 | **0.5248**<br>**0.4979** | 0.7699<br>0.7670 | 0.9460<br>0.9424 | ✓ | ✓ | WD sweep: wd 4e-1. Best mean min-F1 of the wd magnitudes (0.498), clears 0.5 on wrist_smash for the best serial.
2026-06-01 | 48 | `run_20260601`<br>`_023543_278210` | une_v1_14 | split_v2 | 0.7496<br>0.7394 | 0.4454<br>0.4868 | 0.7692<br>0.7597 | 0.9398<br>0.9393 |  |  | Gate+focal-revert arm: une_v1_14 / v2, the taxonomy the alpha-revert was built to target.
2026-06-01 | 49 | `run_20260601`<br>`_141054_760522` | shuttleset_18 | split_v2 | 0.6634<br>0.6545 | 0.0000<br>0.0000 | 0.7556<br>0.7534 | 0.9307<br>0.9288 |  |  | wd 1e-2, gate off: shuttleset_18 / v2. Floor is driven_flight (n/a); macro tracks the standard.
2026-06-01 | 50 | `run_20260601`<br>`_153738_936523` | shuttleset_18 | split_v2 | **0.6721**<br>**0.6587** | 0.0000<br>0.0000 | 0.7646<br>0.7580 | 0.9393<br>0.9346 | ✓ | ✓ | wd 4e-1, gate off: shuttleset_18 / v2. Small macro nudge over the standard; floor n/a.
2026-06-01 | 51 | `run_20260601`<br>`_171005_492538` | bst_24 | split_v2 | 0.8179<br>0.8114 | 0.6264<br>0.5566 | 0.8275<br>0.8228 | 0.9550<br>0.9586 |  |  | wd 1e-2, gate off: bst_24 / v2. Exclusion lifts the floor over the all-layers standard; 4e-1 lifts it more.
2026-06-01 | 52 | `run_20260601`<br>`_184437_030823` | bst_24 | split_v2 | **0.8217**<br>**0.8130** | **0.6406**<br>**0.5880** | 0.8315<br>0.8250 | 0.9607<br>0.9577 | ✓ | ✓ | wd 4e-1, gate off: bst_24 / v2. Best floor here, +5.4% mean min over the standard; the keeper, new bst_24/v2 best.
2026-06-01 | 53 | `run_20260601`<br>`_202022_522780` | bst_12 | split_v2 | 0.8177<br>**0.8141** | 0.6231<br>**0.6119** | 0.8258<br>0.8252 | 0.9607<br>0.9585 |  | ✓ | wd 1e-2, gate off: bst_12 / v2. Exclusion grabs the floor; the bst_12 keeper (4e-1 adds nothing).
2026-06-01 | 54 | `run_20260601`<br>`_215358_806530` | bst_12 | split_v2 | 0.8150<br>0.8095 | **0.6364**<br>0.6029 | 0.8218<br>0.8208 | 0.9557<br>0.9565 | ✓ |  | wd 4e-1, gate off: bst_12 / v2. No gain over 1e-2, slightly over-regularises.
2026-06-01 | 55 | `run_20260601`<br>`_232557_387981` | bst_25 | split_bst_baseline | 0.8268<br>0.8155 | 0.6250<br>0.5481 | 0.8437<br>0.8357 | 0.9633<br>0.9639 |  |  | wd 1e-2, gate off: bst_25 / baseline. Keep-unknown: exclusion drops the floor below the standard.
2026-06-02 | 56 | `run_20260602`<br>`_011036_539898` | bst_25 | split_bst_baseline | 0.8289<br>0.8222 | 0.6525<br>0.6141 | 0.8428<br>0.8389 | 0.9687<br>0.9670 |  |  | wd 4e-1, gate off: bst_25 / baseline. Recovers most of 1e-2's drop, still under the standard; bst_25 stays on it.
2026-06-02 | 57 | `run_20260602`<br>`_025726_461760` | bst_24 | split_bst_baseline | 0.8339<br>0.8295 | 0.6358<br>0.6239 | 0.8459<br>0.8434 | 0.9688<br>0.9661 |  |  | wd 1e-2, gate off: bst_24 / baseline. Big floor lift over the standard; 4e-1 edges it further.
2026-06-02 | 58 | `run_20260602`<br>`_044754_361908` | bst_24 | split_bst_baseline | 0.8347<br>**0.8323** | **0.6557**<br>**0.6312** | 0.8426<br>0.8437 | 0.9718<br>0.9689 | ✓ | ✓ | wd 4e-1, gate off: bst_24 / baseline. Best floor, +6.3% mean min; the keeper, new bst_24/baseline best.
2026-06-02 | 59 | `run_20260602`<br>`_063203_601676` | shuttleset_18 | split_v2 | 0.6522<br>0.6484 | 0.0000<br>0.0000 | 0.7499<br>0.7461 | 0.9284<br>0.9276 |  |  | wd 1e-2, gate on (focal_alpha_revert): shuttleset_18 / v2. No gain over gate-off; floor n/a.
2026-06-02 | 60 | `run_20260602`<br>`_075831_129574` | shuttleset_18 | split_v2 | 0.6584<br>0.6478 | 0.0000<br>0.0000 | 0.7575<br>0.7468 | 0.9329<br>0.9271 |  |  | wd 4e-1, gate on (focal_alpha_revert): shuttleset_18 / v2. No gain over gate-off; floor n/a.
2026-06-02 | 61 | `run_20260602`<br>`_092522_446222` | bst_25 | split_bst_baseline | 0.8262<br>0.8219 | 0.6087<br>0.5951 | 0.8414<br>0.8376 | 0.9613<br>0.9656 |  |  | wd 1e-2, gate on (focal_alpha_revert): bst_25 / baseline. Recovers the bare-1e-2 floor drop, still under the standard.
2026-06-02 | 62 | `run_20260602`<br>`_111410_233326` | bst_25 | split_bst_baseline | 0.8286<br>0.8207 | 0.6298<br>0.5736 | 0.8394<br>0.8377 | 0.9636<br>0.9666 |  |  | wd 4e-1, gate on (focal_alpha_revert): bst_25 / baseline. No better than gate-off 4e-1; under the standard.
2026-06-02 | 63 | `run_20260602`<br>`_130421_390854` | une_v1_14 | split_v2 | 0.7553<br>0.7378 | 0.4818<br>0.4781 | 0.7715<br>0.7581 | 0.9434<br>0.9396 |  |  | wd 1e-2, gate on (focal_alpha_revert): une_v1_14 / v2. Floor below the plain wd 4e-1 (#47).
2026-06-02 | 64 | `run_20260602`<br>`_143618_156220` | une_v1_14 | split_v2 | 0.7526<br>**0.7483** | 0.5029<br>0.4819 | 0.7727<br>0.7663 | 0.9438<br>0.9388 |  | ✓ | wd 4e-1, gate on (focal_alpha_revert): une_v1_14 / v2. Top une mean macro (0.748, 4 serials) but floor trails #47; alpha-revert retired.

---

## 2. Per taxonomy

Grouped by taxonomy; split shown as a bold separator row within each. Run numbers stay global (`*` = unknown ghost channel; see appendix).


### bst_25 (25-class, sided, keep-unknown), incl. legacy `merged_25`

Date | # | Run ID | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---
**split_bst_baseline** | | | | | | | | |
2026-04-17 | 1 | `bst_cg_ap_base`<br>`_17_04_2026` | 0.8230<br>0.8177 | 0.5850<br>0.5670 | 0.8410<br>0.8373 | 0.9630<br>0.9607 |  |  | BST paper hparams verbatim, 1600ep. TrackNetV3 *with* inpaint (vs paper's without) beats the paper. Baseline reference.
2026-04-17 | 2 | `run_20260417`<br>`_191851` | 0.8300<br>0.8263 | 0.6270<br>0.6073 | 0.8440<br>0.8420 | 0.9640<br>0.9633 |  |  | Retuned LR: 120ep, cosine cycles 0.25→0.5 so LR reaches 0 in-window, warmup 400→100, patience 300→40.
2026-04-18 | 3 | `run_20260418`<br>`_104152` | 0.8300<br>0.8243 | 0.6260<br>0.5860 | 0.8460<br>0.8433 | 0.9640<br>0.9633 |  |  | First CG/AP aux-loss anneal, gentle (cosine fade to 0 by ep60); most seeds picked val before the fade bit.
2026-04-18 | 4 | `run_20260418`<br>`_151139` | **0.8310**<br>0.8288 | 0.6000<br>0.5996 | 0.8500<br>0.8442 | 0.9680<br>0.9642 | ✓ |  | Aggressive CG/AP anneal (fade to 0 by ep15, then pure backbone); first 5-serial run. Top macro of the schedule arms.
2026-04-18 | 5 | `run_20260418`<br>`_174238` | 0.8280<br>0.8262 | 0.6030<br>0.5768 | 0.8440<br>0.8448 | 0.9630<br>0.9630 |  |  | CG/AP ablation, always-on arm (aux held 1.0 all 80ep).
2026-04-18 | 6 | `run_20260418`<br>`_234822` | 0.8300<br>0.8221 | 0.5860<br>0.5777 | 0.8419<br>0.8398 | 0.9633<br>0.9609 |  |  | CG/AP ablation, control arm: the auxiliary loss pinned to 0 from ep1 (CG/AP fully off).
2026-04-29 | 11 | `run_20260429`<br>`_202144` | 0.8307<br>**0.8314** | 0.5769<br>0.5775 | 0.8495<br>0.8477 | 0.9676<br>0.9687 |  | ✓ | Phase-2 sanity 1/3: re-run of the bst_25 baseline combo (drop-unknown) on the unified 32,203-stem sticky-anchor clean dir.
2026-05-30 | 35 | `run_20260530`<br>`_210600_435552` | 0.8303<br>0.8242 | **0.6562**<br>**0.6203** | 0.8431<br>0.8408 | 0.9647<br>0.9654 | ✓ | ✓ | Multi-taxon batch: bst_25 / baseline, keep-unknown (paper-faithful).
2026-05-31 | 43 | `run_20260531`<br>`_225619_826430` | 0.8261<br>0.8193 | 0.6522<br>0.6148 | 0.8396<br>0.8382 | 0.9676<br>0.9666 |  |  | Gate+focal-revert arm: bst_25 / baseline, keep-unknown.
2026-06-01 | 55 | `run_20260601`<br>`_232557_387981` | 0.8268<br>0.8155 | 0.6250<br>0.5481 | 0.8437<br>0.8357 | 0.9633<br>0.9639 |  |  | wd 1e-2, gate off: bst_25 / baseline. Keep-unknown: exclusion drops the floor below the standard.
2026-06-02 | 56 | `run_20260602`<br>`_011036_539898` | 0.8289<br>0.8222 | 0.6525<br>0.6141 | 0.8428<br>0.8389 | 0.9687<br>0.9670 |  |  | wd 4e-1, gate off: bst_25 / baseline. Recovers most of 1e-2's drop, still under the standard; bst_25 stays on it.
2026-06-02 | 61 | `run_20260602`<br>`_092522_446222` | 0.8262<br>0.8219 | 0.6087<br>0.5951 | 0.8414<br>0.8376 | 0.9613<br>0.9656 |  |  | wd 1e-2, gate on (focal_alpha_revert): bst_25 / baseline. Recovers the bare-1e-2 floor drop, still under the standard.
2026-06-02 | 62 | `run_20260602`<br>`_111410_233326` | 0.8286<br>0.8207 | 0.6298<br>0.5736 | 0.8394<br>0.8377 | 0.9636<br>0.9666 |  |  | wd 4e-1, gate on (focal_alpha_revert): bst_25 / baseline. No better than gate-off 4e-1; under the standard.

### bst_24 (24-class, sided, drop-unknown)

Date | # | Run ID | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---
**split_bst_baseline** | | | | | | | | |
2026-05-30 | 36 | `run_20260530`<br>`_225714_593038` | **0.8421**<br>0.8265 | 0.6122<br>0.5683 | 0.8534<br>0.8422 | 0.9640<br>0.9661 | ✓ |  | Multi-taxon batch: bst_24 / baseline, drop-unknown. Top macro overall; dropping unknown lifts the 24 real strokes.
2026-06-01 | 46 | `run_20260601`<br>`_005010_962006` | 0.8336<br>0.8286 | 0.5652<br>0.5726 | 0.8480<br>0.8451 | 0.9673<br>0.9669 |  |  | Gate+focal-revert arm: bst_24 / baseline, drop-unknown.
2026-06-02 | 57 | `run_20260602`<br>`_025726_461760` | 0.8339<br>0.8295 | 0.6358<br>0.6239 | 0.8459<br>0.8434 | 0.9688<br>0.9661 |  |  | wd 1e-2, gate off: bst_24 / baseline. Big floor lift over the standard; 4e-1 edges it further.
2026-06-02 | 58 | `run_20260602`<br>`_044754_361908` | 0.8347<br>**0.8323** | **0.6557**<br>**0.6312** | 0.8426<br>0.8437 | 0.9718<br>0.9689 | ✓ | ✓ | wd 4e-1, gate off: bst_24 / baseline. Best floor, +6.3% mean min; the keeper, new bst_24/baseline best.
**split_v2** | | | | | | | | |
2026-05-30 | 33 | `run_20260530`<br>`_174818_410060` | 0.8156<br>0.8093 | 0.5714<br>0.5336 | 0.8277<br>0.8221 | 0.9603<br>0.9575 |  |  | Multi-taxon batch: bst_24 / v2. v2 val overshoots test ~3.9%.
2026-05-31 | 39 | `run_20260531`<br>`_193021_308927` | 0.8108<br>0.8065 | 0.5636<br>0.5294 | 0.8258<br>0.8206 | 0.9612<br>0.9573 |  |  | Gate+focal-revert arm: bst_24 / v2.
2026-06-01 | 51 | `run_20260601`<br>`_171005_492538` | 0.8179<br>0.8114 | 0.6264<br>0.5566 | 0.8275<br>0.8228 | 0.9550<br>0.9586 |  |  | wd 1e-2, gate off: bst_24 / v2. Exclusion lifts the floor over the all-layers standard; 4e-1 lifts it more.
2026-06-01 | 52 | `run_20260601`<br>`_184437_030823` | 0.8217<br>0.8130 | 0.6406<br>0.5880 | 0.8315<br>0.8250 | 0.9607<br>0.9577 |  |  | wd 4e-1, gate off: bst_24 / v2. Best floor here, +5.4% mean min over the standard; the keeper, new bst_24/v2 best.

### bst_12 (12-class, no-sides, drop-unknown)

Date | # | Run ID | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---
**split_v2** | | | | | | | | |
2026-05-30 | 34 | `run_20260530`<br>`_192738_970644` | **0.8210**<br>0.8116 | 0.6314<br>0.5990 | 0.8284<br>0.8221 | 0.9591<br>0.9577 | ✓ |  | Multi-taxon batch: bst_12 / v2 (fewest classes; top macro of the batch).
2026-05-31 | 41 | `run_20260531`<br>`_211838_567072` | 0.8136<br>0.8093 | 0.6008<br>0.5870 | 0.8282<br>0.8222 | 0.9584<br>0.9578 |  |  | Gate+focal-revert arm: bst_12 / v2.
2026-06-01 | 53 | `run_20260601`<br>`_202022_522780` | 0.8177<br>**0.8141** | 0.6231<br>**0.6119** | 0.8258<br>0.8252 | 0.9607<br>0.9585 |  | ✓ | wd 1e-2, gate off: bst_12 / v2. Exclusion grabs the floor; the bst_12 keeper (4e-1 adds nothing).
2026-06-01 | 54 | `run_20260601`<br>`_215358_806530` | 0.8150<br>0.8095 | **0.6364**<br>0.6029 | 0.8218<br>0.8208 | 0.9557<br>0.9565 | ✓ |  | wd 4e-1, gate off: bst_12 / v2. No gain over 1e-2, slightly over-regularises.

### une_v1_14 (14-class, no-sides, drop-unknown), incl. legacy `une_merge_v1_nosides`

Date | # | Run ID | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---
**split_v2** | | | | | | | | |
2026-04-25 | 10* | `run_20260425`<br>`_185421` | 0.7462<br>0.7427 | 0.4238<br>0.3969 | 0.7651<br>0.7658 | 0.9396<br>0.9376 |  |  | Nosides collapse of #9 (28→14 cls by dropping Top_/Bottom_); collapse rescues wrist_smash min-F1.
2026-04-30 | 13* | `run_20260430`<br>`_170325` | 0.7466<br>0.7419 | 0.4027<br>0.3751 | 0.7703<br>0.7673 | 0.9398<br>0.9378 |  |  | Phase-2 sanity 3/3: une_v1_14 + v2 on the unified clean dir. LS=0.1 (default of the era).
2026-04-30 | 14* | `run_20260430`<br>`_213933` | 0.7403<br>0.7433 | 0.4044<br>0.3591 | 0.7661<br>0.7681 | 0.9405<br>0.9392 |  |  | Label-smoothing ablation: LS=0.0 vs #13's LS=0.1.
2026-05-01 | 15* | `run_20260501`<br>`_073430` | 0.7525<br>0.7471 | 0.4482<br>0.4168 | 0.7730<br>0.7686 | 0.9396<br>0.9378 |  |  | Label-smoothing LS=0.15.
2026-05-01 | 16* | `run_20260501`<br>`_110525` | 0.7508<br>0.7478 | 0.5179<br>0.4221 | 0.7656<br>0.7695 | 0.9357<br>0.9360 |  |  | Class-weighting smoke test: LS=0.15 + class_weights{wrist_smash:2.0, smash:2.0}.
2026-05-01 | 17 | `run_20260501`<br>`_164658` | 0.7530<br>0.7432 | 0.4863<br>0.4621 | 0.7692<br>0.7617 | 0.9403<br>0.9351 |  |  | First CDB-F1 run: adaptive focal (tau=1, gamma=1, momentum=0.9, warm_up=5), LS dropped to 0.
2026-05-01 | 18 | `run_20260501`<br>`_192113` | 0.7425<br>0.7401 | 0.4938<br>0.4342 | 0.7582<br>0.7585 | 0.9357<br>0.9354 |  |  | CDB-F1 follow-up: gamma 1→0, the per-sample focusing term (per-class alpha shape unchanged).
2026-05-01 | 19 | `run_20260501`<br>`_192519` | 0.7533<br>0.7452 | 0.3670<br>0.4119 | 0.7720<br>0.7665 | 0.9403<br>0.9389 |  |  | CDB-F1 follow-up: tau 1→0.5, softening the per-class alpha weighting (range narrows ~0.48–1.44).
2026-05-01 | 20 | `run_20260501`<br>`_230252` | 0.7480<br>0.7402 | 0.4181<br>0.4105 | 0.7675<br>0.7604 | 0.9331<br>0.9344 |  |  | CDB-F1 + alpha pair-cap forcing alpha[smash] ≥ 0.7×alpha[wrist_smash] each epoch.
2026-05-02 | 21 | `run_20260502`<br>`_075808` | 0.7330<br>0.7359 | 0.4873<br>0.4207 | 0.7494<br>0.7559 | 0.9350<br>0.9346 |  |  | CDB-F1 gamma 1→2 (Lin et al. focal default); didn't lift the floor.
2026-05-03 | 22 | `run_20260503`<br>`_104300` | 0.7434<br>0.7414 | 0.4449<br>0.4138 | 0.7570<br>0.7604 | 0.9365<br>0.9319 |  |  | Capacity bump: MLP head hidden 400→1200 (encoder untouched). CDB-F1 parity test.
2026-05-03 | 23 | `run_20260503`<br>`_172922` | **0.7559**<br>0.7481 | 0.4935<br>0.4741 | 0.7684<br>0.7653 | 0.9334<br>0.9353 | ✓ |  | Shuttle-unzeroing (wipe_drop): stop zeroing shuttle on keypoint-fail (~14k frames, 0.84%). Project best at the time; smash & wrist_smash lift together.
2026-05-03 | 24 | `run_20260503`<br>`_192718` | 0.7456<br>0.7440 | 0.4899<br>0.4568 | 0.7646<br>0.7630 | 0.9391<br>0.9365 |  |  | Shuttle-mask (mask_wiring): wipe_drop + a post-TCN shuttle_missing channel fused via mask_proj + shuttle_fuse.
2026-05-04 | 25 | `run_20260504`<br>`_152529` | 0.7398<br>0.7401 | 0.4848<br>0.4301 | 0.7584<br>0.7586 | 0.9348<br>0.9365 |  |  | Jitter-off ablation (RandomTranslation prob=0) vs the wipe_drop best; min-F1 −4.4.
2026-05-05 | 26 | `run_20260505`<br>`_111211` | 0.7341<br>0.7388 | 0.5034<br>0.4750 | 0.7573<br>0.7591 | 0.9405<br>0.9402 |  |  | Augmentation framework v1: coupled centreline flip + pos/shuttle constrained jitter (p_flip0.5, p_jitter0.2). Replaces the broken joints-only jitter.
2026-05-05 | 27 | `run_20260505`<br>`_154907` | 0.7479<br>0.7447 | 0.5147<br>0.4778 | 0.7675<br>0.7635 | 0.9407<br>0.9394 |  |  | Aug v1, p_jitter 0.2→0.3. The current-best aug config; sweep reference.
2026-05-05 | 28 | `run_20260505`<br>`_213008_504674` | 0.7427<br>0.7402 | 0.4785<br>0.4783 | 0.7615<br>0.7602 | 0.9391<br>0.9383 |  |  | Aug sweep cell `p_flip_25`: p_flip 0.5→0.25 (recover cross_court_net_shot?).
2026-05-05 | 29 | `run_20260505`<br>`_233645_734631` | 0.7349<br>0.7339 | 0.4803<br>0.4587 | 0.7558<br>0.7539 | 0.9386<br>0.9344 |  |  | Aug sweep cell `cap_bump`: cap_y0.05→0.075, cap_x0.10→0.15. Killed at S4 on the wrapper's macro tolerance.
2026-05-06 | 30 | `run_20260506`<br>`_011851_522295` | 0.7506<br>0.7426 | 0.4760<br>0.4822 | 0.7684<br>0.7610 | 0.9455<br>0.9409 |  |  | Aug sweep cell `p_jitter_40`: p_jitter 0.3→0.4.
2026-05-06 | 31 | `run_20260506`<br>`_032632_652587` | 0.7447<br>0.7389 | 0.5231<br>0.4569 | 0.7625<br>0.7588 | 0.9391<br>0.9385 |  |  | Aug sweep cell `p_flip_25 × p_jitter_30`: override matched base, so config collided with #28 (same aug, different seeds).
2026-05-31 | 37 | `run_20260531`<br>`_005535_005154` | 0.7514<br>0.7423 | 0.4851<br>0.4714 | 0.7703<br>0.7619 | 0.9405<br>0.9386 |  |  | Multi-taxon batch: une_v1_14 / v2. Splitting wrist_smash off smash and passive_drop off drop is expensive on the parents.
2026-05-31 | 40 | `run_20260531`<br>`_201350_026614` | 0.7525<br>0.7447 | 0.5009<br>0.4848 | 0.7663<br>0.7636 | 0.9403<br>0.9377 |  |  | WD sweep: wd 1e-2 *excluding* norms/bias/embeddings from decay. New une_v1_14 high (mean macro/min +0.003/+0.014).
2026-05-31 | 42 | `run_20260531`<br>`_214009_170864` | 0.7451<br>0.7399 | 0.4873<br>0.4753 | 0.7680<br>0.7612 | 0.9410<br>0.9396 |  |  | WD sweep: wd 5e-2.
2026-05-31 | 44 | `run_20260531`<br>`_231403_971803` | 0.7335<br>0.7380 | 0.4923<br>0.4816 | 0.7520<br>0.7575 | 0.9336<br>0.9366 |  |  | WD sweep: wd 1e-1.
2026-06-01 | 45 | `run_20260601`<br>`_003918_078077` | 0.7487<br>0.7437 | 0.4830<br>0.4752 | 0.7646<br>0.7600 | 0.9438<br>0.9408 |  |  | WD sweep: wd 2e-1.
2026-06-01 | 47 | `run_20260601`<br>`_021234_940276` | 0.7479<br>0.7463 | **0.5248**<br>**0.4979** | 0.7699<br>0.7670 | 0.9460<br>0.9424 | ✓ | ✓ | WD sweep: wd 4e-1. Best mean min-F1 of the wd magnitudes (0.498), clears 0.5 on wrist_smash for the best serial.
2026-06-01 | 48 | `run_20260601`<br>`_023543_278210` | 0.7496<br>0.7394 | 0.4454<br>0.4868 | 0.7692<br>0.7597 | 0.9398<br>0.9393 |  |  | Gate+focal-revert arm: une_v1_14 / v2, the taxonomy the alpha-revert was built to target.
2026-06-02 | 63 | `run_20260602`<br>`_130421_390854` | 0.7553<br>0.7378 | 0.4818<br>0.4781 | 0.7715<br>0.7581 | 0.9434<br>0.9396 |  |  | wd 1e-2, gate on (focal_alpha_revert): une_v1_14 / v2. Floor below the plain wd 4e-1 (#47).
2026-06-02 | 64 | `run_20260602`<br>`_143618_156220` | 0.7526<br>**0.7483** | 0.5029<br>0.4819 | 0.7727<br>0.7663 | 0.9438<br>0.9388 |  | ✓ | wd 4e-1, gate on (focal_alpha_revert): une_v1_14 / v2. Top une mean macro (0.748, 4 serials) but floor trails #47; alpha-revert retired.

### une_merge_v1 (early sided 28-class drop-unknown runs; no clean canonical name)

Date | # | Run ID | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---
**split_bst_baseline** | | | | | | | | |
2026-04-20 | 7* | `run_20260420`<br>`_141629` | **0.7715**<br>**0.7664** | 0.3810<br>0.3489 | 0.7907<br>0.7873 | 0.9451<br>0.9410 | ✓ | ✓ | First une_merge_v1 run (sided, 28-class) and first CSV flat-pipeline run; drop-unknown, BST split. Schedule mirrors #4.
**split_v2** | | | | | | | | |
2026-04-20 | 8* | `run_20260420`<br>`_171101` | 0.7428<br>0.7407 | **0.4315**<br>**0.3894** | 0.7656<br>0.7663 | 0.9367<br>0.9363 | ✓ | ✓ | Same as #7 (sided, 28-class), split swapped to split_v2 (player-overlap-minimised).
2026-04-25 | 9* | `run_20260425`<br>`_150548` | 0.7550<br>0.7476 | 0.3518<br>0.3326 | 0.7801<br>0.7737 | 0.9426<br>0.9418 |  |  | Sanity check that the re-extraction worked: re-extracted and sticky-anchored only the 1,716 worst (hit-zone-busted) clips, symlinked the rest. Sided 28-class.
2026-04-30 | 12* | `run_20260430`<br>`_110101` | 0.7431<br>0.7393 | 0.2447<br>0.3172 | 0.7703<br>0.7664 | 0.9438<br>0.9384 |  |  | Phase-2 sanity 2/3: une sided (28-class) + v2 on the unified clean dir.

### shuttleset_18 (18-class, no-sides, drop-unknown)

Date | # | Run ID | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---
**split_v2** | | | | | | | | |
2026-05-30 | 32 | `run_20260530`<br>`_161525_131279` | 0.6652<br>0.6544 | 0.0000<br>0.0000 | 0.7656<br>0.7535 | 0.9372<br>0.9290 |  |  | Multi-taxon batch: shuttleset_18 / v2 (finest 18-class cut). min-F1=0 is driven_flight, a single-test-clip dice-roll with no real signal.
2026-05-31 | 38 | `run_20260531`<br>`_163906_107348` | 0.6670<br>0.6538 | 0.0000<br>0.0000 | 0.7634<br>0.7535 | 0.9341<br>0.9290 |  |  | Gate+focal-revert arm: shuttleset_18 / v2. min=0 driven_flight dice-roll again.
2026-06-01 | 49 | `run_20260601`<br>`_141054_760522` | 0.6634<br>0.6545 | 0.0000<br>0.0000 | 0.7556<br>0.7534 | 0.9307<br>0.9288 |  |  | wd 1e-2, gate off: shuttleset_18 / v2. Floor is driven_flight (n/a); macro tracks the standard.
2026-06-01 | 50 | `run_20260601`<br>`_153738_936523` | **0.6721**<br>**0.6587** | 0.0000<br>0.0000 | 0.7646<br>0.7580 | 0.9393<br>0.9346 | ✓ | ✓ | wd 4e-1, gate off: shuttleset_18 / v2. Small macro nudge over the standard; floor n/a.
2026-06-02 | 59 | `run_20260602`<br>`_063203_601676` | 0.6522<br>0.6484 | 0.0000<br>0.0000 | 0.7499<br>0.7461 | 0.9284<br>0.9276 |  |  | wd 1e-2, gate on (focal_alpha_revert): shuttleset_18 / v2. No gain over gate-off; floor n/a.
2026-06-02 | 60 | `run_20260602`<br>`_075831_129574` | 0.6584<br>0.6478 | 0.0000<br>0.0000 | 0.7575<br>0.7468 | 0.9329<br>0.9271 |  |  | wd 4e-1, gate on (focal_alpha_revert): shuttleset_18 / v2. No gain over gate-off; floor n/a.

---

## 3. Per sweep / ablation series

Ten chronological series, derived from the manifests' `ablation_id`, config knobs, dates and hosts. Each run belongs to exactly one. Series that mix taxonomies/splits carry both columns.


### Series A: BST paper baseline & LR / CG-AP schedule

Apr 17–18. Reproduce the BST paper on bst_25 (keep-unknown, baseline split), retune the cosine LR so decay bites in-window, then ablate the CG/AP auxiliary-loss schedule (gentle / aggressive anneal, always-on, null).

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-04-17 | 1 | `bst_cg_ap_base`<br>`_17_04_2026` | bst_25 [merged_25] | split_bst_baseline | 0.8230<br>0.8177 | 0.5850<br>0.5670 | 0.8410<br>0.8373 | 0.9630<br>0.9607 |  |  | BST paper hparams verbatim, 1600ep. TrackNetV3 *with* inpaint (vs paper's without) beats the paper. Baseline reference.
2026-04-17 | 2 | `run_20260417`<br>`_191851` | bst_25 [merged_25] | split_bst_baseline | 0.8300<br>0.8263 | **0.6270**<br>**0.6073** | 0.8440<br>0.8420 | 0.9640<br>0.9633 | ✓ | ✓ | Retuned LR: 120ep, cosine cycles 0.25→0.5 so LR reaches 0 in-window, warmup 400→100, patience 300→40.
2026-04-18 | 3 | `run_20260418`<br>`_104152` | bst_25 [merged_25] | split_bst_baseline | 0.8300<br>0.8243 | 0.6260<br>0.5860 | 0.8460<br>0.8433 | 0.9640<br>0.9633 |  |  | First CG/AP aux-loss anneal, gentle (cosine fade to 0 by ep60); most seeds picked val before the fade bit.
2026-04-18 | 4 | `run_20260418`<br>`_151139` | bst_25 [merged_25] | split_bst_baseline | **0.8310**<br>**0.8288** | 0.6000<br>0.5996 | 0.8500<br>0.8442 | 0.9680<br>0.9642 | ✓ | ✓ | Aggressive CG/AP anneal (fade to 0 by ep15, then pure backbone); first 5-serial run. Top macro of the schedule arms.
2026-04-18 | 5 | `run_20260418`<br>`_174238` | bst_25 [merged_25] | split_bst_baseline | 0.8280<br>0.8262 | 0.6030<br>0.5768 | 0.8440<br>0.8448 | 0.9630<br>0.9630 |  |  | CG/AP ablation, always-on arm (aux held 1.0 all 80ep).
2026-04-18 | 6 | `run_20260418`<br>`_234822` | bst_25 [merged_25] | split_bst_baseline | 0.8300<br>0.8221 | 0.5860<br>0.5777 | 0.8419<br>0.8398 | 0.9633<br>0.9609 |  |  | CG/AP ablation, control arm: the auxiliary loss pinned to 0 from ep1 (CG/AP fully off).

### Series B: New-taxonomy migration (une_merge_v1, split_v2, sticky-anchor)

Apr 20–25. Switch from BST's 25-class merge to the UNE merge, move to the CSV flat pipeline, trial split_v2 over the BST split, fold in sticky-anchor pose cleaning, and test the no-sides collapse.

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-04-20 | 7* | `run_20260420`<br>`_141629` | une_merge_v1 | split_bst_baseline | **0.7715**<br>**0.7664** | 0.3810<br>0.3489 | 0.7907<br>0.7873 | 0.9451<br>0.9410 | ✓ | ✓ | First une_merge_v1 run (sided, 28-class) and first CSV flat-pipeline run; drop-unknown, BST split. Schedule mirrors #4.
2026-04-20 | 8* | `run_20260420`<br>`_171101` | une_merge_v1 | split_v2 | 0.7428<br>0.7407 | **0.4315**<br>0.3894 | 0.7656<br>0.7663 | 0.9367<br>0.9363 | ✓ |  | Same as #7 (sided, 28-class), split swapped to split_v2 (player-overlap-minimised).
2026-04-25 | 9* | `run_20260425`<br>`_150548` | une_merge_v1 | split_v2 | 0.7550<br>0.7476 | 0.3518<br>0.3326 | 0.7801<br>0.7737 | 0.9426<br>0.9418 |  |  | Sanity check that the re-extraction worked: re-extracted and sticky-anchored only the 1,716 worst (hit-zone-busted) clips, symlinked the rest. Sided 28-class.
2026-04-25 | 10* | `run_20260425`<br>`_185421` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7462<br>0.7427 | 0.4238<br>**0.3969** | 0.7651<br>0.7658 | 0.9396<br>0.9376 |  | ✓ | Nosides collapse of #9 (28→14 cls by dropping Top_/Bottom_); collapse rescues wrist_smash min-F1.

### Series C: Phase-2 unified-data sanity (3 combos)

Apr 29–30. Re-run three representative combos on the unified 32,203-stem sticky-anchor clean directory to confirm the full-extract data matches the Phase-1 baselines.

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-04-29 | 11 | `run_20260429`<br>`_202144` | bst_25 [merged_25] | split_bst_baseline | **0.8307**<br>**0.8314** | **0.5769**<br>**0.5775** | 0.8495<br>0.8477 | 0.9676<br>0.9687 | ✓ | ✓ | Phase-2 sanity 1/3: re-run of the bst_25 baseline combo (drop-unknown) on the unified 32,203-stem sticky-anchor clean dir.
2026-04-30 | 12* | `run_20260430`<br>`_110101` | une_merge_v1 | split_v2 | 0.7431<br>0.7393 | 0.2447<br>0.3172 | 0.7703<br>0.7664 | 0.9438<br>0.9384 |  |  | Phase-2 sanity 2/3: une sided (28-class) + v2 on the unified clean dir.
2026-04-30 | 13* | `run_20260430`<br>`_170325` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7466<br>0.7419 | 0.4027<br>0.3751 | 0.7703<br>0.7673 | 0.9398<br>0.9378 |  |  | Phase-2 sanity 3/3: une_v1_14 + v2 on the unified clean dir. LS=0.1 (default of the era).

### Series D: Regularisation & loss sweep on une_v1_14

Apr 30 – May 3. Hold the une_v1_14 / v2 baseline and sweep the loss: label smoothing (0.0 / 0.1 / 0.15), class weighting, the adaptive-focal CDB-F1 family (tau / gamma / alpha pair-cap), and an MLP-head capacity bump.

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-04-30 | 14* | `run_20260430`<br>`_213933` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7403<br>0.7433 | 0.4044<br>0.3591 | 0.7661<br>0.7681 | 0.9405<br>0.9392 |  |  | Label-smoothing ablation: LS=0.0 vs #13's LS=0.1.
2026-05-01 | 15* | `run_20260501`<br>`_073430` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7525<br>0.7471 | 0.4482<br>0.4168 | 0.7730<br>0.7686 | 0.9396<br>0.9378 |  |  | Label-smoothing LS=0.15.
2026-05-01 | 16* | `run_20260501`<br>`_110525` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7508<br>**0.7478** | **0.5179**<br>0.4221 | 0.7656<br>0.7695 | 0.9357<br>0.9360 | ✓ | ✓ | Class-weighting smoke test: LS=0.15 + class_weights{wrist_smash:2.0, smash:2.0}.
2026-05-01 | 17 | `run_20260501`<br>`_164658` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7530<br>0.7432 | 0.4863<br>**0.4621** | 0.7692<br>0.7617 | 0.9403<br>0.9351 |  | ✓ | First CDB-F1 run: adaptive focal (tau=1, gamma=1, momentum=0.9, warm_up=5), LS dropped to 0.
2026-05-01 | 18 | `run_20260501`<br>`_192113` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7425<br>0.7401 | 0.4938<br>0.4342 | 0.7582<br>0.7585 | 0.9357<br>0.9354 |  |  | CDB-F1 follow-up: gamma 1→0, the per-sample focusing term (per-class alpha shape unchanged).
2026-05-01 | 19 | `run_20260501`<br>`_192519` | une_v1_14 [une_merge_v1_nosides] | split_v2 | **0.7533**<br>0.7452 | 0.3670<br>0.4119 | 0.7720<br>0.7665 | 0.9403<br>0.9389 | ✓ |  | CDB-F1 follow-up: tau 1→0.5, softening the per-class alpha weighting (range narrows ~0.48–1.44).
2026-05-01 | 20 | `run_20260501`<br>`_230252` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7480<br>0.7402 | 0.4181<br>0.4105 | 0.7675<br>0.7604 | 0.9331<br>0.9344 |  |  | CDB-F1 + alpha pair-cap forcing alpha[smash] ≥ 0.7×alpha[wrist_smash] each epoch.
2026-05-02 | 21 | `run_20260502`<br>`_075808` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7330<br>0.7359 | 0.4873<br>0.4207 | 0.7494<br>0.7559 | 0.9350<br>0.9346 |  |  | CDB-F1 gamma 1→2 (Lin et al. focal default); didn't lift the floor.
2026-05-03 | 22 | `run_20260503`<br>`_104300` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7434<br>0.7414 | 0.4449<br>0.4138 | 0.7570<br>0.7604 | 0.9365<br>0.9319 |  |  | Capacity bump: MLP head hidden 400→1200 (encoder untouched). CDB-F1 parity test.

### Series E: Data-quality: shuttle-unzeroing & mask-wiring

May 3–4. Stop zeroing the shuttle track on keypoint-fail (wipe_drop), add a post-TCN shuttle-missing channel (mask_wiring), and check the augmentation jitter actually helps (jitter-off).

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-05-03 | 23 | `run_20260503`<br>`_172922` | une_v1_14 [une_merge_v1_nosides] | split_v2 | **0.7559**<br>**0.7481** | **0.4935**<br>**0.4741** | 0.7684<br>0.7653 | 0.9334<br>0.9353 | ✓ | ✓ | Shuttle-unzeroing (wipe_drop): stop zeroing shuttle on keypoint-fail (~14k frames, 0.84%). Project best at the time; smash & wrist_smash lift together.
2026-05-03 | 24 | `run_20260503`<br>`_192718` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7456<br>0.7440 | 0.4899<br>0.4568 | 0.7646<br>0.7630 | 0.9391<br>0.9365 |  |  | Shuttle-mask (mask_wiring): wipe_drop + a post-TCN shuttle_missing channel fused via mask_proj + shuttle_fuse.
2026-05-04 | 25 | `run_20260504`<br>`_152529` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7398<br>0.7401 | 0.4848<br>0.4301 | 0.7584<br>0.7586 | 0.9348<br>0.9365 |  |  | Jitter-off ablation (RandomTranslation prob=0) vs the wipe_drop best; min-F1 −4.4.

### Series F: Augmentation framework v1 sweep

May 5–6. Coupled centreline-flip + pos/shuttle jitter on the wipe_drop substrate, then a round-1 hparam sweep over p_flip / p_jitter / jitter caps.

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-05-05 | 26 | `run_20260505`<br>`_111211` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7341<br>0.7388 | 0.5034<br>0.4750 | 0.7573<br>0.7591 | 0.9405<br>0.9402 |  |  | Augmentation framework v1: coupled centreline flip + pos/shuttle constrained jitter (p_flip0.5, p_jitter0.2). Replaces the broken joints-only jitter.
2026-05-05 | 27 | `run_20260505`<br>`_154907` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7479<br>**0.7447** | 0.5147<br>0.4778 | 0.7675<br>0.7635 | 0.9407<br>0.9394 |  | ✓ | Aug v1, p_jitter 0.2→0.3. The current-best aug config; sweep reference.
2026-05-05 | 28 | `run_20260505`<br>`_213008_504674` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7427<br>0.7402 | 0.4785<br>0.4783 | 0.7615<br>0.7602 | 0.9391<br>0.9383 |  |  | Aug sweep cell `p_flip_25`: p_flip 0.5→0.25 (recover cross_court_net_shot?).
2026-05-05 | 29 | `run_20260505`<br>`_233645_734631` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7349<br>0.7339 | 0.4803<br>0.4587 | 0.7558<br>0.7539 | 0.9386<br>0.9344 |  |  | Aug sweep cell `cap_bump`: cap_y0.05→0.075, cap_x0.10→0.15. Killed at S4 on the wrapper's macro tolerance.
2026-05-06 | 30 | `run_20260506`<br>`_011851_522295` | une_v1_14 [une_merge_v1_nosides] | split_v2 | **0.7506**<br>0.7426 | 0.4760<br>**0.4822** | 0.7684<br>0.7610 | 0.9455<br>0.9409 | ✓ | ✓ | Aug sweep cell `p_jitter_40`: p_jitter 0.3→0.4.
2026-05-06 | 31 | `run_20260506`<br>`_032632_652587` | une_v1_14 [une_merge_v1_nosides] | split_v2 | 0.7447<br>0.7389 | **0.5231**<br>0.4569 | 0.7625<br>0.7588 | 0.9391<br>0.9385 | ✓ |  | Aug sweep cell `p_flip_25 × p_jitter_30`: override matched base, so config collided with #28 (same aug, different seeds).

### Series G: Multi-taxonomy baseline batch (taxon-pinned)

May 30–31. One clean cell per taxonomy on the taxon_pinned_w_preds collation: shuttleset_18, bst_24, bst_12, bst_25, une_v1_14, across v2 and baseline splits. Establishes the per-taxonomy reference on the final data.

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-05-30 | 32 | `run_20260530`<br>`_161525_131279` | shuttleset_18 | split_v2 | 0.6652<br>0.6544 | 0.0000<br>0.0000 | 0.7656<br>0.7535 | 0.9372<br>0.9290 |  |  | Multi-taxon batch: shuttleset_18 / v2 (finest 18-class cut). min-F1=0 is driven_flight, a single-test-clip dice-roll with no real signal.
2026-05-30 | 33 | `run_20260530`<br>`_174818_410060` | bst_24 | split_v2 | 0.8156<br>0.8093 | 0.5714<br>0.5336 | 0.8277<br>0.8221 | 0.9603<br>0.9575 |  |  | Multi-taxon batch: bst_24 / v2. v2 val overshoots test ~3.9%.
2026-05-30 | 34 | `run_20260530`<br>`_192738_970644` | bst_12 | split_v2 | 0.8210<br>0.8116 | 0.6314<br>0.5990 | 0.8284<br>0.8221 | 0.9591<br>0.9577 |  |  | Multi-taxon batch: bst_12 / v2 (fewest classes; top macro of the batch).
2026-05-30 | 35 | `run_20260530`<br>`_210600_435552` | bst_25 | split_bst_baseline | 0.8303<br>0.8242 | **0.6562**<br>**0.6203** | 0.8431<br>0.8408 | 0.9647<br>0.9654 | ✓ | ✓ | Multi-taxon batch: bst_25 / baseline, keep-unknown (paper-faithful).
2026-05-30 | 36 | `run_20260530`<br>`_225714_593038` | bst_24 | split_bst_baseline | **0.8421**<br>**0.8265** | 0.6122<br>0.5683 | 0.8534<br>0.8422 | 0.9640<br>0.9661 | ✓ | ✓ | Multi-taxon batch: bst_24 / baseline, drop-unknown. Top macro overall; dropping unknown lifts the 24 real strokes.
2026-05-31 | 37 | `run_20260531`<br>`_005535_005154` | une_v1_14 | split_v2 | 0.7514<br>0.7423 | 0.4851<br>0.4714 | 0.7703<br>0.7619 | 0.9405<br>0.9386 |  |  | Multi-taxon batch: une_v1_14 / v2. Splitting wrist_smash off smash and passive_drop off drop is expensive on the parents.

### Series H: Val-improvability-gate + focal-alpha-revert across taxa

May 31 – Jun 1 (bourbaki). The val-gate + focal_alpha_revert_overallocated arm run across five taxonomies/splits. Ran interleaved with series I as one batch.

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-05-31 | 38 | `run_20260531`<br>`_163906_107348` | shuttleset_18 | split_v2 | 0.6670<br>0.6538 | 0.0000<br>0.0000 | 0.7634<br>0.7535 | 0.9341<br>0.9290 |  |  | Gate+focal-revert arm: shuttleset_18 / v2. min=0 driven_flight dice-roll again.
2026-05-31 | 39 | `run_20260531`<br>`_193021_308927` | bst_24 | split_v2 | 0.8108<br>0.8065 | 0.5636<br>0.5294 | 0.8258<br>0.8206 | 0.9612<br>0.9573 |  |  | Gate+focal-revert arm: bst_24 / v2.
2026-05-31 | 41 | `run_20260531`<br>`_211838_567072` | bst_12 | split_v2 | 0.8136<br>0.8093 | 0.6008<br>0.5870 | 0.8282<br>0.8222 | 0.9584<br>0.9578 |  |  | Gate+focal-revert arm: bst_12 / v2.
2026-05-31 | 43 | `run_20260531`<br>`_225619_826430` | bst_25 | split_bst_baseline | 0.8261<br>0.8193 | **0.6522**<br>**0.6148** | 0.8396<br>0.8382 | 0.9676<br>0.9666 | ✓ | ✓ | Gate+focal-revert arm: bst_25 / baseline, keep-unknown.
2026-06-01 | 46 | `run_20260601`<br>`_005010_962006` | bst_24 | split_bst_baseline | **0.8336**<br>**0.8286** | 0.5652<br>0.5726 | 0.8480<br>0.8451 | 0.9673<br>0.9669 | ✓ | ✓ | Gate+focal-revert arm: bst_24 / baseline, drop-unknown.
2026-06-01 | 48 | `run_20260601`<br>`_023543_278210` | une_v1_14 | split_v2 | 0.7496<br>0.7394 | 0.4454<br>0.4868 | 0.7692<br>0.7597 | 0.9398<br>0.9393 |  |  | Gate+focal-revert arm: une_v1_14 / v2, the taxonomy the alpha-revert was built to target.

### Series I: Weight-decay sweep (gate off, decay exclusion)

May 31 – Jun 2 (carmack). AdamW weight decay with norms / bias / embeddings held out of decay, val-gate off. The une_v1_14 / v2 magnitude sweep (1e-2 through 4e-1) plus the two endpoints (1e-2, 4e-1) across the other five taxon / split cells. wd 4e-1 lifts the floor where it started lowest (bst_24 both splits, une); flat-to-down elsewhere. Default optimiser setting going forward: wd 4e-1 with the exclusion.

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-05-31 | 40 | `run_20260531`<br>`_201350_026614` | une_v1_14 | split_v2 | 0.7525<br>0.7447 | 0.5009<br>0.4848 | 0.7663<br>0.7636 | 0.9403<br>0.9377 |  |  | WD sweep: wd 1e-2 *excluding* norms/bias/embeddings from decay. New une_v1_14 high (mean macro/min +0.003/+0.014).
2026-05-31 | 42 | `run_20260531`<br>`_214009_170864` | une_v1_14 | split_v2 | 0.7451<br>0.7399 | 0.4873<br>0.4753 | 0.7680<br>0.7612 | 0.9410<br>0.9396 |  |  | WD sweep: wd 5e-2.
2026-05-31 | 44 | `run_20260531`<br>`_231403_971803` | une_v1_14 | split_v2 | 0.7335<br>0.7380 | 0.4923<br>0.4816 | 0.7520<br>0.7575 | 0.9336<br>0.9366 |  |  | WD sweep: wd 1e-1.
2026-06-01 | 45 | `run_20260601`<br>`_003918_078077` | une_v1_14 | split_v2 | 0.7487<br>0.7437 | 0.4830<br>0.4752 | 0.7646<br>0.7600 | 0.9438<br>0.9408 |  |  | WD sweep: wd 2e-1.
2026-06-01 | 47 | `run_20260601`<br>`_021234_940276` | une_v1_14 | split_v2 | 0.7479<br>0.7463 | 0.5248<br>0.4979 | 0.7699<br>0.7670 | 0.9460<br>0.9424 |  |  | WD sweep: wd 4e-1. Best mean min-F1 of the wd magnitudes (0.498), clears 0.5 on wrist_smash for the best serial.
2026-06-01 | 51 | `run_20260601`<br>`_171005_492538` | bst_24 | split_v2 | 0.8179<br>0.8114 | 0.6264<br>0.5566 | 0.8275<br>0.8228 | 0.9550<br>0.9586 |  |  | wd 1e-2, gate off: bst_24 / v2. Exclusion lifts the floor over the all-layers standard; 4e-1 lifts it more.
2026-06-01 | 52 | `run_20260601`<br>`_184437_030823` | bst_24 | split_v2 | 0.8217<br>0.8130 | 0.6406<br>0.5880 | 0.8315<br>0.8250 | 0.9607<br>0.9577 |  |  | wd 4e-1, gate off: bst_24 / v2. Best floor here, +5.4% mean min over the standard; the keeper, new bst_24/v2 best.
2026-06-02 | 57 | `run_20260602`<br>`_025726_461760` | bst_24 | split_bst_baseline | 0.8339<br>0.8295 | 0.6358<br>0.6239 | 0.8459<br>0.8434 | 0.9688<br>0.9661 |  |  | wd 1e-2, gate off: bst_24 / baseline. Big floor lift over the standard; 4e-1 edges it further.
2026-06-02 | 58 | `run_20260602`<br>`_044754_361908` | bst_24 | split_bst_baseline | **0.8347**<br>**0.8323** | **0.6557**<br>**0.6312** | 0.8426<br>0.8437 | 0.9718<br>0.9689 | ✓ | ✓ | wd 4e-1, gate off: bst_24 / baseline. Best floor, +6.3% mean min; the keeper, new bst_24/baseline best.
2026-06-01 | 53 | `run_20260601`<br>`_202022_522780` | bst_12 | split_v2 | 0.8177<br>0.8141 | 0.6231<br>0.6119 | 0.8258<br>0.8252 | 0.9607<br>0.9585 |  |  | wd 1e-2, gate off: bst_12 / v2. Exclusion grabs the floor; the bst_12 keeper (4e-1 adds nothing).
2026-06-01 | 54 | `run_20260601`<br>`_215358_806530` | bst_12 | split_v2 | 0.8150<br>0.8095 | 0.6364<br>0.6029 | 0.8218<br>0.8208 | 0.9557<br>0.9565 |  |  | wd 4e-1, gate off: bst_12 / v2. No gain over 1e-2, slightly over-regularises.
2026-06-01 | 55 | `run_20260601`<br>`_232557_387981` | bst_25 | split_bst_baseline | 0.8268<br>0.8155 | 0.6250<br>0.5481 | 0.8437<br>0.8357 | 0.9633<br>0.9639 |  |  | wd 1e-2, gate off: bst_25 / baseline. Keep-unknown: exclusion drops the floor below the standard.
2026-06-02 | 56 | `run_20260602`<br>`_011036_539898` | bst_25 | split_bst_baseline | 0.8289<br>0.8222 | 0.6525<br>0.6141 | 0.8428<br>0.8389 | 0.9687<br>0.9670 |  |  | wd 4e-1, gate off: bst_25 / baseline. Recovers most of 1e-2's drop, still under the standard; bst_25 stays on it.
2026-06-01 | 49 | `run_20260601`<br>`_141054_760522` | shuttleset_18 | split_v2 | 0.6634<br>0.6545 | 0.0000<br>0.0000 | 0.7556<br>0.7534 | 0.9307<br>0.9288 |  |  | wd 1e-2, gate off: shuttleset_18 / v2. Floor is driven_flight (n/a); macro tracks the standard.
2026-06-01 | 50 | `run_20260601`<br>`_153738_936523` | shuttleset_18 | split_v2 | 0.6721<br>0.6587 | 0.0000<br>0.0000 | 0.7646<br>0.7580 | 0.9393<br>0.9346 |  |  | wd 4e-1, gate off: shuttleset_18 / v2. Small macro nudge over the standard; floor n/a.

### Series J: Weight-decay endpoints x focal-alpha-revert (gate on)

Jun 2 (carmack). The val-improvability gate + focal_alpha_revert_overallocated arm crossed with the two wd endpoints (1e-2, 4e-1), on bst_25 / baseline, une_v1_14 / v2 and shuttleset_18 / v2. Never the best config for any of the three; second batch after series H to show alpha-revert earns nothing, so it can be retired.

Date | # | Run ID | Taxonomy [legacy] | Split | Macro F1 (best / mean) | Min F1 (best / mean) | Acc (best / mean) | Top-2 (best / mean) | best | mean | Description
---|---|---|---|---|---|---|---|---|---|---|---
2026-06-02 | 61 | `run_20260602`<br>`_092522_446222` | bst_25 | split_bst_baseline | 0.8262<br>**0.8219** | 0.6087<br>**0.5951** | 0.8414<br>0.8376 | 0.9613<br>0.9656 |  | ✓ | wd 1e-2, gate on (focal_alpha_revert): bst_25 / baseline. Recovers the bare-1e-2 floor drop, still under the standard.
2026-06-02 | 62 | `run_20260602`<br>`_111410_233326` | bst_25 | split_bst_baseline | **0.8286**<br>0.8207 | **0.6298**<br>0.5736 | 0.8394<br>0.8377 | 0.9636<br>0.9666 | ✓ |  | wd 4e-1, gate on (focal_alpha_revert): bst_25 / baseline. No better than gate-off 4e-1; under the standard.
2026-06-02 | 63 | `run_20260602`<br>`_130421_390854` | une_v1_14 | split_v2 | 0.7553<br>0.7378 | 0.4818<br>0.4781 | 0.7715<br>0.7581 | 0.9434<br>0.9396 |  |  | wd 1e-2, gate on (focal_alpha_revert): une_v1_14 / v2. Floor below the plain wd 4e-1 (#47).
2026-06-02 | 64 | `run_20260602`<br>`_143618_156220` | une_v1_14 | split_v2 | 0.7526<br>0.7483 | 0.5029<br>0.4819 | 0.7727<br>0.7663 | 0.9438<br>0.9388 |  |  | wd 4e-1, gate on (focal_alpha_revert): une_v1_14 / v2. Top une mean macro (0.748, 4 serials) but floor trails #47; alpha-revert retired.
2026-06-02 | 59 | `run_20260602`<br>`_063203_601676` | shuttleset_18 | split_v2 | 0.6522<br>0.6484 | 0.0000<br>0.0000 | 0.7499<br>0.7461 | 0.9284<br>0.9276 |  |  | wd 1e-2, gate on (focal_alpha_revert): shuttleset_18 / v2. No gain over gate-off; floor n/a.
2026-06-02 | 60 | `run_20260602`<br>`_075831_129574` | shuttleset_18 | split_v2 | 0.6584<br>0.6478 | 0.0000<br>0.0000 | 0.7575<br>0.7468 | 0.9329<br>0.9271 |  |  | wd 4e-1, gate on (focal_alpha_revert): shuttleset_18 / v2. No gain over gate-off; floor n/a.

---

## Appendix: the unknown ghost-channel era

From late April to 1 May, every drop-unknown run on a taxonomy that carried an `unknown` class trained with a dead `unknown` output slot. Worth knowing which runs, because it puts that batch in a slightly different architecture from everything after.

**What was happening.** The model head was always sized to `taxonomy.n_classes`, and every taxonomy in the registry lists `unknown`. `drop_unknown=True` only told the collator to drop `raw_type_en == 'unknown'` rows; it never shrank the head. So a dropunk run on an unknown-bearing taxonomy got an `unknown` output channel that saw no positive samples, still ate a softmax slot, still took a label-smoothed target on every sample, and (in the one class-weighted run) pushed the class-weight renorm onto an n+1 basis. Never populated, never in per-class F1: a ghost. It turned up in the class-weighted run (#16, `run_20260501_110525`) on the morning of 1 May, where the live loss printout listed `unknown weight=0.882` for a class that never trained.

**Which runs carried the ghost** (head dim read straight off the saved `mlp_head.mlp.mlp.3.weight`):
- #7, #8, #9, #12 (sided `une_merge_v1`): 29-channel head, 28 real classes.
- #10, #13, #14, #15, #16 (nosides `une_merge_v1_nosides`): 15-channel head, 14 real classes.

All drop-unknown, all on or before 1 May morning. The fix doc names #13–16; the weights show the same ghost back to #7.

**The one that looks like a ghost but isn't.** #11 (`run_20260429_202144`, merged_25 dropunk) has a 25-channel head reporting 24, but index-0 is fed by the 52 `driven_flight` rows (the merge map sends `driven_flight` to `unknown`), so that channel is trained, not dead. It reports 24 only because the baseline test split happens to carry no `driven_flight` clip. Its numbers stand. The genuine keep-unknown runs (#1–6 merged_25, #35/#43 bst_25) likewise train `unknown` as a real 25th class and are fine.

**Fixed in effect (1 May, #17 on).** The head started being derived from the classes actually present in the train labels, so dropunk une collapsed from 15 to 14. #17 (`run_20260501_164658`) is the first 14-channel run, the same afternoon as the morning catch; #17–31 are all clean.

**Fully fixed (pinned collations, #32 on, 30 May).** The whole band-aid came out with the canonical taxonomy registry, where each taxon spells out its class list and excluded types directly, so the ghost can't recur by construction rather than by a runtime derivation.

**Does it move the numbers?** The dead logit rarely wins an argmax, so the practical hit to the affected runs' macro / min / acc is small, but #7–10 and #12–16 dropunk do sit in a different architecture era from #17 on: not a like-for-like head against the clean runs. The weights are what they are, so there's nothing to redo.

Full diagnosis and fix design: [`architecture_notes/unknown_channel_fix_review.md`](architecture_notes/unknown_channel_fix_review.md).

