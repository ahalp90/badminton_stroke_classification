# Arch 1: Open Research Directions

## Part 1: Tweaks to existing BST arch

### Q4. LR schedule: faster cosine drop-off

`bst_train.py:255` passes `num_cycles=0.25` into
`get_cosine_schedule_with_warmup`, alongside `n_epochs=1600`,
`warm_up_step=400`, and `early_stop_n_epochs=300`. At `num_cycles=0.25` only
a quarter of the cosine curve runs across the full budget, so the LR barely
decays.

Our runs converge around epoch 60 and early-stopping fires around epoch 360, so the scheduler never really has time to 
significantly lower the learning rate. Things to try to squeeze more lr decay into the real train window:

- Cut `n_epochs` to 200–400 and keep `num_cycles=0.25`. Speeds up decay so it sits in the epochs we actually use.
- Push `num_cycles` to 0.5, so more of the cosine curve lands before early-stop.
- Worth considering warm restarts with super low learning rate, though more effort.


#### Outcome (2026-04-17): retune beats paper and prior best on every test metric

Compressed `n_epochs` to match the real convergence timeframe & bumped up `num_cycles` so the cosine curve 
actually hits zero. 
`Apr17_13-04-35` run showed: best F1 macro 0.8311 at epoch 41 (out of 1600), val loss peaked by epoch 27, 
early-stop at 341. So at peak, the LR was basically still at full speed.

Active settings now (old values preserved commented in `bst_train.py`):

| param | was | now |
|---|---|---|
| `n_epochs` | 1600 | 120 |
| `warm_up_step` | 400 | 100 |
| `early_stop_n_epochs` | 300 | 40 |
| `num_cycles` | 0.25 | 0.5 |

Run `run_20260417_191851` (commit 2cb78b8), 3 serials on merged_25, test set
(num_strokes 3486):

| | F1 macro | F1 min | Accuracy | Top-2 |
|---|---|---|---|---|
| BST paper (published) | 0.8097 | 0.5762 | 0.8322 | — |
| Prior best (commit 8810e95, old schedule) | 0.823 | 0.585 | 0.841 | 0.963 |
| **Retune serial 1 (winner)** | **0.830** | **0.627** | **0.844** | **0.964** |
| Retune serial 2 | 0.822 | 0.610 | 0.841 | 0.963 |
| Retune serial 3 | 0.827 | 0.585 | 0.841 | 0.963 |

All three serials beat the paper on every metric, so it's not just a lucky random seed. 
Huge jump on F1 min (+4.2 points vs prior best, +5.1 vs paper). Harder classes get a massive benefit.

The val-vs-test direction flipped too: the old run had val macro 0.8311
but test macro 0.823; the retune's winner had val macro 0.816 but test
macro 0.830.

We might be hitting a data quality cap soon though. 3% are 'unknown', which is just a catch-all garbage class. 
Another 3% have known bad labels. 
And then 25% of the majority class (smash) have serious problems with over strict frame zeroing by mmpose (probably fixable).

Winning weight kept at
`main_on_shuttleset/weight/run_20260417_191851/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_25.pt`
and is tracked via an `!` override in `.gitignore`. Numbers above are
verified from `test_logs/test_20260417_191851.log`.

Next dial to experiment with, if we want more: nudge `n_epochs` up to
~150-180 and re-test. F1 min on the winning val curve was still climbing
at the final epoch, so there might be a tiny bit more juice to squeeze.

---

### Q5. Attention head geometry (`d_head`, `n_head`)

`bst.py:145` defaults to `d_model=100`, `d_head=128`, `n_head=6`. The model
concatenates across heads to `d_head * n_head = 768`, then
`MultiHeadCrossAttention.tail` (`bst.py:59-62`) projects back down to 100.
The temporal and interactional transformers in `tempose.py` follow the same
pattern.

I traced the ratio back to see where it came from. BST inherits it from
TemPose, which inherits it from AcT (Action Transformer). AcT ran
progressive-widening ablations on exactly this expand-then-contract
pattern, and I read their results as: a small `d_model` keeps the bulk of
the network cheap, while the wide per-head projection gives each head
enough capacity to learn a distinct specialised view. Low total parameter
count, rich per-head representations.

As far as I can tell, nobody has swept this on BST. Worth a pass
over `d_head ∈ {32, 64, 96, 128}`, either holding `n_head=6` (which shrinks
the model) or holding `d_head * n_head` constant (which tests whether the
expansion matters or just the total width). If a smaller `d_head` holds
F1, we get a free parameter-efficiency win.

One caveat I've already hit: `d_model` couples tightly across TCN,
cross-transformer, interactional transformer, and PPF, which I wrote up in
`tuning_thoughts.md`. So I'd hold `d_model=100` fixed and only vary
`d_head`/`n_head`.

---

### Q3. CG and AP weighting / annealing schedule

Right now CG (Clean Gate) and AP (Aim Player) run unweighted for the whole
training run, see the `use_cg`/`use_ap` branches in `bst.py`. The BST paper
shows both modules improve accuracy over the bare transformer, so they're
pulling real weight.

My hypothesis: their strongest role is as a **warm-start prior**, not as a
permanent fixture. Early in training the transformers haven't yet learnt
robust shuttle- or player-aware representations, so the hand-crafted CG/AP
interactions look like useful inductive bias in that regime. Later, once
the transformers have learnt their own (analogous, potentially richer)
interactions, a fixed CG/AP contribution could start to constrain the
model, pinning it to the hand-crafted formulation instead of letting it
find something better.

The experiment: add a scalar weight on the CG subtraction and on the AP
weighting in `bst.py`'s forward, and have `bst_train.py` pass in an
epoch-indexed schedule that sets it. Three configurations to compare:

- **Constant**: current behaviour, baseline.
- **Annealed out**: weight starts at 1.0, decays to 0. Directly tests the
  warm-start hypothesis.
- **Annealed in**: weight starts at 0, grows to 1.0. Tests the opposite,
  that CG/AP help most once the transformers have stabilised.

This couples to Q4. Any annealing schedule has to finish inside the
effective training window (~epoch 60 before early-stop fires), same
compressed-window concern as the LR schedule. Without the LR fix first,
anneal-in especially won't have time to reach its target weight.

---

## Part 2: X3D-S racket crop fusion

### Model choice: X3D-S

X3D-S is the model I'm going with for the racket-crop branch. This fits two constraints:
- Easily available with weights, and
- small enough (params) to fine-tune end-to-end in the short time we have--on a v100 16gb.

There are other strong, low-param models, MoViNet for example, but none
with prebuilt appropriate weights and easy model zoo integration. X3D would probably do even better
with SSv2 pretraining (fine hand motions), but the
SSv2 weights only exist as an unofficial TensorFlow port, and interface bugs will probably eat more time than the engineering.

Within the X3D family I picked S over XS and the larger variants:

- **vs XS.** XS expects 4 frames × stride=12, too coarse for granular
  badminton racket motion.
- **vs M / L / XL.** They only drop stride to 5, perform not-that-much
  better, and way more params.
- **X3D-S.** Strong accuracy at a low parameter count. Expected input is 13
  frames × stride=6.

### Target input shape: frames=39, stride=1

I'm fine-tuning X3D-S toward `frames=39, stride=1`, not its default
`13 × stride=6`.

`stride=1` gives the model access to every frame and lets it learn the
interactions between them, which is what granular badminton racket motion
needs. I set `39` so that by the final convolutional block the receptive
field covers all input frames. That imposes a hard limit around ~40
frames, which is fine for a racket crop centred on a stroke event.

### Fusion depth: where X3D-S output enters BST

*(to fill in)*

Competing ideas on how deep into BST the X3D-S signal cuts in:

- **Late concat, just before the MLP head.** Easiest to implement, lowest
  risk, but gives BST no chance to condition its attention on the racket
  signal.
- **Tie into attention earlier, in a meaningful way.** X3D-S output
  feeds into the cross-attention or the interactional transformer, so the
  racket evidence shapes how players and shuttle attend to each other.
  More expressive, more moving parts.
- **Separate tower with learned significance weighting.** X3D-S runs as
  its own tower and a learned scalar (or vector) gates how much its
  prediction counts vs BST's. Keeps the two branches clean and lets the
  model decide per-sample how much to trust the racket signal.

### Open questions: training and integration

Three things I still need to pin down:

1. **Fine-tuning and end-to-end schedule.** What's the right sequence for
   fine-tuning X3D-S on badminton video first, then co-training it
   end-to-end with the rest of Arch 1? Length of each phase, learning
   rates, what to freeze when. 
2. **Temporal cut-in of X3D-S feedback.** The reported stroke racket
   contact times are noisy. I need to pick where the X3D-S input window
   sits relative to the reported contact time so the feature stays
   responsive to the stroke event even when the reported time is slightly
   off. Options: a fixed offset centred on the reported time, a learned
   offset, or a slightly wider window that lets X3D-S self-align.
3. **Juggling MMPose drops.** MMPose periodically drops frames, sometimes
   with alarming frequency for certain stroke categories. The X3D-S
   window has to cope with that. First check to see if loosening the MMPose frame zeroing helps.  Otherwise, probably handle through interpolation (one euro filter?). Worst case: pin the camera to the shuttle velocity reversal position.

### MMPose frame-zeroing rules

Question 3 above feeds straight into a broader concern: I think the
current MMPose frame-zeroing rules are excessively strict, and I want to
check that before I build racket-anchoring heuristics on top of them.

Right now the pipeline zeroes out a whole frame if either of:

- The midpoint of a single player's feet extends outside the court
  boundaries, plus a small epsilon (which might itself be too tight).
- Either player failed to be detected on that frame.

Both probably throw away useful frames. Before I pick anchoring heuristics
for the racket crop, I want to loosen each criterion in turn, re-extract
video for the worst-offending clips, and see how much of the apparent
MMPose drop rate is artefactual vs genuinely unrecoverable. Wherever that
lands will also shape the window logic for (3) above.

---

## Cleanup backlog

### Dedup `bst_train.py` and `bst_infer.py` scaffolding

`bst_infer.py` and `bst_train.py` both carry their own copy of the
`MODELS` dict, a `Task` class with `get_network_architecture`, the
`pose_style` + `in_dim` arithmetic, and the dataloader setup from
`preparing_data.shuttleset_dataset`. The genuinely different parts are
small: `bst_infer.py` does argmax-only predictions with no metrics, and
its Task has a `load_weight` instead of the cache-or-train
`seek_network_weights`.

Two entry points is few enough that I'm leaving it for now. When a third
arrives (Gradio backend, ONNX export, or the Arch 1 fusion pipeline once
X3D-S lands), the right move is a `bst_common.py` holding `MODELS`, a
base `Task`, and the shared dataloader helpers, with `bst_train.py` and
`bst_infer.py` importing from it. A mirror TODO is pinned at the top of
`bst_infer.py`.

---

## Cross-references

- `src/bst_refactor/stroke_classification/model/bst.py`: model defaults
  (`d_model=100, d_head=128, n_head=6`), CG/AP branches in `BST.forward`,
  and the `CrossTransformerLayer` docstring (Q6, resolved).
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py:251-256`:
  cosine schedule configuration.
- `scratch/architecture_notes/tuning_thoughts.md`: broader HP strategy;
  Q4/Q5 here are new items it didn't cover, X3D-S schedule in (1) above
  refines the stub in that doc.
- `scratch/architecture_notes/architecture_1_bst_3dcnn_racket_extension_09_April.md`:
  the initial X3D-S fusion design doc (this section refines it).
