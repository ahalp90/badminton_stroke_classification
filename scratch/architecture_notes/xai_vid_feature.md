# XAI Video Visualisation Feature

Comprehensive plan for the IG-modulated skeleton-overlay video, scoping a one-shot artefact (correct-smash vs misclassified-smash A/B) before deciding whether to generalise. Not a project priority lever — explanatory artefact for supervisor / paper audience use.

## Goal

A single side-by-side mp4 showing two test clips from S5 of `run_20260505_154907`:

1. A high-confidence correctly-called smash.
2. A high-confidence smash that the model called wrist_smash.

Each panel renders the source clip pixels with the model's input layered on top:

- 17-joint COCO skeleton + 19 bone edges per player, hitter slot only (sticky_anchor's pick).
- Shuttle dot in pixel space.
- Visual intensity of joints, bone edges, and the shuttle dot modulated by Integrated Gradients attribution to the predicted-class logit at each frame.
- Per-frame attribution-summary strip above each panel: cumulative IG magnitude over time.

The viewer simultaneously sees: the raw rally, the abstracted view the model received, the model's prediction, and which input features drove the call. The pair-confusion thesis (smash↔wrist_smash collapses on the pose stream alone) gets a visceral A/B.

## Approach summary

- Integrated Gradients via Captum 0.9.0 (already installed in venv-bst on engelbart).
- Attribution targets the predicted-class logit, not the true class. For the misclassified clip this gives the "this is what made it look like wrist_smash" story rather than "this is what should have but didn't".
- Zero baseline for IG, with a mean-pose fallback if convergence-delta gates fail (see Risks).
- Per-joint resolution recoverable from the gradient on the model's flat-input tensor of shape `(B, T, n_players, in_dim)` where `in_dim = (J+B)*2 = (17+19)*2 = 72`. L2 over the two coord dims yields per-joint and per-bone scalars per frame per player.
- Render layer: opencv on local cicd venv, source clip + raw mmpose kps + raw shuttle xy pulled back from engelbart. Final mp4 lives at `scratch/presentation_prep/ig_overlay_smash_vs_ws.mp4`.

## Data flow

```
                  on engelbart                                         on local
                  ============                                         ========

[BST.forward weights]                                                  [predictions/serial_5.pt]
  └─ run_<id>/weights/                                                      ⬇
                                                                       choose two clips (pred-class confident)
[collated test arrays]                                                      ⬇
  └─ npy_wipe_drop/test/ (via /scratch symlink)                         emit clip_stems + row indices
                                                                            ⬇
[Captum IntegratedGradients]                                                ⬇
  ├─ load weights into BST                                              git pull predictions/attributions
  ├─ load test arrays, slice picked rows
  ├─ ig.attribute(inputs=(JnB,), additional_forward_args=...)               ⬇
  └─ save attributions/{clip_stem}.pt                                  [render layer]
       ⬇                                                                ├─ open source mp4 (rsync from /scratch)
   commit + push                                                        ├─ read raw_kps + raw_shuttle .npys
                                                                        ├─ per real frame: composite
                                                                        └─ encode mp4

[source clip mp4s]              ──── rsync ────►                       [scratch/presentation_prep/
  └─ /scratch/comp320a/ShuttleSet/clips/...                              ig_overlay_<...>.mp4]

[raw mmpose .npys]              ──── rsync ────►                       (local working copies)
  └─ /scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/

[raw shuttle .npys]             ──── rsync ────►
  └─ /scratch/comp320a/ShuttleSet/shuttle_npy_flat/
```

Two engelbart-side scripts; one local render script. Predictions, attributions, and render output all under git.

## Phase 0 — pre-flight verification

Three unknowns. All can be resolved by reading code + a small ls on engelbart. Spawn one verification agent on the local repo + ask Ariel to run two commands on engelbart.

### 0.1 — Test-row → clip_stem mapping (resolved during scoping)

Verified via an Explore agent against the collation code. No clip_stems sidecar is written by `collate_npy` (`preparing_data/prepare_train_on_shuttleset.py:714`). The order is recoverable in two stages because there's a second filter applied at runtime by the dataset class:

**Stage A — collation-order list (from `prepare_train_on_shuttleset.py:760-800`)**:
1. `clips_df = pd.read_csv(clips_master_csv)` (line 760).
2. Filter `clips_df[split_v2 == 'test']` (line 766).
3. If `drop_unknown`: filter `clips_df[raw_type_en != 'unknown']` (line 768).
4. Iterate `clip_stem` in resulting pandas row order (line 782-785).
5. Skip rows where `{raw_data_root}/{clip_stem}_pos.npy` does not exist on disk (line 798-800).

That gives the collation-order list of clip_stems, matching the order of rows in `npy_wipe_drop/test/labels.npy`.

**Stage B — runtime-order list (from `shuttleset_dataset.py:186-195`)**:
The dataset class loads `labels.npy` then drops any row where `videos_len[i] == 0`. The model only ever sees the runtime-filtered list, so `predictions/serial_5.pt`'s row indices are in this filtered order.

To recover:
```python
# Stage A: collation-order list
clips_df = pd.read_csv(clips_master_csv)
clips_df = clips_df[clips_df["split_v2"] == "test"]
if drop_unknown:
    clips_df = clips_df[clips_df["raw_type_en"] != "unknown"]
collation_stems = [
    s for s in clips_df["clip_stem"]
    if (raw_root / f"{s}_pos.npy").exists()
]

# Stage B: drop zero-length
videos_len = np.load(collated_dir / "test" / "videos_len.npy")
test_stems = [s for s, v in zip(collation_stems, videos_len) if v > 0]
assert len(test_stems) == len(y_pred)
```

Determinism check: re-run the stage-A filter twice and assert equal lists. pandas iteration on a filtered DataFrame is row-order stable in Python 3.11.

Risk that survives: the `_pos.npy` existence check at stage A.5 runs against engelbart's raw-data dir. The local repo doesn't have those files (they're under `/scratch/comp320a/`). To compute the mapping locally, we either (a) rsync a `clip_stems.txt` from engelbart after deriving it there, or (b) trust the test count from videos_len + clips_master filter and skip the existence check (works only if no raw-pos files are missing from the test split — likely true post-Phase-2 but not guaranteed). Default: derive on engelbart, commit the resulting `test_clip_stems.txt` to the run dir.

### 0.2 — Sticky_anchor chosen-slot per clip per frame

The training data was collated from sticky_anchor's per-frame slot pick. For the overlay to show "what the model received", we draw that exact slot. If the chosen-slot record isn't recoverable, fallback is rendering both detection slots (with one of them being the visual hitter).

- Check: `ls /scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/` to see file naming. Look for a per-clip `*_chosen_slot.npy` or `*_slot.npy` or `*_ndet.npy` (the per-frame detection count, which alongside bbox proximity to court pos may be enough to back out the slot).
- Code check: `src/bst_x/preparing_data/apply_heuristic.py` (the sticky_anchor entrypoint per memory) for whether it writes a chosen-slot sidecar.

If absent, fallback rendering shows both slots faded equally; attribution still maps onto the slot that fed the model (the one indexed by `n_players` axis in JnB), we just lose the visual disambiguation.

### 0.3 — Bone-index layout

Verified during scoping. Recorded here for reference:

- `POSE_BONE_MULTIPLIER['JnB_bone'] = 1` (`preparing_data/shuttleset_dataset.py:25`).
- `get_bone_pairs('coco')` returns 19 ordered pairs (`shuttleset_dataset.py:28-40`): head (5 pairs), ears-to-shoulders (2), arms (4), torso (4), legs (4).
- `create_bones` (`shuttleset_dataset.py:79-89`) produces `(t, m, 19, 2)` bone tensors, zeroed where either endpoint joint is zero (sentinel for missing).
- In-dim layout in the flat model input: `[joints (17 × 2)][bones (19 × 2)]` per (frame, player). Source: training-loop reshape at `bst_x_train.py:239` applied to a tensor whose shape ends in `(..., J+B=36, 2)`.

So `attr.reshape(B, T, n_players, 36, 2)` gives a clean unflattened tensor for downstream per-joint / per-bone slicing.

## Phase 1 — pick the A/B clips

Local script at `scratch/presentation_prep/pick_ig_clips.py`:

```python
# pseudocode
payload = torch.load("predictions/serial_5.pt")
y_true, y_pred, classes = payload["y_true"], payload["y_pred"], payload["active_class_list"]
SMASH = classes.index("smash"); WS = classes.index("wrist_smash")
true_smash = (y_true == SMASH)
correct = true_smash & (y_pred == SMASH)
misclassed = true_smash & (y_pred == WS)
# Need softmax confidences too — pick from one extra eval pass that dumps logits, OR proxy with logit margin if logits saved
# For now: dump logits in the eval script and pick by confidence
```

Decision: the existing `predictions/serial_5.pt` only stores argmax preds. We need softmax confidence to pick "highest-confidence" rows. Cleanest fix: extend `eval_dump_predictions.py` to also save the softmax probabilities (`y_softmax: (n, n_classes)`). One-line addition.

Alternative: skip confidence ranking, pick the first correct + first misclassed clip in row order. Loses the "headline-clean" framing but cheaper.

Pick produces: `(correct_row_idx, correct_clip_stem)`, `(misclassed_row_idx, misclassed_clip_stem)`.

## Phase 2 — Captum attribution

Engelbart-side script at `scratch/presentation_prep/ig_attribute.py`. Runs once per clip.

### 2.1 — Model load

Mirror `eval_dump_predictions.py`'s pattern: read the run's `manifest.yaml`, derive collated_dir, load weights from `<run_dir>/weights/bst_..._5.pt`. Same active-class-list remap. Set `model.eval()` to freeze dropout and BatchNorm running stats. Required for IG completeness because:

- BatchNorm1d inside TCN (`tempose.py:143`) needs running-stats mode, not batch-stats mode (single-clip batch of size 1 would otherwise have zero variance and divide by zero).
- Dropout in attention `self.attend = nn.Sequential(Softmax, Dropout)` (`tempose.py:62-65` and `bst.py:47-50`) needs to be off, else the IG path integral picks up dropout noise.
- The model's outer `pre_dropout` and `mlp_head`'s dropout also need to be off.

`model.eval()` handles all of these in one call. Verify post-load:
```python
assert not model.training
print(f"cg_factor={model.cg_factor.item():.3f} ap_factor={model.ap_factor.item():.3f}")
```

`cg_factor` and `ap_factor` are `register_buffer` on BST (`bst.py:206-207`) and persist in state_dict. The CG/AP aux schedule fades them from 1.0 to 0.0 by epoch 15 (per the manifest's `aux_fade_end_epoch=15`); a fully-trained S5 checkpoint will have both at 0.0. Gradient through CG/AP branches will then be near-zero by construction, so attribution sits on the PPF and main TCN+transformer paths. If the print shows non-zero values (e.g. early-stopped before epoch 15), the CG/AP branches are still live and will pick up attribution too. Either case is fine; the talk track changes slightly.

### 2.2 — Captum forward wrapper

BST.forward signature: `(JnB, shuttle, pos, video_len)`. Captum's IG accepts inputs as a tuple, with the rest as `additional_forward_args`. Verified against captum 0.9.0 source: only the `inputs` tuple is in the autograd target set; `additional_forward_args` participate in the forward graph but don't receive attribution even when they have `requires_grad=True` (captum docstring: "Note that attributions are not computed with respect to these arguments").

Cleanest pattern is to attribute over `(JnB, shuttle)` jointly in a single call:

```python
# JnB and shuttle need explicit requires_grad - .npy loads yield leaf tensors without it.
JnB = JnB.clone().requires_grad_(True)
shuttle = shuttle.clone().requires_grad_(True)
pos = pos.clone()  # no requires_grad needed; passed as additional arg

ig = IntegratedGradients(model)
(attr_jnb, attr_shuttle), delta = ig.attribute(
    inputs=(JnB, shuttle),
    additional_forward_args=(pos, video_len),
    target=pred_class_idx,
    baselines=(torch.zeros_like(JnB), torch.zeros_like(shuttle)),
    n_steps=50,
    internal_batch_size=10,
    return_convergence_delta=True,
)
```

**Important**: `JnB.clone().requires_grad_(True)` is mandatory. `.npy` loads produce leaf tensors with `requires_grad=False`; Captum needs grad-enabled inputs and will error or silently produce zero attribution otherwise.

Convergence delta gate: assert `abs(delta) / abs(attr_jnb.sum() + attr_shuttle.sum()) < 0.05`. If higher, switch baseline from zero to mean-pose (see Risks 1).

The convergence check is per-clip; record `delta` in the attribution payload so we can audit it from the .pt file later.

### 2.3 — Reduction

```python
# attr_jnb shape: (1, T, n_players=2, in_dim=72)
attr = attr_jnb.reshape(1, T, 2, 36, 2)         # unflatten
attr_per_node = attr.pow(2).sum(-1).sqrt()       # (1, T, 2, 36); L2 over xy
attr_joint = attr_per_node[..., :17]             # (1, T, 2, 17)
attr_bone  = attr_per_node[..., 17:]             # (1, T, 2, 19)

# shuttle attr shape: (1, T, 2). L2 over xy.
attr_shuttle = attr_shuttle.pow(2).sum(-1).sqrt()  # (1, T)
```

### 2.4 — Save

Write to `<run_dir>/attributions/{clip_stem}.pt` (sibling to `predictions/`). Bundle: `{"attr_joint", "attr_bone", "attr_shuttle", "pred_class_idx", "true_class_idx", "softmax_prob", "video_len", "clip_stem", "run_id", "serial_no"}`.

Snapshot-guard pattern from `eval_dump_predictions.py` re-used: refuse any write outside `attributions/`.

## Phase 3 — render layer

Local script at `scratch/presentation_prep/ig_overlay_render.py`. Uses cicd venv (matplotlib + numpy + opencv-python). Confirm opencv presence: `~/.venvs/badminton-cicd/bin/pip install --dry-run opencv-python-headless` — install if needed.

### 3.1 — Per-clip data fetch

Local working copies of:
- Source mp4 (rsync from engelbart): `/tmp/ig_render/{clip_stem}.mp4`
- Raw kps: `/tmp/ig_render/{clip_stem}_raw_kps.npy` plus siblings
- Raw shuttle: `/tmp/ig_render/{clip_stem}_shuttle.npy`
- Attribution: pulled via `git pull` from the engelbart-side commit.

### 3.2 — Per-frame render

For each real frame `f in 0..video_len-1`:

```python
frame = read_frame(cap, f)  # source pixel BGR
for player_idx in [hitter_slot]:  # sticky_anchor pick, or both as fallback
    for joint_i in range(17):
        x, y = raw_kps[f, slot_idx[f], joint_i]
        if np.isnan(x): continue
        intensity = scale(attr_joint[f, player_idx, joint_i])  # 0..1
        draw_circle(frame, (x, y), radius=base_r + intensity * delta_r, alpha=intensity)
    for bone_i, (j_start, j_end) in enumerate(BONE_PAIRS):
        # endpoints from raw_kps for this player slot
        # linewidth + alpha proportional to attr_bone[f, player_idx, bone_i]
        ...
# shuttle
sx, sy = raw_shuttle[f]
intensity = scale(attr_shuttle[f])
draw_circle(frame, (sx, sy), ..., alpha=intensity)
```

Scaling: per-clip normalisation (`attr / attr.max()` over the whole clip; then optional gamma to make modest attributions visible). Shared scale across the A/B panels by computing the max over both clips first.

Colour: viridis or cividis (protanopia-safe sequential). The per-joint dot uses the colourmap mapping of intensity. Shuttle uses a contrasting colour (e.g. white circle, alpha by attribution) so it doesn't lose against bird-yellow pixels in the source.

### 3.3 — Per-frame summary strip

Above each panel: a thin horizontal time-axis strip with a moving cursor at the current frame, total-attribution-this-frame as a line. Lets the viewer see "where the model concentrated" at a glance, on top of the spatial heatmap reading from the skeleton.

### 3.4 — Encoding

opencv VideoWriter at the source clip's framerate. Output: per-panel mp4 first, then composition.

## Phase 4 — composition

Side-by-side via numpy hstack per frame OR ffmpeg post-render. Top label per panel: `clip_stem · true class · predicted class · softmax confidence`.

Optional polish (defer unless quick):
- Slow-mo on the impact frame.
- Caption strip below explaining the colour-map scale.

## File touchpoints

### New files

- `scratch/architecture_notes/xai_vid_feature.md` (this file).
- `scratch/presentation_prep/pick_ig_clips.py` — local A/B picker.
- `scratch/presentation_prep/ig_attribute.py` — engelbart-side Captum runner.
- `scratch/presentation_prep/ig_overlay_render.py` — local renderer.
- `<run_dir>/attributions/{stem}.pt` (×2) — engelbart-side, committed.
- `scratch/presentation_prep/ig_overlay_smash_vs_ws.mp4` — final video, committed.

### Modified files

- `scratch/presentation_prep/eval_dump_predictions.py` — add softmax-probability save alongside argmax preds (one-line extension to the existing dump dict, plus a `softmax` call).

### Read-only references

- `src/bst_x/stroke_classification/model/bst.py` (forward + CG/AP buffers).
- `src/bst_x/stroke_classification/model/tempose.py` (attention impl).
- `src/bst_x/stroke_classification/preparing_data/shuttleset_dataset.py` (POSE_BONE_MULTIPLIER, get_bone_pairs, create_bones).
- `src/bst_x/stroke_classification/main_on_shuttleset/bst_x_train.py:239` (view-flatten that defines in_dim layout).
- `src/bst_x/stroke_classification/main_on_shuttleset/experiments/run_20260505_154907/manifest.yaml`.
- `src/bst_x/stroke_classification/main_on_shuttleset/experiments/run_20260505_154907/predictions/serial_5.pt`.
- `/scratch/comp320a/ShuttleSet/clips/**.mp4` (engelbart-only).
- `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/{stem}_raw_kps.npy` + siblings.
- `/scratch/comp320a/ShuttleSet/shuttle_npy_flat/{stem}.npy`.

### Untouched

- Training pipeline (no changes).
- Manifest, best_model_id.txt, tb/, weights/, test_log of the run dir (snapshot-guard pattern from the eval script).
- All existing presentation_prep visuals.

## Risks

Ordered by likelihood × severity.

### High likelihood

1. **Zero baseline → completeness failure** (medium severity). The pad sentinel for short clips is zero. After normalisation, court-origin is also at (0, 0). Sending a zero baseline through LayerNorm at the start of the transformer may produce degenerate distributions that break the linearity assumption IG relies on. Mitigation: convergence-delta gate at `|delta|/|attr.sum()| < 0.05`. If fails, switch to a class-conditional JnB-mean baseline computed by averaging JnB over real frames of N=10 calibration clips of the predicted class.

   Framing nuance (per the Captum review): the fallback baseline only swaps JnB; `pos` and `shuttle` are still passed per-clip via `additional_forward_args`. So the completeness identity reads "F(real clip) - F(class-mean JnB but real pos and shuttle)" rather than a pure-neutral baseline. That's valid IG, just label it as "class-conditional JnB baseline" in the talk track, not "neutral baseline". The PPF multiplicative coupling (`bst.py:280-286`) means a non-zero mean-JnB baseline produces a non-zero PPF activation at the baseline; that's a tradeoff against the zero-baseline LayerNorm risk, not a strict improvement.

   For publication-strength figures (later, not now): `NoiseTunnel(IntegratedGradients(model))` over a zero baseline gives SmoothGrad-IG, which sidesteps the single-baseline issue by averaging over jittered inputs. Cheaper than externally looping multi-baseline averaging. Captum has it built in.

2. **Sticky_anchor chosen-slot not recorded** (low-medium severity). Fallback render shows both detection slots with equal weight, which conceptually muddies "what the model saw". Mitigation: derive the slot from bbox-centroid proximity to the model's input `pos` (court-position) at each frame. Same logic sticky_anchor used. Adds ~30 min.

3. **TCN boundary attribution leak** (low severity, cosmetic). The TCN has receptive field 17 frames (kernel=5 with dilations 1 + 3). Padded-frame inputs at positions `[video_len:T]` are zero but participate in the conv as boundary context. Attribution at the last ~17 real frames will pick up a small tail from the padded zone via the conv, biasing late-clip attribution slightly low. The self-attention path correctly zero-attributes padded positions via `-inf` masking, so this only affects the TCN-mediated component. Not blocking; cosmetic only on the trailing edge of the clip.

### Medium likelihood

4. **PPF coupling caveat** (low severity, interpretation issue). `use_ppf=True` means JnB and pos are multiplicatively coupled at the input (`bst.py:280-286`). Attribution over JnB alone tells you "given this pos, which JnB drove the call". For the supervisor-facing claim ("the model focused on these joints") this is fine. For a stronger "pose vs position split" claim, attribute over `(JnB, pos)` jointly. Default: JnB + shuttle only, mention the coupling caveat in the talk track.

5. **Per-clip normalisation collapses cross-panel comparison** (low severity). Shared-scale normalisation across both panels (recommended by the review agent) needs the attribution magnitudes to be comparable. They should be, since both clips are real smashes and the model's gradient magnitudes are class-conditioned. Sanity: print mean/max attribution per clip after Phase 2.

6. **Bone attribution rendering as edges** vs folding into joints (cosmetic). Edge-rendering with linewidth ∝ bone-attribution is the cleaner visual (per review). The bones are derived as joint-pair offsets, so bone-attribution carries genuinely extra information (the model learned to weight pair-relations distinctly from joint positions). Default: render bones as edges with linewidth + alpha by bone-attribution; joints with radius + alpha by joint-attribution.

7. **`videos_len` filter at runtime can change predictions row count** (low severity, already accounted for in the recovery flow). Stage B of the clip_stem recovery requires loading `videos_len.npy` from the collated test dir; otherwise the row count doesn't match `len(y_pred)`. Already in the recovery script.

### Low likelihood

8. **Captum `internal_batch_size` memory split** (informational). Captum's IG splits the `n_steps` path integral into smaller batches if `internal_batch_size` is set. With `n_steps=50` and one clip, set to 10 to be safe; no real memory pressure at T=100 batch=1.

9. **MultiHeadCrossAttention residual omission** (cosmetic interpretation note). `bst.py:117-120` replaces `x1` rather than adding it to attention output. Doesn't break IG; biases attribution toward the shuttle stream in the cross-attention path. Mention in the talk track if questioned.

10. **Source mp4 framerate mismatch** with the model's per-frame stride (cosmetic). The clip mp4 frame index should align 1:1 with the MMPose extraction frame index, but verify on the first smoke render.

11. **Stage A `_pos.npy` existence check on local box** (low severity, deferred to engelbart). The collation-order recovery filter requires checking `(raw_root / f"{stem}_pos.npy").exists()`. Raw .npy files live on `/scratch/comp320a/` on engelbart, not locally. Workaround: derive the `test_clip_stems.txt` on engelbart in Phase 0.1 and commit it to the run dir, so local Phase 1 picking can read the file directly.

## Open decisions

For Ariel to reconcile before Phase 2 kicks off:

1. **Attribution target for the misclassified clip**: predicted class (default, "this is what made it look like ws") or true class ("this is what should have but didn't"). Default = predicted. Alternative = both as a 2×2 grid panel.
2. **Clip selection criterion**: highest-confidence pair (cleanest narrative, default) or mid-confidence pair (more honest about the boundary). Default = highest-confidence.
3. **Render scope**: hitter slot only or both detection slots. Default = hitter slot if Phase 0.2 resolves; fallback = both.
4. **Colour palette**: viridis (broader contrast) or cividis (more muted, slightly less differentiation). Either is protanopia-safe.
5. **Output framerate**: source rate or slow-mo. Default = source rate for the supervisor-facing version; slow-mo as a stretch.
6. **Side-by-side composition method**: numpy hstack in Python (one less dep, fine for ~10s clips) or ffmpeg `hstack` filter post-render (cleaner if the clips have different lengths). Default = ffmpeg if both clips lengths differ by >10 frames.

## Time estimate

| Phase | Estimate | Heaviest active-attention |
|---|---|---|
| 0 | ~20 min | reading agent recon output |
| 1 | ~15 min | running pick script, inspecting picks |
| 2 | ~30-45 min | engelbart wall-clock for IG runs (~minutes) + git transit |
| 3 | ~1-1.5 hr | colour/alpha taste iteration |
| 4 | ~15 min | composition checks |
| **Total** | **2.5-3.5 hr** | Phase 3 |

Add 30-60 min if Phase 0.1 (clip_stem mapping) requires a re-collation pass. Add 30 min if Phase 0.2 fallback (derive sticky_anchor slot from bbox proximity).

## Verification gates between phases

- **After 0**: confirm clip_stem mapping path is identified and reproducible. If fallback, confirm the re-derivation matches at least 3 known clips end-to-end.
- **After 1**: confirm the picked clips have the desired confidence properties (correct call's pred-class softmax is the max; misclassified call's pred-class softmax dominates the true-class softmax by ≥0.1).
- **After 2 (per clip)**: confirm `|delta|/|attr.sum()| < 0.05`. If not, switch baseline and re-run.
- **After 3 (smoke frame)**: render the impact-vicinity frame with full overlay. Visually confirm: skeleton positions match the player's pose, brightness varies across joints, no NaN/blank dots.
- **After 4**: side-by-side check that both panels share scale and timing.

## Branch / commit cadence

Continue on `presentation_prep`. Discrete commits:

1. `test_clip_stems.txt` under the run dir (engelbart-derived stage-A+B mapping sidecar).
2. `eval_dump_predictions.py` softmax-save extension.
3. `pick_ig_clips.py` + the chosen-clip metadata.
4. `ig_attribute.py` (engelbart-side script).
5. `attributions/{stem}.pt` ×2 (from engelbart).
6. `ig_overlay_render.py`.
7. `ig_overlay_smash_vs_ws.mp4`.

Push after each so engelbart and local stay in sync.
