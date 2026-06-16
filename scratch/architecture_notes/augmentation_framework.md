# Augmentation framework

*Extracted from `hparams_sweep_speculations.md` on 2026-05-04 once the
augmentation analysis grew large enough to live on its own. Companion
doc to Isiah's writeup at `scratch/research/Augmentation.pdf`. The
locked decisions are anchored against that PDF; the
project-side filtering, code traces, and implementation outlines are
this doc.*

*Code traces are linked inline as `(see Aj)` and consolidated in
[Appendix A](#appendix-a-verified-code-traces) at the end of this
doc, so the body keeps reading prose-first while every load-bearing
claim has a verifiable trace one click away.*

## TLDR

- **Locked Task 2 set (2026-05-04; jitter raised to 0.3 on
  2026-05-05)**: centreline flip (p=0.5, coupled, COCO bilateral
  joint-index swap) plus a corrected pos+shuttle constrained-jitter
  (p=0.3, ±0.05y / ±0.10x cap, layered conditional bounds,
  joints/bones untouched, zero-frame preservation, shuttle
  off-screen mirroring). Replaces the broken `RandomTranslation_batch`
  (joints-only, decoupled, body-deforming). Jitter first locked at
  0.2; raised to 0.3 after run_20260505_154907 (0.3) beat
  run_20260505_111211 (0.2) by +0.6 macro / +0.3 min, gains in
  rush/push/drive/lob and no class dropping.
- **Jitter-off ablation result (2026-05-04, `run_20260504_152529`)**:
  turning the broken `RandomTranslation_batch` off (`prob=0.0`)
  regressed against the wipe_drop best (`run_20260503_172922`) by
  macro -0.8, min -4.4, acc -0.7. Conceptually wrong, empirically
  regularising. Defaults restored at `bst_x_train.py:375`; replace
  via the locked corrected formulation rather than disabling.
- **Out for Task 2**: temporal speed jitter (Phase 3 candidate),
  Gaussian joint jitter, random joint masking,
  `WeightedRandomSampler`, net flip.
- **Phase 3 / trimester 2 candidates**: temporal speed jitter
  (uniform [1.0, 2.0] coupled with shuttle-velocity downweight
  cost flagged), rotation / scaling / shearing for amateur
  cameras, per-joint adaptive focal as a loss-side research
  direction.
- **Coordinate spaces verified**: pos in court frame
  (post-homography), shuttle in camera frame, joints in
  bbox-centre-relative frame. PPF fuses pos into JnB at the
  input, before the TCN.
- **X3D-S hit-frame metadata** derivable without re-extraction
  via `clips_master.csv` correlation (Method A, faithful to
  annotation, susceptible to annotator drift) or shuttle
  horizontal-velocity sign reversals (Method B, independent
  verification with ±5-frame ceiling on soft shots, well within
  X3D-S's ±19-frame window).
- **Aug round 1 sweep findings (2026-05-06)**: 5-serial run-mean
  seed envelope is ~0.1% on macro/accuracy/top-2 and ~2% on min F1
  (see "Seed-noise envelope on run means" section below);
  smash/wrist_smash picked-serial F1 has flattened to ~0.51 /
  0.51-0.60, reading as pose-only features running dry for the
  pair (detail at [`hparams_sweep_speculations.md`](hparams_sweep_speculations.md),
  "Smash/wrist_smash F1 split" section).

## Coordinate spaces of the three streams (verified 2026-05-04)

The three input streams live in three different coordinate frames.
Spatial augmentations have to apply the right transform to each.

- **`pos` (player court position)** is in **court-relative [0,1]²
  post-homography**. Trace: `prepare_train_on_shuttleset.py:217-227`
  projects the bbox-bottom (or ankle midpoint) through the
  homography via `to_court_coordinate`, then `normalize_position`
  (lines 135-147) divides by court borders
  (`border_R - border_L`, `border_D - border_U`). x=0 is court-left,
  x=1 is court-right; y=0 is far baseline, y=1 is near baseline.
  See [A1](#a1-court-coord-normalisation-of-pos).
- **`shuttle`** is in **camera-resolution-normalised [0,1]² (raw
  video frame, NOT court-projected)**. Trace: `normalize_shuttlecock`
  at `prepare_train_on_shuttleset.py:190-199` does
  `arr[:,0]/v_width`, `arr[:,1]/v_height`. Shuttle is in pixel
  coords scaled to [0,1] by video resolution. The homography is
  not applied to the shuttle stream. See [A2](#a2-camera-resolution-normalisation-of-shuttle).
- **`human_pose` (joints, J+B)** is in **per-player bbox-relative
  coords scaled by bbox diagonal**. Trace: `normalize_joints` at
  `prepare_train_on_shuttleset.py:150-187` with the active
  `center_align=True` path computes
  `(joint - bbox_center) / bbox_diag`. Joint magnitudes sit
  roughly in [-0.5, 0.5] of "bbox-diagonal units"; the absolute
  player-on-court position is carried by `pos`, not by the joint
  coords. Bones are joint-pair differences computed in this same
  bbox-relative space (`add_bone_at_center` and friends), so they
  inherit whatever transform applies to the joints. Bones aren't
  in a separate measurement space; they're a derivative of the
  joint space. See [A3](#a3-bbox-relative-normalisation-of-joints).

### Implications for centreline flip

| Stream | Flip operation |
| --- | --- |
| `pos` | x → 1 - x (mirror in court frame) |
| `shuttle` | x → 1 - x (mirror in camera frame) |
| `human_pose` joints | x → -x (mirror around each player's own bbox centre) |
| `human_pose` bones | inherit from joints; bilateral pair joint-index swap required |

Each stream flips around its own coord origin, and the cross-modal
relationship the model has learned during training maps cleanly
through the mirror regardless of how well the camera centreline
aligns with the court centreline. The model already learned the
perspective mapping (camera-frame shuttle ↔ court-frame player) per
training distribution; flipping each stream in its own frame just
runs that mapping through its mirror image. Holds for ShuttleSet
straight-on broadcasts and for slightly-tilted amateur footage; only
breaks if a Phase 3 dataset has so much perspective distortion that
the perspective mapping itself is no longer well-approximated as
mirror-symmetric, which is unlikely for any reasonable badminton
broadcast or recording.

### Out-of-court `pos` values flow through unchanged. Important.

Sticky_anchor (`sticky_anchor.py:62`, `generous_margin=0.15`, see
[A5](#a5-sticky_anchor-generous_margin-parameter)) accepts picks
landing in [-0.15, 1.15] on both axes; only when both slots fall
outside that band does the rally-presence check zero the frame. Within the band, the actual normalised values land in
`pos.npy` unclamped — there's no clamp step anywhere in the
collation path. A player on the buffer or chasing a wide shot sits
at e.g. -0.05 or 1.08 in `pos` and the model receives those values
at face value. Two consequences worth flagging explicitly:

- The model has had to learn handling of slight-out-of-court
  positions as a normal part of training, not as an edge case.
- The constrained-jitter formulation has to compute its shift
  bounds against the actual per-clip `pos` values (which may
  exceed [0,1]), not against [0,1] as a hard guarantee. The
  "keep player within own half" constraint becomes "keep player
  within [-eps, 0.5 + eps]" or "[0.5 - eps, 1 + eps]" with
  `eps ≈ 0.15` to match the homography-grid acceptance band.

### Pose-to-pos linkage in the model: PPF at the input, before the TCN

The Pose-Position Fusion module fires *first* in the forward pass
(`bst.py:280-286`, see [A4](#a4-ppf-pose-position-fusion-at-the-input)),
before the TCN, before any transformer. `mlp_positions` projects
the 2D court xy through a 256-hidden MLP to match `in_dim` (the
per-frame, per-player skeleton feature width). The fusion is
multiplicative-with-residual: `JnB` keeps its original signal and
adds a `pos`-modulated component on top, computed as
`JnB = JnB * pos_impact + JnB` which equals `JnB * (1 + pos_impact)`.

So the pose-position relationship is the first thing the model
learns: every other downstream operation (TCN, temporal
transformer, cross-transformer, interactional transformer, CG, AP)
sees pose features that have already been position-modulated. PPF
is on by default for every active BST variant except `BST_0`
(`bst.py:428-432`, see [A4](#a4-ppf-pose-position-fusion-at-the-input)).
CG and AP are *additional* mechanisms layered on top of PPF, not
replacements for it. The Q3 ablations exercised the CG and AP
layering decisions; PPF was load-bearing in every arm except the
reference `BST_0` which never had it.

## Locked Task 2 set (in)

### 1. Centreline flip

Mirror across court centreline, x → 1-x in each stream's own coord
frame, with COCO bilateral joint-index swap. p=0.5. Coupled across
pose+shuttle+court. Court is already collation-baked in normalised
[0,1]² via homography (`prepare_train_on_shuttleset.py:217-227`);
shuttle is in camera-resolution-normalised [0,1]²; joints are
bbox-centre-relative. Each stream flips around its own coord origin
(court-centre for pos, camera-centre for shuttle, bbox-centre for
joints). For ShuttleSet broadcasts the camera and court centrelines
are close enough that the cross-modal relationship the model has
learned during training maps cleanly through the mirror; off-axis
amateur footage works the same way because the model relearns the
perspective mapping per training distribution.

**Bilateral joint-index swap (and equivalent bone-endpoint swap)
on flip.** BST's joint slot identity is encoded in the input
feature-dim ordering: `JnB` enters the TCN as `(b, t, n, input_dim)`
where `input_dim = (J + B) × 2` for J+B style, and the TCN's
channel-wise kernels treat each feature-dim position as a fixed
semantic slot ("channel 18 is slot 9 x", etc.). There's no
per-joint positional embedding; the slot-to-channel mapping is
static across training.

Without an index swap on flip, channel 18 carries the anatomical
left-wrist's coords on unflipped samples and a coordinate-mirrored
right-wrist's coords on flipped samples, with the channel still
encoding "slot 9 = left wrist". The TCN can't learn a consistent
per-channel semantic from a channel whose meaning flips half the
time. Bones are worse: the bone vector "left-shoulder →
left-elbow" (slots 5→7 say) keeps the same channel encoding but
its spatial direction reverses under coord flip — the bone says
"I'm on the left arm" but its xy points to the right arm.

With the swap, slot 9 carries left-side data and slot 10 carries
right-side data both originally and post-flip-with-swap. A
right-handed player's smash (right wrist = racket wrist, slot 10
active) becomes a left-handed player's smash post-flip+swap (slot
9 now carries the racket-wrist data because the swap put the
post-flip racket-wrist there). The slots consistently mean "body
part on the player's right" vs "body part on the player's left",
and the augmentation teaches the model handedness-invariant stroke
patterns. That's the actual generalisation goal and it matters for
amateur transfer where left-handed players are more common than in
the pro broadcast pool.

Index-swap pairs (COCO-17): (5,6), (7,8), (9,10), (11,12), (13,14),
(15,16), eyes (1,2), ears (3,4). Non-bilateral joints (nose 0) just
mirror in xy with no slot swap.

**Bones need both transforms applied: x-component sign flip and
bilateral slot swap.** Two effects, both required:

- *X-component sign flip on every bone.* Bone `b = j_a → j_b` has
  xy components `(j_b.x - j_a.x, j_b.y - j_a.y)`. After joint
  x-flip, both `j_a.x` and `j_b.x` negate, so the bone's x-component
  negates too: `bone.x → -bone.x`, `bone.y` unchanged. Magnitude
  is preserved (|(-x, y)| = |(x, y)|); only direction reverses
  across the y-axis. This applies uniformly to all bones —
  bilateral, cross-body, midline — because it's just the geometric
  consequence of the joint coord flip, regardless of which joints
  the bone connects.
- *Bilateral slot swap.* Bones connecting bilateral joint pairs
  (left-arm bones ↔ right-arm bones, left-leg ↔ right-leg) swap
  their slot positions in the input feature-dim ordering, exactly
  like the joint slot swap. Cross-body bones (e.g. left-shoulder
  → right-shoulder) become their own mirror after the joint swap
  and don't need a separate slot swap. Midline bones (e.g. nose →
  shoulder-mid) likewise need no slot swap.

**Default implementation: recompute bones from the post-flip+
post-swap joints.** Both transforms happen automatically because
bones are a deterministic function of joints (`bone = j_b - j_a`).
Cost is negligible: a fancy-indexed subtraction over
(t=100, n=2, B=19, xy=2) is ~7,600 float ops per clip, single-digit
microseconds on CPU and lost in the rounding error on GPU; the
data loader is disk-bound, not aug-compute-bound. Conceptual safety
is the larger win: one well-defined operation (subtraction over the
existing `bone_pairs` table) replaces a per-bone sign flip plus a
bilateral-only slot swap, eliminating the bone-side error surface
entirely. The augmented bones share their definition with the
unaugmented ones by construction, since both come from the same
`bone_pairs` source of truth. The manual two-transform rules in
the bullet points above are documented for completeness; they're
not the recommended path.

### 2. Corrected constrained pos+shuttle jitter

Replaces the broken `RandomTranslation_batch`. The current aug is
doubly broken:

- It shifts *joints* rather than pos. Joints are bbox-centre-relative
  (`normalize_joints` `center_align=True` path stores
  `(joint - bbox_center) / bbox_diag`); adding a constant shift to
  joint coords does not simulate "player at a different court
  position", because court position is carried by `pos`, not by
  joints. What a joint shift actually does: deforms the body around
  its own bbox centre — telling the model that sometimes people are
  taller than they are, off-centred from their own bbox, or with
  body parts shifted into the floor / above the head. As
  regularisation against pose-keypoint noise, this is redundant:
  MMPose already supplies natural per-keypoint jitter at 3-10 px /
  ~6-19 cm of real positional noise, and the homography fit itself
  adds a small per-clip projection error. Adding ±0.3 of
  bbox-diagonal synthetic noise on top is gross overkill against an
  already noisy signal.
- It doesn't couple shuttle (or pos). Even if the joint shift were
  doing its intended job, the cross-modal alignment would break for
  ~30% of batches.

So the corrected position-invariance jitter operates only on `pos`
and `shuttle`, and leaves joints and bones alone. Bones are
joint-pair differences and are translation-invariant (a bone vector
is the *difference* between two shifted joints = same as the
difference between two unshifted joints), so even a hypothetical
pos-driven body translation wouldn't need a bone transform. Joints
in the bbox-centre-relative frame are court-position-invariant by
construction. Only pos and shuttle carry court-position information,
so only they need to jitter.

#### Implementation outline (layered conditional bounds)

Per-clip uniform shift, not per-frame: one `(dx, dy)` sampled once
per selected clip and applied identically to every frame. Per-frame
independent jitter would sawtooth the shuttle's parabolic arc and
teleport the player between frames; the one-shift-per-clip structure
preserves trajectory continuity across the clip while still moving
the whole event around the court. Same broadcasting structure as the
inherited `RandomTranslation_batch` (one shift per batch element);
only the target streams (pos+shuttle instead of joints) and the
bounds (per-clip dynamic instead of fixed ±0.3) change.

Compute the shift directly off `pos` (the only stream in court
frame). The constraint logic is conditional on per-clip pre-existing
state: the augmentation never *introduces* a centreline-crossing or
band-exceeding artefact, but clips where such states already exist
naturally still flow through unaugmented. So per-axis bounds layer
like this:

```
# eps matches sticky_anchor's generous_margin = 0.15;
# this is the rally-presence acceptance band on either axis.
# Note: sticky_anchor's sanity_ceiling = 0.6 (Euclidean distance from
# anchor) is upstream of us — natural data may have e.g. a top player
# at y ≈ 0.85 because that's still ≤ 0.6 from the top anchor (0.5, 0.25)
# and inside [-eps, 1+eps]. Our jitter constraint is the rally-presence
# band on the post-shift coords, applied only to players who respect
# it pre-shift.
eps = 0.15

# Per-clip y-extremes, both players:
y_top_min, y_top_max = pos[:, 0, 1].min(), pos[:, 0, 1].max()
y_bot_min, y_bot_max = pos[:, 1, 1].min(), pos[:, 1, 1].max()

# Layered y-axis upper bound on dy. Skip a constraint where the
# player already violates it pre-shift (pre-existing crosser or
# pre-existing out-of-band → no new constraint added by us):
dy_max_constraints = []
if y_top_max <= 0.5:                # top player respects centreline pre-shift
    dy_max_constraints.append(0.5 - y_top_max)
if y_bot_max <= 1 + eps:            # bot player respects far-baseline pre-shift
    dy_max_constraints.append(1 + eps - y_bot_max)
dy_max = min(dy_max_constraints) if dy_max_constraints else 0.0

# Symmetric layered y-axis lower bound on dy:
dy_min_constraints = []
if y_top_min >= -eps:               # top player respects far-baseline pre-shift
    dy_min_constraints.append(-eps - y_top_min)
if y_bot_min >= 0.5:                # bot player respects centreline pre-shift
    dy_min_constraints.append(0.5 - y_bot_min)
dy_min = max(dy_min_constraints) if dy_min_constraints else 0.0

# x-axis has no centreline constraint, just the band:
x_min, x_max = pos[:, :, 0].min(), pos[:, :, 0].max()
dx_max = (1 + eps - x_max) if x_max <= 1 + eps else 0.0
dx_min = (-eps - x_min) if x_min >= -eps else 0.0
```

- Sample `(dx, dy)` uniformly in `[dx_min, dx_max] × [dy_min, dy_max]`.
  If either axis has a degenerate range (`min ≥ max`, which happens
  when the layered constraints intersect tightly), fall back to
  no-shift on that axis. No rerolling — the bounds are deterministic
  from the clip's own pos extremes; the sample is always feasible by
  construction.
- Apply `(dx, dy)` to `pos` in court frame.
- Apply the *same* `(dx, dy)` to `shuttle` in camera frame as the
  coarse approximation (court and camera centrelines align closely
  enough on ShuttleSet straight-on broadcasts). For Phase 3 amateur
  off-axis cameras, the more correct path is to project `(dx, dy)`
  from court frame to camera frame via the per-clip inverse
  homography. Code change required; defer until amateur footage is
  in scope.
- **Preserve pre-existing zeroed frames through the shift.**
  Approximately 1% of clips have at least one frame where pose and
  pos were zeroed by sticky_anchor's rally-presence check; shuttle
  has its own per-frame TrackNet-failed zeros. A naive shift would
  turn `(0, 0)` into `(dx, dy)` and lose the "missing" signal.
  Implementation: build per-stream zero-masks pre-shift (for pos:
  frames where all xy entries are zero; for shuttle: same), apply
  the shift, then restore zeros at masked positions:
  ```
  pos_zero_mask = (pos == 0).all(axis=-1)        # (t, n)
  shuttle_zero_mask = (shuttle == 0).all(axis=-1) # (t,)
  pos_shifted = pos + (dx, dy)
  shuttle_shifted = shuttle + (dx, dy)
  pos_shifted[pos_zero_mask] = 0
  shuttle_shifted[shuttle_zero_mask] = 0
  ```
- **Shuttle out-of-bounds post-shift → zero, mirroring the TrackNet
  off-screen convention.** Where the shifted shuttle leaves
  `[0, 1]²` in camera frame, set its xy to `(0, 0)`. Same sentinel
  value the existing collation pipeline writes when TrackNet declares
  the shuttle off-screen, so the model recognises the induced
  "shuttle off-screen" state through the same input pattern it
  already handles for natural off-screen shuttle. No second sentinel
  needed:
  ```
  shuttle_oob = (shuttle_shifted < 0).any(axis=-1) | (shuttle_shifted > 1).any(axis=-1)
  shuttle_shifted[shuttle_oob] = 0
  ```
  *Verified empirically (2026-05-05) at the locked
  cap_y=0.05 / cap_x=0.10 / p_jitter=0.2:* per-frame off-screen rate
  is 0.65% (fewer than 1 in 150 real shuttle frames replaced with
  the sentinel by the shift). When the shift fires on a clip,
  7.21% of those clips have at least one frame go off-screen;
  the remaining 92.8% keep all their shuttle frames in-bounds.
  Run script: `scripts/estimate_shuttle_oob_rate.py`. The rate is
  small enough that the model's existing tolerance for natural
  off-screen shuttle absorbs the aug-induced cases. The
  cross_court_net_shot regression in `run_20260505_111211` (5.7%
  on 5-serial mean, 0.65 on S5 specifically near the wipe_drop
  baseline) is therefore not attributable to the OOB sentinel rule;
  the seed-to-seed variance of cross_court_net_shot in the run
  (0.57, 0.58, 0.59, 0.65 across S1-S5 with S4 missing from the
  cross-checked set) is the dominant explanation.
- Joints and bones stay untouched throughout.

No rerolling cost because the bounds are deterministic from per-clip
pos. The loop "compute layered bounds → sample dx,dy → apply with
mask preservation" is ~constant time per clip. Per-axis fallback to
no-shift when the layered constraints produce a degenerate range is
a feature, not a bug: clips where players are already pinned
shouldn't get jittered.

##### What happens to clips where a player was already out of band

A player who was already past their band before the shift (e.g. the
top player reaching y = 0.6, past the centreline) doesn't contribute
a constraint on that side anymore. The other player's constraint
still applies, so the shift size for the clip is bounded by whatever
the in-band player can absorb, intersected with the cap. Both
players and the shuttle shift together by the same `(dx, dy)`, so
the already-out-of-band player moves with everyone else. The most
they can drift further out by is the cap (currently 0.05 on y, 0.10
on x).

This is by design. sticky_anchor's `generous_margin = 0.15` lets
out-of-band-but-realistic positions through unclamped already, since
they happen in real play (diving returns, players chasing wide
shots). Pulling them back toward the band would actively reshape the
data distribution, which we avoid for the same reason sticky_anchor
uses a permissive margin. The current rule lets the out-of-band
player drift a little further out (no more than the cap), rather
than introducing a constraint that would pull them back in.

##### If we ever raise cap_y

At `cap_y = 0.05` the worst-case extra drift is small enough not to
matter in practice. If a future experiment raises the y cap and that
drift starts looking like a problem (e.g. a top player at y = 0.6
drifting to y = 0.75 starts looking like an off-court position rather
than a wide reach), the y axis should switch to a stricter rule: if
either player is already out of band on y before the shift, set the
y-shift to 0 for that clip and keep only the x-shift. That's a
stricter form of distribution reshaping than the current permissive
rule (clips with extreme-y players lose their y-jitter entirely),
but it's even-handed across both directions of out-of-band rather
than the current "drift outward only" behaviour. The x axis can stay
permissive since `cap_x` is loose by design and x carries less class
information.

Park this as a future tune; it's not a concern at the locked
`cap_y = 0.05`.

##### Cap and per-clip bound: which one limits the shift

Each axis has two limits on how far the shift can go:

- *Per-clip bound*: how far the shift can go before pushing a player
  out of their band. Computed from the clip's own min and max pos
  values. A clip with players near the centre of the court has a
  wide bound; a clip with a player near the centreline has a narrow
  one.
- *Cap*: a fixed maximum shift size (`cap_y`, `cap_x`), the same
  for every clip.

The actual shift is drawn from the intersection of the two:
`dy_lo = max(dy_min, -cap_y)`, `dy_hi = min(dy_max, +cap_y)`, and
the same for x. When the per-clip bound is wider than the cap
(typical centre-court clip), the cap is what limits the shift, so
dy is sampled from `[-cap_y, +cap_y]`. When the per-clip bound is
narrower than the cap (high-movement clip near a constraint), the
per-clip bound limits the shift, so dy is sampled from the smaller
window. This is what makes the jitter magnitude-adaptive: clips
with players near the edges of their band get a smaller injected
shift than centre-of-court clips, which absorb the full cap.

#### Effective augmentation rate vs nominal `p`

The lottery roll is fired before the bounds compute, so a
fully-degenerate clip (both axes have zero envelope) lands in the
augmented pool nominally but receives `(dx, dy) = (0, 0)` and passes
through unmodified. Effective aug rate = nominal `p` × P(clip has at
least one non-degenerate axis). Worth being explicit about three
distinct cases under "constraint-tight":

1. *Fully degenerate.* Both axes zero range. Effectively a skip;
   counts against the effective rate.
2. *One axis degenerate.* Partial shift on the non-degenerate axis.
   Aug fires; smaller than nominal magnitude; doesn't count as a
   skip.
3. *Both axes tight but non-zero.* Small shift in both axes drawn
   from the narrow envelope. Aug fires.

Cases 2 and 3 produce magnitude-adaptive jitter (high-movement clips
with players near constraint edges get smaller injected aug;
low-movement clips with players in the centre of court absorb the
full envelope). That adaptivity is closer to defensible than a
hardcoded uniform magnitude across all clips. Case 1 is the only
one that subtracts from effective `p`.

No rerolling fix exists because rerolling within a zero envelope
still produces zero. Two options:

- **Log + calibrate.** Add a TensorBoard scalar `effective_aug_rate`
  per epoch (count of non-degenerate samples / count of selected
  samples), monitor it on a few baseline runs, then tune the nominal
  `p_roll` to achieve a target effective rate. Cleanest because the
  failure rate is data-dependent and unknown ahead of time.
- **Compensate upfront.** Set nominal `p_roll = 0.35-0.4` to target
  ~0.30 effective on the assumption of a 5-15% case-1 rate. Faster
  to ship; less precise.

*Structural bias to flag*: case-1 clips concentrate on high-movement
scenarios (both players already at constraint edges) or
already-out-of-band clips (layered constraints excluded everything).
Those are systematically different from the augmented pool. Probably
doesn't bite hard in practice because the natural-data signal on
those clips is already information-dense, but worth keeping in mind
when reading per-class results — high-movement classes (smashes,
net-rushes) may be under-augmented relative to centre-of-court
classes (services, drives).

#### Magnitude distribution

Sample by continuous random draw from a uniform distribution within
the per-clip envelope, with a fixed magnitude cap on top:
`dx, dy ~ Uniform(-min(cap, |bound|), +min(cap, |bound|))` per axis.
Continuous in the technical-stats sense: any real value in the
bounded interval is equally likely as a probability density
(`dx = -0.072` is as likely as `dx = +0.034` or `dx = -0.10`). Not
discretised, not stepped at any granularity. Function-call shape:
`dx = numpy.random.uniform(-cap_x, cap_x)`, continuous float output.
Standard skeleton-aug literature uses continuous uniform; no benefit
to discretisation.

Without the cap, a low-movement clip (both players at centre court)
has a wide envelope (~±0.35 in y) and a uniform sample could send a
centre-court player back to the baseline — an unusually large
training perturbation that risks introducing patterns the
in-distribution data doesn't contain. With a cap, low-movement clips
top out at the cap; high-movement clips with tighter envelopes get
the smaller envelope-bounded shift. Uniform-within-bounds is the
skeleton-aug literature norm (PYSKL, AimCLR, Shap-Mix, the inherited
`RandomTranslation_batch` itself); no clear win for
Gaussian-within-bounds.

**Cap default: asymmetric ±0.05 on y, ±0.10 on x.** Rationale:
badminton shot classes encode positional information, with y
carrying more class signal than x. y carries court-zone info
directly (back-baseline clears vs midcourt drives vs front net
shots) and on un-collapsed taxonomies also carries Top/Bottom slot
identity, so the y cap stays tight to preserve class signal. x
carries less class information directly (shot direction is encoded
by trajectory rather than static x position), so the x cap is
looser. Possibly slightly conservative; tune up if the first
ablation shows the model isn't getting enough perturbation signal.

**x-axis side-baseline contact (considered, not enforced).**
Jittering a player to contact-position outside [0, 1] x bounds
simulates "player reached for a ball about to go out". For
non-serves this happens in real play (diving returns, desperate digs
on wide shots) but is unusual. For serves it's nonsense (the contact
must be in the diagonal service box), but serves are a minority of
clips and the constraint detection cost is non-trivial. Trying to
exclude this case requires per-class branching (serves vs non-serves
vs first-contact-of-rally) plus per-frame contact-detection logic,
and the failure mode it would prevent is rare. Diminishing returns;
not enforced. Flag to monitor: post-aug class-specific regressions
on serves or serve-adjacent classes (long_service, short_service).
Also monitor for regressions on long-shuttle-arc classes (clear,
lob, passive_drop, smash) where shuttle physics is most visible and
any cross-modal misalignment from the straight-on-broadcast
approximation would surface first.

Magnitude cap sweep (±0.025/±0.05, ±0.05/±0.10, ±0.10/±0.15 for y/x
respectively, plus a symmetric ±0.10 control) only after the
corrected formulation is locked and shown to be at-or-above the
no-aug baseline.

#### Frequency

Default `p_jitter_roll = 0.2`, paired with `p_flip_roll = 0.5`.
Independent rolls per clip per epoch.

*Mental model: stochastic per-epoch, not expanded dataset.* The
model sees one version of each clip per epoch — whichever version
the per-epoch aug rolls land on. PyTorch standard practice:
`Dataset.__getitem__` is called once per sample per epoch with
whatever fresh aug roll happens that time. The "split table" below
is the *per-epoch population statistic across all clips*, not what
each individual clip sees in a single epoch. Each individual clip
cycles through the variants stochastically across the 80-epoch
training budget, accumulating different aug-parameter draws at each
augmented exposure.

*Why flip and jitter rates are different.* They're different kinds
of augmentation and admit different reasoning:

- **Flip is a binary, parameter-free, label-preserving transform**
  that produces a useful mirror image essentially equivalent to
  "the same shot played by the mirror player". p=0.5 is the
  literature norm specifically because at 0.5 the augmentation
  **effectively doubles the training set**: each clip's mirror is
  a near-orthogonal training signal that the model can absorb
  cleanly without contamination. PYSKL, AimCLR, Shap-Mix, and most
  contrastive-learning skeleton lit all use 0.5 specifically
  because of this dataset-doubling rationale. Lower p means the
  model sees one orientation dominantly and weakly prefers it
  rather than truly being flip-invariant. No tunable magnitude that
  could be too aggressive — only the frequency, and 0.5 is the
  natural rate for "model sees both orientations equally". **Worth
  ablating a lower flip rate (e.g. p=0.25) once the locked
  formulation is in**: the literature norm rationale assumes flip
  provides a near-orthogonal signal that fully justifies
  dataset-doubling, but ShuttleSet specifics (e.g. side-coordinate
  carrying class identity on un-collapsed taxonomies,
  post-homography canonical coords already collapsing some
  symmetry) might make the effective signal less than fully
  orthogonal. If a 0.25 flip ablation matches or beats 0.5 on macro
  / min-F1, the lower rate wins on principle (more clean exposure
  preserved).
- **Jitter has a magnitude knob and a continuous parameter
  distribution**, so each augmented exposure is a different sample
  anyway. With ±0.05y/±0.10x cap and p=0.2, each clip sees ~16
  jittered variants across 80 epochs — workable because the
  random-magnitude variance accumulates across exposures even at
  low frequency. p=0.3 (TemPose-inherited) is also fine. p=0.5 is
  the literature norm but not load-bearing for jitter the way it is
  for flip.

Combined four-way per-epoch population split with `p_flip = 0.5`,
`p_jitter = 0.2`:

| flip | jitter | per-epoch rate across clips |
| --- | --- | --- |
| no | no | 40% |
| yes | no | 40% |
| no | yes | 10% |
| yes | yes | 10% |

Total raw-flip-only exposure per clip across 80 epochs ≈ 80 × 0.5 =
40 flipped exposures (good doubling), 80 × 0.2 = 16 jittered
exposures with different random shifts each time (workable).
Conservative starting point that keeps clean exposure ample and the
model's anchor on the actual data manifold strong; tune jitter up to
0.3 or 0.5 if the train/val gap stays high after the first runs.

Calibrate against case-1 jitter dropout via the `effective_aug_rate`
TB scalar. If observed effective rate sits at e.g. 0.16 instead of
the targeted 0.20, raise nominal `p_jitter_roll` to ~0.24 to hit
0.20 effective.

## Skipped (out)

### Temporal speed jitter — Phase 3 amateur-generalisation candidate

Originally proposed slow-only [0.85, 1.0] coupled with linear interp
on all three streams. Removed from the Task 2 locked set on review.

**Trade-off to be explicit about when proposing for Phase 3.** Even
uniform stretch teaches the model that "high-arc shot shape" can
come with "slow apparent shuttle velocity" — unphysical for a single
pro shot type, where a high-energy strike produces a fast-trajectory
shuttle. The likely model adaptation is to **downweight
shuttle-trajectory velocity contribution** to keep predictions
consistent across stretched and unstretched samples. That's a real
cost on the pro inference distribution because the shuttle stream
does carry signal for several classes (passive vs aggressive drops,
smash vs clear, anything where the shuttle's flight characteristics
distinguish the stroke). The compensating benefit is on amateur
footage: amateurs play whole rallies more slowly (incoming opponent
shot was slower, so incoming shuttle is slower; outgoing self-shot
is slower, so outgoing shuttle is slower). Coupled uniform stretch
is internally consistent at the rally level for amateur play, just
not at the within-shot level for pros.

**Predicted outcome on adoption**: small drop on pro test
performance (the velocity-downweight cost), gain on amateur transfer
(the rally-level pacing match). Net value depends on whether the
project goal weights amateur generalisation above pro test ceiling.
For Task 2 (pro ceiling is the headline), net-negative or
break-even. For Phase 3 (amateur generalisation is the goal), worth
running with a held-out amateur sample as the gate.

Architecture 2 prior research at
`scratch/architecture_notes/architecture_2_research_10_April.md` §9
carries the same trade-off acknowledgement and arrives at uniform
stretch [1.0, 2.0] for Phase 3 by default. If we adopt it,
configuration: scale ∈ [1.0, 2.0] (slow-only, since pros are the
fast end), p=0.3-0.5, coupled across all three streams, linear
interp throughout. No asymmetric within-stroke variant (see
"Non-uniform temporal augmentation" section below: no clean
physics-preserving asymmetric formulation exists).

### Gaussian joint jitter

PDF recommended σ=0.005, pose-only, p=0.5. Skipped. MMPose already
supplies natural per-frame jitter at 3-10 px (per
`03_feature_validation` work, ~6-19 cm of real positional noise at
typical broadcast resolution); the PDF's σ=0.005 sits at the low end
of that distribution, i.e. matches noise the model is already
exposed to. PYSKL's supervised ablation (Duan et al. ACM MM 2022)
found Gaussian noise net-negative on clean skeletons. The PDF itself
flags this as the first to drop if an ablation says it doesn't help;
we're skipping the ablation step and dropping it upfront.

### Random joint masking

PDF recommended 10% of 34 joints, p=0.5, pose-only. Skipped. MMPose
always reports keypoint values for every joint per frame (with
noise), so there is no genuine missing-joint pattern in the data to
teach robustness against. Sticky_anchor reduced natural frame
zeroing to 0.93%, so the "(0, 0) means missing" pattern the PDF was
hedging against is already vanishingly rare. No sentinel-value
question to resolve because we're not masking.

### Sampler

`WeightedRandomSampler` with √-frequency weights also skipped.
CDB-F1 (`adaptive_focal` loss with α = (1 - F1_c)^τ, EMA-smoothed)
already does class-aware reweighting dynamically based on actual
per-class difficulty, which is strictly more informative than static
1/√(count). Stacking a √-frequency sampler on top would double-count
the lever and risk overcorrecting on rare classes that CDB-F1 has
already lifted (e.g. wrist_smash up +8.7 pp on the LS=0.1 baseline
at the floor-lift sweet spot).

## First aug ablation slot: fix or remove `RandomTranslation_batch`

Live aug is joints-only ±0.3 with p=0.3 (verified at
`bst_x_train.py:198-205`, see [A6](#a6-randomtranslation_batch-joints-only-call-site));
shuttle and court are not shifted. That violates Rule 1 of the
PDF §3 (spatial transforms apply to all three streams or none) and
is actively mis-training the cross-attention on ~30% of batches. Three options to A/B against the current baseline:

- **Remove**: disable the transform. Tells you whether the decoupled
  aug was net-positive, net-negative, or noise.
- **Couple at current magnitude**: apply the same per-sample shift
  to pose, shuttle, and court in the train loop. Tests whether
  position-invariance jitter helps when the multi-modal coupling
  actually holds.
- **Couple and tighten**: range to (-0.15, 0.15), p=0.5. The PDF
  notes ±0.3 is large relative to the half-court height of 0.5 (a
  0.3 shift can land a Bottom player on the Top court), and with
  side-coordinate carrying class signal, the smaller magnitude is
  closer to the "small projection-calibration jitter" job.

**Remove arm result (2026-05-04, `run_20260504_152529`)**: the
disable-and-see arm has been run with `RandomTranslation_batch(prob=0.0)`
and otherwise identical hparams to the wipe_drop best
(`run_20260503_172922`). Mean macro -0.8, min -4.4, acc -0.7, top-2 +0.1.
Wrist_smash mean 0.4742 → 0.4301; S4/S5 floor at 0.39 / 0.36; min F1
also fell below the first CDB-F1 baseline (`run_20260501_164658`). The
decoupled body-deforming jitter is empirically regularising despite
being structurally wrong. Defaults restored at `bst_x_train.py:375`.
**Practical takeaway: don't disable, replace.** The corrected
pos+shuttle constrained-jitter (above) is the path; the
"couple-at-magnitude" and "couple-and-tighten" arms are subsumed by
it. Detail: bst_x_overview.md (jitter-off ablation section).

Note on the magnitude question: even with coupled streams, ±0.3 is
aggressive given that 30% prob × the player's own coords means a
Bottom player can shift halfway up the court. Constraining the shift
to keep each player on their own half would require asymmetric
per-slot shift bounds (and stays a question even with shuttle and
court paired). Keeping the streams paired is the most important
correction; magnitude is a second-order tune.

## Seed-noise envelope on run means (round 1 sweep finding)

Aug hparam sweep round 1 (`sweep_20260505_211814_aug_v1_round_1`, 2026-05-06) ran four configs against the prior best (`run_20260505_154907`). The most useful result was a YAML slip: `p_flip_25_x_p_jitter_30`'s `p_jitter: 0.3` override matched the sweep base, so it ran the same aug as `p_flip_25`. Two 5-serial runs, identical config, different seeds.

5-serial mean spread between the two:

| Metric | p_flip_25 | replicate | spread |
| --- | --- | --- | --- |
| macro F1 | 0.7402 | 0.7389 | 0.13% |
| min F1 | 0.4783 | 0.4569 | 2.14% |
| accuracy | 0.7602 | 0.7588 | 0.14% |
| top-2 | 0.9383 | 0.9385 | 0.02% |

That's the 5-serial seed envelope. Macro and accuracy sit tight at ~0.1%, top-2 is essentially deterministic, min F1 swings two whole percentage points off seed alone.

What this changes about reading sweeps:

Macro and accuracy are the reliable signals at 5 serials. A run-mean macro difference of ~0.5%+ is genuinely outside seed noise; smaller moves are inside it. Min F1 at 5 serials is too noisy to drive hparam decisions: the 2% spread across identical-config runs is the same order as the differences we see between runs in a sweep, so big min F1 moves (3%+) are real and smaller ones aren't separable. Practical read: optimise the search on macro, where signal-to-noise is tractable, and read min F1 alongside as the success criterion (it's what we want to lift on the floor class) without using it as the hparam decision signal.

The wrapper's 0.7% macro kill threshold sits right around one run-mean spread, so borderline runs (0.5-1% macro deficit) carry false-kill risk. Clear losers (1%+) get killed correctly: cap_bump's 1.08% deficit triggered exactly as designed at S4. Tighten the threshold only if false-kills become a problem in practice.

Per-class noise on individual classes is much wider than the run-mean spread suggests. Cross_court_net_shot per-serial values across the prior reference's 5 serials: 0.625, 0.567, 0.556, 0.669, 0.613. Spread of 0.11 on one class from seed alone, larger than any cross-run movement we've observed on cross_court. The "cross_court regressed under augmentation" reading is inside its own seed envelope at this sample size, so we can't pin it on any one aug knob.

Companion finding from the same sweep: smash/wrist_smash F1 has flattened, reading as a signal limit on pose-only features. Detail at [`hparams_sweep_speculations.md`](hparams_sweep_speculations.md), "Smash/wrist_smash F1 split" section.

## Application order in `__getitem__`

Per PDF §5.3: flip first (deterministic clean spatial transform),
then temporal speed jitter (deterministic temporal transform).
Reasoning for the PDF's flip → speed → jitter → mask order doesn't
fully apply because we're dropping the noise-and-mask half; but the
spatial-before-temporal order still holds because they don't
interact and flipping a resampled clip is the same as resampling a
flipped one.

## Trimester 2 (amateur generalisation) candidates

Rotation (±15-30° around court centre), scaling (×[0.9, 1.1]),
shearing (±0.1) all flagged as candidates for the amateur-camera
regime, not this in-distribution pass. PDF §4.1 rejects all three
for this regime because court-relative coords are already on a
canonical top-down frame: rotation moves Top players' feet off the
baseline and into the stands, scaling teaches the model that broken
homographies are normal, shearing deforms court geometry. Those
arguments lose force when the camera *is* heterogenous (amateur
phone footage, off-axis broadcast, hand-held), which is Phase 3
territory. **Coupling note for any of these three at Phase 3:**
they'd need to apply identically to pose, shuttle, court, *and* the
bone vectors (which are differences between joint pairs and
transform identically to joints under linear maps, but explicitly
need to be transformed since BST doesn't recompute them at training
time from the aug'd joints).

## MMPose dropout addressed at extraction

Sticky_anchor (Phase 1 landed 2026-04-25, Phase 2 landed
2026-04-29) cut the overall frame-zeroing rate from 5.38% to 0.93%.
The shuttle-on-pose-fail collation wipe was then dropped 2026-05-01
(`run_20260501_164658` baseline carries it; subsequent
shuttle-wipe-drop branch verified recovery of ~14k frames without
regression). Bringing natural frame dropout *down* via these two
interventions has been net positive every time. So adding synthetic
frame dropout (PDF §4.2's "frame dropout" candidate) on top would
directly counter the extraction-side cleanup work; not adopting it.

## X3D-S integration consideration (flag, don't build yet)

The temporal-speed-jitter resample changes the relationship between
clip frame indices and the original video timeline. If the X3D-S
branch consumes a ±19-frame window centred on the reported hit
frame, the augmentation needs to update the hit-frame index under
the resample mapping (or the X3D-S window pulls the wrong frames
relative to contact). Hit-frame metadata is not currently in the
collated `.npy` files; the bridging design needs to come from the
upstream CSVs that identify the hit frame per clip, tied to total
match-video runtime, then reapplied through the
`between_2_hits_with_max_limits` 100-frame windowing rule with
reference to the hits either side. That collation work needs a test
suite covering: clips with hits both sides, clips with a hit only
one side (buffer on the other), clips where the windowing truncates
near the start or end of the match video. Out of scope for the
augmentation runbook itself; flagging here so the X3D-S build picks
it up when fusion lands.

### How hit-frame metadata would get derived (related to X3D-S)

The data flow: each match has a CSV identifying hit frames and
stroke labels at match-video time; clip extraction cuts a window
around each stroke per the `between_2_hits_with_max_limits` rule
(centred near the stroke's hit frame, bounded by the previous and
next hits or by the `seq_len=100` cap, whichever is tighter).
Currently the in-clip hit-frame index is implicit in that windowing
logic but not preserved explicitly in the collated tensors.
**Re-extraction is not required**; two cheaper paths are available
that work over the existing collated artefacts.

*Method A — CSV correlation, deterministic.* Read
`clips_master.csv`, recover each clip's stroke event ID and match
ID from the filename, look up the stroke's hit frame at match-video
time in the source set CSV, look up the previous and next hit frames
for the same match, apply the `between_2_hits_with_max_limits` rule
to recover the clip's start frame at match-video time, then the
in-clip hit-frame index is `(match_hit_frame - clip_start_frame)`.
Write a sidecar `hit_frame_idx.npy` per split (train/val/test),
indexed alongside the existing collated tensors. CPU minutes, no
re-extraction. **Most faithful path to the videos as extracted**:
derives directly from the source-set annotations the clips were cut
against, so it reproduces whatever the ShuttleSet annotators marked
as the hit frame. **Susceptible to annotation noise**: if the
original annotation drifted by 1-2 frames (which is plausible for
fast strokes where the contact frame is hard to identify by eye),
Method A inherits that drift. Test suite covers: clips with hits
both sides (typical), clips with hit only one side (start of rally /
end of rally), edge-of-match-video windowing where the windowing
rule truncates near the start or end of the source video.

*Method B — shuttle trajectory inversion, independent.* Detect
shuttle horizontal-velocity sign reversals in the clip's
`shuttle.npy` stream; each reversal is a hit (sign of horizontal
velocity flips at contact regardless of which player). Reading: 3
reversals → previous opponent hit, this player's hit, next opponent
hit; map to the second (the labelled stroke). 2 reversals → either
(previous + this) or (this + next), depending on which side the
windowing rule put the buffer; resolve via the windowing-epsilon
rule plus the pre-vs-post-contact frame counts. 1 reversal → just
the labelled stroke's hit (clip is short on both sides).

TrackNet noise concern is mild on this dataset. TrackNetV3 reports
recall 99.33% and precision 97.79% on the official benchmarks, and
the existing collation runs trajectory inpaint to fill on-screen
gaps, so per-frame sign-of-derivative detection on the resulting
shuttle stream is not fighting much noise. A light low-pass smooth
before differencing (3-5 frame moving average) is enough to suppress
residual jitter. Soft shots (net shots, slow drops) have small
velocity reversals but the contact frame still falls within ±5
frames of the detection by construction; for X3D-S window centring
at ±19 frames around contact, a ±5-frame detection error eats <30%
of the window on either side and the contact event remains
comfortably inside. So trajectory inversion is a strong anchor for
X3D-S even on the soft-shot tail; the X3D-S branch can absorb that
slack as error-correction rather than as a hard failure mode.

**Recommendation: build Method A first** (cheap, deterministic, uses
existing ground truth), then run Method B as a verification pass on
top. Disagreements between the two methods flag clips where either
the CSV annotation drifted by 1-2 frames or TrackNet's velocity
inversion is unreliable; both are useful diagnostic signal
regardless of which lookup the X3D-S branch ends up using. Method
B's accuracy ceiling is bounded by TrackNet's own contact-frame
resolution (probably ±1-2 frames at typical broadcast frame rates
given the inpaint-smoothed signal), so where it disagrees with
Method A the right reading is usually "Method A inherited annotator
drift; Method B is closer to physical contact", not the other way
around. Test-suite cases for Method B: at least one clip each of
(3-reversal, 2-reversal-prev, 2-reversal-next, 1-reversal) with
hand-checked hit frames as ground truth.

If a temporal aug ever needs the hit-frame index, the
augmentation-side update is simple: when applying the resample
mapping that turns F frames into F' frames before subsampling F
back, push the hit-frame index through the same mapping (and round
to nearest output frame, since the index has to land on an integer
position).

## Non-uniform temporal augmentation (analysis, not adopted)

Phase 3 amateur-generalisation interest in asymmetric pre-vs-post
contact stretch surfaced via the prior research at
`scratch/architecture_notes/architecture_2_research_10_April.md` §9:
biomechanical evidence puts amateur-vs-pro at ~2:1 overall, but the
gap concentrates on the preparation phase; the forward-swing phase
(60-100ms before contact) is only ~1.5x slower for amateurs.
Reasonable to want to model this asymmetry in aug.

**The physics constraint is harder than it looks.** If we apply
asymmetric pose stretch (slow prep, normal swing) and couple the
shuttle to the same time-warp:

- Pre-contact: shuttle moves slowly through the air (unphysical;
  shuttle should travel at near-constant horizontal velocity minus
  drag, regardless of the receiver's prep timing).
- At contact: a slow racket hits a slow shuttle.
- Post-contact: shuttle moves slowly away from contact (unphysical;
  hit shuttle speed depends on racket-head speed at contact, which
  in our asymmetric warp is the *fast* swing phase, so the shuttle
  should be *fast* post-contact, not slow).

So coupled asymmetric stretch breaks shuttle parabolic motion in two
places. Decoupling shuttle (leave it on real timing while warping
pose asymmetrically) preserves shuttle physics but loses cross-modal
frame alignment: pose frame n and shuttle frame n stop corresponding
to the same real-world moment, so the cross-attention sees "racket
is in early-prep but shuttle is already at contact zone", which
doesn't correspond to anything in real footage.

**Literature search status.** I'm not aware of a skeleton-AR
augmentation paper that does phase-aware non-uniform temporal
stretch while preserving cross-modal physics. PYSKL, Shap-Mix, JMDA,
AS-CAL all stick to uniform stretch. SpecAugment-style time warping
exists in audio but is single-modality. The architecture_2 prior
research flagged "non-uniform pre-vs-post-contact stretch" as a
follow-up search dimension (§9.6) but didn't actually prescribe a
mechanism for it; the actual recommendation in §9.2 is uniform
stretch s ∈ [1.0, 2.0]. So the prior research arrived at the same
default we're locking: uniform per-clip stretch, accept the loss of
within-stroke timing realism.

**Uniform stretch IS reasonable for amateur generalisation.** The
internal asymmetry (amateur prep slower than amateur swing) is a
within-stroke pose-pattern feature, not a cross-modal timing fact.
Across a single 100-frame clip, what makes amateur footage "look
amateur" at the rally level is that *both* the incoming shuttle and
the outgoing shuttle are slower (because the previous shot was a
slow amateur shot, and the current shot is a slow amateur shot too).
Uniform stretch with shuttle coupled correctly models this: slow
rally, slow shuttle, slow racket. The amateur internal-pose
asymmetry is genuinely lost, but the cross-modal physics holds and
the rally-level pacing is right. That's a defensible compromise.

**Asymmetric stretch is parked indefinitely.** The Phase 3 default
should be uniform stretch in a wider band ([1.0, 2.0] per the
architecture_2 recommendation), with shuttle coupled and linear
interp on all three streams. Asymmetric stretch only worth
revisiting if a Phase 3 amateur-validation pass shows the residual
gap is specifically on prep-phase pose dynamics rather than
rally-level pacing — and even then, the implementation would need to
choose explicitly between breaking physics (couple shuttle) or
breaking alignment (decouple shuttle).

## Per-joint adaptive focal (Phase 3 / trimester 2 research direction, low priority)

Sketched here so the idea doesn't get lost over the trimester break.
Strict research-direction status; not on the active runbook.

CDB-F1 currently runs **per-class** adaptive α: each of K classes
gets a single scalar weight α_c = (1 - F1_c)^τ, EMA-smoothed across
epochs, applied uniformly to every joint of every sample of that
class. The lever is "rare/hard classes get more loss weight". The
mechanism that produces F1_c is the classifier-head output on the
clip-level label.

A natural extension: **per-joint × per-class adaptive weighting.**
Each (class, joint) pair gets its own weight α_{c,j} =
(1 - importance_{c,j})^τ, and per-frame loss for class c includes
each joint's contribution scaled by α_{c,j}. The intuition is that
which joints discriminate a class is itself class-dependent: wrist
position is critical for distinguishing wrist_smash from smash, hip
rotation might be the distinguisher for clear vs lob, foot position
for service classes, and so on. A per-joint-per-class weight could
let the model focus on the joints that actually carry the class
signal rather than all 17 equally.

The architectural problem: **BST has no per-joint prediction head.**
It's a clip-to-class classifier; the per-frame, per-joint
information has been pooled away by the time the loss is computed.
To get a per-joint signal you'd need either (a) an auxiliary
per-joint reconstruction or per-joint classification task with its
own head, providing a per-joint loss decomposition that the adaptive
weighting can drive off, or (b) a Shapley-style attribution loop
that estimates per-joint contribution to the class output and uses
that in place of "F1 per joint". Both are non-trivial: (a) adds a
head and a multi-task loss balance question; (b) adds a Shapley
estimation cost per epoch.

Parameter count is a real concern: K classes × J joints = K×J
weights. For nosides it's 14 × 17 = 238 weights, manageable. For
amateur-generalisation taxonomies that might split classes further,
it grows linearly. Not a blow-out per se but more than the current
K-vector.

The closest existing work is **Shap-Mix's** Shapley-importance
estimation per joint per class (used to drive a *mixing* policy
rather than a *loss-weight* policy). Re-purposing the same Shapley
machinery as the per-joint-per-class CDB driver is a sensible
implementation direction and would inherit Shap-Mix's "amplify
important joints, downweight unimportant" framing.

Status: low-priority Phase 3 / trimester 2 research direction.
Implementation cost is non-trivial; benefit is speculative; the
existing per-class CDB has already mapped most of the available
loss-side ceiling on nosides per the 2026-05-02 status block. Most
likely revisited only if a future signal-side gain (X3D-S fusion,
wider taxonomy from amateur data) reopens the per-class ceiling and
the limit looks specifically per-joint rather than per-class.

## Appendix A: verified code traces

All snippets and line numbers verified 2026-05-04 against the
working tree at HEAD. Line numbers may drift if the referenced
files are edited; the snippets and structural claims (which
function, which block, which behaviour) are the authoritative
reference. If line numbers are stale at read-time, search for the
reproduced snippet text and update the inline citations.

### A1. Court-coord normalisation of `pos`

File: `src/bst_x/stroke_classification/preparing_data/prepare_train_on_shuttleset.py`

`to_court_coordinate` projects camera-frame foot positions through
the per-clip homography; `normalize_position` then divides by the
court borders to land in court-relative [0, 1]² (with the
acceptance-band-induced overshoot tolerated; see A5).

Court projection inside `check_pos_in_court`, lines 217-228:

```python
feet_court = to_court_coordinate(
    feet_camera, vid=vid, all_court_info=all_court_info, res_df=res_df
)
feet_court = feet_court.reshape(2, n_people, -1)
# feet_court: (2, m, J)

pos_court = feet_court.mean(axis=-1)  # middle point between feet
# pos_court: (2, m)
pos_court_normalized = normalize_position(
    pos_court, court_info=all_court_info[vid]
).T
# pos_court_normalized: (m, 2)
```

`normalize_position` definition, lines 135-147:

```python
def normalize_position(arr: np.ndarray, court_info: dict):
    """
    Normalized by court boundary.
    `arr`: (2, N). Output: (2, N). Every 'x', 'y' in-court should be in [0, 1].
    """
    x_dist = court_info["border_R"] - court_info["border_L"]
    y_dist = court_info["border_D"] - court_info["border_U"]

    x_normalized = (arr[0, :] - court_info["border_L"]) / x_dist
    y_normalized = (arr[1, :] - court_info["border_U"]) / y_dist
    return np.stack((x_normalized, y_normalized))
```

Confirms: `pos.x = 0` is `border_L`, `pos.x = 1` is `border_R`; same
for y on the court borders. There is no clamp on the output.

### A2. Camera-resolution normalisation of `shuttle`

File: `src/bst_x/stroke_classification/preparing_data/prepare_train_on_shuttleset.py`. Lines 190-199:

```python
def normalize_shuttlecock(arr: np.ndarray, v_width, v_height):
    """
    Normalized by the video resolution.
    `arr`: (t, 2). Output: (t, 2). Every 'x', 'y' in-court should be in [0, 1].
    """
    x_normalized = arr[:, 0] / v_width
    y_normalized = arr[:, 1] / v_height
    return np.stack((x_normalized, y_normalized), axis=-1)
```

Confirms: shuttle is normalised by `v_width` and `v_height` (raw
video resolution), not by court borders. The homography is NOT
applied to the shuttle stream.

### A3. Bbox-relative normalisation of joints

File: `src/bst_x/stroke_classification/preparing_data/prepare_train_on_shuttleset.py`. Lines 150-187:

```python
def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    v_height=None,
    center_align=False,
):
    """
    `arr`: (m, J, 2), m=2.
    `bbox`: (m, 4), m=2.
    Output: (m, J, 2), m=2.
    """
    if v_height:
        dist = v_height / 4
    else:  # bbox diagonal dist
        dist = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)

    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / dist, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / dist, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_normalized = (center - bbox[:, :2]) / dist
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)
```

The CLI invocation in `main()` overrides `center_align=True` and
keeps `v_height=None` (per the docstring note: "Signature defaults
preserved verbatim from BST upstream for canonical accuracy. The
CLI invocation in `main()` below overrides `center_align` to
True"). Active code path therefore stores
`(joint - bbox_center) / bbox_diag`. Bones, computed downstream
by `add_bone_at_center`-family helpers, are joint-pair differences
in this same bbox-relative space.

### A4. PPF: Pose-Position Fusion at the input

File: `src/bst_x/stroke_classification/model/bst.py`.

Forward-pass call site, lines 280-286 (inside `BST.forward`,
immediately after `JnB` reshape and before `self.tcn_pose`):

```python
if self.use_ppf:
    pos = self.mlp_positions(pos)
    # pos: (b, t, n, in_dim)
    pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
    # pos_impact: (b*n, in_dim, t)
    JnB = JnB * pos_impact + JnB
    # Multiplicative fusion with residual: JnB * (1 + pos_impact)
```

Confirms: PPF runs before the TCN (`self.tcn_pose(JnB)` follows
immediately at line 291). The fusion is multiplicative-with-residual.

`mlp_positions` is constructed as `MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)`
when `use_ppf=True` (`bst.py:155`), so it projects 2D court xy
through a 256-hidden MLP to match the per-frame skeleton feature
width.

Variant defaults, lines 428-432:

```python
BST_0     = partial(BST, use_ppf=False, use_cg=False, use_ap=False)
BST_PPF   = partial(BST, use_ppf=True,  use_cg=False, use_ap=False)
BST_CG    = partial(BST, use_ppf=True,  use_cg=True,  use_ap=False)
BST_AP    = partial(BST, use_ppf=True,  use_cg=False, use_ap=True)
BST_CG_AP = partial(BST, use_ppf=True,  use_cg=True,  use_ap=True)
```

Confirms: every active variant except the reference `BST_0` has
`use_ppf=True`. CG and AP are layered on top of PPF, not in place
of it.

### A5. Sticky_anchor `generous_margin` parameter

File: `src/bst_x/stroke_classification/preparing_data/heuristics/sticky_anchor.py`.

`StickyAnchorParams` dataclass field at line 62:

```python
generous_margin: float = 0.15
```

Drives the rally-presence check in the heuristic flow (per the
module docstring, step D): "if both slots picked but neither pick
lands inside `[-generous_margin, 1 + generous_margin]` on both
axes, zero both". So the acceptance band is [-0.15, 1.15]; in-band
values pass through unclamped to `pos.npy` at collation time.

For comparison the other coord-related parameter, `sanity_ceiling
= 0.6`, is a Euclidean distance from each slot's anchor (not a
coord band); a top candidate at y ≈ 0.85 is ~0.60 court-units
from the top anchor (0.5, 0.25), so passes the sanity_ceiling
check while also sitting inside [-0.15, 1.15]. Natural cross-centre
clips therefore exist in the dataset, bounded by `sanity_ceiling`
above the centreline rather than by `generous_margin` directly.

### A6. RandomTranslation_batch joints-only call site

Class definition at
`src/bst_x/stroke_classification/preparing_data/shuttleset_dataset.py`,
lines 121-137:

```python
class RandomTranslation_batch(v2.Transform):
    '''Same as RandomTranslation in TemPose.'''
    def __init__(self, trans_range=(-0.3, 0.3), prob=0.3) -> None:
        super().__init__()
        self.trans_range = trans_range
        self.p = prob

    def __call__(self, x: Tensor):
        # x: (n, t, m, J, d)
        n = x.shape[0]
        d = x.shape[-1]
        shift = torch.from_numpy(
            np.random.uniform(*self.trans_range, size=(n, d)).astype(np.float32)
        ).to(x.device)
        if np.random.uniform(0, 1) < self.p:
            x = x + shift.view(n, 1, 1, 1, d)
        return x
```

Confirms: per-sample shift drawn `np.random.uniform(-0.3, 0.3, size=(n, d))`
(uniform within bounds, not Gaussian, not fixed). Whole-batch
single-flip on `p=0.3`.

Call site at
`src/bst_x/stroke_classification/main_on_shuttleset/bst_x_train.py`,
lines 196-205 (inside `train_one_epoch`):

```python
# Apply random translation augmentation to joints only (not bones,
# because bone vectors are relative and translation-invariant)
if n_bones == 0:
    human_pose = random_shift_fn(human_pose)
else:
    joints = human_pose[:, :, :, :-n_bones, :].contiguous()
    bones = human_pose[:, :, :, -n_bones:, :]

    joints = random_shift_fn(joints)
    human_pose = torch.cat([joints, bones], dim=-2)
```

Confirms: only the joint slice of `human_pose` is passed to
`random_shift_fn`. Bones are excluded (the existing code's
self-justification: "bone vectors are relative and
translation-invariant"; correct as far as it goes, but the
joint shift on bbox-centre-relative coords doesn't simulate
position-invariance — see body of doc for the analysis). Shuttle
and `pos` (`pos: Tensor` at line 192) are not in this code path
at all and pass straight through to the model unmodified.
