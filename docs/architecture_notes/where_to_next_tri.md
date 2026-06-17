# Where to Next: Trimester 2 Model Plan

Drafted 2026-06-04.


## Where we are

BST-X currently achieves 0.84/0.83 macro F1 on the original BST taxonomy (without/with 'unknown' class), and ~0.75 macro on the 14-class no-sides taxonomy, with the `wrist_smash` finally clearing 0.5 min-F1. The smash-wrist smash confusion is a "not enough discrimination in the signal" problem: 2D keypoints don't carry the information needed to distinguish a wrist-flick from a full-arm swing, and the 3D keypoints tested in the original BST paper did no better. We tested loss, capacity, and augmentation in T1 to compensate.

BRIC currently achieves 0.73 macro/0.41 min F1 on the 14-class taxonomy. This is impressive for a non-specialised model. However, at 10x the parameter count and significant work to optimise it to match BST-X, next trimester's work should be built on BST-X. That is, unless one of the joining team-mates wants to take on the optimisation challenge.

The pipeline is fully automated end-to-end (video, pose, collation, training), we can swap taxonomies by name (no hardcoded class lists), and the frontend (FE) and backend (BE) talk through the JSON files the model writes.

Trimester 2 (T2) team is **Ariel + Curtis**. Two more team-mates expected for COSC595, though their strength areas aren't known. Curtis would like to lean more ML in T2 but will keep an important FE/integration role, especially in the second half when new ML features need connecting into the user interface (UI).

Two new ML-leaning team-mates would let us do real new model work. Two new FE-leaning members would find a comprehensive UI that's already mostly there: live inference on user-uploaded videos is the main missing feature, and beyond it the FE work is limited by whatever the ML side can deliver. Scope below considers MVP based on Ariel + Curtis' continuing roles, and treats the rest of the handover as flexible.


## Trimester 2 objective

### Major pieces

1. **X3D-S wrist-crop fusion.** BST-X's first new input channel. Stage 2.
2. **Classifier that handles amateur footage.** Self-supervised learning (SSL) pretrain on amateur skeletons, fine-tune over the expanded pro training set. Stage 3.
3. **CrossTrainer-style stroke-quality proof of concept (PoC).** Ego-Exo4D-pretrained multimodal language model (LM) head, badminton vocabulary fine-tune, zero-shot to per-stroke clips. Stage 4.

Rally-level work is shelved unless a second ML team-member joins.

### Deliverables

- rtmlib pose stack migration (no MMPose dependency tree)
- X3D-S wrist-crop fusion for BST-X
- Expanded pro training set: ShuttleSet + ShuttleSet22 + VideoBadminton (~74k strokes)
- Amateur video collection from YouTube
- Self-supervised BST-X pretrain on amateur skeletons
- Classification fine-tune over the expanded training set, with the amateur-pretrained encoder
- CrossTrainer-style stroke-quality PoC: skill-attributes + actionable feedback + proficiency, zero-shot to badminton
- Live inference on user-uploaded videos (FE/integration)
- (from the Hunter Badminton Association & Arthur) Annotated clips of amateur badminton footage, ideally a dozen samples of all our taxonomy classes. Even if this is too small to train on, it can be our (dual-purpose) val and test set for generalisation. We can do homography and timestamps, but we can't accurately identify the strokes by name.
- Final report + handover docs for next year's team
- Stretch/alternative: badminton expert-commentary dataset pipeline (parked)


## Timeline

| Weeks | Ariel | Curtis | New team-mates |
|-------|----|--------|----------------|
| 1 | rtmlib + parity test | Data expansion: SS22 + VideoBadminton crossover; dataset scouting | Onboarding |
| 1-3 | X3D-S stages 1-6 (config sweep on ShuttleSet) | Amateur video scraper | See Headcount-conditional scope |
| 3-4 | Final fused X3D-S + BST-X training run on the full pro data | Scraper wrap-up; Stage 4 setup (encoder bridge + training scripts) | See Headcount-conditional scope |
| 4-6 | SSL pretrain BST-X on amateur; classification fine-tune | Live inference on user-uploaded videos starts | See Headcount-conditional scope |
| 5-10 | Stage 4 quality stack | Live inference + connecting new ML outputs to the UI | See Headcount-conditional scope |
| 10-12 | Loose ends, model docs, final writeup | Live inference wrap; demo prep | Demo polish + accessibility + user docs + deployment runbooks (FE-skilled); rally model wrap or Stage 4 contributions wrap (ML-skilled) |

Stage 4 overlaps Stage 3 from week 5 onward; we can write the encoder-bridge code (the glue between BST-X output and the CrossTrainer mapper) while SSL runs.


## Headcount-conditional scope

After ~1 week of onboarding (codebase + pipeline + project state), new members can take one of two paths.

### Path A: Own a feature (probably more sensible)

Clear scope and accountability without merge-conflict friction on the main critical path.

- **ML-skilled**: own the rally model end-to-end. RallyTemPose as the entry point, BST-X stroke embeddings into its decoder, shot-selection quality coming straight from what stroke the model expected next. Activates once Stage 2 BST-X embeddings are stable (around week 4-5). The first 1-2 weeks after arrival go to reading the paper and getting familiar with the BST-X codebase.
- **FE-skilled**: own live inference on user-uploaded videos as a complete feature. The main outstanding UI piece; clean hand-over from Curtis if a strong FE-skilled member arrives.

### Path B: Speed up the MVP

Pair work on the existing critical path. Faster on those threads but more time spent keeping up with each others' changes.

- **ML-skilled**: pair on Stage 4 (CrossTrainer-style stack). Split Step A (Ego-Exo4D pretrain pipeline) and Step B (badminton vocab fine-tune), or one runs Stage 4 while the other handles parallel X3D-S side experiments (fusion points, capacity Run 2, augmentation v2).
- **FE-skilled**: pair with Curtis on UI as new ML outputs ship — X3D-S classifier, CrossTrainer-style feedback panel, polish, accessibility, demo prep, user-facing docs, deployment runbooks.

Most likely outcome: one ML-leaning + one FE-leaning. Each picks their own path.


## Risks

**X3D-S fusion runs to 4-5 weeks.** Medium probability. Sweep surprises on fusion point or co-training schedule. Compresses SSL pretrain and pushes Stage 4 later.

**CrossTrainer reimplementation eats into the wrap-up window.** Medium probability. Stage 4's 6-week window fits the high end of the 5-7 week estimate tight, and the LLM-extracted skill-attribute supervision has details the paper glosses over. Mitigation: email the authors week 1 for weights; have the encoder-bridge code ready by week 5 so Step A starts the moment SSL pretrain delivers a usable encoder.

**Amateur scraper produces low-quality skeletons.** Medium probability. YouTube amateur footage varies wildly. SSL tolerates noise better than supervised training does, but if skeleton quality is genuinely poor, the transfer benefit goes away. Mitigation: time-box at 2 weeks; the Hunter clips give a measurable transfer-benefit check by week 6 — fall back to supervised + pseudo-labelling if it hasn't moved.

**Curtis pulled to FE earlier than planned.** Medium probability. Worst case the scraper falls to Ariel; ~1-week delay in the SSL window.


## Stage 1: Foundation (week 1, parallel)

### rtmlib migration (~3 days, Ariel, first)

Swap the slow, brittly-pinned MMPose / RTMPose-L step (which requires its own venv) for a more efficient implementation. rtmlib is a more modern keypoint library wrapper that's still maintained. It includes RTMPose-L, so the minimal fix is a straight swap. Run some quick tests on an extract to make sure the keypoints only drift within reasonable GPU noise. If that's okay, run a couple of full 5-serial trains on the `bst_25` and `une_v1_14` taxonomies for a full sanity check.

### Data expansion (~1 week, Curtis, parallel)

In priority order:

**ShuttleSet22** (`wywyWang/CoachAI-Projects/.../ShuttleSet22`). 33,612 strokes / 3,992 rallies / 35 players. Same 18-class taxonomy and metadata schema as ShuttleSet; video bundled. Integration is mostly adding a couple of dataclasses, then plug-and-play re-extract, retrain, and measure performance. Nearly doubles the training data for free.

**VideoBadminton** (Li et al. 2024, arXiv 2403.12385). 7,822 clips, 60fps, 19 players (15M/4F), self-recorded National Central University (NCU) practice footage. 18 Badminton World Federation (BWF) classes on a **different taxonomy** than ShuttleSet: some map cleanly (Smash, Clear, Drive, Lift→Lob, Long/Short Serve), others split or merge (Tap Smash vs Smash; Drop Shot vs ShuttleSet's drop/passive_drop). Curtis writes the mapping table before collation.

**BadmintonDB** (kwban/badminton-db): **skip**. Only 2 players.

**Worth a look**:
- **FineBadminton** (arXiv 2508.07554, Aug 2025) — multi-level fine-grained badminton; taxonomy and release status not yet verified.
- **Badminton Olympic Dataset** — 10 YouTube videos, 12 stroke categories. Small, pro-only.

Integration depends on video availability, as well as how well the classes map to ours.


## Stage 2: X3D-S wrist-crop fusion (weeks 1-3, Ariel)

BST-X plateaus at smash vs wrist_smash because pose-2D doesn't see the racket-pixel motion that separates them. X3D-S adds a small video stream on the player's wrist region — racket angle, wrist snap, shuttle contact. Fused with BST-X, this should split the confusion.

Architecture is set: X3D-S Kinetics-400 (K400)-pretrained, 39 frames × stride 1. Still open: where to fuse the two streams, and how best to interpolate the ~0.6% of frames where MMPose gets dropped mid-clip.

How to schedule the training: probably best to fine-tune pure X3D-S on wrist RGB alone first. Otherwise the combined model will probably just auto-anneal out the 3D Convolutional Neural Network (3D-CNN) signal, since the core BST-CG-AP branch almost certainly converges faster independently and already does so to a high F1. Worth testing separate multi-stage vs end-to-end integrated train approaches if time permits — maybe the latter gives more useful specialisation of the RGB wrist-cam module.

Existing 6-stage plan at `docs/architecture_notes/x3d_integration_macro_plan/` (stages 1 and 2 already detailed):

1. Hit-frame derivation (Method A CSV correlation, Method B' shuttle direction reversal cross-referenced with wrist-velocity peaks per Liu et al. 2023)
2. Wrist-loss assessment (per-class ±19-frame loss rates, dominant wrist)
3. Crop sizing + dominant-wrist heuristic
4. Wrist-crop extraction pipeline (raw uint8 stacked .npy on /scratch; rsync engelbart→bourbaki)
5. Solo X3D-S baseline (14-class on wrist crops alone; the floor that fusion has to beat to be worthwhile)
6. Fused training (injection point + co-train schedule sweep)

Run the X3D-S config sweep on the original ShuttleSet for fast iteration. Then one proper full training of the fused X3D-S + BST-X model on the expanded pro data (SS + SS22 + VideoBadminton), from random init, using the winning config from the sweep. Capacity Run 2 (d_model 100→192, d_head 128→32) parked at T1 close; could run as a low-cost side experiment alongside the sweep if the encoder looks like it needs more capacity.

Optimistic 1 week, realistic 2-3 weeks.


## Stage 3: Amateur generalisation (weeks 2-6)

Amateur strokes look different from pro: different movement patterns, different camera angles than broadcast. A classifier trained only on broadcast pro footage won't transfer. Two parts to this: build a data scraper for amateur data, and then do SSL to learn its patterns. SSL can't happen until the scraper is built.

### Amateur video scraper (~1-2 weeks, Curtis)

YouTube amateur badminton, cropped to individual players, skeletons extracted through the rtmlib pipeline. Goal: skeletons usable from roughly-cropped player footage. Player detection via modified sticky_anchor; rough temporal segmentation via ffmpeg; no labels. Shuttle via TrackNetV3 if it holds up on amateur footage (broadcast-trained; needs a quick test on amateur footage and manual verification). Court homography skipped — amateur camera angles vary too much to reliably extract court corners. More people = more footage; source-finding and quality-filtering both split easily.

### Self-supervised BST-X pretrain (~1-2 weeks, Ariel, follows scraper)

Masked-reconstruct within the BST-X architecture:

1. Swap the classification head for a reconstruction head matching input channel width.
2. Mask random joints, frames, and channels in the input (multi-stream: pose, court-position, shuttle).
3. Loss = Mean Squared Error (MSE) against the unmasked sequence.
4. Pretrain on amateur footage (rtmlib skeletons + TrackNetV3 shuttle from the scraper; position channel zeroed since court homography isn't reliable on amateur).
5. Swap the head back. Fine-tune on the expanded pro training set for classification. The annotated Hunter clips (Deliverables) become the held-out amateur eval — the measure of whether SSL transfer worked.

BST-X SSL is the easier option. Heavier alternative: SkeletonMAE + teacher-student, if the lightweight version is disappointing on amateur videos.

Worth trying frame re-ordering too. Quick to build (literally just `np.random.shuffle()`). Either run it after masking as a separate pass, or jointly via a second prediction head (more plumbing but no extra train time). The two train different things: masking trains content, re-ordering trains sequence.

### Encoder double-duty into Stage 4

The amateur-pretrained BST-X encoder does two jobs: classification fine-tune above, and the input encoder for the Stage 4 movement quality assessments (frozen at that stage). The same BST-X reads skeletons extracted from Ego-Exo4D video through rtmlib, replacing CrossTrainer's frozen EgoVLPv2/CLIP video encoder.

CrossTrainer report "minimal benefit" from pose. But their main concern is related to the increased compute of extraction, and our keypoints are already there. Also, they stride just one input per second, whereas we map a median 30fps.


## Stage 4: Quality assessment PoC (weeks 5-12, Ariel)

Per stroke clip: three outputs — skill-attributes (open-vocabulary list of what to improve), actionable feedback (free-form text), proficiency level (4-class). Trained on Ego-Exo4D + Qualcomm Exercise Videos Dataset (QEVD) expert commentary; applied zero-shot to badminton. The model follows CrossTrainer (Ashutosh & Grauman, NeurIPS 2025, arXiv 2511.13993) with one substitution — see below.

### Architecture

Frozen encoder → 2-layer Multi-Layer Perceptron (MLP) mapper → Llama-3.1-8B-Instruct tuned with LoRA (rank 128, alpha 256) → three output heads. Substitution: where CrossTrainer uses frozen EgoVLPv2/CLIP video features, we feed BST-X embeddings (Stage 3, frozen) over rtmlib skeletons extracted from Ego-Exo4D video. Everything else follows the paper.

### Supervision

CrossTrainer's training signals come from expert commentary, not manual labels:

1. **Skill-attributes Ŝ** — phrases extracted from expert commentary by LLM prompt ("balance", "hand positioning", "follow-through"). Free-text, not from a fixed class list. The expert text is the real human signal; the LLM is a structured extractor. CrossTrainer used ~34k commentary strings.
2. **Actionable feedback T** — the original expert commentary text, paired with the clip around its timestamp.
3. **Proficiency P** — 4-class label from Ego-Exo4D (novice / intermediate / early expert / late expert).

### Two training steps

**Step A: Ego-Exo4D pretrain.** Build the CrossTrainer training pipeline on Ego-Exo4D (publicly available): download the videos, extract poses through rtmlib, LLM-extract skill-attributes from the expert commentary, then LoRA-train the mapper and LM heads against those labels. Focus on the Ego-Exo4D domains useful for skill transfer to badminton — soccer, basketball, rock climbing, dance (structured limb motion + balance under load). Plus QEVD if compute allows. CrossTrainer's own train was 1-3 hours on a single GH200; plan for 2-3× that with the encoder substitution. Output: trained mapper + LoRA weights + proficiency head.

**Step B: Badminton vocabulary fine-tune.** LoRA-fine-tune only the LM (the mapper, encoder, and proficiency head stay frozen) on unaligned badminton coaching text: YouTube coaching channels, BWF technique articles, transcribed coaching books. No stroke alignment, no per-clip labels. A few thousand paragraphs is enough; LoRA on Llama-3.1-8B trains in hours. The fine-tune teaches the LLM badminton vocabulary so the generated feedback is sport-specific.

Freeze the proficiency head during Step B. Every badminton text source is effectively "late expert" content (pros teaching pros); training the proficiency head on that would collapse it to predicting "late expert" for everything.

### Inference

Per stroke: extract 39-frame window → rtmlib → BST-X encoder → mapper → LLM with prompt *"Here is a video of a person doing badminton. Highlight up to k key concept areas where the person can improve: ..."* Generate skill-attributes, then actionable feedback, then proficiency. **Zero-shot to badminton** in the CrossTrainer sense: no badminton clips train the multimodal LLM; only the LM head sees badminton text.

Hunter clips give a real-world generalisation measurement: what does the model actually say about an amateur smash?

### Base-model contingency

CrossTrainer's GitHub repo (`thechargedneutron/CrossTrainer`) is empty as of 2026-06-04. Email the authors week 1. If they release weights mid-T2, use them directly instead of Step A and save 2-3 weeks. Otherwise reimplement: build the MLP mapper, set up LoRA-tuning on Llama-3.1-8B, write the LLM-extraction step that turns expert commentary into skill-attribute labels. The architecture and hparams are in the paper. TechCoach (arXiv 2411.17130) is the only other Ego-Exo4D-derived coaching-feedback model worth checking; code status unconfirmed.

Realistic scope: 5-7 weeks for Stage 4 if everything goes roughly in order. The available window is 6 weeks (weeks 5-10 before wrap-up), so this fits tight if we reimplement from scratch — comfortable if we find a pre-built base to fork.


## Parked for later

- **Rally-level model.** RallyTemPose (Ibh et al. CVPRW 2024, "A stroke of genius") as the entry point. Skeleton + LM-text-embedding decoder; code released, no weights. ShuttleSet 10-class acc 54.3% / acc-3 92.5%. Picked up if a second ML joiner is available.
- **Badminton expert-commentary dataset** from YouTube broadcasts. Novel contribution (no racket-sport Action Quality Assessment (AQA) dataset exists in Awesome-AQA). Pipeline: BWF + tournament + coaching channels → Automatic Speech Recognition (ASR) → BST-X stroke-timestamp alignment → LLM-as-aligner for commentary chunks → technique-only filtering → clip extraction. ~3 weeks. Train only F_a (skill-attributes) + F_t (feedback); freeze F_p on the Ego-Exo4D distribution since the badminton data is all "late expert" content.
- **Full CrossTrainer / TechCoach reimplementation** trained on the badminton commentary above. The heavy version of Stage 4.
- **Calibrated quality scoring.** Contrastive pro vs amateur same-stroke, or quality labels on a small amateur set with active learning.
- **Production-efficient inference.** Llama-3.1-8B at inference isn't real-time; distillation or quantisation when latency matters.
- **Match-level analytics, multi-camera robustness, full autograder integration.** T3/T4.


## Bottom line

If all three major deliverables are complete by week 10, T2 puts the core model in place. The full autograder we want to build sits on top of it in later trimesters.


## Anchor state at T1 close

- BST-X best mean macro 0.742 / mean min-F1 0.471 on `une_v1_14` (run #37); per-taxonomy bests in `bst_x_overview.md`
- Optimiser: AdamW wd 4e-1 with decay exclusion on norms/bias/embeddings
- Loss: CDB-F1 adaptive focal (tau 1, gamma 1, momentum 0.9, warm-up 5 epochs)
- Augmentation: centreline flip (p 0.5) + constrained jitter (p 0.3)
- Collation: `taxon_pinned_w_preds` with shuttle-unzeroing
- Detection: sticky_anchor (Phase-2 full-clean MMPose extract, 32,203 clips)
- X3D-S baseline: K400-pretrained via `torch.hub`, 39 frames × stride 1


## References

- **CrossTrainer** (Ashutosh & Grauman, NeurIPS 2025, arXiv 2511.13993). PDF: `COSC594/learning_skill_attributes_for_transferrable_assessments_in_video.pdf`. Empty repo at `github.com/thechargedneutron/CrossTrainer`.
- **RallyTemPose** (Ibh et al. CVPRW 2024). PDF in `COSC594/`. Code: `github.com/MagnusPetersenIbh/RallyTempose`.
- **TemPose** (Ibh et al. CVPRW 2023). PDF in `COSC594/research/`. Same lead author.
- **VideoBadminton** (Li et al. 2024, arXiv 2403.12385). PDF in `COSC594/research/`.
- **TechCoach** (arXiv 2411.17130). Ego-Exo4D-derived coaching feedback; code status unconfirmed.
- **FineBadminton** (arXiv 2508.07554). Multi-level fine-grained; release status unconfirmed.
- **rtmlib** v0.0.15+ (Tau-J/rtmlib).
