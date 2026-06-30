# BST-X: issues and bugs squashed

This document covers major bugs and bug-like issues. It doesn't cover most of the theory issues and explorations that shaped the model build.

## Bug-shaped issues

### sticky_anchor: frame zeroing at the moment of contact (16-22 Apr)

I was trying to figure out what was causing the high frame failure rate (~25% smash exclusion on the train split, per the 16 April hit-zone heatmap; `d2477b2`), so I did up a script to overlay bounding box detections and keypoints on raw clips based on the mmpose extracts (`71d7f6f`). I watched the videos on loop for a while until I realised that player keypoint markup seemed to primarily disappear when the top player jumped. This led me to discover the heuristic used for zeroing 'faulty' frames was overly strict. Any frame where either player fell outside the marked white lines of the badminton court (plus a tiny offset) was automatically discarded. I built sticky_anchor to fix this (`914e27b`), recovering ~97% of lost frames.

### Label noise to CDB-F1: regularising then reweighting the loss (30 Apr - 2 May)

Scott and I got talking about contact-frame annotation errors he'd been finding while manually reviewing clips. It occurred to me that if the dataset was noisily annotated, as most are, then regularising the label targets was worth a try. I bumped label smoothing above BST's 0.1 and saw an immediate jump (0.15 won the sweep; `4fdb53a`). That set me thinking about whether the noise hit some classes harder than others, and whether more could be recovered by dynamically reweighting the loss by class. That led to the CDB-F1 focal loss build (`a532967`): a per-class weight driven by a running average of train F1, so the classes that stay weak keep drawing more of the loss. It lifted wrist_smash about 8.7pp and recovered the min-F1 floor, though macro held at 0.74-0.75 (`e30db10`).

### The model registry: an interface layer between ML and FE (14 May)

ML had weights to serve and FE needed to know how to serve them without getting tangled in the moving feast of ML's builds. I sat down and worked out the most decoupled arrangement: ML serves a tight bundle of the core data FE had been asking for, and FE never has to track ML's build churn. That became the registry module, an interface layer where each side meets its obligations (`ebed676`). I built it as a proof of concept, a working skeleton, and handed it to the FE team to build off and improve if it suited them. It has since become the seam both BST-X and BRIC serve through.

### Manifest overwrite on resume (1,30 May)

Resuming a run to re-test or add serials made `track_serial` rewrite the whole manifest, so resuming an existing run could overwrite its original record. I first guarded it with a timestamped `.bak` snapshot taken before any resume work (`63e6953`, 1 May), then fixed it at the root by removing the path that resumed into an existing run: there's no longer any reason, or any code, to write back to a finished run (`251f6ac`, 30 May).

## Bugs squashed

Bugs I dealt with while building BST-X that caused more than a momentary headache.

### FP16 inference dropped fast-shuttle detections (12 Apr)

**Bug.** I'd put the TrackNetV3 shuttle pass under float16 autocast to batch it faster. Fast shuttles (>400 km/h) only leave a faint heatmap response, and FP16 rounding could tip it under the 0.5 visibility threshold and drop the detection. The same batch run also never released its `cv2.VideoCapture` handles, so it ran out of file descriptors and hung after ~374 clips.

**Why it mattered.** It lost the shuttle on smashes, and the leak stalled long extractions.

**Fix.** Back to FP32 for the forwards; `cap.release()` at each site.

**More.** Commit `9987477`.

### MMPose/TrackNet frame-count mismatch crashed collation (14 Apr)

**Bug.** MMPose and TrackNetV3 decode through different video backends, and the two can disagree on a clip's length by a frame or two. `shuttle_result[failed_ls, :] = 0` then threw IndexError when the fail mask and the shuttle array came out different lengths, and `np.stack()` in `make_seq_len_same()` fell over on the mismatched dims.

**Why it mattered.** Collation crashed on any clip where the two disagreed.

**Fix.** Slice every per-clip array to the shorter length before masking or stacking.

This same bug-fix session also split shuttle-CSV reading out of the pose step (decoupling so the artefacts could be separately built/rebuilt).

**More.** Commit `160d8e9`; `preparing_data/mmpose_changes.md`.

### Zero-length clips gave NaN loss (17 Apr)

**Bug.** A clip where MMPose never found two players on any frame collates to `videos_len = 0`, all padding. The transformer masks every position, so the attention softmax runs over all `-inf` and comes back NaN; one such clip takes the loss NaN from epoch 1.

Chang's original BST work must have avoided this because he hand-curated his clips and likely did a manual .isna() pass.

**Why it mattered.** One bad clip kills the whole run. 

**Fix.** Drop the `videos_len == 0` clips at dataset load.

**More.** Commit `ae90c01`. Problem clips depends were initially ~65, though I believe there are now no zeroed frame clips left.

### Overlay renderer hid valid picks (25 Apr)

**Bug.** My sticky_anchor sanity check video overlay greyed out every box whenever the frame was marked failed (gated on the whole-frame `failed[f]`, not per slot). On a partial success, one slot picked and the other empty, it drew the good pick as unpicked too, so the frame looked like a double zero. This didn't actually impact training, but it was a huge problem for my sanity-check pipeline.

**Why it mattered.** It sent me chasing the wrong thing: I'd pinned ~21% of clip `19_2_10_7`'s zeroed frames on a net-crossover case in the detector. Replaying the clip frame by frame showed the detector was fine, and the real cause was an upstream MMPose gap (top player ~85% occluded at the net; unfixable MMPose problem). Net-crossover turned out never to actually happen in the data.

**Fix.** Test each slot's pick on its own (`pos[f, slot].any()`).

**More.** `mmpose_heuristic/historical_mmpose_heuristic_investigation.md`.

### Mystery extra channel on drop-unknown runs (1 May)

**Bug.** The head was sized to `taxonomy.n_classes`, and every taxonomy lists an `unknown` class. On a drop-unknown run there are no `unknown` rows, so that channel existed and still took a slice of every loss, but without any samples that needed to go to it.

**Why it mattered.** The channel never won argmax, so maybe it had no *real* effect. But logically it had to have some effect on the loss distribution, however small. I only discovered it while reading a manifest.yaml and puzzling out why there was an 'unknown' class weighting for a taxonomy that had no 'unknown'-class.

**Fix.** Size the head off the classes actually in the train labels (`derive_active_classes_from_labels`, val/test asserted as subsets). This was originally a brittle band-aid patch, but got hardcoded into the collation rules on 30 May. Now each collation definitively pins its own class list.

**More.** `unknown_channel_fix_review.md`; commit `63e6953`. Pre-fix drop-unknown weights for une/raw_35 don't load post-fix (head shape changed).

### Shuttle track wiped whenever pose failed (3 May)

**Bug.** Collation zeroed the shuttle on any frame where pose extraction failed (`shuttle[failed, :] = 0`), even when TrackNet had the shuttle sitting right there. But pose-fail and shuttle-fail are independent streams, so this was just throwing good data away: ~14k frames (0.84% of the extract).

**Why it mattered.** I had two streams of information and when I lost one I was unnecessarily throwing the other away. Recovering it resulted in a substantial performance improvement (mean min-F1 +1.2, macro +0.5 on the taxonomy tested at that commit).

**Fix.** Delete the shuttle wipe heuristic so that shuttle can flow even if keypoints are zeroed.

**More.** Commit `4e478fc`; lost-frame-recovery entry in `bst_x_overview_technical_appendix.md`.

### Jitter read padding zeros as real positions (5 May)

**Bug.** The jitter works out how far it can safely shift a clip from the min/max of player positions across it. Every clip is shorter than 100 frames, so the loader pads the rest with zeros, and the jitter was reading those padded `(0, 0)` rows as if a player had stood in the corner, which threw off the min/max.

**Why it mattered.** For most clips it made the bottom player's lower-bound check drop out entirely, when it should have been a real constraint.

**Fix.** Drop the padded frames before taking min/max.

**More.** Commit `2291ad8`.

### Eval script looked one directory too high (11 May)

**Bug.** Exactly as it sounds. Not exciting, but quite annoying.

**Fix.** `parents[3]` to `parents[2]`. Later folded into `bst_x_infer --fe`.

**More.** Commit `5693ce4`.

### driven_flight merged into unknown instead of drive (23 May)

**Bug.** The 25-class merge map was folding `driven_flight` into `unknown`, where the BST paper (Table G) folds it into `drive`. The 35-class convention had bled into the 25-class collation.

**Why it mattered.** ~52 (yes, only 52) of ~33k clips mislabelled in every historical merged_25 run. Tiny, but it means old merged_25 weights carry a wrong per-class reading (literally only off-by-one--there's only 1 test driven-flight video).

**Fix.** `driven_flight` to `drive`, matching the paper.

**More.** Commit `1d98949`.

### Registry class list empty for BRIC models (30 May)

**Bug.** The registry read each model's class list only out of BST's `extra.arch.active_class_list` block. BRIC manifests don't carry that block, so a BRIC entry came back with an empty class list and every label lookup returned None. Discovered almost immediately after commit because Scott and I happened to be making changes to the registry in short succession.

**Why it mattered.** A BRIC model would have served with no labels and nothing would have errored. It stayed hidden because the BST models had the block and worked fine.

**Fix.** Read `config.classes` first (it covers both families), fall back to the legacy block.

**More.** Commit `1b69393`.

### Prediction rows misaligned to clip IDs (30 May)

**Bug.** The prediction npz wrote its rows straight from the in-memory dataset, which has already dropped the zero-length clips (and can reorder train by class), but it carried no `clip_stems` column and its docstring claimed the rows lined up with the on-disk `clip_stems.npy`, which still lists every clip. The on-disk row indices woudl have been misaligned at any dropped zero-length clip form the in-memory dataset.

**Why it mattered.** Discovered before any cached predictions were served. Predictions would have silently pinned to the wrong clips, with nothing to flag it, corrupting per-clip evaluations and the FE display.

**Fix.** Write `clip_stems` straight from the in-memory dataset so the join carries its own ground truth; assert the sidecar's there; regression tests over the drop and reorder cases.

**More.** Commit `251f6ac`.
