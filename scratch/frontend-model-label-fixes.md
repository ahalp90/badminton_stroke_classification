# Frontend model-label fixes

The configure/results screens currently describe Model A as a "Spatio-Temporal 3D-CNN" pipeline and Model B as a "Keypoint Graph TCN" with MediaPipe. Neither matches the codebase. This doc records what's actually true, with code citations, and the minimum UI changes to stop overclaiming.

## Ground truth

### Model A — what runs today

The only model with code and inference paths is **BST** (Badminton Stroke-type Transformer):

- Architecture: dilated **TCN** front-end → temporal transformer → cross transformer → interactional transformer → head
- Inputs: MMPose pose keypoints, TrackNetV3 shuttle xy, court positions
- No 3D-CNN. No MediaPipe.

Code citations:

- `src/bst_x/stroke_classification/model/tempose.py:129` — `class TCN` (dilated 1D temporal convolutions)
- `src/bst_x/stroke_classification/model/bst.py:23` — `from model.tempose import TCN, ...`
- `bst.py:161-162` — `self.tcn_pose = TCN(...)`, `self.tcn_shuttle = TCN(...)`
- `bst.py:269` — pipeline comment: `TCN -> Temporal Transformer -> Cross Transformer -> Interactional Transformer -> Head`
- `bst.py:426-430` — variant partials: `BST_0`, `BST_PPF`, `BST_CG`, `BST_AP`, `BST_CG_AP`
- `src/bst_x/data_pipeline_to_model_train.md:346` — "TCN feature extraction: separate TCNs for pose and shuttle"
- Zero occurrences of "3D-CNN" or "MediaPipe" in `src/`

### Model B — there is no concrete Model B

Checked:

- `arch_1_directions.md:1-5` — the project's primary research direction is "BST **+** X3D-S Wrist Crop Fusion". X3D-S is a **fusion branch added to BST**, not a separate model. When built, it becomes part of Model A.
- TemPose (standalone variants TemPose_V/PF/SF/TF) was "excised pre-phase-2" per `src/bst_x/data_pipeline_to_model_train.md:340`; the code now lives in `scratch/architecture_notes/historical_bst.md`. Historical, not roadmap.
- No other competing model architecture appears in `src/` or `scratch/architecture_notes/`.

Conclusion: there is no concrete second model. The Model B card must be labelled TBD, not given a fabricated architecture.

### Backend caveat

`src/api/inference.py` is still a stub returning two hardcoded strokes. `/api/upload` accepts a single `model` query param (one checkpoint per job), so even the "select multiple models" framing in the UI doesn't match the API. Out of scope for this label fix; flagged separately.

### Out of scope for label changes

- `scratch/architecture_notes/augmentation_framework.md` — training augmentations. No UI impact.
- `scratch/architecture_notes/class_f1_focal_design.md` — CDB-F1 loss design. No UI impact.

## Updates

### `frontend/hba-stroke-classifier/configure-screen.jsx`

**Lines 5–30 — `MODELS` array**

Model A card (matches what runs today):

- `name`: `'Model A — BST'`
- `subtitle`: `'TCN + Transformer (pose + shuttle)'`
- `description`: "Dilated TCN front-end over MMPose pose keypoints and TrackNetV3 shuttle trajectory, then a temporal / cross / interactional transformer stack. Variants: BST_0, PPF, CG, AP, CG_AP."
- tags: drop "3D-CNN"; add "TCN", "Transformer", "Pose+Shuttle"

Model B card (`disabled: true`, TBD placeholder — do not invent an architecture):

- `name`: `'Model B — TBD'`
- `subtitle`: `'Second model — to be confirmed'`
- `description`: "Reserved slot for a second classification model. No architecture committed yet."
- tags: none, or a single neutral "Reserved" badge
- Keep the card visually disabled exactly as it is today.

**Line 143** — adjust copy so it doesn't imply Model B is a real reference model:

> "Only Model A is currently available for inference. A second model slot is reserved for future work."

**Lines 214–215 — activity log strings**

- `'Model A (3D-CNN): inference started'` → `'Model A (BST): inference started'`
- `'Model A: inference complete (847 strokes)'` — fine as-is

### `frontend/hba-stroke-classifier/results-screen.jsx`

**Line 323**

- `sub: 'Spatio-Temporal 3D-CNN'` → `sub: 'BST (TCN + Transformer)'`

### `frontend/hba-stroke-classifier/README.md`

**Lines 14–17 — models table**

- Model A row: architecture → `MMPose pose + TrackNetV3 shuttle → TCN → Transformer (BST)`; target accuracy / inference timing — leave the current numbers only if they're real BST numbers, otherwise mark "TBD"
- Model B row: architecture → `TBD`; status → "Reserved / no architecture committed"
