# Task Breakdown: Weeks 3–4 (April 14–27, 2026)

**Assumptions:** ~8–12 hours per person per week (3 other subjects + life). Tasks are sized accordingly.

**Critical path:** The dataset pipeline is processing TrackNet shuttle positions on engelbart (ETA ~Tuesday April 15). Once complete, Ariel & Scott run the phased baseline retraining sequence. Isiah's updated splits are needed before Phase 2. Everything else can proceed in parallel.

---

## Where We Are: End of Week 2

### Project status

- **Data pipeline**: Operational end-to-end. 33k stroke clips generated on merged_25 taxonomy. Currently running TrackNetV3 shuttle position extraction (long compute — ETA Tuesday). Pipeline orchestrator (`build_dataset.py`) handles download → clip extraction → class merging → shuttle extraction with configurable skip flags.
- **Model code**: BST-CG-AP refactored into clean PyTorch modules. Training loop, inference script, and evaluation utilities ready.
- **Architecture designs**: Arch 1 (BST + X3D-S wrist crop fusion) is designed and documented. Arch 2 is in active development — Scott is working through the design, currently thinking 2+1D RGB through YOLO player detection with a torch model zoo 2+1D model, plus player bounding box and shuttlecock trajectory features.
- **Frontend**: React + Vite app with page routing (upload, analysis, results), reusable components, and API client stubs. Living on a separate branch.
- **Backend**: FastAPI stub endpoints returning mock data (upload, status, results, models).
- **Infrastructure**: Dockerfile and docker-compose.yml exist but haven't been verified end-to-end against the project. A GitHub Actions CI config exists (`ci.yml`) that runs pytest on push/PR — worth checking it's actually running and passing.
- **Data analysis**: Comprehensive EDA notebook with class distributions, split strategy, class weights, and data quality findings. Separate hyperparameter optimisation research. Scott has done manual data exploration identifying timing discrepancies and produced the revised class taxonomy.

### What's been delivered (Weeks 1–2)

| Person | Key deliverables |
|--------|-----------------|
| **Ariel** | Data pipeline (download → clips → shuttles), TrackNetV3 batching + InpaintNet integration, BST codebase refactor, HPC environment setup, Arch 1 research + design docs |
| **Scott** | Revised and substantiated class taxonomy, manual data exploration + sanity checking (identified timing discrepancies), split strategy analysis (player representation concerns), Arch 2 research + ongoing design |
| **Isiah** | EDA notebook with class/split/quality analysis, train/val/test split assignments (32k strokes), class weights for loss function, hyperparameter optimisation research (Optuna + TPE recommendation with trial budgets) |
| **Curtis** | Project scaffolding, FastAPI backend stubs with mock data, MLflow experiment tracking setup, PR reviews |
| **Kiri** | React frontend MVP: Vite setup, page routing, Header/Button/FileUploader components, API client stubs, CORS middleware |
| **Jared** | GitHub Actions CI config (pytest on push/PR) |
| **Ethan** | Dockerfile + docker-compose.yml, README setup instructions |

### What's next: the 4-phase plan

We're at the Phase 1 gate. The phases are designed to isolate variables — each one tells us something specific:

1. **Phase 1 — BST sanity check**: Verify the refactored BST-CG-AP reproduces published results on the merged_25 taxonomy. Confirms our pipeline and code are correct.
2. **Phase 2 — New taxonomy + new splits**: Retrain BST-CG-AP using `une_merge_v1` (14 types, 29 classes) with revised splits (minimised player overlap). Gives us the baseline to beat.
3. **Phase 3 — Augmentation**: Retrain with temporal speed and camera augmentations to generalise to amateur players. Expected mild performance loss.
4. **Phase 4 — Novel architectures**: Build and evaluate Arch 1 and Arch 2 on the same augmented dataset.

---

## Week 3: April 14–20 — Baselines & Research

### Ariel & Scott (ML Core)

**Primary tasks:**

1. **Phase 1 — BST sanity check** (April 15–17) ⚠️ *Critical path — gates all subsequent phases*
   - Pipeline completes → run MMPose collation → train BST-CG-AP on merged_25.
   - Compare accuracy and macro-F1 against published BST results. Document any delta and diagnose if significant.

2. **Phase 2 — New taxonomy baseline** (April 17–20) ⚠️ *Critical path — establishes the benchmark everything is measured against*
   - Rebuild the dataset using `une_merge_v1` taxonomy with Isiah's updated splits.
   - Retrain BST-CG-AP. This establishes the benchmark for expected performance under our preferred taxonomy and fair evaluation splits.

3. **Scott — Arch 2 development** ⚠️ *Critical path — one of our two novel contributions*
   - Continue developing the Arch 2 design and implementation. Some things worth considering:
     - Locking down the YOLO player detection → bounding box crop pipeline early, since crop quality feeds everything downstream.
     - How the three feature streams (player position from bounding boxes, shuttlecock trajectory, RGB frame convolution) get combined — temporal sequencing across the streams will matter for classification.
   - Nominal target: some Arch 2 code in the repo by end of this week so the team can see the data flow.

---

### Isiah (Data Science)

**Primary tasks:**

1. **Revised splits — minimise player overlap** (~3 hrs) ⚠️ *Critical path — needed before Phase 2 retrain*

   The current splits are match-level (no rally leakage — good), but some players appear across train/val/test partitions. Scott also identified this concern: the model could learn player-specific movement patterns rather than general stroke patterns. Fully player-disjoint splits aren't feasible — with ~40 matches and players like Axelsen in 10 of them, the player graph is too connected. The goal is to **minimise** player leakage, especially between train and test.

   - Re-run the split stratification to reduce player overlap across partitions as much as possible while maintaining reasonable class distributions and gender balance.
   - Where there's a tradeoff between val and test set size, **prioritise val** — that's what gets used for checkpointing and early stopping during training. Test is an additional held-out evaluation.
   - Quantify the remaining overlap (e.g., what % of test-set players also appear in train, and how many strokes are affected).
   - Deliver updated `shuttleset_splits.csv` by ~April 17 so Ariel can use it for Phase 2.

2. **Feature validation notebook** (~4 hrs)
   - Once Phase 1 collation runs on HPC: verify the keypoint sequences are well-formed.
   - Check: shapes, value ranges, NaN/zero rates, pose estimation confidence scores.
   - Visualise a sample of extracted keypoints overlaid on source frames.
   - Output: `03_feature_validation.ipynb`

---

### Curtis (XAI & Deployment)

Curtis's focus shifts to research this fortnight — figuring out explainability, deployment, and the video cropping pipeline. This work feeds directly into the frontend and into how we demo the final system.

**Primary tasks:**

1. **Explainable AI techniques survey** (~6 hrs)

   Research which XAI methods suit our model types and how they'd map to the web UI:
   - **Grad-CAM / Grad-CAM++** for the X3D CNN streams — produces heatmaps over video frames showing which spatial regions drove the prediction.
   - **Attention rollout / attention weight visualisation** for the BST transformer — shows which frames and joints the model attended to.
   - **SHAP or integrated gradients** for trajectory inputs — feature importance for shuttle/player position data.
   - For each: what does the output look like? What's the right UI component? (Heatmap overlay on video? Attention timeline? Feature importance chart?)
   - Output: a research document with recommendations and rough UI sketches.

2. **Deployment architecture research** (~4 hrs) ⚠️ *Critical path — gates frontend integration, container decisions, and Jared+Ethan's deployment work*

   Figure out where and how we serve inference for the demo:
   - **First step**: ask the uni whether we can serve inference from HPC, or if there's a hosted option. Don't spend time designing around constraints we haven't confirmed.
   - Then evaluate realistic options: HPC GPU via SSH tunnel, Gradio hosted on HPC, cloud VM, etc.
   - How does the frontend call the model? Direct API? Queue-based async?
   - Output: architecture decision document with recommendation.

---

### Kiri (Frontend)

Frontend scaffolding is in place with sensible architectural decisions (React + Vite, component separation, API client stubs). Ready to build out into a full user flow.

This fortnight builds out the two most important missing pieces: video cropping (so users can select the relevant part of a video) and XAI visualisations (our key differentiator).

**Primary tasks:**

1. **Video cropping UI** (~6 hrs)
   - **Spatial crop**: canvas overlay on the uploaded video where the user draws a bounding box to select the court/player region.
   - **Temporal crop**: timeline scrubber to set start and end times.
   - Send crop parameters to the backend along with the uploaded video.
   - Coordinate with Curtis on validation constraints — minimum pixel sizes, aspect ratios.

2. **XAI visualisation placeholders** (~3 hrs)
   - Based on Curtis's XAI research: build placeholder components for the results page.
   - Heatmap/Grad-CAM overlay container (on video frames or a still image).
   - Attention timeline visualisation (which frames mattered most for the prediction).
   - Confidence breakdown chart per stroke.
   - These can use mock data initially, but the structure should match the expected XAI output format so integration is straightforward later.

---

### Jared & Ethan (DevOps & Documentation)

This fortnight's focus is on two things: (1) figuring out the real compute and deployment requirements for our system, and (2) getting a head start on the final report.

**Collaborative task: Inference compute analysis & deployment requirements**

1. **Inference compute requirements** (~5 hrs each)

   Calculate the actual compute requirements for our deployed inference pipeline. This is important for deployment decisions and for the report.

   - **Current pipeline**: BST-CG-AP + TrackNetV3 with InpaintNet + MMPose (RTMPose). What GPU memory does each component need? What's the expected latency per video? What throughput can we expect?
   - **With Arch 1 wrist crop addition**: The wrist crop model (X3D-S) might be integrated in one of two ways — (a) a simpler late concatenation before the MLP head, or (b) a cross-attention/transformer fusion layer. These have very different compute profiles. Analyse both. You'll want to read the architecture notes in `scratch/architecture_notes/` and talk to Ariel to get at this meaningfully.
   - **Target deployment platforms**: Consider the compute profile for (a) whatever remote GPU deployment we end up using, (b) a run-of-the-mill business laptop, and (c) a decent midrange phone. Not all of these may be realistic for every model — document what's feasible where.
   - Start from actual model specs — parameter counts, FLOPs, input sizes, batch sizes. Not generic estimates.
   - Scott's Arch 2 pipeline can be analysed once his implementation takes shape.

2. **Deployment & monitoring requirements** (~4 hrs each)

   - What hosting options exist for serving inference? Evaluate HPC tunnel, cloud VM, serverless GPU (e.g., Replicate, Modal). Pros and cons for our specific context.
   - What monitoring and logging do we need **beyond MLflow** (which Curtis set up for experiment tracking)? Consider: inference request logging, error rates, latency monitoring, GPU utilisation tracking in production.
   - Do we actually need containers for deployment, or can we serve from Python venvs on HPC? Document the tradeoffs — this depends on Curtis's deployment architecture decision.
   - What does CI/CD look like for deploying a new model version?
   - If Curtis's research concludes we need containers, Jared + Ethan will own building them.

**Individual tasks: Report sections**

**Jared:**
- **Dataset Description** (~3 hrs): ShuttleSet overview, our taxonomy decisions (why we collapsed from 18 to 14 types — draw on Scott's taxonomy report for this), class statistics, split strategy rationale. Source material: Isiah's EDA notebook, Scott's taxonomy report, `pipeline/config.py`.
- **Experimental Setup** (~3 hrs): Hardware specs (engelbart V100), training hyperparameters, evaluation metrics, baselines we compare against. Source: `bst_x_train.py`, architecture research notes in `scratch/`.

**Ethan:**
- **System Architecture diagram + writeup** (~3 hrs): End-to-end system diagram showing frontend → API → preprocessing → model inference → results. Technology stack summary. Should draw from the compute analysis above.
- **Methodology: Data Pipeline** (~3 hrs): Document the preprocessing pipeline steps — video download, clip extraction, class merging, pose estimation (RTMPose), shuttle extraction (TrackNetV3 + InpaintNet), collation. Source: `data_pipeline_to_model_train.md`, pipeline README.

---

## Week 4: April 21–27 — Augmentation & Integration

### Ariel & Scott

4. **Phase 3 — Augmentation experiments**
   - Temporal speed augmentation to simulate amateur-level play. Amateurs swing roughly 50% slower than pros, but the augmentation should be **variable** — applied to a subset of samples, not uniformly, and ideally non-uniform within each sample (the swing deceleration profile differs from post-contact physics). This mimics real amateur variation rather than just globally slowing everything down.
   - Camera angle, quality, and stability augmentations for amateur recording conditions.
   - Retrain BST on augmented data. Document the performance delta.
   - Small hyperparameter search to accommodate the shifted data distribution.

5. **Begin Arch 1 & 2 data preparation**
   - Ariel: X3D-S wrist crop extraction pipeline with adaptive crop logic.
   - Scott: Player detection + tracking → crops, temporal windowing around contact point.

---

### Isiah

3. **Performance metrics research** (~4 hrs)
   - Research what performance measures are actually appropriate for this task, beyond the standard accuracy and F1 reported in prior work. Consider:
     - Whether class-level metrics (per-class F1/precision/recall) are more informative than aggregates given our class imbalance.
     - Whether top-k accuracy matters for deployment (e.g., showing the user the top 3 predicted stroke types).
     - Whether there are domain-specific evaluation approaches used in sports action recognition.
   - This shapes how we evaluate and report results for all phases.

4. **Amateur vs pro data distribution analysis** (~4 hrs)
   - What does the amateur data distribution look like compared to pro? Swing speed differences, trajectory patterns, positional tendencies.
   - Identify realistic augmentation ranges — even where source data is limited, literature and biomechanics reasoning can inform sensible bounds.
   - This feeds directly into Ariel & Scott's augmentation parameter choices.

5. **Literature comparison table** (~3 hrs)
   - Compile: model name, dataset, number of classes, reported metrics, key methodology notes.
   - Goes straight into the Related Work section of the final report.

---

### Curtis

3. **Video cropping backend validation** (~5 hrs)
   - Backend logic to enforce minimum pixel sizes for model input (224×224 for X3D, sufficient resolution for pose estimation). Warn the user rather than silently degrading quality.
   - Temporal crop validation: ensure clips meet minimum frame count for the model.
   - Coordinate with Kiri on how the UI sends crop parameters.

4. **Inference API hookup** (~3 hrs)
   - Wire `bst_x_infer.py` (or a wrapper) into the existing FastAPI endpoints.
   - Replace mock responses with real BST model predictions.
   - This is the bridge from "demo with fake data" to "demo with real inference."

---

### Kiri

3. **Results dashboard improvements** (~4 hrs)
   - Stroke timeline visualisation (visual timeline of strokes within a rally).
   - Confidence visualisation improvements (colour-coded thresholds).
   - Integrate with real API once Curtis wires up inference.

4. **Report drafting** (~3 hrs)
   - Introduction section.
   - Frontend architecture decisions documentation.
   - Screenshots of current UI state for the report.

---

### Jared & Ethan

- Complete remaining report sections (see individual assignments above).
- Finalise compute analysis and deployment requirements documents.
- If Curtis's deployment architecture decision requires containers: begin building them.

---

## End-of-Week Checkpoints

**End of Week 3 (April 20):**
- [ ] Ariel & Scott: Phase 1 BST result documented, Phase 2 retrain underway or complete
- [ ] Isiah: Revised splits delivered ⚠️ *(critical path)*, feature validation notebook
- [ ] Curtis: XAI survey document shared, deployment architecture researched
- [ ] Kiri: Video cropping UI component, XAI visualisation placeholders
- [ ] Jared & Ethan: Inference compute analysis, first report sections drafted

**End of Week 4 (April 27):**
- [ ] Ariel & Scott: Phase 3 augmentation results documented, Arch 1 & 2 data prep underway
- [ ] Isiah: Performance metrics research, amateur vs pro analysis, literature comparison table
- [ ] Curtis: Cropping backend validation, inference API returning real predictions
- [ ] Kiri: Results dashboard improved, report sections drafted
- [ ] Jared & Ethan: DevOps requirements complete, remaining report sections, containers if needed

This positions the team to begin **Phase 4 (novel architectures)** by end of April, with ~2.5 weeks for implementation, training, and evaluation before the May 17 milestone.

---

## Key Dependencies

```
Pipeline finishes (~Apr 15)
    │
    ├── Phase 1: BST sanity check (Ariel, Apr 15-17)
    │       │
    │       ├── Isiah: revised splits — minimised player overlap (by Apr 17)
    │       │
    │       └── Phase 2: New taxonomy + new splits retrain (Ariel+Scott, Apr 17-20)
    │               │
    │               └── Phase 3: Augmentation retrain (Ariel+Scott, Apr 21-25)
    │                       │
    │                       └── Phase 4: Novel architectures (Apr 25+)
    │
    ├── Curtis: XAI + deployment research (independent, Apr 14-20)
    │       │
    │       └── Kiri: XAI viz + cropping UI (partially dependent on Curtis)
    │
    └── Jared & Ethan: compute analysis + report sections (independent)
                │
                └── Container work (if needed) depends on Curtis's deployment decision
```
