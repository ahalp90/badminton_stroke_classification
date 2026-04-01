# Task Breakdown: Weeks 1–2 (April 1–14, 2026)

**Assumptions:** ~8–12 hours per person per week (3 other subjects + life). Tasks are sized accordingly. "Stretch" items are genuinely optional — do them if you finish early, skip them guilt-free if not.

**Critical path:** Scott & Ariel finalising architecture + adapting the BST pipeline code. Everything else is designed to be useful regardless of exactly where the architecture lands.

---

## Key Principles

1. **Nobody waits on Ariel & Scott.** Every task below can begin before the model architecture is locked.
2. **No SQL database.** A JSON manifest (or CSV index) is sufficient for 30k videos. Experiment tracking uses MLflow/W&B or a simple JSON log — not a custom DB.
3. **Model B (keypoint-based) is the priority model.** If time gets tight, a working lightweight model beats two half-built models.
4. **Code reviews are cool (and documentation too)** If you're writing code that others have to build around, ask for code reviews for major components. No-one's a pro here, but hopefully we can work together to make sure the codebase is easy to work in.<br><br>
->If using an LLM for code review, please ask it not to overengineer.

---

## Week 1: April 1–7 — Foundation & Parallel Workstreams

### Scott & Ariel (ML Architecture)

**Primary tasks:**
1. **Finalise architecture decisions and write a data contract document (Days 1–3)**
   - Even before the architecture is fully locked, write a brief spec covering what inputs each model expects. This unblocks Jared and Isaiah immediately.
   - Model A: e.g., "video clips of ~30 frames at 224×224, 30fps"
   - Model B: e.g., "MediaPipe keypoint sequences of length T, 17 joints × 2D/3D coords"
   - Shuttle trajectory format from ShuttleTracker
2. **Begin adapting BST pipeline code to your 14-class taxonomy and project needs (Days 3–7)**
   - This is architecture work, not a handoff task — it requires understanding both the existing codebase and your design decisions.
   - Map the BST data loader to the proposed 14-class taxonomy (use Isaiah's class-collapse mapping as a reference once ready).
   - Identify what needs to change vs. what can be reused as-is.

---

### Isaiah (Data Analysis & EDA)

> *Isaiah, these tasks are about producing findings and recommendations, not production-quality code. If it's not code others need to build around, feel free to work in R--we just need md/pdf/html outputs that everyone can view. *

*RE code that builds project sections: if as an R guy you're not comfortable doing production Python then just ask for code review and restructuring from Scott/Ariel/Curtis/anyone happy to take this on.*

**Primary tasks (do in order):**

1. **ShuttleSet Exploratory Data Analysis notebook (Days 1–4, ~6 hrs)**

   Load the ShuttleSet dataset annotations and produce a reference notebook the team can use throughout the project. This is pure data exploration — no deep learning involved.

   <details>
   <summary>Granular steps (click to expand)</summary>

   - Load the ShuttleSet CSV/JSON annotation files into a pandas DataFrame (if you're more comfortable, you can prototype in R first and translate — but the deliverable should be a `.ipynb` notebook).
   - Plot class distributions for the original 18 stroke types (bar chart with counts and percentages).
   - Apply the 14-class collapse mapping (see the taxonomy document — it's a simple remapping of labels) and plot the collapsed distribution.
   - Summarise rally-level statistics: rally lengths (number of strokes per rally), match counts, player counts.
   - Flag any data quality issues: missing labels, duplicate entries, annotation inconsistencies.
   - Output: a single Jupyter notebook titled `01_shuttleset_eda.ipynb` with markdown commentary explaining each finding.

   **Helpful pandas equivalents for R users:**
   | R (tidyverse) | Python (pandas) |
   |---|---|
   | `read_csv()` | `pd.read_csv()` |
   | `group_by() %>% summarise()` | `df.groupby().agg()` |
   | `ggplot() + geom_bar()` | `plt.bar()` or `df.plot.bar()` |
   | `mutate()` | `df['col'] = ...` |
   | `filter()` | `df[df['col'] == val]` |

   </details>

2. **Train/validation/test split strategy (Days 4–6, ~4 hrs)**

   Research and propose how to split the dataset. This is a data science decision, not a coding task.

   <details>
   <summary>Granular steps</summary>

   - **Key constraint:** You cannot split randomly by stroke, because consecutive strokes within a rally are not independent (temporal autocorrelation). Splitting must be at the match level or at minimum the rally level.
   - Propose a match-stratified split (e.g., 70/15/15 or 80/10/10 by match, ensuring similar class distributions across partitions).
   - Implement the split assignment in your notebook and verify that class proportions are roughly balanced across train/val/test.
   - Document the rationale in markdown cells.
   - Output: a section in your EDA notebook (or a second notebook `02_split_strategy.ipynb`) with the proposed split and supporting analysis.

   </details>

3. **Class imbalance analysis (Days 6–7, ~3 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Using the split from task 2, compute per-class sample counts in each partition.
   - Calculate imbalance ratios (largest class / smallest class).
   - Research and briefly describe 2–3 mitigation strategies: class weighting in the loss function, oversampling (e.g., random oversampling of minority classes), data augmentation. You don't need to implement these — just describe them and recommend which to try first.
   - Output: a short section in your notebook with a recommendation.

   </details>

**Stretch (only if time permits):**
- Read the ShuttleSet paper's Section 3 (dataset construction) and summarise any caveats about annotation quality or inter-annotator agreement that the team should know about.

---

### Curtis (Project Infrastructure & Backend)

> *Curtis, your testing and backend skills are where the team is weakest. You're setting up the scaffolding that everyone else builds on.*

**Primary tasks:**

1. **Project scaffolding and testing framework (Days 1–3, ~4 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Set up the Python project structure. Suggested layout:
     ```
     badminton-stroke-classifier/
     ├── src/
     │   ├── data/          # data loading, preprocessing
     │   ├── models/        # model definitions
     │   ├── training/      # training loops, evaluation
     │   └── api/           # backend API
     ├── tests/
     ├── notebooks/         # Isaiah's EDA, experiments
     ├── configs/           # hyperparameter configs (YAML/JSON)
     ├── scripts/           # utility scripts
     ├── requirements.txt
     └── README.md
     ```
   - Set up pytest with a basic test that just verifies imports work.
   - Add a `requirements.txt` with known dependencies: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `mediapipe`, `pytest`.
   - Write a brief `README.md` with setup instructions.

   </details>

2. **Backend API skeleton (Days 3–6, ~5 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Scaffold a FastAPI application with stub endpoints. These don't need to do anything real yet — they define the contract that the front-end will code against:
     - `POST /api/upload` — accept a video file (return a job ID)
     - `GET /api/status/{job_id}` — return job status (queued/processing/complete)
     - `GET /api/results/{job_id}` — return classification results (return mock JSON for now)
     - `GET /api/models` — list available models (return hardcoded list: Model A, Model B)
   - Return sensible mock data from each endpoint so Kiri's front-end can develop against it.
   - Write basic pytest tests for each endpoint.

   </details>

3. **Experiment tracking setup (Days 6–7, ~3 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Set up MLflow locally (or Weights & Biases if the team prefers — both have free tiers).
   - Write a minimal example script that logs a fake training run: hyperparameters, a loss curve, and a final accuracy metric.
   - Document setup instructions so Ariel & Scott can plug into it when training begins.
   - If MLflow/W&B feels like overkill, a simple JSON-lines log file (`experiments.jsonl`) with one JSON object per run is honestly fine for this project's scale.

   </details>

**Stretch:**
- Write a utility function for loading and validating the JSON manifest (video path → metadata mapping) once Isaiah's split assignments exist.

---

### Ethan (Containers & Environment)

> *Ethan, your containerisation scope is real but bounded at this stage. The bigger value you can add now is making sure the team's dev and HPC environments actually work smoothly.*

**Primary tasks:**

1. **Docker dev environment (Days 1–3, ~4 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Create a `Dockerfile` for local development: Python 3.10+, PyTorch (CPU version for local dev), all project dependencies from `requirements.txt`.
   - Create a `docker-compose.yml` that mounts the project directory and the data directory as volumes.
   - Test that the container builds and runs, and that `import torch` and `import mediapipe` work inside it.
   - Document in the README: "How to run locally with Docker."

   </details>

2. **UNE-HPC project-specific quickstart guide (Days 3–5, ~4 hrs)**

   <details>
   <summary>Granular steps</summary>

   - The uni provides general HPC documentation, but nobody has written a guide specific to *this project's* workflow. Write one that covers:
     - How to connect and authenticate
     - Where to store the ShuttleSet data on HPC
     - How to submit a PyTorch GPU training job (sample SLURM/PBS script with the correct CUDA version, module loads, and conda/venv activation)
     - How to monitor job status and retrieve logs
     - Common gotchas (e.g., memory limits, job queue times, storage quotas)
   - Test it end-to-end: submit a trivial PyTorch script that creates a tensor on GPU and prints `torch.cuda.is_available()`.
   - Output: a markdown document `hpc_quickstart.md` in the project repo.

   </details>

3. **ShuttleSet data access pattern (Days 5–7, ~3 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Figure out where the ShuttleSet video files will live (local, HPC storage, cloud bucket?) and how they'll be accessed during training.
   - If the videos need downloading or organising, write a script for it.
   - Coordinate with Jared on the preprocessing pipeline's input expectations.

   </details>

---

### Kiri (Front-end & Documentation)

> *Kiri, your exploratory work becomes the team's visual spec this week. You're also the technical writer — start capturing decisions early rather than reconstructing them later.*

**Primary tasks:**

1. **Wireframes and component inventory (Days 1–4, ~5 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Produce low-fidelity wireframes (Figma, pen-and-paper photos, whatever is fastest) for the key screens from the functional requirements:
     - **Video selection screen:** browse/upload match videos, apply court metadata, identify target player
     - **Model selection & analysis:** choose model (A or B), monitor analysis progress
     - **Results dashboard:** stroke classification results, confidence scores, heatmap/explainability visualisations
   - List all UI components needed (video player, progress bar, results table, heatmap overlay, etc.)
   - Share wireframes with the team for feedback by end of Day 4.

   </details>

2. **Front-end tech stack decision and scaffold (Days 4–7, ~5 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Recommend a framework. React is the natural choice given team skills (Curtis and Jared are both competent in full-stack web).
   - Initialise the project (`create-react-app` or Vite).
   - Set up basic routing for the key screens (even if they're empty placeholder components).
   - Create a stub API client that hits Curtis's mock endpoints and displays the returned data.

   </details>

**Stretch:**
- Begin a `decisions_log.md` capturing key technical decisions made this week (architecture choices, split strategy, tech stack) with dates and rationale. This feeds directly into the final report.

---

### Jared (DevOps & Preprocessing Pipeline)

> *Jared, you're the bridge between infrastructure and ML. Your feature engineering and CI/CD skills are both needed this week.*

**Primary tasks:**

1. **CI/CD pipeline (Days 1–2, ~3 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Set up GitHub Actions for the repo:
     - On push/PR: run `pytest`, run a linter (`ruff` or `flake8`), check that the Docker image builds
   - Keep it simple — the goal is to catch broken code early, not build a production deployment pipeline yet.
   - Coordinate with Curtis on the test framework so CI runs his tests.

   </details>

2. **Video preprocessing pipeline (Days 2–7, ~7 hrs)**

   <details>
   <summary>Granular steps</summary>

   - This is the most important Week 1 task outside of Ariel & Scott's architecture work. The pipeline takes raw ShuttleSet videos and produces model-ready inputs.
   - **Architecture-agnostic components (start immediately):**
     - Frame extraction from rally videos at a consistent FPS (30fps is standard)
     - Clip segmentation: given stroke annotation timestamps, extract a window of N frames centred on each stroke event
     - Resolution normalisation (e.g., resize to 224×224 — Ariel/Scott can confirm exact dimensions via the data contract)
     - Output as a structured directory: `processed/{split}/{match_id}/{rally_id}/{stroke_id}/frames/`
   - **Architecture-dependent components (start once data contract is available, likely Day 3+):**
     - For Model B: run MediaPipe pose estimation on extracted frames, output keypoint sequences as `.npy` files
     - For Model A: stack frames into tensors, apply any required normalisation
   - Write the pipeline as a CLI script: `python preprocess.py --input raw/ --output processed/ --config config.yaml`
   - Include basic validation: log how many strokes were processed, flag any that failed.

   </details>

**Stretch:**
- Begin investigating ShuttleTracker: clone the repo, check dependencies, try running it on a sample video. Both models use shuttle trajectory data, so this will be needed in Week 2.

---

## Week 2: April 8–14 — Integration & Early Model Code

By now, Ariel & Scott should have the architecture largely locked and the BST pipeline partially adapted. The team pivots from independent workstreams toward integration.

### Ariel & Scott

1. **Finish BST pipeline adaptation (Days 8–10)**
   - Complete the data loading, label mapping, and feature extraction pipeline for at least Model B.
2. **Model skeleton code (Days 10–13)**
   - Write the PyTorch model class (`nn.Module`) for Model B: forward pass, layer definitions. Doesn't need to train well — just needs to be structurally correct and accept the right input shapes.
   - If time permits, draft Model A skeleton too.
3. **Training loop scaffold (Days 13–14)**
   - Basic training loop: loss function, optimiser, metric logging, integration with Curtis's experiment tracking.
   - Validate end-to-end: data → preprocessing → DataLoader → model forward pass → loss → backward pass.

---

### Isaiah

4. **Feature extraction validation (Days 8–11, ~5 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Once Jared's preprocessing pipeline and/or Ariel & Scott's adapted BST pipeline are producing outputs, verify them statistically:
     - Are the keypoint sequences the expected shape and range?
     - Are there NaN/zero values indicating failed pose estimation?
     - What percentage of frames have confident pose detections vs. low-confidence/missing keypoints?
   - Visualise a sample of extracted keypoints overlaid on source frames to sanity-check alignment.
   - Output: a validation notebook `03_feature_validation.ipynb`.

   </details>

5. **Literature review of comparable results (Days 11–14, ~4 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Review the BST paper and ShuttleSet paper's reported classification accuracies.
   - Compile a comparison table: model, dataset, number of classes, reported accuracy/F1, and any notes on methodology.
   - This directly feeds into the benchmarking objective and the final report's related work section.
   - Output: a markdown document or notebook section with the comparison table.

   </details>

---

### Curtis

4. **PyTorch Dataset and DataLoader (Days 8–11, ~5 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Write a PyTorch `Dataset` class that reads the processed features (from Jared's pipeline or the adapted BST loader) and returns `(input_tensor, label)` pairs.
   - Write a `DataLoader` wrapper with configurable batch size, shuffling, and worker count.
   - This is a well-defined task with excellent documentation — follow the official PyTorch `Dataset` tutorial.
   - Include a test that loads a batch and prints shapes.
   - Coordinate with Ariel & Scott on the expected tensor shapes.

   </details>

5. **Integration testing (Days 11–14, ~4 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Write end-to-end tests for the pipeline:
     - Raw annotation → class collapse → split assignment → preprocessing → DataLoader → dummy model forward pass
   - The goal is to verify that all the pieces connect. If any step fails, the test tells you exactly where.
   - This is where your testing expertise is most valuable to the team.

   </details>

---

### Ethan

4. **GPU training container (Days 8–11, ~4 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Now that dependencies are more settled, create a training-specific Dockerfile with PyTorch + CUDA.
   - Ensure it works on UNE-HPC (or at least document any incompatibilities).
   - Test: container can run a minimal PyTorch GPU training script.

   </details>

5. **Deployment architecture document (Days 11–14, ~3 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Document how the full system will be containerised for the May 17 milestone:
     - Training container (PyTorch + CUDA + data pipeline)
     - Inference container (lighter, model serving)
     - Front-end container (React app)
     - How they communicate (API calls between front-end and inference containers)
   - This is a planning document, not implementation — it guides the team's Docker work in later weeks.

   </details>

---

### Kiri

4. **Results dashboard prototype with mock data (Days 8–12, ~6 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Build a working prototype of the results view using mock classification data from Curtis's API.
   - Focus on the explainability components — these are the client-facing differentiator:
     - Stroke-by-stroke classification table with confidence scores
     - Visual timeline of strokes within a rally
     - Placeholder for heatmap/Grad-CAM overlay (actual heatmaps come later — just design the container and use a placeholder image)
   - Use the wireframes from Week 1 as the spec.

   </details>

5. **Report and documentation skeleton (Days 12–14, ~3 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Set up the final report structure (sections, subsections, page budget).
   - Begin drafting the Introduction, Related Work, and Methodology sections using decisions already made.
   - Capture Week 1–2 decisions in the decisions log.
   - This is genuinely important and gets neglected if left to the end.

   </details>

---

### Jared

3. **ShuttleTracker integration (Days 8–11, ~5 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Both models use shuttle trajectory data. Clone the ShuttleTracker repo, install dependencies, run it on a sample ShuttleSet video.
   - Wrap it as a reusable module: `get_shuttle_trajectory(video_path) → trajectory_array`.
   - Document any issues (dependency conflicts, speed, failure cases).

   </details>

4. **End-to-end preprocessing pipeline (Days 11–14, ~4 hrs)**

   <details>
   <summary>Granular steps</summary>

   - Connect your video preprocessing with the feature extraction (whether from your Week 1 work, Isaiah's validation, or Ariel & Scott's adapted BST code) into a single runnable pipeline.
   - Deliverable: `python preprocess.py --input raw_videos/ --output features/ --config config.yaml` that the training code can consume directly.
   - Run it on a subset of ShuttleSet and verify outputs with Isaiah's validation notebook.

   </details>

---

## PyTorch Ramp-Up (Everyone, Week 1)

Nobody knows PyTorch, so invest 2–3 hours in Week 1 on self-study. Suggested resources by role:

| Person | What to cover | Resource |
|---|---|---|
| Everyone | General concepts | [PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) (~2 hrs) |
| Ariel, Scott | `nn.Module`, training loops | [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) |
| Curtis | `Dataset` and `DataLoader` | [Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) (~1 hr) |
| Isaiah | Tensors and basic ops (for validation) | Just the "What is PyTorch?" section of the Blitz |

---

## End-of-Week Checkpoints

**End of Week 1 (April 7):**
- [ ] Ariel & Scott: data contract document shared, BST pipeline adaptation underway
- [ ] Isaiah: EDA notebook complete, split strategy proposed
- [ ] Curtis: project scaffolded with tests, API stubs returning mock data
- [ ] Ethan: Docker dev environment working, HPC quickstart guide written
- [ ] Kiri: wireframes shared, front-end project initialised
- [ ] Jared: CI running, video preprocessing pipeline functional for basic frame extraction

**End of Week 2 (April 14):**
- [ ] Ariel & Scott: Model B skeleton code written, training loop scaffolded
- [ ] Isaiah: feature validation notebook complete, literature comparison table drafted
- [ ] Curtis: PyTorch DataLoader working, integration tests passing
- [ ] Ethan: GPU training container tested, deployment architecture documented
- [ ] Kiri: results dashboard prototype with mock data, report skeleton started
- [ ] Jared: ShuttleTracker running on sample data, end-to-end preprocessing pipeline functional

This positions the team for the **April 19 milestone** (pipeline and architectures finalised) with 5 days of buffer.
