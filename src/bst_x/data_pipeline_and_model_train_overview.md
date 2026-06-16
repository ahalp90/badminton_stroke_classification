# Data Pipeline and Model Training: Overview

How a raw match video becomes a trained stroke classifier. It covers what the pipeline does, what was inherited (the original BST research code), and what changed.

For more detail: `data_pipeline_to_model_train.md` is the module-by-module reference, and `pipeline/README.md` is the step-by-step run guide.

## What it does

It downloads matches from YouTube, cuts out every individual stroke clip (frame exact re-encode), and turns each one into three things the model can read: where each player's body is (skeletal pose), where they stand on the court, and where the shuttle travels.

```
raw match video
   |   cut into per-stroke clips, track the shuttle
   v
labelled clips  +  shuttle trajectories
   |   add skeletal pose, pack into arrays
   v
pose + court position + shuttle,  per split
   |   train and test
   v
stroke classifier  +  per-class scores
```

It runs in two halves. The first downloads the videos, cuts them into labelled clips, and runs a shuttle tracker over each clip. The second adds player pose and packs all three streams into the arrays the model trains on. The whole thing is automated, called from the CLI with specific args depending on what you need (e.g., if you want to skip stages or tweak a setting).

It has a single source of truth config file (`pipeline/config.py`): the splits, the stroke names, the rules for grouping. It also sets the class collapse per taxonomy; for example, whether a "wrist_smash" counts as its own class, or gets merged with "smash". These options can be set at the CLI.

The pipeline is totally separate from the model build. It's just a modular data pipeline that eats video urls and CSVs and saves out CSVs and numpy arrays. It's model agnostic.

The model it feeds is BST-X: a transformer that fuses the three streams into a stroke prediction. You train it on a chosen taxonomy with one command (`bst_x_train`), over several serial trials so one seed can't flatter the result, and score it on macro and min-class F1; inference is a separate small script (`bst_x_infer`). The architecture and the results live in `bst_x_overview.md`.

## The code it was built on

Firstly, **huge** thanks to Jing-Yuan Chang for providing the original BST source code with data pipeline, and helping me out with e-mail questions to get started on this project.

The original BST repo code was published to support a research paper, so its model results could be reproduced.

It was not intended to be a fully independent pipeline, though it provided the essential foundations for this rebuild.

The original code required:
- Manually running scripts several times to cut clips and match the player to the split.
- Data prep involved progressively uncommenting blocks of code, with file paths hardwired (Windows root).
- Quality control was manual: delete the bad clips by hand by looking at a spreadsheet, then compare per-class counts eyeball.

I'm immensely appreciative to Jing-Yuan for the preparation steps he mapped out and the rigorous curation of misclassified clips that he did. This is the bin of ~1000 'unknown'-class clips, which were not identified by the core ShuttleSet release, and whose excision from the dataset is now automated.

## Key features

### Data pipeline

- **Resumable one-command build.** The clip-and-shuttle stage runs from a single command and picks up where it left off if it crashes, so a long extraction needs no babysitting or restart from scratch.
- **Batched shuttle tracking (TrackNetV3 with inpaint).** The shuttle tracker loads once and runs over clips in bulk.
- **Automatic player identification.** Pose extraction keeps the two players inside the court boundary and flags the frames where it can't find them, so per-clip skeletons stay clean with no manual player tagging.
- **Extraction and heuristics are decoupled.** The pipeline builds raw csv data for the shuttle, and npy files for the mmpose extracts. These are then refined and collated at separate stages, so that the original (very long) extract is protected even if business rules change.
- **CSV-driven splits.** Each clip's split and label are applied from a CSV at collation time. Trying out new train/val/test split means just changing a column and a couple of minutes of recollation.
- **Loud failure on bad data.** Empty or malformed clips are dropped or raise errors; this will not break an extraction, but it gives warning.
- **Cluster-ready and verified.** Run across 3 HPC configs and verified to produce identical input.

### Handoff to training

All of this is model-agnostic: any architecture plugs into the same loaders, validation scripts and run tracker.

- **Ready-to-train arrays + loaders.** Collation saves a handful of npy files per split (pose, court position, shuttle, labels, lengths), plus a ready-made PyTorch Dataset and DataLoader. Training reads these collated arrays, never the raw clips.
- **Validation at load.** Validation scripts report detection-failure rates before training starts, and fail loudly if the failure would poison the data quality.
- **Run tracker (manifest.yaml).** Every training run records its code version, machine, settings, and a fingerprint of its data, so any result stays traceable months later.
- **Cross-run comparison.** `run_overview.py` pulls every run's manifest into one table (macro and min-class F1, accuracy, spread across serials), so runs compare at a glance. Optional Aim UI on top if you want the graphs.

## Where the detail lives

- **Module-by-module reference, with CLI and tensor shapes**: `data_pipeline_to_model_train.md`
- **Running it, step by step**: `pipeline/README.md`
- **The design decisions and the full bug log**: `../../scratch/architecture_notes/` (`bst_x_overview.md`, `bst_x_issues_and_bugs_squashed.md`)
