# Archived development aids

The git branch contains `archive/development_aids_2026-09-06.tar.gz`, a small historical bundle of 16 files from the completed closing-pass work: about 12 KB compressed and 32 KB unpacked.

The bundle stays here for reference; current reports and saved results live outside the archive.

The archive contains:

- `analysis/` — one script for comparing per-video gains, losses, costs and acceptance; it requires the saved broader results and cached chooser inputs;
- `launches/` — seven HPC command templates for training, broader comparisons, boundary fixes and the visual-model experiment;
- `audits/` — checked findings and decisions that explain useful fixes and rejected concerns; three records were copied from related external-audit folders.

These are historical development aids, not the current entry point. Update machine paths and output locations before reusing a script. Current results begin at [the main report](../README.md).

**Contents**  
[Auditor notes retained in the archive](#auditor-notes-retained-in-the-archive)

From the repository's `archive/` directory, inspect the original bundle with:

```bash
tar -tzf development_aids_2026-09-06.tar.gz
```

## Auditor notes retained in the archive

| Agent | Recorded value in this closing pass |
|---|---|
| Gemini 3.8 Flash High | Concrete fixes: consistent saved-model keys, preserved player guesses on unmatched starts, and separate accepted/all counters. Its learning audits also produced several rejected claims. |
| Opus 4.6 Thinking | Mostly a second check. It flagged a valid limit on attributing gains to features introduced together; most proposed implementation concerns required no change. |
| DeepSeek Flash | No completed audit found. The learning-audit record says the available catalogue had no DeepSeek model. |

The old `worklog/` was otherwise discarded: caches, transcripts, report drafts, duplicated summaries, obsolete prompts and superseded experimental outputs. Current code, results, models and clip-review evidence were retained in git.
