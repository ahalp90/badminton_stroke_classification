# Contact, serve and high-confidence selection numbers

Note: these tables predate the video-15 exclusion and the follow-up work in #147/#148, so treat them as the closing-pass snapshot rather than the final release benchmark.

Compact reference for the final detector and the **784 of 3,982 proposed clips** that pass the fixed confidence threshold.

**Trusted GT:** 3,422 rallies / 38,218 contacts.  
**All GT:** 3,965 rallies / 43,159 contacts, including the **543 rallies** excluded during label cleaning.

Both reads use the same predictions. The main timing allowance is **±10 frames on a 30 fps clock**; ±5 is kept below as a stricter check.

**Contents**  
[Whole-rally recovery](#whole-rally-recovery)  
[Final contact and rally-start performance](#final-contact-and-rally-start-performance)  
[High-confidence selection](#high-confidence-selection)  
[Tighter ±5 check](#tighter-5-check)  
[Reproduce the reference](#reproduce-the-reference)

## Whole-rally recovery

Each cell is fully correct rallies at **±10 / ±5**.

| Detector | Trusted GT | All GT |
|---|---:|---:|
| Previous model | 995 / 901 | 993 / 900 |
| Serve repair | 1,105 / 1,001 | 1,103 / 1,000 |
| Score possible sequences | 1,435 / 1,224 | 1,433 / 1,223 |
| + one missed later contact | 1,597 / 1,327 | 1,596 / 1,326 |
| + independent added-contact evaluation | 1,622 / 1,350 | 1,621 / 1,349 |
| Rally start/end correction only | 1,732 / 1,404 | 1,732 / 1,403 |
| **Final detector** | **1,763 / 1,430** | **1,763 / 1,429** |
| Wider serve shortlist | 1,767 / 1,425 | 1,767 / 1,424 |

## Final contact and rally-start performance

Metric cells are **precision / recall / F1**.

| Task | Trusted GT | All GT |
|---|---:|---:|
| All contacts | **81.0 / 88.2 / 84.5%** | **90.1 / 86.9 / 88.4%** |
| Contacts + correct player | **78.5 / 85.5 / 81.8%** | **87.2 / 84.0 / 85.6%** |
| Start is the serve | **70.4 / 76.7 / 73.4%** | **72.1 / 67.7 / 69.8%** |
| Start + correct server | **68.1 / 74.1 / 71.0%** | **68.5 / 64.4 / 66.4%** |

Recall by labelled contact type:

| Contact type | Trusted GT: timing / + player | All GT: timing / + player |
|---|---:|---:|
| Serve | **81.3% / 77.4%** | **72.0% / 67.3%** |
| Non-serve | **88.9% / 86.3%** | **88.4% / 85.7%** |

## High-confidence selection

A fully correct clip has the whole rally, every contact, and every player right. Whole-rally discovery ignores local contact mistakes inside an otherwise correct clip.

| Selection task | Trusted GT | All GT |
|---|---:|---:|
| Exact annotation precision | **616 / 740 = 83.2%** | **615 / 784 = 78.4%** |
| Exact annotation recall | **616 / 3,422 = 18.0%** | **615 / 3,965 = 15.5%** |
| Exact annotation F1 | **29.6%** | **25.9%** |
| Whole-rally discovery precision | **728 / 740 = 98.4%** | **739 / 784 = 94.3%** |
| Whole-rally discovery recall | **728 / 3,422 = 21.3%** | **739 / 3,965 = 18.6%** |
| Whole-rally discovery F1 | **35.0%** | **31.1%** |

Counts at ±10:

| Labels | Fully correct | Wrong | Unknown | Contains one whole rally |
|---|---:|---:|---:|---:|
| Trusted GT | 616 | 124 | 44 | 728 |
| All GT | 615 | 140 | 29 | 739 |

Unknown clips get no credit in the conservative all-GT precision.

## Tighter ±5 check

<details>
<summary>Show ±5 results</summary>

### Contacts and rally starts

| Task | Trusted GT | All GT |
|---|---:|---:|
| All contacts | 79.3 / 86.3 / 82.6% | 88.1 / 84.9 / 86.5% |
| Contacts + correct player | 76.9 / 83.7 / 80.2% | 85.5 / 82.4 / 83.9% |
| Start is the serve | 60.0 / 65.3 / 62.5% | 61.0 / 57.3 / 59.1% |
| Start + correct server | 58.2 / 63.3 / 60.6% | 58.5 / 54.9 / 56.6% |

### Recall by labelled contact type

| Contact type | Trusted GT: timing / + player | All GT: timing / + player |
|---|---:|---:|
| Serve | 69.3% / 66.1% | 61.0% / 57.5% |
| Non-serve | 87.9% / 85.5% | 87.4% / 84.9% |

### Selected clips

| Labels | Fully correct | Wrong | Unknown | Contains one whole rally |
|---|---:|---:|---:|---:|
| Trusted GT | 549 | 191 | 44 | 728 |
| All GT | 549 | 207 | 28 | 739 |

</details>

## Reproduce the reference

From the repository root, with the original ShuttleSet22 annotations:

```bash
PYTHONPATH="$PWD/src:$PWD" ~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_closing_pass.scripts.summarise_metrics \
  --annotations /path/to/ShuttleSet22
```

The script rebuilds the counts and figures from saved predictions, checks the trusted-GT results against the saved experiments, and writes `results/metric_summary.json.gz`. It does **not** retrain models or rerun vision.

Clip review notes: `results/selected_clip_review.csv`.
