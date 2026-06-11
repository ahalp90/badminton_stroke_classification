# eval_dump_predictions.py is retired

Folded into `bst_x_infer.py --fe` in the taxon_pinned_w_preds refactor (Step D10).
The old `.pt` dump (`predictions/serial_<n>.pt` with `y_true` / `y_pred` /
`active_class_list`) is gone; the replacement writes the same npz schema
`bst_x_train` emits at end-of-serial.

Dump predictions for an existing run:

```
PYTHONPATH=src/bst_x:src/bst_x/stroke_classification \
    python -m main_on_shuttleset.bst_x_infer --fe \
        --run-dir .../experiments/run_<id> --serial 5 \
        --fe-output-dir /some/dump/root --splits test
```

Output: `<dump-root>/<run_id>/inference_runs/<timestamp>/test_serial_5.npz` (plus an
`inference_manifest.yaml`) with `logits, y_true, y_pred_top1, topk_idx, clip_stems,
class_list, run_id, serial_no, taxonomy_name`.

`confusion_matrix.py` already reads that npz (`y_pred_top1` for argmax preds,
`class_list` for the labels).
