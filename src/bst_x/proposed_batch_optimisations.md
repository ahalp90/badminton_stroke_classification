# BST-X Training Efficiency Assessment

## Context

Reviewed the full training pipeline — `bst_x_train.py`, `shuttleset_dataset.py`, `bst.py`, `tempose.py` — to find inefficient memory loading, batching, or compute patterns that could be fixed without accuracy degradation and with minimal code change.

## Verdict: Already essentially optimal for this scale

The pipeline is well-designed. The key architectural choice — pre-collating all data into large numpy arrays loaded entirely into RAM at init (`Dataset_npy_collated`) — eliminates the #1 training bottleneck (I/O). With that in place:

- **`__getitem__` is trivial**: just numpy array indexing (`self.human_pose[i]`), no disk reads, no transforms
- **Model is small**: d_model=100, seq_len=100, 3 transformer layers, ~25k training samples
- **Batch size is reasonable**: 128, producing ~201 batches/epoch

The items that optimization checklists typically flag are **negligible at this scale**:

| Commonly flagged item | Why it doesn't matter here |
|---|---|
| `num_workers=0` | `__getitem__` is O(1) numpy indexing. Subprocess IPC overhead would likely cancel any benefit. Data is already in RAM. |
| `loss.item()` GPU sync | ~201 calls/epoch, each microseconds. Total: <1ms/epoch. |
| `.cpu()` in validation loop | 4 tiny tensors (n_classes,) per batch, ~33 batches. Negligible. |
| `zero_grad()` vs `set_to_none=True` | Saves fraction of a ms for this model size. |
| `RandomTranslation_batch` numpy->torch | Creates 256 floats. Negligible. |
| Flash attention / `scaled_dot_product_attention` | seq_len=100 means attention matrices are ~10K elements — fits in L2 cache. Flash attention shines at seq_len>512. Refactoring two attention classes is not "minimal change" either. |
| No mixed precision (AMP) | Could help but carries accuracy risk for small models, and isn't a trivial change. |

## One optimization worth doing: `torch.compile()`

If running PyTorch 2.0+, wrapping the model in `torch.compile()` fuses kernels and optimizes the graph automatically. This is:
- **1 line of code**
- **10-30% speedup** typical for transformer models
- **Zero accuracy impact** (mathematically identical)
- Multiplied across 5 serial runs, the time savings add up

### Change (in `bst_x_train.py`)

In `train_network()`, after model is passed in but before training begins (~line 241):

```python
# Add after line 241 (writer = SummaryWriter())
if hasattr(torch, 'compile'):  # PyTorch 2.0+ guard
    model = torch.compile(model)
```

That's it. The guard ensures it's a no-op on older PyTorch.

### Files to modify
- `bst_x_train.py` (1 line added)

## Optional micro-optimization: `set_to_none=True`

Trivial change, marginal benefit, but free:

```python
# Line 106: change
optimizer.zero_grad()
# to
optimizer.zero_grad(set_to_none=True)
```

Deallocates gradient tensors instead of zeroing them. Saves a tiny amount of memory and time.

## Verification

1. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. Run a training experiment and compare epoch times with/without `torch.compile`
3. Verify F1 scores are unchanged