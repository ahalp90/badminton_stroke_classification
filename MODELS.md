# Model Artifacts: Storage and Retrieval

How model weights, training arrays, and TensorBoard logs are made available to
the next team, and why. See `HANDOVER.md` for the project-wide handover.

> **Status — Tier 1 published.** The 6 BST-X run-time weights are tracked in
> git AND uploaded as assets on the `models-v1` GitHub Release (which carries
> 14 assets total: the 6 BST-X plus 8 BRIC weights uploaded alongside).
> `scripts/fetch-models.sh` works against it. The Tier 2 training archive
> already lives on shared institutional storage (which may not persist
> indefinitely; a maintainer backup is the fallback, see Tier 2 below). The
> demo is unaffected either way: it runs from the precomputed predictions
> already in the repo.

## The decision (two tiers)

| Tier | What | Size | Where it goes | How to get it |
| --- | --- | --- | --- | --- |
| 1. Run-time weights | The 6 BST-X `*.pt` the registry serves | ~43 MB | **GitHub Release** asset | `scripts/fetch-models.sh` |
| 2. Training archive | All historical `*.pt` (49), logit `*.npz` (63), TB logs (466) | ~440 MB | **Institutional / HPC bulk storage** | manifest + manual copy |

Neither tier is needed to run the demo. Results screens are driven by the
**precomputed predictions that are already committed** (~11 MB; see
`HANDOVER.md`). You only need Tier 1 for live inference on new uploads, and
Tier 2 only to reproduce or retrain.

## Why not the obvious alternatives

- **Leave them in git (status quo).** Every clone drags ~440 MB and `.git`
  keeps growing as new runs land. This is the problem we are fixing.
- **Git LFS.** Tempting, but GitHub's free LFS tier is ~1 GB storage and
  ~1 GB/month bandwidth; 440 MB plus repeated clones blows it quickly, it adds a
  client dependency (`git lfs`), and actually shrinking the existing `.git`
  still needs a history rewrite. Not worth it for a capstone handover.
- **One giant zip on a drive link.** Opaque, no per-file integrity, link rot.

GitHub Releases win for Tier 1: free, up to 2 GB per asset, no LFS quota, no
history rewrite, and a tiny fetch script keyed off the registry. The bulky
historical archive (Tier 2) is reproducibility material, so it belongs in the
institutional storage the team already has (the HPC scratch it was trained on,
or university cloud), referenced by a manifest.

## Tier 1: publish the run-time weights (one-time, current team) — DONE

> Already published to the `models-v1` release via `scripts/publish-models.sh`.
> The steps below are kept for reference / re-publishing a new tag. The manifest's
> `sha256` column is filled, so fetches are integrity-verified.

The deployed weights are listed in `scripts/model_manifest.tsv` (generated from
each registry entry's `weights_path`). It currently holds the **6 BST-X**
weights that exist on this server. BRIC's `weights_path` is declared in the
registry but the file was never committed and is not on the server; BRIC serves
precomputed predictions, so the demo does not need it. The manifest documents
this in a comment, with how to add BRIC back if live BRIC inference is wanted.

1. Create a release tag, e.g. `models-v1`.
2. Upload each file in the manifest's first column as a release asset, named by
   the second column (`asset_name`). For example with the GitHub CLI:
   ```bash
   gh release create models-v1 --title "Model weights v1" --notes "Deployed weights for the registry."
   while IFS=$'\t' read -r dest asset sha; do
     [ -z "$dest" ] || [ "${dest#\#}" != "$dest" ] && continue
     gh release upload models-v1 "$dest#$asset"   # uploads dest under the asset name
   done < scripts/model_manifest.tsv
   ```
3. (Recommended) fill the `sha256` column in the manifest so fetches are
   verified: `sha256sum <file>` for each, paste the hash into column 3.
4. Commit the updated manifest.

## Tier 1: fetch the run-time weights (next team)

```bash
MODELS_BASE_URL=https://github.com/ahalp90/badminton_stroke_classification/releases/download/models-v1 \
  ./scripts/fetch-models.sh
```
The script reads the manifest and drops each weight back into its registry
`weights_path`. Skips files already present; `--force` re-downloads.

## Tier 2: the training archive

The full set of historical checkpoints, `*.npz` logit dumps, and TB event logs
is for reproducibility and retraining only.

- **Location.** A full copy (~440 MB total: 49 `*.pt`, 63 `*.npz`, and 466
  TensorBoard event files) already lives on shared institutional storage. That store may not persist
  indefinitely, so treat it as the primary copy but not guaranteed permanent.
- **Fallback.** If the archive is no longer present on the shared store, open an
  issue on the repo or reach Curtis Martin on GitHub (@curtislmartin), who keeps
  a backup and can provide a copy or access.
- TODO (next owner): record the exact shared-storage path + access here, and keep
  a manifest (path + size + sha256) alongside it so completeness can be verified.

## Stop the bleed (this branch)

`.gitignore` now excludes new TB logs, `*_serial_*.npz`, and `weights/*.pt`
going forward, so fresh training runs do not re-bloat the repo. This does **not**
remove the copies already in history.

To actually reclaim the ~440 MB already in `.git`, a maintainer must rewrite
history once (coordinate with the team first, since it changes commit hashes):
```bash
# example, review before running
git filter-repo --strip-blobs-bigger-than 1M
```
TODO (team): decide whether to do this now or leave history as-is until the
project is archived.
