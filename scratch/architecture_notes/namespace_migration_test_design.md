# Namespace migration test suite: design

2026-06-11. Test design for the BST to BST-X rebrand (env vars, module renames, the `src/bst_x/` move, and the Step 6b `BST_X` switchover with its weight-file rename). Design only; no test code in this pass.

Sources: `/home/ariel/Documents/COSC594/bst_x_rebrand.md` (assessment, bugs #A-#T, safety review, Round 2 lockstep list, the "Updates from chat 2026-06-11" overlay, and the 2026-06-11 consistency-pass section at its tail), `/home/ariel/Documents/COSC594/rebrand_next_session.md` (Steps 0-10), and the decisions locked in the 2026-06-11 planning chat. Updated in the 2026-06-11 consistency pass: decisions 4-5 revised (Chang baseline lowercases in place; `scripts/model_manifest.tsv` joins the lockstep), T6/T10/T11/H1 amended, open questions 3-4 resolved; change log at the tail of this doc.

Locked decisions this suite verifies:

1. `MODELS` dict in `bst_x_common.py` gains `'BST_X': BST_CG_AP` as an alias; the five Chang keys stay.
2. Default `model_name` flips `'BST_CG_AP'` to `'BST_X'` at `bst_x_train.py:942,1394` and `bst_x_infer.py:99,175,317`.
3. The case-mix branch at `bst_x_train.py:979-985` becomes `save_name = self.model_name.lower()`; new runs write `bst_x_*.pt`.
4. Every `bst_CG_AP_*.pt` under `experiments/run_*/weights/` is renamed to `bst_x_*.pt`. The Chang baseline at `experiments/bst_cg_ap_base_17_04_2026/weights/` keeps its three files but lowercases their prefix in place (`bst_CG_AP_` → `bst_cg_ap_`), with its manifest and .txt notes following, so the new save-name rule resumes it. (Revised 2026-06-11; was "its three files stay".)
5. Manifest `weights_path` lines, `docs/models_registry.yaml`, `scripts/model_manifest.tsv` `dest_path` basenames (the models-v1 release assets keep their pre-rebrand names; only the tsv's dest column moves), and the `src/api/bst_x_inference.py:53` literal update in lockstep.
6. `fe_jsons/*.json.gz` and `predictions/*.npz` carry no model_name strings; untouched.
7. TB logs untouched.
8. venv renames to `venv-bst-x` (out-of-repo ops).
9. The aim repo is deleted and remade (out-of-repo ops).

## Table of contents

- [Ground rules](#ground-rules)
- [Repo ground truth (captured 2026-06-11)](#repo-ground-truth-captured-2026-06-11)
- [Shared test scaffolding](#shared-test-scaffolding)
- [Code-contract tests](#code-contract-tests)
  - [T1: MODELS alias integrity](#t1-models-alias-integrity)
  - [T2: default model-name flip](#t2-default-model-name-flip)
  - [T3: weight save-name round trip](#t3-weight-save-name-round-trip)
  - [T4: env-var legacy fallback](#t4-env-var-legacy-fallback)
  - [T5: architecture wire format](#t5-architecture-wire-format)
- [Artefact-integrity tests](#artefact-integrity-tests)
  - [T6: registry lockstep](#t6-registry-lockstep)
  - [T7: live-inference weight literal](#t7-live-inference-weight-literal)
  - [T8: sidecar schema invariants](#t8-sidecar-schema-invariants)
  - [T9: Chang KEEP set, code identifiers](#t9-chang-keep-set-code-identifiers)
  - [T10: Chang baseline run untouched](#t10-chang-baseline-run-untouched)
- [Tree-hygiene tests](#tree-hygiene-tests)
  - [T11: staged orphan-string scan](#t11-staged-orphan-string-scan)
  - [T12: subprocess and dynamic module strings resolve](#t12-subprocess-and-dynamic-module-strings-resolve)
- [Harness](#harness)
  - [H1: artefact inventory capture and verify](#h1-artefact-inventory-capture-and-verify)
- [Landing order](#landing-order)
- [Disposition of the sketched test list](#disposition-of-the-sketched-test-list)
- [What the existing backstop already covers](#what-the-existing-backstop-already-covers)
- [Open questions](#open-questions)
- [Change log (2026-06-11 consistency pass)](#change-log-2026-06-11-consistency-pass)

## Ground rules

**Environment.** Laptop `~/.venvs/badminton-cicd`: CPU-only, full model stack, pytest, fastapi TestClient. No GPU, no HPC. Everything below must run there.

**Two homes.** Pytest tests live in `tests/test_namespace_migration.py` and run in CI (`.github/workflows/ci.yml` runs the whole suite on every push). The pre/post comparison harness lives in `scratch/rebrand_smoke/` as a plain script, mirroring the `scratch/post_tidy_smoke/` precedent, because it holds state across two tree states and needs the local artefact superset.

**Tracked-artefact rule.** Pytest asserts only against git-tracked artefacts (64 run manifests, 64 tracked weight `.pt`, 165 sidecar `.json.gz`, 111 prediction `.npz`, the registry YAML). A fresh CI clone has all of these. The 122 untracked local-only weight files are the harness's job, not pytest's.

**Self-gating.** The migration lands over many commits. Each pytest test either holds on today's main or gates on a sentinel that detects its step has landed (`pytest.skip` until then). The gate must be a different signal from the asserted property wherever possible, so a botched step can't gate its own verification off. Where the step IS the contract (T5), the test lands in that step's commit instead of gating.

**Fail loud.** Where an assert can be exact (a key set, a filename, a partial's flag table), pin it exactly. If an exact assert trips on a legitimate future change, the fix is to update the pin in the same commit, not to loosen the assert.

## Repo ground truth (captured 2026-06-11)

Numbers the specs below rely on. Re-verify any that look stale at implementation time.

| Fact | Value |
|---|---|
| Experiment dirs under `main_on_shuttleset/experiments/` | 66 (64 `run_*` + `bst_cg_ap_base_17_04_2026` + `aug_hparam_sweep`) |
| Dirs with a `manifest.yaml` | 64 (`run_20260503_063338` and `aug_hparam_sweep` have none) |
| `weights_path` entries across manifests | 312; 137 resolve on disk, 175 pruned (prune-to-best) |
| Weight `.pt` files on disk under `*/weights/` | 186, all prefixed `bst_CG_AP_`; 64 git-tracked |
| Chang baseline weights | exactly 3 files; the serial-1 file has NO serial suffix (`..._merged_25.pt`) |
| Chang baseline manifest path convention | `experiments/bst_cg_ap_base_.../weights/...` (relative to `main_on_shuttleset/`), unlike run manifests which are repo-root-relative `src/bst_x/...` |
| `fe_jsons/` dirs | 33; every one holds exactly `clip_index.json.gz`, `test.json.gz`, `val.json.gz`, `perclass_stats_test.json.gz`, `perclass_stats_val.json.gz` |
| Prediction `.npz` | 111, schema uniform across all |
| Registry entries | 6 `architecture: bst-x` + 1 `bric` |
| Files containing `bst_refactor` (py/md/yaml/toml, outside excluded zones) | 168 |
| `notebooks/clips_master.csv` | git-tracked; carries `split_bst_baseline` |
| `aug_hparam_sweep/` | sweep state only (config + state.json + manifest.md); no weights, no manifest.yaml; outside the rename's weight glob by construction |

## Shared test scaffolding

Top of `tests/test_namespace_migration.py`:

- A module-resolution helper: `_first_importable(*names)` returns the first module that imports from an ordered candidate list, e.g. `('main_on_shuttleset.bst_x_common', 'main_on_shuttleset.bst_x_common')`. Every test that touches a renamed module goes through it, so the suite collects green on every commit between Step 0 and Step 8. Raise (don't skip) if neither imports: that state is a broken tree, not a not-yet state.
- The switchover sentinel: `SWITCHOVER_LANDED = 'BST_X' in MODELS`. Tests gated on Step 6b skip with a clear message until it's true.
- `REPO_ROOT = Path(__file__).resolve().parents[1]` and `EXPERIMENTS = REPO_ROOT / 'src' / <pkg> / 'stroke_classification' / 'main_on_shuttleset' / 'experiments'` where `<pkg>` is whichever of `bst_x` / `bst_refactor` exists (same first-existing pattern as the import helper).

Imports of `pipeline.*` / `main_on_shuttleset.*` / `model.*` resolve through `conftest.py`'s sys.path inserts, same as every existing test. The Step 8 commit updates conftest in lockstep, so this file needs no path logic of its own beyond `<pkg>` detection.

## Code-contract tests

### T1: MODELS alias integrity

- **Purpose / bug class.** Alias drift: `BST_X` missing from `MODELS`, pointed at the wrong partial, or silently diverging from `BST_CG_AP` after a later edit. Also catches constructor breakage in any of the six keys.
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing, post-switchover. Gate: skip unless `SWITCHOVER_LANDED`.
- **Spec.**
  - Assert `set(MODELS) == {'BST_0', 'BST', 'BST_CG', 'BST_AP', 'BST_CG_AP', 'BST_X'}`. Exactly six: catches both a missed alias and an accidental drop of a Chang key.
  - Assert `MODELS['BST_X'] is MODELS['BST_CG_AP']`. The locked decision is alias-by-identity, not a re-declared equivalent partial.
  - For every key: build via the shared builder (`build_bst_x_network` or its renamed successor, resolved through the import helper) with `n_joints=17, pose_style='JnB_bone', in_channels=2, n_class=14, seq_len=100, device='cpu'`. Assert the return is an `nn.Module` and `sum(p.numel() for p in net.parameters()) > 0`. This exercises the full dispatch path, not just dict membership.
  - Behavioural backstop for the alias: `torch.manual_seed(0)`, build `'BST_X'`; `torch.manual_seed(0)`, build `'BST_CG_AP'`; both `.eval()` (dropout off); assert equal param counts AND `torch.equal` on the forward output of one seeded batch (batch 2, the same input-shaping the `test_taxonomy.py` forward smoke uses, including the `human_pose.view(*shape[:-2], -1)` flatten and `set_schedule_factors(1.0, 1.0)`).
- **Dependencies.** torch CPU; the builder; no artefacts.
- **Brittleness.** The `is` assert is stricter than behaviour: if someone later re-declares `BST_X` as its own `partial(BST, use_ppf=True, use_cg=True, use_ap=True)`, identity fails while the forward check passes. That strictness is deliberate; an alias that stops being an alias should be a conscious decision. The forward-identity check relies on identical construction order under a fixed seed, which holds for two builds of the same partial on CPU.

### T2: default model-name flip

- **Purpose / bug class.** A half-flipped default is silent: `'BST_CG_AP'` still resolves through `MODELS`, everything runs, and new runs quietly write old-prefix weight files (or the `--fe` dump builds the right net under the wrong label). Nothing crashes, so only an explicit pin catches it.
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing, post-switchover. Gate: `SWITCHOVER_LANDED`.
- **Spec.**
  - `inspect.signature` checks on importable surfaces: `bst_x_train.Task.get_network_architecture` parameter `model_name` default `== 'BST_X'`; same for `bst_x_infer.Task.get_network_architecture` and `bst_x_infer.dump_run_predictions`.
  - The two non-importable sites (`bst_x_train.py:1394` call literal inside `__main__`; `bst_x_infer.py:317` argparse default) get a source-level check: read each module's `__file__` text and assert the regexes `model_name\s*=\s*'BST_CG_AP'` and `default\s*=\s*'BST_CG_AP'` have zero matches. Comments that mention BST_CG_AP (the optimiser notes at `bst_x_train.py:591-592,637`, the `--model-name` help text lineage) survive because the regexes target assignment/keyword forms only.
  - Positive pin for the argparse site: assert `default='BST_X'` appears in the `bst_x_infer` source within the `--model-name` `add_argument` call (regex over a window around `--model-name`).
- **Dependencies.** Import of the two entry modules (already done by `test_train_surface.py` / `test_inference_smoke.py`, so module-level import is known CPU-safe).
- **Brittleness.** The source regexes couple to single-quote style and `default=` spelling. Low risk in this codebase (consistent quoting), and the failure mode is a false failure at edit time, not a silent pass. If the argparse block is ever refactored into a builder function, replace the regex with a signature check in the same commit.

### T3: weight save-name round trip

- **Purpose / bug class.** Decision 3 rewrites the filename branch. The bug class is writer/loader disagreement: trainer writes one name, resume-by-name looks for another, and every "resume" silently retrains. A revert to the old case-mix branch would map `'BST_X'` to `bst_X_...` while the on-disk rename produced `bst_x_...`: exactly the mismatch this catches.
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing, post-switchover. Gate: `SWITCHOVER_LANDED`.
- **Spec.**
  - Build a tiny net for a 3-class throwaway `Taxonomy` (the `test_train_surface.py` `TAX3` pattern) via the builder with `model_name='BST_X'`, CPU.
  - Compose the expected filename by the new rule: `'bst_x' + '_JnB_bone' + '_' + tax.name + '_2'` (use `serial_no=2` so the serial suffix is deterministic; serial 1 drops the suffix). Save the net's `state_dict` to `tmp_path/weights/<expected>.pt`.
  - Assemble a minimal Task stand-in (`types.SimpleNamespace` carrying `net`, `model_name='BST_X'`, `pose_style='JnB_bone'`, `taxonomy`, `weight_dir=tmp_path/'weights'`, `device='cpu'`) and call the real unbound method: `bt.Task.seek_network_weights(ns, model_info='', serial_no=2)`.
  - Assert it returns `(True, None)` (the load path fired, no training) and `ns.weight_path.name == expected`. The load path firing proves name-resolution agreement end to end through the production code, with zero training.
- **Dependencies.** torch CPU, `tmp_path`, the trainer module. The method body must not touch loaders on the load path (it doesn't today; `train_network` is only reached when the file is absent).
- **Brittleness.** Couples to `seek_network_weights`'s attribute usage; if the method grows a new `self.` dependency, the namespace gets one more field. The `test_train_surface.py` helpers already accept this trade for the same method family.
- **Note for the doc trail, not a test.** The new lowercase rule changes the Chang lowercasings too: `'BST_CG_AP'` now maps to `bst_cg_ap`, not `bst_CG_AP`. Resolved 2026-06-11: the Chang baseline's files lowercase in place at 6b.2 (revised decision 4), so resume-by-name keeps resolving. Open question 4 is closed.

### T4: env-var legacy fallback

- **Purpose / bug class.** Bug #B: a renamed var without a fallback breaks live deploys whose `.env` (bourbaki, engelbart, prod compose) still carries the legacy name. Failure is silent in the worst paths (API falls through to a `None` default and serves empty).
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing through the fallback window; the legacy-only case flips its expectation when the fallback is removed (that commit updates the test).
- **Gate.** Skip unless the rename mapping exists (see dependency below).
- **Spec.**
  - Step 4's implementation shape is LOCKED (2026-06-11 round 2): a single module-level mapping of new name to legacy name (`ENV_VAR_RENAMES = {'BST_X_CLIPS_DIR': 'BST_CLIPS_DIR', ...}`) next to a shared resolve helper in `pipeline/data_access.py`, with a small stdlib-only twin in `src/api/config.py` for the API-side vars (`BST_X_REPO_ROOT`, `BST_X_REGISTRY_PATH`, `BST_X_LOCAL_CLIPS_DIR`, `BST_X_CLIPS_DIR`, `BST_X_INPUTS_DIR`). The test parametrises over the mapping(s), so adding a var to the mapping adds it to the matrix for free. The mapping is branch-lifetime only: the Step 9b cleanup commit deletes it and flips this test's legacy-case expectation in the same diff.
  - Matrix per var, via `monkeypatch.setenv` / `delenv` on the helper directly:
    - legacy set, new unset: resolves to the legacy value AND emits the deprecation signal once.
    - new set, legacy unset: resolves to the new value, no signal.
    - both set (different values): new wins.
    - neither: default passes through (the helper's default arg, or `None` for the `_or_none` flavour).
  - Signal capture: `pytest.warns(DeprecationWarning)` — LOCKED (2026-06-11 round 2): the helper uses `warnings.warn(..., DeprecationWarning)`, because `data_access` consumers don't configure logging and pytest captures warnings natively.
  - Module-import-time reads (`src/api/config.py`, the inputs-dir block in the API inference module): one targeted case each, `importlib.reload` with patched env inside a fixture that reloads the module back to pristine afterwards. Assert only on the reloaded module's own attributes, not on downstream importers; binding propagation is the existing API tests' job.
  - `BST_DATA_DIR` is excluded: its only consumer is `tests/test_integration.py` reading `os.environ` for its own skip logic. Renaming there is a test-internal edit with no production fallback to verify.
- **Dependencies.** monkeypatch, importlib; no artefacts.
- **Brittleness.** Reload-based tests order-couple with `tests/test_api.py`'s module-level `TestClient` (importing `src.api.main` freezes config values at first import). Containing each reload in a restore-fixture and asserting only on the reloaded module keeps the blast radius zero. Note in the test docstring.

### T5: architecture wire format

- **Purpose / bug class.** Bug #A: the Pydantic `Literal` drifting from the registry/FE wire value `"bst-x"`. Catches a one-sided edit (Markup fixed, LibraryPredictRequest missed) and pins `'bst'` as rejected.
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Lands in the Step 1 commit (the test asserts the post-Step-1 contract, so it cannot gate on a separate sentinel: the step is the contract). Standing thereafter.
- **Spec.**
  - For BOTH `src.api.main.Markup` and `src.api.main.LibraryPredictRequest`: extract the `architecture` field's `Literal` values (`typing.get_args` through the `Optional` wrapper; write a small helper). Assert the two models carry the identical value set.
  - Assert `'bric' in values` and `'bst-x' in values`.
  - No back-compat phase (2026-06-11 round 2: `'bst'` drops in the Step 1 commit itself; the FE flips in the same diff and no independent client exists). Assert `'bst' not in values` and `pytest.raises(ValidationError)` on `Markup(architecture='bst')`. No flag machinery.
  - Always: `Markup(architecture='bst-x')` validates; `Markup(architecture='bogus')` raises.
  - Wire-level smoke via the existing `TestClient(app)` pattern: POST `/api/library_predict` with `{"clip_stem": "nonexistent", "architecture": "bst-x"}` asserts `status_code != 422` (404/503 are fine; only request validation is under test). Same body with `"architecture": "nonsense"` asserts 422.
- **Dependencies.** fastapi TestClient (already used by `tests/test_api.py`); pydantic.
- **Brittleness.** None notable; the test pins the final wire format from the Step 1 commit onward.

## Artefact-integrity tests

### T6: registry lockstep

- **Purpose / bug class.** Decision 5's three-way lockstep (weight files, manifests, registry YAML) breaking on the FE-facing surface: a registry entry whose `weights_path` 404s, or whose manifest disagrees about the weight filename. Today nothing loads registry weights at boot (Tier 1 reads metrics only), so this drift is silent until Tier 2 or a teammate pulls a weight by registry path.
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing, both pre- and post-rename (it holds on today's main). The prefix assert gates on `SWITCHOVER_LANDED`.
- **Spec.**
  - Load `docs/models_registry.yaml`. Assert at least 7 entries and every `architecture` value is in `{'bst-x', 'bric'}`.
  - Per entry: `REPO_ROOT / manifest_path` exists; `REPO_ROOT / weights_path` exists.
  - Per `bst-x` entry: parse the manifest, find the serial whose `serial_no` matches the entry; assert one exists; assert `Path(serial['weights_path']).name == Path(entry['weights_path']).name`. This is the manifest/registry filename agreement that the rename script must preserve.
  - Exactly one `bst-x` entry carries `is_default: true` (existing registry convention the FE relies on; cheap to pin here since the file is already parsed).
  - `scripts/model_manifest.tsv` agreement (added 2026-06-11, bug #T): every non-comment row's `dest_path` exists on disk and equals some registry entry's `weights_path`; `sha256` is 64 hex chars. Don't assert the `asset_name` format — release assets keep their pre-rebrand names by design.
  - Post-switchover only: every `bst-x` entry's weight filename starts `'bst_x_'`, and the string `'bst_CG_AP'` does not appear anywhere in the registry file. The tsv's `dest_path` basenames start `'bst_x_'` too (the `asset_name` column is exempt — frozen at the old names).
- **Dependencies.** PyYAML; tracked artefacts only (all six referenced weights are git-tracked best serials).
- **Brittleness.** Pinning "at least 7 entries" rather than exactly 7 tolerates new model registrations mid-migration. The filename-agreement assert assumes run manifests stay repo-root-relative in `weights_path`; it compares basenames only, so it survives the Step 8 prefix rewrite either way.

### T7: live-inference weight literal

- **Purpose / bug class.** The hardcoded checkpoint path at `src/api/bst_x_inference.py:53` (decision 5's third edge). The model loads lazily at first predict, so an import-green API can still be a dead live-inference path; no existing test asserts the file exists.
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing, both. Prefix assert gates on `SWITCHOVER_LANDED`.
- **Spec.**
  - Import the API inference module through the resolution helper (`src.api.bst_x_inference` then `src.api.bst_x_inference`).
  - Assert `module.RUN_DIR.is_dir()` and `module.WEIGHTS_PATH.is_file()`.
  - Post-switchover: `module.WEIGHTS_PATH.name.startswith('bst_x_')`.
  - Do NOT load the checkpoint; existence is the contract under test, and the forward pass needs tensors this environment doesn't mount.
- **Dependencies.** The weight at `run_20260505_154907/weights/` is git-tracked, so the assert holds on a fresh clone and in CI.
- **Brittleness.** If the live-inference run is ever re-pointed at a new run dir, the test follows automatically (it reads the module constants, not pinned strings).

### T8: sidecar schema invariants

- **Purpose / bug class.** Decision 6's safety claim, made standing: the FE sidecars and prediction dumps carry no model-name strings, so the rename cannot touch them; and their schemas stay exactly what the API and FE consume. Catches a rename script that "helpfully" rewrites sidecars, and any future writer drift.
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing, both (identical expectations pre and post; the files must not change at all, which H1 enforces byte-level). Lifecycle (2026-06-11 round 3): full-corpus through the migration branch only — the Step 9b cleanup commit slims the walk to the six registry-anchored run dirs, keeping the FE schema pin (the gzipped fe_jsons shape the API serves) at a few seconds' cost. Fresh-dump npz schema stays covered by `test_inference_smoke.py`.
- **Spec.** Walk `EXPERIMENTS/*/fe_jsons/` and `EXPERIMENTS/*/predictions/*.npz` (this exact scope; the legacy run-root `clip_index.json` of `run_20260505_154907` is a different, pre-fe_jsons artefact and stays out of scope).
  - Every `fe_jsons/` dir holds exactly the five files: `clip_index.json.gz`, `test.json.gz`, `val.json.gz`, `perclass_stats_test.json.gz`, `perclass_stats_val.json.gz`.
  - `test.json.gz` / `val.json.gz`: top-level keys exactly `{run_id, serial_no, split, class_list, clips}`; every element of `clips` has keys exactly `{clip_stem, softmax, top_k_idx, top_k_prob, y_pred, y_true}`; `run_id` equals the run dir name; `split` matches the filename.
  - `perclass_stats_*.json.gz`: top-level keys exactly `{class_list, n_clips, per_class, split}`; every `per_class` value has keys exactly `{f1, precision, recall, support_pred, support_true, top5_when_pred, top5_when_true}`.
  - `clip_index.json.gz`: top-level `{clips}`; every entry has keys exactly `{ball_round, match, player_side, rally, raw_type_en, set_id, split, video_path}`.
  - Every `.npz` (`allow_pickle=True`; `clip_stems`/`class_list` are object arrays): field set exactly `{logits, y_true, y_pred_top1, topk_idx, clip_stems, class_list, run_id, serial_no, taxonomy_name}`.
  - The string `'model_name'` appears as a key at no level walked above.
  - Anchor assert so an empty walk can't pass vacuously: every `bst-x` registry entry's run dir has a complete five-file `fe_jsons/` set.
- **Dependencies.** gzip/json/numpy; tracked artefacts. The 2026-06-11 scan confirmed all 33 dirs and all 111 npz already conform with zero exceptions, so the exact sets are safe to pin.
- **Brittleness.** Runtime: parsing 66 split jsons (4-15 MB each) plus 111 npz headers runs in the tens of seconds. Acceptable as-is; if it ever annoys, sample `clips` per file instead of full-walking, but keep the full key-absence scan (it's cheap once parsed). If a legitimate schema addition lands later (e.g. a new per-clip field), update the pin and regenerate stale sidecars with `build_fe_stats_jsons.py` rather than loosening to subset asserts.

### T9: Chang KEEP set, code identifiers

- **Purpose / bug class.** An over-eager sed in Steps 6/8/6b renaming the do-not-touch identifiers: Chang's class and variant partials, the `MODELS` Chang keys, the taxonomy constants, the split column. Each is load-bearing for on-disk data or weight dispatch (bugs #J, #K, #L).
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing, both; holds on today's main. No gate.
- **Spec.** Import-based asserts, not greps (imports are formatting-proof and catch semantic breakage):
  - `from model.bst import BST, BST_0, BST_PPF, BST_CG, BST_AP, BST_CG_AP`. Assert each variant is a `functools.partial` with `.func is BST` and `.keywords` exactly matching the pinned flag table: `BST_0` (False, False, False), `BST_PPF` (True, False, False), `BST_CG` (True, True, False), `BST_AP` (True, False, True), `BST_CG_AP` (True, True, True) for `(use_ppf, use_cg, use_ap)`.
  - The five Chang `MODELS` keys map to those exact objects: `MODELS['BST'] is BST_PPF`, `MODELS['BST_0'] is BST_0`, etc. (the `'BST'`-to-`BST_PPF` mapping is the one a mechanical rename is most likely to mangle).
  - `from pipeline.config import TAXONOMY_BST_25, TAXONOMY_BST_24, TAXONOMY_BST_12, resolve_taxonomy`; assert `.name` values `'bst_25'`/`'bst_24'`/`'bst_12'`, `n_classes` 25/24/12, and `resolve_taxonomy('bst_25') is TAXONOMY_BST_25`.
  - `from shared.dataset import SPLITS_BST_BASELINE`; assert it is a dict with key set `{'train', 'val', 'test'}` (confirm exact keys at implementation) and non-empty list values.
  - `pd.read_csv(REPO_ROOT / 'notebooks' / 'clips_master.csv', nrows=0)`; assert `'split_bst_baseline'` and `'split_v2'` in columns. Header-only read keeps it fast.
  - One grep-shaped assert, kept because it's prose with no import surface: the first line of `model/bst.py` contains `Original BST by Jing-Yuan Chang` (attribution is part of the KEEP contract).
- **Dependencies.** pandas (in the venv); conftest sys.path.
- **Brittleness.** Minimal. The flag table duplicates `bst.py`'s five lines, which is the point: two independent statements of the variant contract.

### T10: Chang baseline run untouched

- **Purpose / bug class.** Decision 4 violated: a blanket rename script taking the baseline to `bst_x_*`, a glob that doesn't know the no-serial filename dropping one of the three files, or the in-place lowercase (revised decision 4) going missing or half-applied (files renamed, manifest not).
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing, both; holds today. The expected triple flips at the 6b.2 commit. Gate the post-rename expectation on a different signal from the asserted property, per the self-gating rule: `any(EXPERIMENTS.glob('run_*/weights/bst_x_*.pt'))` — the run-dir rename lands in the same 6b.2 commit as the baseline lowercase.
- **Spec.**
  - `BASE = EXPERIMENTS / 'bst_cg_ap_base_17_04_2026'`. Assert the weights dir contains exactly three files, pinned as literals for the current state:
    - Gate signal false (pre-6b.2): `bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_25.pt` / `..._merged_25_2.pt` / `..._merged_25_3.pt` (mixed case).
    - Gate signal true (post-6b.2): the same three names with the prefix lowercased to `bst_cg_ap_`. Also assert NO `bst_x_*.pt` in the dir — the baseline must never take the project prefix.
  - Pin literals, not a glob: a `bst_CG_AP_*_merged_25_*.pt` glob misses the serial-1 file (no `_<n>` suffix). A rename script written off that glob would miss one of the three.
  - Parse `BASE / 'manifest.yaml'`; assert the three serials' `weights_path` basenames equal the currently-expected triple (the manifest lowercases in the same 6b.2 commit — this catches the files-renamed-manifest-missed half-application). Note this manifest's paths are `experiments/`-relative (not `src/...`-relative); compare basenames so the test is convention-agnostic.
- **Dependencies.** Tracked artefacts (all three .pt are tracked).
- **Brittleness.** Minimal; the dir is otherwise frozen, and the single allowed transition (the 6b.2 lowercase) is exactly what the gate models.

## Tree-hygiene tests

### T11: staged orphan-string scan

- **Purpose / bug class.** Stragglers after each rename stage: dead module paths in docstrings, env-var names in error messages and UI strings, stale extras-group names. Individually cosmetic, collectively how the next person loses an afternoon (the bug #C/#D documentation tail).
- **Location.** `tests/test_namespace_migration.py`, one parametrised test over a STAGES table.
- **Runs.** Each stage skips until its sentinel lands, then becomes a standing zero-hits assert.
- **Spec.**
  - Corpus: `git ls-files` filtered to text extensions (`.py .md .yaml .yml .toml .sh .ipynb .txt .tsv .jsx .js .env.example .gitignore` plus the docker-compose files). Tracked-only keeps CI deterministic and skips venvs/caches by construction. Scan `.ipynb` as plain text (cell JSON greps fine). `.tsv` earned its place on 2026-06-11: `scripts/model_manifest.tsv` is where every earlier review pass missed the release pipeline (bug #T).
  - Global allowlist: `scratch/project_history/**`, `local_scratch/**` (paper transcript; mostly untracked anyway). No other standing exclusions: the rebrand docs live outside the repo, so they need no carve-out.
  - STAGES table (sentinel; pattern; extra allowlist; notes):
    1. Sentinel: `src/bst_x` does not exist AND `src/bst_x` (or the chosen name) does. Pattern: `bst_refactor`. Today's baseline is 168 files, so this skips until Step 8 lands.
    2. Sentinel: `bst_x_common` importable. Patterns: `main_on_shuttleset\.bst_(train|infer|common)\b` and `\bbuild_bst_network\b`.
    3. Sentinel: `docker-compose.dev.yml` mentions `bst_x_inputs`. Pattern: `\bbst_inputs\b` (widened 2026-06-11 from `scratch/bst_inputs\b` — the container target `/app/bst_inputs` and the code fallbacks at `bst_x_inference.py:66-70` rename too, and the narrow pattern would miss stragglers there).
    4. Sentinel: `pyproject.toml` defines `bst-x-runtime`. Pattern: `\bbst-runtime\b`. No allowlist — the alias group was dropped (2026-06-11), so post-Step-5 the pattern should hit nothing.
    5. Sentinel: the Step 4 fallback mapping has been REMOVED (mapping dict absent — the Step 9b end-of-branch cleanup commit). Pattern: `\bBST_(CLIPS_DIR|CLIPS_CSV|SHUTTLE_NPY_DIR|MMPOSE_NPY_DIR|INPUTS_DIR|DATA_DIR|LOCAL_CLIPS_DIR|REPO_ROOT|REGISTRY_PATH|SHUTTLE_CSV_DIR)\b`. The `\b` after `BST_` family names keeps `BST_X_CLIPS_DIR` from matching.
    6. Sentinel: `SWITCHOVER_LANDED`. Pattern: `bst_CG_AP` restricted to `docs/**`, `src/api/**`, `scripts/model_manifest.tsv` dest_path column, manifest files under `experiments/run_*/`, AND the Chang baseline dir (its files + manifest lowercase at 6b.2, so a mixed-case hit there is a missed rename; the tsv's frozen `asset_name` column is the one deliberate survivor — exempt it by column, not by file). Allowlist starts empty; the two trainer lines that used to carry lowercase `bst_CG_AP` examples (`bst_x_train.py:965,979`) are edited or deleted in 6b.1. Scope this stage narrowly rather than repo-wide: scratch history and the runs ledger legitimately mention the old filenames.
    7. Sentinel: none in-repo (the venv rename is host ops). Pattern: `venv-bst(?!-x)`. Gate on an env flag (`RENAME_SCAN_VENV=1`) rather than auto-detect; run it manually after decision 8's ops step. The lookahead matters: `\b` alone would match inside `venv-bst-x`.
  - On failure, report `file:line:match` for every hit; the test message is the work list.
- **Dependencies.** git CLI (available in CI); stdlib re.
- **Brittleness.** Tracked-only scanning misses untracked scratch scripts (today: the two untracked `presentation_prep` files carry `bst_refactor` paths). Accept the gap in pytest; H1's local run and a pre-Step-8 manual `grep -r` over `scratch/` cover it. Stage 6's allowlist needs care at implementation: start narrow, widen only on confirmed-legitimate hits.

### T12: subprocess and dynamic module strings resolve

- **Purpose / bug class.** Bug #C's dynamic sites: pytest collection import-checks every static import, but the subprocess `-m` strings in `collation_runner.py:45` and `hparam_sweep.py:463` and the `importlib.import_module` literal in the verify-target script fail only when a sweep launches on HPC, mid-batch. Cheapest possible canary.
- **Location.** `tests/test_namespace_migration.py`.
- **Runs.** Standing, both; rename-agnostic (it validates whatever string is currently in the source, so it holds today and after Step 6). No gate.
- **Spec.**
  - Resolve the two runner modules' file paths through the import helper (importing them is fine; both are import-safe).
  - Regex-extract every module literal following a `'-m'` element: `'-m',\s*'([\w\.]+)'`. Assert at least one capture per file (guards against the regex rotting silently).
  - For each capture: `importlib.util.find_spec(capture) is not None`. find_spec imports the parent package only, not the target module, so it stays cheap and side-effect-light.
  - Same treatment for the dynamic-import literal in `verify_bst_train_target.py` (or its renamed successor): extract `import_module\('([\w\.]+)'\)` captures from source, find_spec each.
- **Dependencies.** None beyond the package tree.
- **Brittleness.** Couples to single-quote style in the subprocess lists. The at-least-one-capture assert converts quoting drift into a loud failure instead of a vacuous pass.

## Harness

### H1: artefact inventory capture and verify

- **Purpose / bug class.** The pre/post instrument for the destructive steps (6b weight rename, Step 8 dir move). Catches: a weight file renamed wrongly or dropped; a manifest `weights_path` edited out of sync with its file; content corruption during the move; sidecars or TB logs touched when decisions 6 and 7 say untouched; the local untracked-weight superset that pytest deliberately ignores. This is the sketched "manifest weights_path integrity" test rebuilt around the prune-to-best reality: 175 of 312 manifest entries already point at deleted files on purpose, so "every weights_path resolves" is not a valid invariant. The valid invariant is "the resolution map is preserved under the rename".
- **Location.** `scratch/rebrand_smoke/artefact_inventory.py` plus a short `scratch/rebrand_smoke/README.md` (mirror the `post_tidy_smoke` README shape: what it closes, how to run). Script, not pytest: it holds state across two tree states, needs the untracked superset, and hashes a few GB.
- **Runs.** Capture immediately before Step 6b; verify after Step 6b; re-capture; verify again after Step 8 with the source-prefix map. Verifying after non-destructive steps is a free no-op pass.
- **Spec.**
  - `--capture out.json`: walk `EXPERIMENTS` and record:
    - Per manifest (the 64 with one), per serial: `(manifest_relpath, serial_no, weights_path_string, resolves: bool)`. Resolution rule: try repo-root-relative; if that misses, try `main_on_shuttleset/`-relative (the Chang baseline manifest's convention).
    - Every `*/weights/*.pt`: relpath, size, sha256.
    - Every `*/fe_jsons/*` and `*/predictions/*` file: relpath, size, sha256.
    - Every `*/tb/**` file: relpath, size only (no hash; listings + sizes catch touches at a fraction of the cost, which is all decision 7 needs).
    - `docs/models_registry.yaml`: per entry, `(id, manifest_path resolves, weights_path resolves)`.
    - `scripts/model_manifest.tsv`: per non-comment row, `(dest_path, resolves: bool)`. Verify applies the same expected-name map to dest_path (added 2026-06-11, bug #T).
    - Write sorted JSON. Record the git SHA and a timestamp in the header.
  - `--verify baseline.json [--src-map src/bst_x=src/bst_x]`: recompute the walk, then compare against the baseline under the expected-name mapping:
    - Weight files: a baseline path under `run_*/weights/` with basename `bst_CG_AP_<rest>.pt` maps to expected basename `bst_x_<rest>.pt`; inside the Chang baseline dir the map is `bst_CG_AP_<rest>.pt` → `bst_cg_ap_<rest>.pt` (lowercase in place, revised decision 4); `--src-map` rewrites the dir prefix when verifying Step 8. Assert a one-to-one match between baseline and current sets under the map, with identical sha256 and size per pair. Report extras and missing by name.
    - Manifests: per `(manifest_relpath under map, serial_no)`, assert `resolves` is unchanged. A resolved entry going missing means the file rename and the manifest edit went out of sync; a missing entry starting to resolve means a stray file appeared.
    - Manifest `weights_path` strings: assert the basename matches the mapped expectation (so a manifest edited to a typo'd filename fails even if some file happens to exist).
    - Sidecars and predictions: byte-identical (same relpath under `--src-map`, same sha256). TB: identical name+size listings.
    - Registry: all resolve-flags still true.
    - Exit non-zero on any mismatch; print a per-category summary plus offender lists.
  - Baselines land under `scratch/rebrand_smoke/baselines/` and get a `.gitignore` line (they encode local-superset state; not meaningful on other machines).
- **Dependencies.** stdlib + PyYAML. Runtime: sha256 over roughly 2-3 GB of weights plus sidecars; a couple of minutes locally, run twice per destructive step.
- **Brittleness.** The expected-name map duplicates the rename rule, deliberately: the harness is the independent statement of what the rename script should have done. If the chosen Step 8 target dir differs from `src/bst_x`, the `--src-map` argument carries it; nothing else changes.

## Landing order

| Item | Lands | Gate |
|---|---|---|
| T6, T7, T8, T9, T10, T12 | now, before any migration commit (all pass on today's main) | none |
| H1 | before Step 6b; first capture taken on the pre-rename tree | n/a (manual cadence) |
| T5 | in the Step 1 commit | none |
| T4 | with the first Step 4 commit (needs the mapping/helper) | mapping exists |
| T11 | now | per-stage sentinels |
| T1, T2, T3 | now or with Step 6b | `SWITCHOVER_LANDED` |

Landing the standing tests first gives every later migration commit a richer red/green signal than the existing backstop alone, and the pre-rename green run IS the baseline for everything except the artefact inventory (which H1 captures explicitly).

## Disposition of the sketched test list

1. MODELS round-trip: kept as T1, plus the alias-identity assert (`is`), which is cheaper and stricter than forward-equality alone.
2. Env fallback per var: kept as T4, reshaped from per-var bespoke tests to one helper matrix plus reload spot-checks. Ten vars times three cases through importlib.reload is the expensive way to test one resolution rule. `BST_DATA_DIR` dropped (test-internal consumer only).
3. Manifest weights_path resolve-all: reshaped into H1 plus T6. The absolute version fails on today's main: 175 of 312 entries point at pruned files by design. The preserved invariant is the resolution map, which needs pre/post state, hence a harness. T6 keeps a standing absolute assert for the seven registry-referenced paths, where existence IS the contract.
4. fe_jsons schema: kept as T8 with the exact observed schemas. Corrections to the sketch: the per-clip rank fields are `top_k_idx`/`top_k_prob` (not a `top_k_*` family), and the npz is not "pure numeric": it carries `clip_stems`/`class_list`/`run_id`/`taxonomy_name` string arrays. The load-bearing assert is "no model_name key", not "no strings".
5. Chang KEEP grep set: kept as T9, greps replaced by imports plus a `partial.keywords` flag table (formatting-proof, and catches a semantically wrong re-definition that a grep would bless). One prose grep survives (the attribution line).
6. Chang baseline untouched: kept as T10 with the three filenames pinned as literals; the sketched glob misses the no-serial file.
7. Orphan bst_refactor scan: kept as T11 stage 1, widened into the staged table so every rename family gets the same treatment, each self-gated. Corpus narrowed to tracked files for CI determinism; the untracked-scratch gap is noted and handled manually pre-Step-8.
8. Pydantic Literal: kept as T5, extended to assert both request models agree and to exercise the 422 path through TestClient.

Added beyond the sketch: T2 (the silent-default class), T3 (writer/loader filename agreement), T7 (the lazy-load weight literal nothing currently checks), T12 (dynamic module strings, the one bug #C class pytest collection cannot see), and H1's TB/sidecar byte-identity (decisions 6 and 7 as checks rather than assumptions).

## What the existing backstop already covers

Not duplicated here:

- Static import breakage on Steps 6/8: `tests/test_train_surface.py`, `test_inference_smoke.py`, `test_taxonomy.py`, `test_sticky_anchor.py`, `test_dataset.py`, `test_adaptive_focal.py`, `test_hparam_sweep.py` import the renamed modules directly; pytest collection fails loudly on a partial rename once their import lines are updated in the lockstep commits. `conftest.py` itself is the Step 8 canary.
- Registry serving and pinned metrics: `tests/test_api.py` (`bst_x_une_v1_14_v2` lookups, macro-F1 pins, sidecar serving, mock-prediction degradation).
- Registry/manifest field semantics: `tests/test_api_registry.py`.
- Trainer/infer behaviour: `test_train_surface.py`, `test_inference_smoke.py` (npz schema for fresh dumps; T8 extends the same invariant across all historical on-disk artefacts).
- HPC-side env wiring: `tests/test_remote_preflight.py` plus `verify_env_paths.py` on the nodes (out of scope for the laptop suite; the runbook's per-step verify list owns it).

Out of scope entirely: decision 8 (venv rename; T11 stage 7 is the only repo-side trace) and decision 9 (aim repo delete-and-remake; no repo surface).

## Open questions

1. **Step 4 shape.** RESOLVED 2026-06-11 round 2: the mapping dict + resolve helper (per module family: pipeline and api). T4's parametrised matrix stands as specced.
2. **Deprecation signal channel.** RESOLVED 2026-06-11 round 2: `warnings.warn(DeprecationWarning)`.
3. **Step 8 manifest rewrite.** RESOLVED 2026-06-11: yes — all 64 run manifests get `weights_path` AND `tb_dir` prefixes rewritten in the Step 8 commit (the pickup doc's process step 5b). H1's verify assumption stands. The Chang baseline manifest (`experiments/`-relative paths) is untouched at Step 8; its 6b.2 lowercase is a separate, earlier edit.
4. **Chang baseline resume gap.** RESOLVED 2026-06-11: the baseline's three weights lowercase in place at 6b.2 (manifest + .txt notes following), so `model_name='BST_CG_AP'` resumes it under the new rule. No case-insensitive lookup in `seek_network_weights`. T3 tests the new rule; T10 pins the lowercase triple post-6b.2.
5. **Back-compat windows.** RESOLVED 2026-06-11 round 2: none exist. The `'bst'` wire value never gets a window (dropped in the Step 1 commit alongside the FE flip), the extras alias was dropped outright, and the env-var fallbacks are branch-only — deleted in the Step 9b end-of-branch cleanup commit, gated on `verify_env_paths.py` green on all hosts. Main never carries compat; T4/T5/T11 pin final states.
6. **Baseline files.** Keep `scratch/rebrand_smoke/baselines/*.json` gitignored (the design above) or commit the pre-rename capture as a record? Committing the first capture has some provenance value; the JSON is small relative to the repo.
7. **Glob scope for the weight rename.** Decision 4 says `experiments/run_*/weights/`. Confirmed on disk: nothing outside `run_*` and the Chang baseline holds weights (`aug_hparam_sweep` carries sweep state only), so the glob is complete today. If any future non-`run_` dir grows a weights dir before the rename executes, H1's one-to-one set match will flag it; no action needed now.

## Change log (2026-06-11 consistency pass)

Fable review pass over the two planning docs and this design, after the baseline-lowercase decision and the bug #T discovery. Changes to this doc:

- Header source note: the "appendix was never written" claim was stale — the assessment's "Updates from chat 2026-06-11" overlay exists (plus a consistency-pass section at its tail); authority pointer fixed.
- Decision 4: Chang baseline weights lowercase in place (`bst_CG_AP_` → `bst_cg_ap_`, manifest + txt following) instead of staying mixed-case. Decision 5: `scripts/model_manifest.tsv` dest_path basenames join the lockstep (release assets keep pre-rebrand names).
- T6: gains the tsv↔registry agreement assert and a post-switchover `bst_x_` pin on tsv dest_path basenames (asset_name column exempt).
- T10: reshaped from a frozen-dir pin to a gated pre/post pin — mixed-case triple before 6b.2, lowercase triple plus a no-`bst_x_` assert after, gated on `run_*` weights carrying `bst_x_*` (different signal, same commit). Manifest-agreement assert now also catches a files-renamed-manifest-missed half-application.
- T11: corpus gains `.tsv` (the extension every earlier pass skipped — bug #T); stage 3 pattern widened to `\bbst_inputs\b` (container target + code fallbacks rename too); stage 4 allowlist dropped (no extras alias); stage 6 scope now includes the baseline dir and the tsv dest column, allowlist starts empty.
- H1: capture records tsv dest_path resolution; verify maps the baseline dir through the in-place lowercase instead of identity.
- T3 doc-trail note and open questions 3-4: marked resolved (manifest rewrite happens at Step 8 incl. `tb_dir`; baseline resume gap closed by the lowercase). Open question 5 trimmed (alias window gone).

Round 2 (same day, pre-execution talk-through):

- T5: back-compat flag machinery removed — `'bst'` drops in the Step 1 commit itself (FE flips in the same diff, no independent clients), so the test pins the final wire format from day one. Landing-order row updated.
- T4: Step 4 shape locked (mapping dict + resolve helper, api/config stdlib twin) and signal channel locked (`warnings.warn(DeprecationWarning)`); mapping is branch-lifetime, deleted in the new Step 9b cleanup commit which flips T4's legacy expectation in the same diff.
- T11 stage 5: sentinel now names Step 9b as the commit that arms it.
- Open questions 1, 2, and 5: resolved (shape, channel, and "no back-compat windows exist — all compat dies on the branch").

Round 3 (same day, final sign-off):

- Design signed off as specced (T1-T12 + H1); standing tests land in the pickup doc's new Step 0b, before any migration commit.
- T8 gains a lifecycle: full-corpus through the migration branch, slimmed to the six registry-anchored run dirs in the Step 9b cleanup commit. The fe_jsons schema pin survives; the migration-era runtime doesn't.
