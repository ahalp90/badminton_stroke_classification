#!/usr/bin/env python3
"""Pre/post artefact-inventory harness for the BST to BST-X rebrand (H1).

Two modes:

  --capture out.json [--root REPO_ROOT]
      Walk every run dir + the registry yaml + the model_manifest.tsv, record
      sha256 + size for weights and FE sidecars, manifest weights_path
      resolution flags, and TB file listings (sizes only). Output is sorted
      JSON. Run TWICE per destructive step (Step 6b.2 + Step 8): once before
      the change, once after.

  --verify baseline.json [--src-map src/bst_x=src/bst_x]
      Re-walk and compare against baseline.json under the expected rename
      rules (bst_CG_AP_* -> bst_x_* inside run_*/weights/, and
      bst_CG_AP_* -> bst_cg_ap_* inside the Chang baseline dir;
      ``--src-map`` rewrites the dir prefix when verifying Step 8). Asserts
      one-to-one weight match (sha256 + size identical under the map), manifest
      resolution flags unchanged, sidecars + predictions byte-identical, TB
      name+size listings identical, registry + tsv resolve.

Run from the badminton-cicd venv (stdlib + PyYAML are all that's needed).
Baselines land under ``scratch/rebrand_smoke/baselines/`` and are gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return ''


def experiments_dir(root: Path) -> Path:
    candidate = root / 'src' / 'bst_x' / 'stroke_classification' / 'main_on_shuttleset' / 'experiments'
    if candidate.is_dir():
        return candidate
    raise SystemExit(f'No experiments dir under src/bst_x in {root}')


def resolve_manifest_weight(root: Path, manifest_path: Path, weights_path_str: str) -> bool:
    """Resolution rule: try repo-root-relative first; if that misses, try
    main_on_shuttleset/-relative (the Chang baseline convention)."""
    if (root / weights_path_str).exists():
        return True
    msroot = manifest_path.parent.parent.parent  # .../main_on_shuttleset/
    if (msroot / weights_path_str).exists():
        return True
    return False


def capture(root: Path) -> dict:
    exp = experiments_dir(root)
    inv = {
        'git_sha': git_sha(root),
        'root': str(root),
        'experiments_root': str(exp.relative_to(root)),
        'manifests': [],
        'weights': [],
        'sidecars': [],
        'predictions': [],
        'tb': [],
        'registry': [],
        'tsv': [],
    }

    for manifest_path in sorted(exp.glob('*/manifest.yaml')):
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest_rel = str(manifest_path.relative_to(root))
        for serial in manifest.get('serials') or []:
            wp = serial.get('weights_path')
            if wp is None:
                continue
            inv['manifests'].append({
                'manifest_relpath': manifest_rel,
                'serial_no': serial.get('serial_no'),
                'weights_path': wp,
                'resolves': resolve_manifest_weight(root, manifest_path, wp),
            })

    for w in sorted(exp.glob('*/weights/*.pt')):
        rel = str(w.relative_to(root))
        inv['weights'].append({
            'relpath': rel,
            'size': w.stat().st_size,
            'sha256': sha256_of(w),
        })

    for fe_dir in sorted(exp.glob('*/fe_jsons')):
        for f in sorted(fe_dir.iterdir()):
            inv['sidecars'].append({
                'relpath': str(f.relative_to(root)),
                'size': f.stat().st_size,
                'sha256': sha256_of(f),
            })

    for npz in sorted(exp.glob('*/predictions/*.npz')):
        inv['predictions'].append({
            'relpath': str(npz.relative_to(root)),
            'size': npz.stat().st_size,
            'sha256': sha256_of(npz),
        })

    for tb_file in sorted(exp.glob('*/tb/**/*')):
        if tb_file.is_file():
            inv['tb'].append({
                'relpath': str(tb_file.relative_to(root)),
                'size': tb_file.stat().st_size,
            })

    registry_path = root / 'docs' / 'models_registry.yaml'
    if registry_path.exists():
        for entry in yaml.safe_load(registry_path.read_text())['models']:
            inv['registry'].append({
                'id': entry['id'],
                'manifest_path': entry['manifest_path'],
                'manifest_resolves': (root / entry['manifest_path']).exists(),
                'weights_path': entry['weights_path'],
                'weights_resolves': (root / entry['weights_path']).exists(),
            })

    tsv_path = root / 'scripts' / 'model_manifest.tsv'
    if tsv_path.exists():
        for line in tsv_path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            cols = line.split('\t')
            dest_path = cols[0]
            inv['tsv'].append({
                'dest_path': dest_path,
                'resolves': (root / dest_path).is_file(),
            })

    return inv


def apply_weight_name_map(rel: str) -> str:
    """Map a baseline weight path to its expected post-rename path.

    Rules: inside ``run_*/weights/``, ``bst_CG_AP_<rest>.pt`` -> ``bst_x_<rest>.pt``.
    Inside the Chang baseline dir, ``bst_CG_AP_<rest>.pt`` -> ``bst_cg_ap_<rest>.pt``
    (lowercase in place; never bst_x_).
    """
    p = Path(rel)
    if 'bst_cg_ap_base_' in rel:
        m = re.match(r'(bst_CG_AP)(_.+\.pt)$', p.name)
        if m:
            return str(p.with_name(f'bst_cg_ap{m.group(2)}'))
    m = re.match(r'bst_CG_AP_(.+\.pt)$', p.name)
    if m and '/run_' in '/' + rel:
        return str(p.with_name(f'bst_x_{m.group(1)}'))
    return rel


def apply_src_map(rel: str, mapping: list[tuple[str, str]]) -> str:
    for src, dst in mapping:
        if rel.startswith(src):
            return dst + rel[len(src):]
    return rel


def map_path(rel: str, src_map: list[tuple[str, str]]) -> str:
    return apply_weight_name_map(apply_src_map(rel, src_map))


def diff_listing(label, baseline, current, key='relpath'):
    """Compare two lists of dicts by ``key``; return a summary diff."""
    base_index = {entry[key]: entry for entry in baseline}
    cur_index = {entry[key]: entry for entry in current}
    missing = sorted(set(base_index) - set(cur_index))
    extra = sorted(set(cur_index) - set(base_index))
    differ = []
    for k in sorted(set(base_index) & set(cur_index)):
        if base_index[k] != cur_index[k]:
            differ.append((k, base_index[k], cur_index[k]))
    return label, missing, extra, differ


def verify(baseline: dict, current: dict, src_map: list[tuple[str, str]]) -> list[str]:
    """Compare current against baseline under the expected-name rules.
    Returns a list of failure messages (empty -> green)."""
    fails: list[str] = []

    expected_weights = {
        map_path(w['relpath'], src_map): w for w in baseline['weights']
    }
    current_weights = {w['relpath']: w for w in current['weights']}
    miss = sorted(set(expected_weights) - set(current_weights))
    extra = sorted(set(current_weights) - set(expected_weights))
    if miss:
        fails.append(f'weights missing under map: {miss}')
    if extra:
        fails.append(f'weights extras not in baseline: {extra}')
    for k in sorted(set(expected_weights) & set(current_weights)):
        exp = expected_weights[k]
        cur = current_weights[k]
        if exp['sha256'] != cur['sha256']:
            fails.append(f'weight sha changed: {k} {exp["sha256"][:12]} != {cur["sha256"][:12]}')
        if exp['size'] != cur['size']:
            fails.append(f'weight size changed: {k} {exp["size"]} != {cur["size"]}')

    expected_manifest_resolutions = {}
    for entry in baseline['manifests']:
        mapped_mp = apply_src_map(entry['manifest_relpath'], src_map)
        mapped_wp = map_path(entry['weights_path'], src_map)
        expected_manifest_resolutions[(mapped_mp, entry['serial_no'])] = {
            'expected_basename': Path(mapped_wp).name,
            'resolves': entry['resolves'],
        }
    current_manifest_entries = {
        (entry['manifest_relpath'], entry['serial_no']): entry
        for entry in current['manifests']
    }
    miss_mf = sorted(set(expected_manifest_resolutions) - set(current_manifest_entries))
    extra_mf = sorted(set(current_manifest_entries) - set(expected_manifest_resolutions))
    if miss_mf:
        fails.append(f'manifest serials missing under map: {miss_mf}')
    if extra_mf:
        fails.append(f'manifest serials new vs baseline: {extra_mf}')
    for key in sorted(set(expected_manifest_resolutions) & set(current_manifest_entries)):
        exp = expected_manifest_resolutions[key]
        cur = current_manifest_entries[key]
        if exp['resolves'] != cur['resolves']:
            fails.append(f'manifest resolve flag flipped for {key}: '
                         f'baseline={exp["resolves"]} current={cur["resolves"]}')
        if Path(cur['weights_path']).name != exp['expected_basename']:
            fails.append(f'manifest weights_path basename mismatched: {key} '
                         f'expected={exp["expected_basename"]} '
                         f'got={Path(cur["weights_path"]).name}')

    for label, items in (('sidecars', 'sidecars'), ('predictions', 'predictions')):
        expected = {
            apply_src_map(item['relpath'], src_map): item for item in baseline[items]
        }
        current_map = {item['relpath']: item for item in current[items]}
        miss = sorted(set(expected) - set(current_map))
        extra = sorted(set(current_map) - set(expected))
        if miss:
            fails.append(f'{label} missing: {miss}')
        if extra:
            fails.append(f'{label} extras: {extra}')
        for k in sorted(set(expected) & set(current_map)):
            exp = expected[k]
            cur = current_map[k]
            if exp['sha256'] != cur['sha256'] or exp['size'] != cur['size']:
                fails.append(f'{label} content changed: {k}')

    expected_tb = {apply_src_map(t['relpath'], src_map): t for t in baseline['tb']}
    current_tb = {t['relpath']: t for t in current['tb']}
    miss = sorted(set(expected_tb) - set(current_tb))
    extra = sorted(set(current_tb) - set(expected_tb))
    if miss:
        fails.append(f'tb listing missing: {miss}')
    if extra:
        fails.append(f'tb listing extras: {extra}')
    for k in sorted(set(expected_tb) & set(current_tb)):
        if expected_tb[k]['size'] != current_tb[k]['size']:
            fails.append(f'tb size changed: {k}')

    baseline_registry = {e['id']: e for e in baseline['registry']}
    for entry in current['registry']:
        base = baseline_registry.get(entry['id'])
        # Flag a resolve flag that flipped vs baseline; pre-existing-missing
        # weights (e.g. BRIC's intentionally-absent best.pt) shouldn't fail.
        if base is None:
            fails.append(f'registry entry new vs baseline: {entry["id"]}')
            continue
        if entry['manifest_resolves'] != base['manifest_resolves']:
            fails.append(
                f'registry manifest flag flipped: {entry["id"]} '
                f'baseline={base["manifest_resolves"]} current={entry["manifest_resolves"]}'
            )
        if entry['weights_resolves'] != base['weights_resolves']:
            fails.append(
                f'registry weights flag flipped: {entry["id"]} '
                f'baseline={base["weights_resolves"]} current={entry["weights_resolves"]}'
            )

    for row in current['tsv']:
        if not row['resolves']:
            fails.append(f'tsv dest_path does not resolve: {row["dest_path"]}')

    return fails


def parse_src_map(spec: str | None) -> list[tuple[str, str]]:
    if not spec:
        return []
    out = []
    for chunk in spec.split(','):
        src, dst = chunk.split('=', 1)
        out.append((src.strip(), dst.strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path.cwd(),
                    help='Repo root (defaults to cwd)')
    sub = ap.add_subparsers(dest='mode', required=True)

    cap = sub.add_parser('capture', help='Write a JSON inventory to OUT')
    cap.add_argument('out', type=Path)

    ver = sub.add_parser('verify', help='Compare current state vs BASELINE')
    ver.add_argument('baseline', type=Path)
    ver.add_argument('--src-map', default='',
                     help='Comma-separated src=dst rewrites applied to baseline paths')

    args = ap.parse_args()
    if args.mode == 'capture':
        inv = capture(args.root.resolve())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(inv, indent=2, sort_keys=True))
        print(f'wrote {args.out} ({len(inv["weights"])} weights, '
              f'{len(inv["sidecars"])} sidecars, {len(inv["predictions"])} npz, '
              f'{len(inv["tb"])} tb files)')
        return

    baseline = json.loads(args.baseline.read_text())
    current = capture(args.root.resolve())
    fails = verify(baseline, current, parse_src_map(args.src_map))
    if fails:
        print('VERIFY FAILED:', file=sys.stderr)
        for line in fails:
            print(f'  {line}', file=sys.stderr)
        sys.exit(1)
    print(f'VERIFY OK ({len(current["weights"])} weights, '
          f'{len(current["sidecars"])} sidecars, {len(current["predictions"])} npz)')


if __name__ == '__main__':
    main()
