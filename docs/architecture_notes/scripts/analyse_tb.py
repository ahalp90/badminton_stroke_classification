"""Build summary tables + charts from extracted TB data."""
import json
import os
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt

DATA_PATH = '/tmp/nosides_tb_extract.json'
EXP_DIR = 'src/bst_x/stroke_classification/main_on_shuttleset/experiments'
CHARTS_DIR = Path('docs/architecture_notes/charts')
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    'net_shot', 'return_net', 'smash', 'wrist_smash', 'lob', 'clear',
    'drive', 'drop', 'passive_drop', 'push', 'rush',
    'cross_court_net_shot', 'short_service', 'long_service',
]

ORDER = [
    ('run_20260425_185421', 'P1 baseline'),
    ('run_20260430_170325', 'LS 0.1*'),
    ('run_20260430_213933', 'LS 0.0'),
    ('run_20260501_073430', 'LS 0.15'),
    ('run_20260501_110525', 'LS 0.15 + cw{ws,sm}=2'),
    ('run_20260501_164658', 'CDB tau=1 gamma=1'),
    ('run_20260501_192113', 'CDB tau=1 gamma=0'),
    ('run_20260501_192519', 'CDB tau=0.5 gamma=1'),
    ('run_20260501_230252', 'CDB tau=1 gamma=1 + cap'),
    ('run_20260502_075808', 'CDB tau=1 gamma=2'),
]

with open(DATA_PATH) as f:
    DATA = json.load(f)


def to_array(pairs):
    """List of [step, value] -> np.array sorted by step."""
    if not pairs:
        return np.array([]), np.array([])
    pairs = sorted(pairs, key=lambda p: p[0])
    steps = np.array([p[0] for p in pairs])
    vals = np.array([p[1] for p in pairs])
    return steps, vals


def serial_train_macro(serial: dict) -> tuple:
    """Compute per-epoch train macro F1 by averaging per-class F1.
    Returns (steps, macro). Empty if not a CDB run."""
    tpc = serial.get('train_per_class')
    if not tpc:
        return np.array([]), np.array([])
    # Stack per class, mean across class axis per step
    arrs = []
    common_steps = None
    for cls in CLASSES:
        if cls not in tpc:
            continue
        s, v = to_array(tpc[cls])
        if common_steps is None:
            common_steps = s
        # Align to common steps (typically same)
        arrs.append(v)
    arrs = np.stack(arrs, axis=0)  # (n_classes, n_steps)
    macro = arrs.mean(axis=0)
    return common_steps, macro


def serial_train_min(serial: dict) -> tuple:
    """Per-epoch min train F1 across classes (specifically wrist_smash usually)."""
    tpc = serial.get('train_per_class')
    if not tpc:
        return np.array([]), np.array([])
    arrs = []
    common_steps = None
    for cls in CLASSES:
        if cls not in tpc:
            continue
        s, v = to_array(tpc[cls])
        if common_steps is None:
            common_steps = s
        arrs.append(v)
    arrs = np.stack(arrs, axis=0)
    minf = arrs.min(axis=0)
    return common_steps, minf


def aggregate_curves(run_id: str, key: str) -> tuple:
    """Mean and std across serials for a given scalar curve. Returns (steps, mean, std)."""
    run = DATA[run_id]
    series = []
    for serial in run['serials']:
        if key in serial:
            s, v = to_array(serial[key])
            series.append((s, v))
        elif key == 'train_macro':
            s, v = serial_train_macro(serial)
            if len(s):
                series.append((s, v))
        elif key == 'train_min':
            s, v = serial_train_min(serial)
            if len(s):
                series.append((s, v))
    if not series:
        return np.array([]), np.array([]), np.array([])
    # Trim to common length
    min_len = min(len(s) for s, _ in series)
    aligned = np.stack([v[:min_len] for _, v in series], axis=0)
    steps = series[0][0][:min_len]
    return steps, aligned.mean(axis=0), aligned.std(axis=0)


def best_val_epoch_per_serial(serial: dict) -> int:
    """The epoch index where val macro peaked (1-indexed to match TB)."""
    if 'best_val_macro_epoch' in serial and serial['best_val_macro_epoch'] is not None:
        return int(serial['best_val_macro_epoch'])
    s, v = to_array(serial.get('val_macro', []))
    if len(v) == 0:
        return -1
    return int(s[v.argmax()])


def value_at_epoch(serial: dict, key: str, epoch: int) -> float:
    """Read value at a specific epoch (1-indexed). Returns nan if missing."""
    if key == 'train_macro':
        s, v = serial_train_macro(serial)
    elif key == 'train_min':
        s, v = serial_train_min(serial)
    else:
        s, v = to_array(serial.get(key, []))
    if len(v) == 0:
        return float('nan')
    # Find closest step <= epoch (TB steps may be 1-indexed)
    idx = np.where(s == epoch)[0]
    if len(idx):
        return float(v[idx[0]])
    # Fallback: closest
    return float(v[np.argmin(np.abs(s - epoch))])


def load_test_means(run_id: str) -> dict:
    """Pull mean test metrics from manifest."""
    mp = os.path.join(EXP_DIR, run_id, 'manifest.yaml')
    m = yaml.safe_load(open(mp))
    serials = m['serials']
    return {
        'test_macro': sum(s['metrics']['macro_f1'] for s in serials) / len(serials),
        'test_min': sum(s['metrics']['min_f1'] for s in serials) / len(serials),
        'test_acc': sum(s['metrics']['accuracy'] for s in serials) / len(serials),
        'test_top2': sum(s['metrics']['top2_accuracy'] for s in serials) / len(serials),
    }


# =============================================================================
# Build summary table
# =============================================================================
print('\nBuilding summary table...\n')

rows = []
for run_id, label in ORDER:
    run = DATA[run_id]
    serials = run['serials']

    # Best epochs across serials
    best_epochs = [best_val_epoch_per_serial(s) for s in serials]
    stopped_epochs = [int(s.get('stopped_epoch', -1)) for s in serials]

    # Train macro at the best val epoch (per serial)
    train_macros = []
    val_macros = []
    train_mins = []
    val_mins = []
    train_losses = []
    val_losses = []
    for s, ep in zip(serials, best_epochs):
        if ep < 0:
            continue
        train_macros.append(value_at_epoch(s, 'train_macro', ep))
        val_macros.append(value_at_epoch(s, 'val_macro', ep))
        train_mins.append(value_at_epoch(s, 'train_min', ep))
        val_mins.append(value_at_epoch(s, 'val_min', ep))
        train_losses.append(value_at_epoch(s, 'train_loss', ep))
        val_losses.append(value_at_epoch(s, 'val_loss', ep))

    test = load_test_means(run_id)

    def safe_mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
        return np.mean(xs) if xs else float('nan')

    rows.append({
        'run_id': run_id,
        'label': label,
        'best_epoch_mean': float(np.mean(best_epochs)),
        'stopped_epoch_mean': float(np.mean(stopped_epochs)),
        'train_macro_at_best': safe_mean(train_macros),
        'val_macro_at_best': safe_mean(val_macros),
        'test_macro_mean': test['test_macro'],
        'train_min_at_best': safe_mean(train_mins),
        'val_min_at_best': safe_mean(val_mins),
        'test_min_mean': test['test_min'],
        'train_loss_at_best': safe_mean(train_losses),
        'val_loss_at_best': safe_mean(val_losses),
    })

# Print as table
print('Run                                       Best  Stop  | Train mac  Val mac  Test mac | Train min  Val min  Test min | Train L  Val L')
print('-' * 150)
for r in rows:
    tm = f"{r['train_macro_at_best']:.4f}" if not np.isnan(r['train_macro_at_best']) else '   --   '
    tn = f"{r['train_min_at_best']:.4f}" if not np.isnan(r['train_min_at_best']) else '   --   '
    print(f"{r['run_id']:30s} {r['label']:25s}  {r['best_epoch_mean']:4.0f}  {r['stopped_epoch_mean']:4.0f}  | "
          f"{tm}    {r['val_macro_at_best']:.4f}   {r['test_macro_mean']:.4f}  | "
          f"{tn}    {r['val_min_at_best']:.4f}   {r['test_min_mean']:.4f}  | "
          f"{r['train_loss_at_best']:.4f}   {r['val_loss_at_best']:.4f}")

# Save rows for the markdown
with open('/tmp/nosides_summary_rows.json', 'w') as f:
    json.dump(rows, f, indent=2)

# =============================================================================
# Charts
# =============================================================================

PALETTE = plt.get_cmap('tab10').colors
RUN_COLOURS = {rid: PALETTE[i % 10] for i, (rid, _) in enumerate(ORDER)}


def chart_loss_curves():
    """Train + val loss curves, mean across serials, per run."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), sharex=True)
    axes = axes.flatten()
    for i, (run_id, label) in enumerate(ORDER):
        ax = axes[i]
        s_tr, m_tr, _ = aggregate_curves(run_id, 'train_loss')
        s_va, m_va, _ = aggregate_curves(run_id, 'val_loss')
        if len(m_tr):
            ax.plot(s_tr, m_tr, label='train', color='tab:blue', linewidth=1.2)
        if len(m_va):
            ax.plot(s_va, m_va, label='val', color='tab:orange', linewidth=1.2)
        ax.set_title(f'{run_id[-15:]}\n{label}', fontsize=9)
        ax.set_ylim(0, 2.0)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(alpha=0.3)
    fig.supxlabel('Epoch')
    fig.supylabel('Loss')
    fig.suptitle('Train / val loss curves (mean across 5 serials per run)', y=0.995)
    fig.tight_layout()
    out = CHARTS_DIR / 'loss_curves.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def chart_val_f1_curves():
    """Val macro and val min F1, all runs overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Val macro
    ax = axes[0]
    for run_id, label in ORDER:
        s, m, std = aggregate_curves(run_id, 'val_macro')
        if len(m):
            ax.plot(s, m, label=label, color=RUN_COLOURS[run_id], linewidth=1.3)
            ax.fill_between(s, m - std, m + std, color=RUN_COLOURS[run_id], alpha=0.12)
    ax.set_title('Val macro F1 (mean ± std across 5 serials)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1')
    ax.set_ylim(0.4, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=7)

    # Val min (= ws typically)
    ax = axes[1]
    for run_id, label in ORDER:
        s, m, std = aggregate_curves(run_id, 'val_min')
        if len(m):
            ax.plot(s, m, label=label, color=RUN_COLOURS[run_id], linewidth=1.3)
            ax.fill_between(s, m - std, m + std, color=RUN_COLOURS[run_id], alpha=0.12)
    ax.set_title('Val min F1 / wrist_smash (mean ± std across 5 serials)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1')
    ax.set_ylim(0.0, 0.55)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=7)

    fig.tight_layout()
    out = CHARTS_DIR / 'val_f1_curves.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def chart_train_vs_val_cdb():
    """For CDB runs (which log per-class train F1), train macro F1 vs val macro F1."""
    cdb_runs = [r for r in ORDER if r[0] in (
        'run_20260501_164658', 'run_20260501_192113',
        'run_20260501_192519', 'run_20260501_230252',
        'run_20260502_075808',
    )]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, (run_id, label) in enumerate(cdb_runs):
        ax = axes[i]
        s_tr, m_tr, std_tr = aggregate_curves(run_id, 'train_macro')
        s_va, m_va, std_va = aggregate_curves(run_id, 'val_macro')
        if len(m_tr):
            ax.plot(s_tr, m_tr, label='train macro', color='tab:blue', linewidth=1.3)
            ax.fill_between(s_tr, m_tr - std_tr, m_tr + std_tr, color='tab:blue', alpha=0.12)
        if len(m_va):
            ax.plot(s_va, m_va, label='val macro', color='tab:orange', linewidth=1.3)
            ax.fill_between(s_va, m_va - std_va, m_va + std_va, color='tab:orange', alpha=0.12)
        # Horizontal references: mean train and val macro at best-val epoch
        train_at_best = []
        val_at_best = []
        for s in DATA[run_id]['serials']:
            ep = best_val_epoch_per_serial(s)
            if ep > 0:
                train_at_best.append(value_at_epoch(s, 'train_macro', ep))
                val_at_best.append(value_at_epoch(s, 'val_macro', ep))
        if val_at_best:
            mv = float(np.nanmean(val_at_best))
            ax.axhline(mv, color='tab:orange', linestyle='--', alpha=0.6, linewidth=0.9,
                       label=f'val macro at saved ckpt ({mv:.3f})')
        if train_at_best:
            mt = float(np.nanmean(train_at_best))
            ax.axhline(mt, color='tab:blue', linestyle='--', alpha=0.6, linewidth=0.9,
                       label=f'train macro at saved ckpt ({mt:.3f})')
        ax.set_title(f'{run_id[-15:]}: {label}')
        ax.set_ylim(0.3, 1.0)
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(alpha=0.3)
    fig.supxlabel('Epoch')
    fig.supylabel('F1')
    fig.suptitle('Train vs val macro F1 (CDB runs — train F1 logged per-class only in adaptive_focal)', y=0.995)
    fig.tight_layout()
    out = CHARTS_DIR / 'train_vs_val_cdb.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def chart_train_vs_val_min_cdb():
    """Train min F1 vs val min F1 for CDB runs (the bottleneck class)."""
    cdb_runs = [r for r in ORDER if r[0] in (
        'run_20260501_164658', 'run_20260501_192113',
        'run_20260501_192519', 'run_20260501_230252',
        'run_20260502_075808',
    )]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, (run_id, label) in enumerate(cdb_runs):
        ax = axes[i]
        s_tr, m_tr, std_tr = aggregate_curves(run_id, 'train_min')
        s_va, m_va, std_va = aggregate_curves(run_id, 'val_min')
        if len(m_tr):
            ax.plot(s_tr, m_tr, label='train min (~ws)', color='tab:blue', linewidth=1.3)
            ax.fill_between(s_tr, m_tr - std_tr, m_tr + std_tr, color='tab:blue', alpha=0.12)
        if len(m_va):
            ax.plot(s_va, m_va, label='val min (~ws)', color='tab:orange', linewidth=1.3)
            ax.fill_between(s_va, m_va - std_va, m_va + std_va, color='tab:orange', alpha=0.12)
        # Horizontal references: mean train and val min at best-val epoch
        train_at_best = []
        val_at_best = []
        for s in DATA[run_id]['serials']:
            ep = best_val_epoch_per_serial(s)
            if ep > 0:
                train_at_best.append(value_at_epoch(s, 'train_min', ep))
                val_at_best.append(value_at_epoch(s, 'val_min', ep))
        if val_at_best:
            mv = float(np.nanmean(val_at_best))
            ax.axhline(mv, color='tab:orange', linestyle='--', alpha=0.6, linewidth=0.9,
                       label=f'val min at saved ckpt ({mv:.3f})')
        if train_at_best:
            mt = float(np.nanmean(train_at_best))
            ax.axhline(mt, color='tab:blue', linestyle='--', alpha=0.6, linewidth=0.9,
                       label=f'train min at saved ckpt ({mt:.3f})')
        ax.set_title(f'{run_id[-15:]}: {label}')
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(alpha=0.3)
    fig.supxlabel('Epoch')
    fig.supylabel('F1')
    fig.suptitle('Train vs val min F1 (~wrist_smash) — CDB runs only', y=0.995)
    fig.tight_layout()
    out = CHARTS_DIR / 'train_vs_val_min_cdb.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def chart_summary_bars():
    """Bar chart of train macro vs val macro vs test macro per run, at best val epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    labels = [r['label'] for r in rows]
    x = np.arange(len(rows))
    width = 0.27

    from matplotlib.patches import Patch

    def _draw_bars(ax_, train_vals, val_vals, test_vals, ylabel, title, ylim, leg_loc):
        # Skip train bar entirely where value is NaN (no bar = no data)
        tx = [x[i] - width for i, v in enumerate(train_vals) if not np.isnan(v)]
        tv = [v for v in train_vals if not np.isnan(v)]
        ax_.bar(tx, tv, width, color='tab:blue')
        ax_.bar(x, val_vals, width, color='tab:orange')
        ax_.bar(x + width, test_vals, width, color='tab:green')
        ax_.set_xticks(x)
        ax_.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax_.set_ylabel(ylabel)
        ax_.set_title(title)
        ax_.set_ylim(*ylim)
        ax_.grid(alpha=0.3, axis='y')
        legend_handles = [
            Patch(facecolor='tab:blue', label='train (CDB only; no bar = not logged)'),
            Patch(facecolor='tab:orange', label='val'),
            Patch(facecolor='tab:green', label='test (5-serial mean)'),
        ]
        ax_.legend(handles=legend_handles, loc=leg_loc, fontsize=8)

    # Macro
    train_m = np.array([r['train_macro_at_best'] for r in rows])
    val_m = np.array([r['val_macro_at_best'] for r in rows])
    test_m = np.array([r['test_macro_mean'] for r in rows])
    _draw_bars(axes[0], train_m, val_m, test_m,
               'Macro F1', 'Macro F1 at saved-checkpoint epoch',
               (0.5, 1.0), 'lower right')

    # Min
    train_mn = np.array([r['train_min_at_best'] for r in rows])
    val_mn = np.array([r['val_min_at_best'] for r in rows])
    test_mn = np.array([r['test_min_mean'] for r in rows])
    _draw_bars(axes[1], train_mn, val_mn, test_mn,
               'Min F1 (~wrist_smash)', 'Min F1 at saved-checkpoint epoch',
               (0, 1.0), 'upper right')

    fig.tight_layout()
    out = CHARTS_DIR / 'summary_bars.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def chart_gap_analysis():
    """Generalisation gap: train - val and val - test per run."""
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    labels = [r['label'] for r in rows]
    x = np.arange(len(rows))
    width = 0.4

    train_m = [r['train_macro_at_best'] for r in rows]
    val_m = [r['val_macro_at_best'] for r in rows]
    test_m = [r['test_macro_mean'] for r in rows]

    # Gap: train - val (positive = overfit signal)
    train_val_gap = []
    for tm, vm in zip(train_m, val_m):
        if isinstance(tm, float) and np.isnan(tm):
            train_val_gap.append(0)
        else:
            train_val_gap.append(tm - vm)
    val_test_gap = [vm - tem for vm, tem in zip(val_m, test_m)]

    from matplotlib.patches import Patch
    # Skip the train-val bar entirely where train F1 wasn't logged
    tv_x = [x[i] - width/2 for i, tm in enumerate(train_m)
            if not (isinstance(tm, float) and np.isnan(tm))]
    tv_vals = [g for tm, g in zip(train_m, train_val_gap)
               if not (isinstance(tm, float) and np.isnan(tm))]
    ax.bar(tv_x, tv_vals, width, color='tab:purple')
    ax.bar(x + width/2, val_test_gap, width, color='tab:cyan')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Macro F1 gap')
    ax.set_title('Generalisation gap (train - val) and val - test, at saved-checkpoint epoch')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.grid(alpha=0.3, axis='y')
    legend_handles = [
        Patch(facecolor='tab:purple', label='train - val (CDB only; no bar = train F1 not logged)'),
        Patch(facecolor='tab:cyan', label='val - test'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    fig.tight_layout()
    out = CHARTS_DIR / 'gap_analysis.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def chart_per_class_train_cdb():
    """For each CDB run, per-class train F1 at best val epoch (heatmap-ish)."""
    cdb_runs = [
        ('run_20260501_164658', 'CDB t1g1'),
        ('run_20260501_192113', 'CDB t1g0'),
        ('run_20260501_192519', 'CDB t0.5g1'),
        ('run_20260501_230252', 'CDB t1g1+cap'),
        ('run_20260502_075808', 'CDB t1g2'),
    ]
    matrix = np.zeros((len(cdb_runs), len(CLASSES)))
    for i, (rid, _) in enumerate(cdb_runs):
        run = DATA[rid]
        for j, cls in enumerate(CLASSES):
            vals = []
            for serial in run['serials']:
                tpc = serial.get('train_per_class', {})
                if cls in tpc:
                    ep = best_val_epoch_per_serial(serial)
                    s, v = to_array(tpc[cls])
                    if len(v):
                        idx = np.where(s == ep)[0]
                        vals.append(v[idx[0]] if len(idx) else v[np.argmin(np.abs(s - ep))])
            matrix[i, j] = np.mean(vals) if vals else np.nan

    # Reorder columns by ascending inter-run mean (worst -> best)
    col_means = np.nanmean(matrix, axis=0)
    order = np.argsort(col_means)
    matrix = matrix[:, order]
    classes_ordered = [CLASSES[k] for k in order]

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)
    ax.set_xticks(np.arange(len(classes_ordered)))
    ax.set_xticklabels(classes_ordered, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(cdb_runs)))
    ax.set_yticklabels([f'{rid[-15:]}: {lbl}' for rid, lbl in cdb_runs], fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                    color='black', fontsize=7)
    ax.set_title('Per-class train F1 at saved-checkpoint epoch (CDB runs); columns sorted worst -> best by inter-run mean')
    fig.colorbar(im, ax=ax, label='Train F1')
    fig.tight_layout()
    out = CHARTS_DIR / 'per_class_train_heatmap_cdb.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


chart_loss_curves()
chart_val_f1_curves()
chart_train_vs_val_cdb()
chart_train_vs_val_min_cdb()
chart_summary_bars()
chart_gap_analysis()
chart_per_class_train_cdb()

print('\nDone.')
