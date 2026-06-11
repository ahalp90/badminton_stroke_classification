"""Per-player split overlap analysis for the nosides + dropunk + split_v2 setup.

Maps each clip to its player NAME via (vid, set, rally, side) -> {winner, loser}
using match.csv (downcourt flag) and the set-3 court switch rule. Then computes
overlap of unique players across split pairs, weighted by clip count.

Output: scratch/research/class_player_split_overlap_exploration.md (and charts/).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path('/home/ariel/Documents/COSC594/badminton_stroke_classification')
CLIPS_CSV = REPO / 'notebooks' / 'clips_master.csv'
MATCH_CSV = REPO / 'src/bst_x/ShuttleSet/set/match.csv'
SET_DIR = REPO / 'src/bst_x/ShuttleSet/set'
FLAW_CSV = REPO / 'src/bst_x/ShuttleSet/flaw_shot_records.csv'
DISCARD_CSV = REPO / 'scratch/research/discard_flags_split_v2_dropunk_nosides.csv'

CHARTS_DIR = REPO / 'scratch/research/charts'
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# UNE_MERGE_V1_MAP from pipeline/config.py: 4 raw -> merged mappings.
# Anything not in this map passes through unchanged (after which the
# une_merge_v1_nosides taxonomy keeps the 14 STROKE_TYPES_14 + unknown).
MERGE_MAP = {
    'defensive_return_lob':   'lob',
    'driven_flight':          'drive',
    'back_court_drive':       'drive',
    'defensive_return_drive': 'drive',
}
ACTIVE_CLASSES_14 = [
    'net_shot', 'return_net', 'smash', 'wrist_smash', 'lob', 'clear',
    'drive', 'drop', 'passive_drop', 'push', 'rush', 'cross_court_net_shot',
    'short_service', 'long_service',
]


# ---------------------------------------------------------------------------
# Player mapping: (vid, set, rally, side) -> player name
# ---------------------------------------------------------------------------
def build_match_info() -> pd.DataFrame:
    """Match metadata: vid -> winner / loser / downcourt / video folder."""
    df = pd.read_csv(MATCH_CSV)
    df = df.rename(columns={'id': 'vid'})
    return df[['vid', 'video', 'winner', 'loser', 'downcourt']]


def find_set3_switch_rally(set3_path: Path) -> int:
    """First rally AFTER the 11-point court switch in set 3.

    Mirrors find_set3_switch_rally in pipeline/player_mapping.py: the switch
    occurs at the next rally after either player first reaches 11 points.
    Returns a rally number (not an index); rally < this -> pre-switch phase,
    rally >= this -> post-switch phase. If neither hits 11 (rare retirement
    edge case) returns infinity so all rallies stay pre-switch.
    """
    df = pd.read_csv(set3_path, usecols=['rally', 'roundscore_A', 'roundscore_B'])
    df = df.dropna(subset=['rally']).copy()
    df['rally'] = df['rally'].astype(int)
    df['roundscore_A'] = df['roundscore_A'].astype(int)
    df['roundscore_B'] = df['roundscore_B'].astype(int)
    # Find first row where either score reaches 11
    hit_eleven = df[(df['roundscore_A'] >= 11) | (df['roundscore_B'] >= 11)]
    if hit_eleven.empty:
        return 10**9  # never switches
    first_hit_rally = hit_eleven.iloc[0]['rally']
    # Return the rally AFTER that one (switch happens between rallies)
    after = df[df['rally'] > first_hit_rally]
    if after.empty:
        return 10**9
    return int(after.iloc[0]['rally'])


def map_side_to_player(
    first_A_is_top: bool,
    set_num: int,
    side: str,
    winner: str,
    loser: str,
) -> str:
    """Resolve (set, side) to a player name using BST's A=winner, B=loser convention.

    From player_mapping.map_players: when (first_A_is_top XOR set_num==2)
    is true, A->Top and B->Bottom; otherwise A->Bottom, B->Top. Combined with
    the convention that A is the winner, we get:
    - true:  Top=winner, Bottom=loser
    - false: Top=loser,  Bottom=winner
    Set 3 callers should pass set_num=1 for the pre-switch phase and
    set_num=2 for the post-switch phase.
    """
    a_is_top = first_A_is_top ^ (set_num == 2)
    if side == 'Top':
        return winner if a_is_top else loser
    return loser if a_is_top else winner


# ---------------------------------------------------------------------------
# Build the augmented clips frame
# ---------------------------------------------------------------------------
def load_clips_with_players() -> pd.DataFrame:
    """Load clips_master.csv and add a `player` column with resolved name."""
    clips = pd.read_csv(CLIPS_CSV)
    matches = build_match_info()
    clips = clips.merge(matches, on='vid', how='left')

    # set_id -> int set number (set1 -> 1, set2 -> 2, set3 -> 3)
    clips['set_num'] = clips['set_id'].str.replace('set', '').astype(int)

    # Pre-compute set 3 switch rallies per vid that has a set3.csv
    switch_rallies: dict[int, int] = {}
    for vid, video in matches[['vid', 'video']].itertuples(index=False):
        s3 = SET_DIR / video / 'set3.csv'
        if s3.exists():
            switch_rallies[vid] = find_set3_switch_rally(s3)

    def resolve_player(row) -> str:
        first_A_is_top = bool(row['downcourt'])
        set_num = row['set_num']
        if set_num == 3:
            sw = switch_rallies.get(row['vid'], 10**9)
            phase_set_num = 1 if row['rally'] < sw else 2
        else:
            phase_set_num = set_num
        return map_side_to_player(
            first_A_is_top, phase_set_num, row['player_side'],
            row['winner'], row['loser'],
        )

    clips['player'] = clips.apply(resolve_player, axis=1)
    return clips


def apply_taxonomy(clips: pd.DataFrame) -> pd.DataFrame:
    """Apply nosides taxonomy + drop_unknown. Adds `class` column = merged type."""
    def merge(t: str) -> str:
        return MERGE_MAP.get(t, t)
    clips = clips.copy()
    clips['class'] = clips['raw_type_en'].apply(merge)
    # Keep only the 14 active classes (drops unknown plus any stragglers)
    clips = clips[clips['class'].isin(ACTIVE_CLASSES_14)].copy()
    return clips


def load_flaw_marked_shots() -> set[tuple[int, int, int, int]]:
    """Per-shot flaw flag from each match's set CSVs.

    The ShuttleSet set1.csv / set2.csv / set3.csv files have a `flaw` column
    that's either NaN or 1 (per-shot quality flag, separate from the
    flaw_shot_records.csv removal log). Shots with `flaw == 1` are flagged
    by the original annotators as having a quality issue but were kept in
    the dataset. This is the filter the user asked for (filter b).
    """
    matches = build_match_info()
    flagged: set[tuple[int, int, int, int]] = set()
    for vid, video in matches[['vid', 'video']].itertuples(index=False):
        for set_n in (1, 2, 3):
            p = SET_DIR / video / f'set{set_n}.csv'
            if not p.exists():
                continue
            df = pd.read_csv(p, usecols=['rally', 'ball_round', 'flaw'])
            sub = df[df['flaw'].notna()]
            for _, row in sub.iterrows():
                flagged.add((int(vid), set_n,
                             int(row['rally']), int(row['ball_round'])))
    return flagged


# ---------------------------------------------------------------------------
# Median top-3 / bottom-4 classes from Phase 2 nosides runs
# ---------------------------------------------------------------------------
PHASE_2_NOSIDES_RUNS = [
    'run_20260430_170325',
    'run_20260430_213933',
    'run_20260501_073430',
    'run_20260501_110525',
    'run_20260501_164658',
    'run_20260501_192113',
    'run_20260501_192519',
    'run_20260501_230252',
    'run_20260502_075808',
]


def compute_median_class_ranks() -> dict:
    """For each class, compute median test F1 across Phase 2 nosides runs."""
    EXP = REPO / 'src/bst_x/stroke_classification/main_on_shuttleset/experiments'
    per_run_per_class = {}
    for run_id in PHASE_2_NOSIDES_RUNS:
        m = yaml.safe_load(open(EXP / run_id / 'manifest.yaml'))
        # Mean across 5 serials per class
        per_class = {}
        for cls in ACTIVE_CLASSES_14:
            vals = [s['metrics']['per_class_f1'][cls] for s in m['serials']]
            per_class[cls] = float(np.mean(vals))
        per_run_per_class[run_id] = per_class

    # Median across the 9 runs per class
    median_f1 = {
        cls: float(np.median([per_run_per_class[r][cls]
                              for r in PHASE_2_NOSIDES_RUNS]))
        for cls in ACTIVE_CLASSES_14
    }
    ranked = sorted(median_f1.items(), key=lambda kv: kv[1])
    return {
        'per_run_per_class': per_run_per_class,
        'median_f1': median_f1,
        'sorted_low_to_high': ranked,
        'top_3': [c for c, _ in ranked[-3:][::-1]],   # highest 3
        'bottom_4': [c for c, _ in ranked[:4]],        # lowest 4
    }


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------
def overlap_for_pair(
    clips: pd.DataFrame,
    cls: str,
    split_a: str,
    split_b: str,
) -> dict:
    """For a class and a split pair, compute clip-weighted overlap proportion.

    Definitions:
      - players_a = unique players who have at least one clip of this class
                    in split_a (similarly for b)
      - common_players = players_a & players_b
      - clips_a, clips_b = total clip counts of the class in each split
      - overlap_clips = clips of this class in (a ∪ b) belonging to common
                        players (i.e. clips of class C in split A by players
                        in common_players, plus same in split B)
      - clip_weighted_overlap = overlap_clips / (clips_a + clips_b)
      - jaccard_players       = |common_players| / |players_a ∪ players_b|
    """
    sub = clips[clips['class'] == cls]
    a = sub[sub['split_v2'] == split_a]
    b = sub[sub['split_v2'] == split_b]
    players_a = set(a['player'].unique())
    players_b = set(b['player'].unique())
    common = players_a & players_b
    union = players_a | players_b

    clips_a = len(a)
    clips_b = len(b)
    overlap_clips = (
        a[a['player'].isin(common)].shape[0]
        + b[b['player'].isin(common)].shape[0]
    )
    total = clips_a + clips_b

    return {
        'class': cls,
        'split_a': split_a,
        'split_b': split_b,
        'clips_a': clips_a,
        'clips_b': clips_b,
        'players_a': len(players_a),
        'players_b': len(players_b),
        'common_players': len(common),
        'union_players': len(union),
        'overlap_clips': overlap_clips,
        'total_clips': total,
        'clip_weighted_overlap': overlap_clips / total if total else float('nan'),
        'jaccard_players': len(common) / len(union) if union else float('nan'),
    }


def gross_overlap(clips: pd.DataFrame, split_a: str, split_b: str) -> dict:
    """Same metric but across all classes pooled (gross figure)."""
    a = clips[clips['split_v2'] == split_a]
    b = clips[clips['split_v2'] == split_b]
    players_a = set(a['player'].unique())
    players_b = set(b['player'].unique())
    common = players_a & players_b
    union = players_a | players_b
    overlap_clips = (
        a[a['player'].isin(common)].shape[0]
        + b[b['player'].isin(common)].shape[0]
    )
    total = len(a) + len(b)
    return {
        'split_a': split_a, 'split_b': split_b,
        'clips_a': len(a), 'clips_b': len(b),
        'players_a': len(players_a), 'players_b': len(players_b),
        'common_players': len(common),
        'union_players': len(union),
        'overlap_clips': overlap_clips,
        'total_clips': total,
        'clip_weighted_overlap': overlap_clips / total if total else float('nan'),
        'jaccard_players': len(common) / len(union) if union else float('nan'),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    print('Loading clips with player resolution...')
    clips_full = load_clips_with_players()
    print(f'  {len(clips_full):,} raw clips loaded')

    clips = apply_taxonomy(clips_full)
    print(f'  {len(clips):,} clips after taxonomy + drop unknown')

    print('\nComputing median class ranks across Phase 2 nosides runs...')
    ranks = compute_median_class_ranks()
    print('  Top 3 by median test F1:    ', ranks['top_3'])
    print('  Bottom 4 by median test F1: ', ranks['bottom_4'])
    target_classes = ranks['top_3'] + ranks['bottom_4']
    print(f'  Target: {target_classes}')

    # Filter levels
    print('\nLoading flaw-flagged shots from per-match set CSVs...')
    flaw_marked = load_flaw_marked_shots()
    print(f'  {len(flaw_marked)} flaw-flagged shots in set CSVs')

    # (a) Raw — clips already filtered to nosides+dropunk above
    clips_raw = clips.copy()

    # (b) After flaw filter — drop clips whose source shot has flaw == 1
    flaw_key = list(zip(clips['vid'], clips['set_num'], clips['rally'], clips['ball_round']))
    is_flaw = pd.Series([k in flaw_marked for k in flaw_key], index=clips.index)
    clips_flaw = clips.loc[~is_flaw].copy()
    n_flaw_drops = int(is_flaw.sum())
    print(f'  Flaw-filter drops: {n_flaw_drops} clips '
          f'(of {len(clips):,}, {n_flaw_drops/len(clips)*100:.2f}%)')

    # (c) After bst_x_train discard filter (videos_len == 0)
    print('\nLoading bst_x_train discard flags from CSV...')
    discard_df = pd.read_csv(DISCARD_CSV)
    zero_stems = set(discard_df.loc[discard_df['videos_len'] == 0, 'clip_stem'].astype(str))
    print(f'  {len(zero_stems)} clips have videos_len == 0 across all splits')
    is_discarded = clips['clip_stem'].astype(str).isin(zero_stems)
    clips_bst_filt = clips.loc[~is_discarded].copy()
    n_bst_drops = int(is_discarded.sum())
    print(f'  bst_x_train-filter drops: {n_bst_drops} clips '
          f'(of {len(clips):,}, {n_bst_drops/len(clips)*100:.2f}%)')

    splits_pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]
    levels = [
        ('raw', clips_raw),
        ('flaw_filtered', clips_flaw),
        ('bst_filtered', clips_bst_filt),
    ]

    results = {'levels': {}, 'classes': target_classes,
               'class_ranks': ranks, 'splits_pairs': splits_pairs,
               'flaw_drops': n_flaw_drops, 'bst_drops': n_bst_drops}

    for level_name, df in levels:
        per_class = []
        for cls in target_classes:
            for sa, sb in splits_pairs:
                per_class.append(overlap_for_pair(df, cls, sa, sb))
        gross = [gross_overlap(df, sa, sb) for sa, sb in splits_pairs]
        results['levels'][level_name] = {
            'per_class': per_class,
            'gross': gross,
        }

    # Dump for the markdown writer
    with open('/tmp/player_overlap_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved /tmp/player_overlap_results.json')

    # ---------- Charts ----------
    plt.rcParams.update({'font.size': 9})

    pair_labels = [f'{a}-{b}' for a, b in splits_pairs]
    x = np.arange(len(target_classes))
    width = 0.27
    # Okabe-Ito-derived palette: distinguishable under red-green colour blindness.
    # Blue / orange / reddish-purple instead of blue / orange / green.
    colours = ['#0072B2', '#E69F00', '#CC79A7']

    def panel_per_class(ax, lvl_name, metric, title):
        per_class = results['levels'][lvl_name]['per_class']
        df_pc = pd.DataFrame(per_class)
        for i, (sa, sb) in enumerate(splits_pairs):
            sub = df_pc[(df_pc['split_a'] == sa) & (df_pc['split_b'] == sb)]
            sub = sub.set_index('class').loc[target_classes]
            ax.bar(x + (i - 1) * width, sub[metric].values,
                   width, label=pair_labels[i], color=colours[i])
        ax.set_xticks(x)
        ax.set_xticklabels(target_classes, rotation=30, ha='right', fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis='y')

    # 1) Per-class clip-weighted overlap, all three filter levels
    level_names = ['raw', 'flaw_filtered', 'bst_filtered']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for ax, lvl in zip(axes, level_names):
        panel_per_class(ax, lvl, 'clip_weighted_overlap', f'{lvl} (clip-weighted)')
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
    axes[0].set_ylabel('overlap clips / total clips in pair')
    axes[0].legend(title='split pair', fontsize=8, loc='upper right')
    fig.suptitle('Clip-weighted player overlap, target classes (top 3 + bottom 4 by median test F1)',
                 y=1.0)
    fig.tight_layout()
    out = CHARTS_DIR / 'overlap_clip_weighted.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # 2) Per-class Jaccard, all three filter levels
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for ax, lvl in zip(axes, level_names):
        panel_per_class(ax, lvl, 'jaccard_players', f'{lvl} (player Jaccard)')
    axes[0].set_ylabel('|players_a ∩ players_b| / |union|')
    axes[0].legend(title='split pair', fontsize=8, loc='upper right')
    fig.suptitle('Unique-player Jaccard, target classes', y=1.0)
    fig.tight_layout()
    out = CHARTS_DIR / 'overlap_jaccard_players.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # 3) Gross overlap (across all 14 classes pooled): clip-weighted + Jaccard, all 3 levels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    pair_x = np.arange(len(splits_pairs))
    width = 0.27
    for i, lvl in enumerate(level_names):
        gross = results['levels'][lvl]['gross']
        cw = [g['clip_weighted_overlap'] for g in gross]
        jc = [g['jaccard_players'] for g in gross]
        axes[0].bar(pair_x + (i - 1) * width, cw, width, label=lvl, color=colours[i])
        axes[1].bar(pair_x + (i - 1) * width, jc, width, label=lvl, color=colours[i])
    for ax in axes:
        ax.set_xticks(pair_x)
        ax.set_xticklabels(pair_labels)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('overlap proportion')
    axes[0].set_title('Clip-weighted (overlap clips / total clips)')
    axes[1].set_title('Jaccard (|common| / |union|, player sets)')
    fig.suptitle('Gross overlap across all 14 classes', y=1.02)
    fig.tight_layout()
    out = CHARTS_DIR / 'gross_overlap.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # 4) Per-class clip counts per split (sanity / sample-size context)
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ws = []
    for sp in ['train', 'val', 'test']:
        cnts = clips[clips['split_v2'] == sp].groupby('class').size().reindex(ACTIVE_CLASSES_14, fill_value=0)
        ws.append((sp, cnts))
    width = 0.27
    cls_x = np.arange(len(ACTIVE_CLASSES_14))
    for i, (sp, cnts) in enumerate(ws):
        ax.bar(cls_x + (i - 1) * width, cnts.values, width, label=sp)
    ax.set_xticks(cls_x)
    ax.set_xticklabels(ACTIVE_CLASSES_14, rotation=30, ha='right')
    ax.set_ylabel('clip count')
    ax.set_title('Clip counts per class per split (taxonomy = nosides + dropunk + split_v2)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    # Highlight target classes. CB-safe: blue for ceiling / purple for floor,
    # plus bold on both so the distinction reads in greyscale too.
    for i, cls in enumerate(ACTIVE_CLASSES_14):
        if cls in ranks['top_3']:
            ax.get_xticklabels()[i].set_color('#0072B2')
            ax.get_xticklabels()[i].set_fontweight('bold')
        elif cls in ranks['bottom_4']:
            ax.get_xticklabels()[i].set_color('#CC79A7')
            ax.get_xticklabels()[i].set_fontweight('bold')
    fig.tight_layout()
    out = CHARTS_DIR / 'class_counts_per_split.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    print('\nDone.')


if __name__ == '__main__':
    main()
