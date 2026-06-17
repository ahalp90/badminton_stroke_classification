"""Single source of truth for ShuttleSet pipeline configuration.

Centralises splits, stroke type definitions (English with Chinese mappings for
CSV I/O), flaw records, merge rules, and default paths. Every other module in
the pipeline imports from here instead of hardcoding these values.
"""
import csv
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Default paths (anchored to project root, not cwd).
# PROJECT_ROOT = src/bst_x/ (this file lives at src/bst_x/pipeline/config.py).
# REPO_ROOT walks up two more levels to the repo top; SHUTTLESET_DIR is the
# shared on-disk data dir at data/shuttleset/ (annotations + symlinked bulk).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent
SHUTTLESET_DIR = REPO_ROOT / 'data' / 'shuttleset'

SET_INFO_DIR = SHUTTLESET_DIR / 'set'
RAW_VIDEO_DIR = SHUTTLESET_DIR / 'raw_video'
CLIPS_OUTPUT_DIR = SHUTTLESET_DIR / 'clips'
SHUTTLE_OUTPUT_DIR = SHUTTLESET_DIR / 'shuttle_npy'
SHUTTLE_CSV_DIR = SHUTTLESET_DIR / 'shuttle_csv'
FLAW_RECORDS_PATH = SHUTTLESET_DIR / 'flaw_shot_records.csv'
RESOLUTION_CSV_PATH = SHUTTLESET_DIR / 'my_raw_video_resolution.csv'

# ---------------------------------------------------------------------------
# English <-> Chinese stroke name mappings
# Chinese names are used ONLY when reading/writing the upstream ShuttleSet CSV
# annotations. All pipeline code, folder names, and logs use English.
# ---------------------------------------------------------------------------
# 19 stroke types as they appear in the CSV annotations (Chinese)
# mapped to their official English translations.
EN_TO_ZH: dict[str, str] = {
    'net_shot':                '放小球',
    'return_net':              '擋小球',
    'smash':                   '殺球',
    'wrist_smash':             '點扣',
    'lob':                     '挑球',
    'defensive_return_lob':    '防守回挑',
    'clear':                   '長球',
    'drive':                   '平球',
    'driven_flight':           '小平球',
    'back_court_drive':        '後場抽平球',
    'drop':                    '切球',
    'passive_drop':            '過渡切球',
    'push':                    '推球',
    'rush':                    '撲球',
    'defensive_return_drive':  '防守回抽',
    'cross_court_net_shot':    '勾球',
    'short_service':           '發短球',
    'long_service':            '發長球',
    'unknown':                 '未知球種',
}

ZH_TO_EN: dict[str, str] = {v: k for k, v in EN_TO_ZH.items()}

# All 19 raw annotation types (English)
STROKE_TYPES_19 = list(EN_TO_ZH.keys())

# The 19 types as Chinese strings, for matching against CSV annotation data
STROKE_TYPES_19_ZH = list(EN_TO_ZH.values())


# ---------------------------------------------------------------------------
# Stroke-type base lists (inputs to the Taxonomy objects below)
# ---------------------------------------------------------------------------
# Naming convention: STROKE_TYPES_<count>_<provenance>. Count is the number of
# unprefixed base types (no Top_/Bottom_, no 'unknown'). Provenance tags where
# the list comes from:
#   _CSV     -- raw ShuttleSet CSV annotation values (the 19)
#   _RAW     -- derived from _CSV by stripping specific raw types
#   _MERGED  -- project-defined merge target (the 12 merged stroke types)
#   _UNE_V1  -- the project's UNE-v1 merge target (14 types; keeps wrist_smash
#              and passive_drop distinct)
# These are inputs to Taxonomy objects; counts here do NOT include 'unknown'
# or side prefixing -- those are applied at Taxonomy construction.

# The 12 merged stroke types (English), in a stable order.
# Used to build the BST 25-class family: bst_25 = 12 * 2 sides + 1 unknown.
STROKE_TYPES_12_MERGED = [
    'net_shot', 'return_net', 'smash', 'lob',
    'clear', 'drive', 'drop', 'push',
    'rush', 'cross_court_net_shot', 'short_service', 'long_service',
]

# The 14 UNE-v1 merged stroke types: keeps wrist_smash and passive_drop as
# distinct classes, folds driven_flight into drive (per UNE_MERGE_V1_MAP).
STROKE_TYPES_14_UNE_V1 = [
    'net_shot', 'return_net', 'smash', 'wrist_smash',
    'lob', 'clear', 'drive', 'drop',
    'passive_drop', 'push', 'rush', 'cross_court_net_shot',
    'short_service', 'long_service',
]

# The 18 raw stroke types: STROKE_TYPES_19 minus 'unknown'. Used by the
# shuttleset_18 taxonomy (raw types, no merge, no sides, no unknown).
STROKE_TYPES_18_RAW = [s for s in STROKE_TYPES_19 if s != 'unknown']


# ---------------------------------------------------------------------------
# Class merging maps: raw_type_en (CSV) -> merged-target name.
# ---------------------------------------------------------------------------
# Paper-faithful merge for the BST 25-class system. Folds rare subtypes into
# parents per BST paper supplementary Table G. Key fix vs the legacy MERGE_MAP:
# driven_flight maps to 'drive', NOT 'unknown'. The legacy 35-class behaviour
# bled into the 25-class collations historically; bst_25 here matches the
# published BST 25-class convention exactly.
MERGE_MAP_25: dict[str, str] = {
    'wrist_smash':            'smash',
    'defensive_return_lob':   'lob',
    'driven_flight':          'drive',
    'back_court_drive':       'drive',
    'passive_drop':           'drop',
    'defensive_return_drive': 'drive',
}

# UNE-v1 merge: keeps wrist_smash and passive_drop distinct (they're high-info
# subtypes for the project's analysis); still folds driven_flight into drive.
UNE_MERGE_V1_MAP: dict[str, str] = {
    'defensive_return_lob':   'lob',
    'driven_flight':          'drive',
    'back_court_drive':       'drive',
    'defensive_return_drive': 'drive',
}


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------
PLAYERS = ('Top', 'Bottom')

# ---------------------------------------------------------------------------
# Side-prefixing rule (label layer)
# ---------------------------------------------------------------------------
# Types that NEVER get Top_/Bottom_ prefixed labels, regardless of taxonomy.
# Consulted by label_for_row() below.
SIDE_AGNOSTIC_TYPES: frozenset[str] = frozenset({'unknown'})

# ---------------------------------------------------------------------------
# Unprefixed types (clip-generation concern, NOT a taxonomy property)
# ---------------------------------------------------------------------------
# These raw ShuttleSet types never get Top_/Bottom_ prefixed folders during
# clip generation, because they lack meaningful player attribution.
# 'driven_flight' is included because it's a transient type that always gets
# merged before training -- but at clip-generation time the raw folder still
# exists, just unprefixed.
UNPREFIXED_TYPES: frozenset[str] = frozenset({'unknown', 'driven_flight'})


# ---------------------------------------------------------------------------
# Taxonomy: contractual class definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Taxonomy:
    """A pinned class taxonomy for training and evaluation.

    Each Taxonomy commits its full class list explicitly (the ``classes``
    field). labels.npy values are in ``[0, len(classes))``; no runtime
    active/full remapping. The collator filters out rows whose ``raw_type_en``
    is in ``excluded_base_stroke_types`` BEFORE merge or side-prefixing, and
    ``__post_init__`` enforces that ``'unknown'`` (if present) always sits at
    index ``-1``.

    :param name: Short identifier, e.g. ``'bst_25'``, ``'une_v1_14'``.
    :param classes: The full ordered class list. With sides, this includes
        ``Top_``/``Bottom_``-prefixed entries. ``'unknown'`` (if present)
        sits at index ``-1``.
    :param merge_map: Maps rare raw types to parent names (e.g.
        ``'driven_flight' -> 'drive'``), or ``None`` if no merging is applied.
        Used by ``label_for_row()`` at the collator's per-row decision point.
    :param has_sides: ``True`` if the taxonomy uses ``Top_``/``Bottom_``
        player prefixes; ``False`` for nosides taxonomies.
    :param excluded_base_stroke_types: Raw types (CSV-level) to drop before
        merge or side-prefixing. e.g. ``frozenset({'unknown'})`` for a
        taxonomy whose class list doesn't include unknown; ``frozenset()``
        for a taxonomy that retains unknown.
    """
    name: str
    classes: tuple[str, ...]
    merge_map: dict[str, str] | None
    has_sides: bool
    excluded_base_stroke_types: frozenset[str]

    def __post_init__(self):
        if 'unknown' in self.classes and self.classes[-1] != 'unknown':
            # Raise rather than assert: assertions strip under python -O,
            # this contract needs to bite in production too.
            raise ValueError(
                f'taxonomy {self.name!r}: unknown must sit at index -1; '
                f'found at index {self.classes.index("unknown")}.'
            )

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def has_unknown(self) -> bool:
        return 'unknown' in self.classes


def _sided_classes(
    base: list[str], with_unknown: bool,
) -> tuple[str, ...]:
    """Build a (Top_..., Bottom_..., 'unknown'?) class tuple from base names.

    Helper for constructing ``Taxonomy.classes`` when ``has_sides=True``.
    Drops in ``'unknown'`` at index ``-1`` if requested.

    :param base: ordered list of unprefixed stroke type names.
    :param with_unknown: if True, append ``'unknown'`` at the end.
    :return: tuple of class names in canonical order.
    """
    side_prefixed = [f'Top_{b}' for b in base] + [f'Bottom_{b}' for b in base]
    if with_unknown:
        side_prefixed = side_prefixed + ['unknown']
    return tuple(side_prefixed)


# ---------------------------------------------------------------------------
# Taxonomy registry
# ---------------------------------------------------------------------------
# Six pinned taxonomies. Each commits its full class list explicitly; the
# Taxonomy.__post_init__ check enforces 'unknown' at index -1 when present.
#
# raw_35 (the BST paper's 35-class with-driven_flight system) is intentionally
# NOT registered here. To reinstate: define merge_map={'driven_flight':
# 'unknown'}, has_sides=True, base = STROKE_TYPES_19 minus {'unknown',
# 'driven_flight'} (17 types Top_/Bottom_ prefixed plus unknown at index -1).
# Note that historical raw_35 collations used the buggy merge convention
# (driven_flight -> unknown) on the BST 25-class taxonomy too; bst_25 here is
# paper-faithful with driven_flight -> drive.

TAXONOMY_BST_25 = Taxonomy(
    name='bst_25',
    classes=_sided_classes(STROKE_TYPES_12_MERGED, with_unknown=True),
    merge_map=MERGE_MAP_25,
    has_sides=True,
    excluded_base_stroke_types=frozenset(),  # keeps unknown rows
)

TAXONOMY_BST_24 = Taxonomy(
    name='bst_24',
    classes=_sided_classes(STROKE_TYPES_12_MERGED, with_unknown=False),
    merge_map=MERGE_MAP_25,
    has_sides=True,
    excluded_base_stroke_types=frozenset({'unknown'}),
)

TAXONOMY_BST_12 = Taxonomy(
    name='bst_12',
    classes=tuple(STROKE_TYPES_12_MERGED),
    merge_map=MERGE_MAP_25,
    has_sides=False,
    excluded_base_stroke_types=frozenset({'unknown'}),
)

TAXONOMY_UNE_V1_14 = Taxonomy(
    name='une_v1_14',
    classes=tuple(STROKE_TYPES_14_UNE_V1),
    merge_map=UNE_MERGE_V1_MAP,
    has_sides=False,
    excluded_base_stroke_types=frozenset({'unknown'}),
)

TAXONOMY_UNE_V1_15 = Taxonomy(
    name='une_v1_15',
    classes=tuple(STROKE_TYPES_14_UNE_V1) + ('unknown',),
    merge_map=UNE_MERGE_V1_MAP,
    has_sides=False,
    excluded_base_stroke_types=frozenset(),
)

TAXONOMY_SHUTTLESET_18 = Taxonomy(
    name='shuttleset_18',
    classes=tuple(STROKE_TYPES_18_RAW),
    merge_map=None,
    has_sides=False,
    excluded_base_stroke_types=frozenset({'unknown'}),
)


TAXONOMIES: dict[str, Taxonomy] = {
    t.name: t for t in (
        TAXONOMY_BST_25, TAXONOMY_BST_24, TAXONOMY_BST_12,
        TAXONOMY_UNE_V1_14, TAXONOMY_UNE_V1_15,
        TAXONOMY_SHUTTLESET_18,
    )
}


# ---------------------------------------------------------------------------
# Back-compat for legacy taxonomy names.
# ---------------------------------------------------------------------------
# Maps historical names (recorded in pre-refactor manifests, YAML registries,
# scratch dirs, and best_model_id.txt notes) to the closest equivalent in the
# new contractual set. Resume + alias lookup uses this; new code should pass
# the canonical names from TAXONOMIES directly.
#
# Intended phase-out: remove entries one-by-one as historical runs retire.
# Removing an entry implies the paired /scratch/comp320a/ShuttleSet_data_<old>/
# dir is no longer needed and can be reclaimed manually.
TAXONOMY_ALIASES: dict[str, str] = {
    'une_merge_v1_nosides': 'une_v1_14',   # current best (run_20260505_154907)
    'une_merge_v1':         'une_v1_15',   # legacy with-unknown 14-class
    'merged_25':            'bst_25',      # legacy; OLD runs used buggy merge
    'raw_35':               'bst_25',      # never collated; aliased for completeness
}


def resolve_taxonomy(name: str) -> Taxonomy:
    """Look up a Taxonomy by name, following TAXONOMY_ALIASES for legacy values.

    :param name: canonical taxonomy name (e.g. ``'bst_25'``) or a legacy alias
        (e.g. ``'une_merge_v1_nosides'``).
    :return: the matched Taxonomy object.
    :raises KeyError: when ``name`` is neither canonical nor an alias.
    """
    if name in TAXONOMIES:
        return TAXONOMIES[name]
    if name in TAXONOMY_ALIASES:
        return TAXONOMIES[TAXONOMY_ALIASES[name]]
    raise KeyError(
        f'taxonomy {name!r} not registered and not aliased; '
        f'known: {sorted(TAXONOMIES)}; aliases: {sorted(TAXONOMY_ALIASES)}'
    )


def label_for_row(
    taxonomy: Taxonomy, raw_type: str, side: str,
) -> int | None:
    """Resolve a per-row class index, or None if the row should be filtered out.

    Single decision point for both the collator and ``_derive_class_label``.
    ``excluded_base_stroke_types`` drops rows before any merge or side-prefix
    step; ``merge_map`` applies next; side-prefixing kicks in when
    ``has_sides=True`` and the merged type is not in ``SIDE_AGNOSTIC_TYPES``.

    :param taxonomy: target Taxonomy.
    :param raw_type: ``raw_type_en`` value from ``clips_master.csv``.
    :param side: ``'Top'`` or ``'Bottom'``. Ignored when ``has_sides=False`` or
        when the merged type is side-agnostic.
    :return: index in ``[0, taxonomy.n_classes)`` or None if filtered out.
    """
    if raw_type in taxonomy.excluded_base_stroke_types:
        return None
    merged = (taxonomy.merge_map or {}).get(raw_type, raw_type)
    if taxonomy.has_sides and merged not in SIDE_AGNOSTIC_TYPES:
        label_str = f'{side}_{merged}'
    else:
        label_str = merged
    try:
        return taxonomy.classes.index(label_str)
    except ValueError as e:
        # Re-raise with full debug context (raw_type, side, taxonomy, derived
        # label). The bare tuple.index ValueError just says "x not in tuple"
        # which is useless when chasing a misconfigured taxonomy / merge_map.
        raise ValueError(
            f"taxonomy {taxonomy.name!r}: derived label {label_str!r} "
            f"(raw_type={raw_type!r}, side={side!r}) not in classes "
            f"{list(taxonomy.classes)}"
        ) from e


# ---------------------------------------------------------------------------
# Clip window
# ---------------------------------------------------------------------------
CLIP_WINDOW = 'between_2_hits_with_max_limits'

# ---------------------------------------------------------------------------
# Homography reference resolution
# The homography matrices in homography.csv were computed at this resolution.
# Coordinates must be scaled to match before applying the homography.
# ---------------------------------------------------------------------------
HOMOGRAPHY_RESOLUTION = (1280, 720)


# ---------------------------------------------------------------------------
# Flaw record parsing -- CSV is the single source of truth for exclusions
# ---------------------------------------------------------------------------
def parse_flaw_records(
    csv_path: Path = FLAW_RECORDS_PATH,
) -> tuple[set[int], set[tuple[int, int, int, int]]]:
    """Parse flaw_shot_records.csv to extract excluded videos and removed shots.

    :param csv_path: Path to flaw_shot_records.csv.
    :return: Tuple of (excluded_video_ids, removed_shot_tuples).
    """
    excluded_videos: set[int] = set()
    removed_shots: set[tuple[int, int, int, int]] = set()

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['measure'] != 'removed':
                continue
            match_id = int(row['match'])
            if row['stroke_type'] == 'whole':
                excluded_videos.add(match_id)
            else:
                removed_shots.add((
                    match_id,
                    int(row['set']),
                    int(row['rally']),
                    int(row['ball_round']),
                ))

    return excluded_videos, removed_shots


def _load_flaw_records() -> tuple[set[int], set[tuple[int, int, int, int]]]:
    """Load flaw records lazily. Returns empty sets if CSV is missing.

    This lets modules import stroke types, merge maps, etc. without
    needing flaw_shot_records.csv to be present. The actual pipeline
    steps (clip generation, verification) will fail with clear errors
    if the data they need is empty.
    """
    try:
        return parse_flaw_records()
    except FileNotFoundError:
        import warnings
        warnings.warn(
            f'{FLAW_RECORDS_PATH} not found. '
            f'EXCLUDED_VIDEOS and REMOVED_SHOTS are empty. '
            f'This is fine for inspecting config, but the pipeline '
            f'will produce incorrect results without this file.',
            stacklevel=2,
        )
        return set(), set()


EXCLUDED_VIDEOS, REMOVED_SHOTS = _load_flaw_records()

# ---------------------------------------------------------------------------
# Match-level train/val/test splits
# Define with full intended ranges -- excluded videos are stripped
# automatically below, so you never need to manually skip them.
# ---------------------------------------------------------------------------
_EXPECTED_SPLIT_KEYS = {'train', 'val', 'test'}

_SPLITS_RAW: dict[str, list[int]] = {
    'train': list(range(1, 35)),
    'val':   list(range(35, 39)) + [41],
    'test':  [39, 40, 42, 43, 44],
}

assert set(_SPLITS_RAW.keys()) == _EXPECTED_SPLIT_KEYS, (
    f'SPLITS keys {set(_SPLITS_RAW.keys())} != expected {_EXPECTED_SPLIT_KEYS}'
)

# Strip excluded videos so SPLITS and EXCLUDED_VIDEOS can never desync.
SPLITS: dict[str, list[int]] = {
    name: [v for v in ids if v not in EXCLUDED_VIDEOS]
    for name, ids in _SPLITS_RAW.items()
}


# ---------------------------------------------------------------------------
# Collated-dir naming
# ---------------------------------------------------------------------------
# Both prepare_train_on_shuttleset.py (writer) and bst_x_train.py (reader) need
# to construct the same collated dir basename for the same config. Single
# source of truth so they stay in lockstep.

def derive_npy_collated_dir_basename(
    *, use_3d_pose: bool, seq_len: int, split_column: str, collation_id: str,
) -> str:
    """Format the collated dir basename:
    ``npy_[3d_][seq{N}_]{split}_{collation_id}``.

    The ``3d_`` prefix appears only when ``use_3d_pose=True``; the ``seq{N}_``
    prefix only when ``seq_len != 100``. ``split`` is the split column with its
    ``split_`` prefix stripped (``split_v2`` -> ``v2``, ``split_bst_baseline``
    -> ``bst_baseline``) and is always present, so two cells that share a
    taxonomy + collation_id but differ by split land in distinct dirs instead
    of clobbering each other. The taxonomy itself lives in the parent dir
    (``ShuttleSet_data_<tax>/``), so it isn't repeated here.

    ``collation_id`` is the collation generation tag (``'taxon_pinned_w_preds'``,
    ``'wipe_drop'``, etc.); it discriminates re-collations of the same taxonomy
    + split on disk and is required (no auto-derive). The legacy auto-derived
    tuple-string (``{taxonomy}_{split_column}_{drop_unknown_tag}``) is gone;
    callers pass the tag explicitly.

    Note: a training-time ``ablation_id`` (different augs / loss / wiring on a
    fixed collation) is a separate, manifest-only field. It does NOT enter the
    path, so it plays no part here. Don't conflate the two.

    :param use_3d_pose: whether the collation holds 3D pose (adds ``3d_``).
    :param seq_len: target clip length; non-100 adds a ``seq{N}_`` tag.
    :param split_column: clips_csv split column (``split_v2`` /
        ``split_bst_baseline``); its ``split_`` prefix is stripped for the tag.
    :param collation_id: collation generation tag; trails the basename.
    :return: the collated dir basename.
    """
    three_d_tag = '3d_' if use_3d_pose else ''
    seq_tag = '' if seq_len == 100 else f'seq{seq_len}_'
    split_tag = split_column.removeprefix('split_')
    return f'npy_{three_d_tag}{seq_tag}{split_tag}_{collation_id}'


def collation_id_from_manifest(manifest: dict) -> str | None:
    """Resolve a run's collation generation tag from its manifest, old or new.

    New-schema manifests carry it directly as ``config.collation_id``.
    Pre-refactor manifests predate that field: the tag lived in
    ``config.ablation_id`` (the Hyp dump, ``None`` for the auto-derived runs)
    and always in ``extra.data_provenance.effective_ablation_id`` (the resolved
    value). Falls back through those in order; ``None`` if none is present.

    For internal analysis scripts that read historical run data. The live FE
    registry does NOT use this: it only ever sees new-schema manifests and reads
    ``config.collation_id`` directly (refactor plan, Step J5).

    Meaning-flip caveat: on a pre-refactor manifest ``config.ablation_id`` is
    the *collation* tag, not a training ablation. A caller that also wants the
    new training ``ablation_id`` must gate it on ``collation_id`` being present
    in ``config`` (new schema); a legacy manifest has no training tag. Reading
    new-schema ``config.collation_id`` first means we never misread a new
    manifest's training ``ablation_id`` as the collation tag.

    :param manifest: a parsed run manifest (e.g. ``yaml.safe_load`` of manifest.yaml).
    :return: the collation generation tag, or None if the manifest carries none.
    """
    config = manifest.get('config') or {}
    if config.get('collation_id'):
        return config['collation_id']
    if config.get('ablation_id'):
        return config['ablation_id']
    provenance = (manifest.get('extra') or {}).get('data_provenance') or {}
    return provenance.get('collation_id') or provenance.get('effective_ablation_id')
