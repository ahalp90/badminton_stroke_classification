"""Single source of truth for ShuttleSet pipeline configuration.

Centralises splits, stroke type definitions (English with Chinese mappings for
CSV I/O), flaw records, merge rules, and default paths. Every other module in
the pipeline imports from here instead of hardcoding these values.
"""
import csv
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Default paths (anchored to project root, not cwd)
# ---------------------------------------------------------------------------
# PROJECT_ROOT = src/bst_x/. REPO_ROOT walks up two more levels to the repo top;
# SHUTTLESET_DIR is the shared on-disk data dir at data/shuttleset/.
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
# English <-> Chinese stroke name mappings (CSV I/O only; pipeline runs in English)
# ---------------------------------------------------------------------------
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

STROKE_TYPES_19 = list(EN_TO_ZH.keys())
STROKE_TYPES_19_ZH = list(EN_TO_ZH.values())


# ---------------------------------------------------------------------------
# Stroke-type base lists (inputs to the Taxonomy objects below)
# ---------------------------------------------------------------------------
# Naming convention: STROKE_TYPES_<count>_<provenance>. Count = unprefixed base
# types only (no Top_/Bottom_, no 'unknown'; both applied at Taxonomy construction).
# Provenance:
#   _RAW     -- derived from the 19 by stripping specific raw types
#   _MERGED  -- BST paper 25-class base set (12 merged stroke types)
#   _UNE_V1  -- project UNE-v1 merge target (14; keeps wrist_smash and passive_drop)

STROKE_TYPES_12_MERGED = [
    'net_shot', 'return_net', 'smash', 'lob',
    'clear', 'drive', 'drop', 'push',
    'rush', 'cross_court_net_shot', 'short_service', 'long_service',
]

STROKE_TYPES_14_UNE_V1 = [
    'net_shot', 'return_net', 'smash', 'wrist_smash',
    'lob', 'clear', 'drive', 'drop',
    'passive_drop', 'push', 'rush', 'cross_court_net_shot',
    'short_service', 'long_service',
]

STROKE_TYPES_18_RAW = [s for s in STROKE_TYPES_19 if s != 'unknown']


# ---------------------------------------------------------------------------
# Class merging maps: raw_type_en (CSV) -> merged-target name
# ---------------------------------------------------------------------------
# Paper-faithful BST 25-class merge per supplementary Table G.
MERGE_MAP_25: dict[str, str] = {
    'wrist_smash':            'smash',
    'defensive_return_lob':   'lob',
    'driven_flight':          'drive',
    'back_court_drive':       'drive',
    'passive_drop':           'drop',
    'defensive_return_drive': 'drive',
}

# UNE-v1: keeps wrist_smash and passive_drop distinct (high-info subtypes for
# the project's analysis); still folds driven_flight into drive.
UNE_MERGE_V1_MAP: dict[str, str] = {
    'defensive_return_lob':   'lob',
    'driven_flight':          'drive',
    'back_court_drive':       'drive',
    'defensive_return_drive': 'drive',
}


# ---------------------------------------------------------------------------
# Player-side rules
# ---------------------------------------------------------------------------
PLAYERS = ('Top', 'Bottom')

# Classes that appear in ``taxonomy.classes`` without a Top_/Bottom_ prefix
# even under a sided taxonomy. Read by derive_class_index when it builds the
# label string from a row's raw type + side.
NOSIDE_CLASSES: frozenset[str] = frozenset({'unknown'})

# Raw stroke types that get one flat folder at clip generation instead of
# split Top_/Bottom_ folders. Disk-layout concern, NOT a taxonomy property.
# 'unknown' lacks meaningful player attribution; 'driven_flight' is a transient
# type that's merged into 'drive' before training, so its raw folder exists
# unprefixed at clip-gen and disappears afterward.
NOSIDE_FOLDERS: frozenset[str] = frozenset({'unknown', 'driven_flight'})


# ---------------------------------------------------------------------------
# Pipeline scalars (clip window, homography reference)
# ---------------------------------------------------------------------------
CLIP_WINDOW = 'between_2_hits_with_max_limits'

# homography.csv matrices were computed at this resolution; coordinates must
# scale to match before applying the homography.
HOMOGRAPHY_RESOLUTION = (1280, 720)


# ---------------------------------------------------------------------------
# Taxonomy: contractual class definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Taxonomy:
    """A pinned class taxonomy for training and evaluation.

    Each Taxonomy commits its class list explicitly; do not live-remap.
    ``__post_init__`` forces ``'unknown'`` (when present) to index ``-1``.

    :param name: e.g. ``'bst_25'``, ``'une_v1_14'``.
    :param classes: ordered class list; if sides, includes ``Top_``/``Bottom_`` entries.
    :param merge_map: rare raw types -> parent names (e.g. ``'driven_flight'``
        -> ``'drive'``), or ``None`` when no merging applies.
    :param has_sides: whether taxonomy uses ``Top_``/``Bottom_`` prefixes.
    :param excluded_base_stroke_types: raw types (CSV-level) to drop before
        merge or side-prefixing, or ``None`` when nothing is dropped.
    """
    name: str
    classes: tuple[str, ...]
    merge_map: dict[str, str] | None
    has_sides: bool
    excluded_base_stroke_types: frozenset[str] | None

    def __post_init__(self):
        if 'unknown' in self.classes and self.classes[-1] != 'unknown':
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

    Helper for ``Taxonomy.classes`` when ``has_sides=True``.

    :param base: ordered unprefixed stroke names.
    :param with_unknown: if True, append ``'unknown'`` at index -1.
    :return: tuple of class names in defined order.
    """
    side_prefixed = [f'Top_{b}' for b in base] + [f'Bottom_{b}' for b in base]
    if with_unknown:
        side_prefixed = side_prefixed + ['unknown']
    return tuple(side_prefixed)


# ---------------------------------------------------------------------------
# Taxonomy registry
# ---------------------------------------------------------------------------
# Six pinned taxonomies. Each commits its class list explicitly; the
# Taxonomy.__post_init__ check enforces 'unknown' at index -1 when present.

TAXONOMY_BST_25 = Taxonomy(
    name='bst_25',
    classes=_sided_classes(STROKE_TYPES_12_MERGED, with_unknown=True),
    merge_map=MERGE_MAP_25,
    has_sides=True,
    excluded_base_stroke_types=None,  # keeps unknown rows
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
    excluded_base_stroke_types=None,
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
# Flaw record parsing -- CSV is the single source of truth for exclusions
# ---------------------------------------------------------------------------
def parse_flaw_records(csv_path: Path = FLAW_RECORDS_PATH,
) -> tuple[set[int], set[tuple[int, int, int, int]]]:
    """Parse flaw_shot_records.csv to extract excluded videos and removed shots.

    :param csv_path: Path to flaw_shot_records.csv.
    :return: Tuple of (excluded_video_ids, removed_shot_tuples).
    :raises FileNotFoundError: if ``csv_path`` is missing.
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


# Load lazily at import so modules can pick up stroke types / merge maps even
# when flaw_shot_records.csv is absent (config-only inspection, isolated tests).
# Pipeline steps that need the records fail loudly when they're empty.
try:
    EXCLUDED_VIDEOS, REMOVED_SHOTS = parse_flaw_records()
except FileNotFoundError:
    import warnings
    warnings.warn(
        f'{FLAW_RECORDS_PATH} not found. '
        f'EXCLUDED_VIDEOS and REMOVED_SHOTS are empty. '
        f'This is fine for inspecting config, but the pipeline '
        f'will produce incorrect results without this file.',
        stacklevel=2,
    )
    EXCLUDED_VIDEOS, REMOVED_SHOTS = set(), set()


# ---------------------------------------------------------------------------
# Match-level train/val/test splits
# ---------------------------------------------------------------------------
# Define with full intended ranges -- excluded videos are stripped automatically
# below, so you never need to manually skip them.
_SPLITS_RAW: dict[str, list[int]] = {
    'train': list(range(1, 35)),
    'val':   list(range(35, 39)) + [41],
    'test':  [39, 40, 42, 43, 44],
}

# Strip excluded videos so SPLITS and EXCLUDED_VIDEOS can never desync.
SPLITS: dict[str, list[int]] = {
    name: [v for v in ids if v not in EXCLUDED_VIDEOS]
    for name, ids in _SPLITS_RAW.items()
}


# ---------------------------------------------------------------------------
# Taxonomy lookup + class-index derivation
# ---------------------------------------------------------------------------

def taxonomy_lookup(name: str) -> Taxonomy:
    """Check the taxonomy is registered.  raises KeyError"""
    if name in TAXONOMIES:
        return TAXONOMIES[name]
    raise KeyError(
        f'taxonomy {name!r} not registered; known: {sorted(TAXONOMIES)}'
    )


def derive_class_index(taxonomy: Taxonomy, raw_type: str, side: str,) -> int | None:
    """The class index a stroke maps to under this taxonomy, or None if dropped.

    Three rules in order: drop the stroke when its raw type is in
    ``excluded_base_stroke_types``; merge rare subtypes via ``merge_map``
    (e.g. ``'driven_flight'`` -> ``'drive'``); then prepend ``Top_``/``Bottom_``
    for sided taxonomies (skipped when the merged type is in
    ``NOSIDE_CLASSES``).

    :param taxonomy: the taxonomy to label under.
    :param raw_type: ``raw_type_en`` from ``clips_master.csv``, e.g. ``'smash'``, ``'driven_flight'``.
    :param side: ``'Top'`` or ``'Bottom'``. Ignored on nosides taxonomies or when the merged type is side-agnostic.
    :return: index in ``[0, taxonomy.n_classes)``, or ``None`` if stroke is filtered out.
    """
    excluded = taxonomy.excluded_base_stroke_types or frozenset()
    merge_map = taxonomy.merge_map or {}

    if raw_type in excluded:
        return None

    merged = merge_map.get(raw_type, raw_type)  # unmapped types pass through
    if taxonomy.has_sides and merged not in NOSIDE_CLASSES:
        label_str = f'{side}_{merged}'
    else:
        label_str = merged

    try:
        return taxonomy.classes.index(label_str)
    except ValueError as e:
        raise ValueError(
            f"taxonomy {taxonomy.name!r}: derived label {label_str!r} "
            f"(raw_type={raw_type!r}, side={side!r}) not in classes "
            f"{list(taxonomy.classes)}"
        ) from e


# ---------------------------------------------------------------------------
# Collated-dir naming -- writer + reader derive the same basename
# ---------------------------------------------------------------------------

def derive_npy_collated_dir_basename(
    *, use_3d_pose: bool, seq_len: int, split_column: str, collation_id: str,
) -> str:
    """Format the collated dir basename: ``npy_[3d_][seq{N}_]{split}_{collation_id}``.

    Taxonomy lives in the parent dir (``ShuttleSet_data_<tax>/``), so isn't
    repeated here. ``seq_len=100`` is canonical and skips the ``seq{N}_`` tag.
    ``split_column`` has its ``split_`` prefix stripped at tag.
    Example ``collation_id`` values: ``'taxon_pinned_w_preds'``, ``'wipe_drop'``.
    """
    three_d_tag = '3d_' if use_3d_pose else ''
    seq_tag = '' if seq_len == 100 else f'seq{seq_len}_'
    split_tag = split_column.removeprefix('split_')
    return f'npy_{three_d_tag}{seq_tag}{split_tag}_{collation_id}'


def collation_id_from_manifest(manifest: dict) -> str | None:
    """Resolve a run's collation generation tag from its manifest, current or pre-2026-06 format.

    New-schema manifests carry it directly as ``config.collation_id``. Pre-refactor
    manifests stored it in ``config.ablation_id`` or
    ``extra.data_provenance.effective_ablation_id`` instead. Reading the
    new-schema field first means a new manifest's training ``ablation_id``
    (different meaning) never gets misread as the collation tag.

    For internal scripts that read historical run data; the live FE registry
    sees new-schema manifests only and reads ``config.collation_id`` directly.

    :param manifest: a parsed run manifest (e.g. ``yaml.safe_load`` of manifest.yaml).
    :return: the collation tag, or None when none is present.
    """
    config = manifest.get('config') or {}
    if config.get('collation_id'):
        return config['collation_id']
    if config.get('ablation_id'):
        return config['ablation_id']
    provenance = (manifest.get('extra') or {}).get('data_provenance') or {}
    return provenance.get('collation_id') or provenance.get('effective_ablation_id')
