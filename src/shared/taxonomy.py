"""Stroke-type taxonomy, mirrored from bst_refactor.pipeline.config.

Canonical for BRIC. Keep values in sync with BST manually for v1; a drift
test is in the v2 backlog.

What's kept here:
  - Stroke type definitions (English + Chinese)
  - Merge maps (raw 19 -> reduced taxonomies)
  - Taxonomy dataclass + the four taxonomies BST defines
  - Players, unprefixed types

Dataset / curation config lives in `shared.dataset`:
  - Annotation paths (SET_INFO_DIR, FLAW_RECORDS_PATH, etc.)
  - EXCLUDED_VIDEOS, REMOVED_SHOTS (parsed from flaw_shot_records.csv)
  - CLIP_WINDOW
  - SPLITS_V2 (active for BRIC) and SPLITS_BST_BASELINE (parity column)

Court geometry config (HOMOGRAPHY_RESOLUTION, court reference points)
lives in `shared.court` alongside the projection math that uses it.

What's NOT mirrored (BST-specific, not relevant to BRIC):
  - derive_ablation_id / derive_npy_collated_dir_basename — BST training infra
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# English <-> Chinese stroke name mappings (Chinese names appear only in the
# upstream ShuttleSet CSV annotations).
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
# Class merging: 19 -> reduced sets
# ---------------------------------------------------------------------------
MERGE_MAP: dict[str, str] = {
    'wrist_smash':            'smash',
    'defensive_return_lob':   'lob',
    'driven_flight':          'unknown',
    'back_court_drive':       'drive',
    'passive_drop':           'drop',
    'defensive_return_drive': 'drive',
}

UNE_MERGE_V1_MAP: dict[str, str] = {
    'defensive_return_lob':   'lob',
    'driven_flight':          'drive',
    'back_court_drive':       'drive',
    'defensive_return_drive': 'drive',
}

# Further collapses biomechanically-similar pairs: wrist_smash and smash share
# the back-court overhead motion (differentiated only by wrist-snap, hard to
# read at 30fps broadcast); drop and passive_drop share the soft-touch back-
# court motion (differentiated by intent rather than mechanics). The net-shot
# cluster is left intact: net_shot and return_net are well-separated by the
# model, and cross_court_net_shot's lower F1 reflects directional confusion
# rather than the kind of mechanical overlap that motivates merging.
UNE_COLLAPSED_V1_MAP: dict[str, str] = {
    **UNE_MERGE_V1_MAP,
    'wrist_smash':  'smash',
    'passive_drop': 'drop',
}

STROKE_TYPES_12_MERGED = [
    'net_shot', 'return_net', 'smash', 'lob',
    'clear', 'drive', 'drop', 'push',
    'rush', 'cross_court_net_shot', 'short_service', 'long_service',
]

STROKE_TYPES_14_UNE_MERGE_V1 = [
    'net_shot', 'return_net', 'smash', 'wrist_smash',
    'lob', 'clear', 'drive', 'drop',
    'passive_drop', 'push', 'rush', 'cross_court_net_shot',
    'short_service', 'long_service',
]

STROKE_TYPES_17_RAW = [s for s in STROKE_TYPES_19 if s not in ('unknown', 'driven_flight')]


PLAYERS = ('Top', 'Bottom')

# These raw types never get Top_/Bottom_ folders (no meaningful player attribution).
UNPREFIXED_TYPES: frozenset[str] = frozenset({'unknown', 'driven_flight'})


# ---------------------------------------------------------------------------
# Taxonomy: single source of truth for class grouping schemes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Taxonomy:
    """A stroke-type grouping scheme for training and evaluation."""

    name: str
    merge_map: dict[str, str] | None
    base_types: tuple[str, ...]
    standalone_types: tuple[str, ...]
    unknown_first: bool

    @property
    def n_classes(self) -> int:
        return len(self.base_types) * 2 + len(self.standalone_types)

    @property
    def standalone_set(self) -> frozenset[str]:
        return frozenset(self.standalone_types)

    @property
    def has_unknown(self) -> bool:
        return 'unknown' in self.standalone_types

    def class_list(self, side: str = 'Both') -> list[str]:
        """Build the full class label list with Top_/Bottom_ prefixes.

        Includes 'unknown' if the taxonomy carries it. For training, see
        `trainable_class_list` (filters unknown to match BST's
        ``drop_unknown=True`` convention).
        """
        base = list(self.base_types)
        standalone = list(self.standalone_types)
        match side:
            case 'Both':
                prefixed = (
                    [f'Top_{s}' for s in base]
                    + [f'Bottom_{s}' for s in base]
                )
            case 'Top':
                prefixed = [f'Top_{s}' for s in base]
            case 'Bottom':
                prefixed = [f'Bottom_{s}' for s in base]
            case _:
                raise ValueError(f"side must be 'Both', 'Top', or 'Bottom', got {side!r}")
        if side == 'Both' and self.unknown_first:
            return standalone + prefixed
        return prefixed + standalone

    def trainable_class_list(self, side: str = 'Both') -> list[str]:
        """Class list with 'unknown' filtered.

        Matches BST's ``drop_unknown=True`` training convention — the
        ShuttleSet 'unknown' bucket is heterogeneous (mislabels,
        ambiguous swings, partial strokes) and training on it teaches
        the model to be confused, not to predict any specific stroke.
        """
        return [c for c in self.class_list(side) if c != 'unknown']

    @property
    def n_trainable_classes(self) -> int:
        """Number of classes the model actually trains on (post-unknown-drop)."""
        return len(self.trainable_class_list())


TAXONOMY_MERGED_25 = Taxonomy(
    name='merged_25',
    merge_map=MERGE_MAP,
    base_types=tuple(STROKE_TYPES_12_MERGED),
    standalone_types=('unknown',),
    unknown_first=True,
)

TAXONOMY_UNE_MERGE_V1 = Taxonomy(
    name='une_merge_v1',
    merge_map=UNE_MERGE_V1_MAP,
    base_types=tuple(STROKE_TYPES_14_UNE_MERGE_V1),
    standalone_types=('unknown',),
    unknown_first=True,
)

# Top_/Bottom_ collapsed; every type is standalone. BST's current default
# and BRIC's v1 target. 14 stroke types + 'unknown' = 15 total when
# keep_unknown; 14 when drop_unknown.
TAXONOMY_UNE_MERGE_V1_NOSIDES = Taxonomy(
    name='une_merge_v1_nosides',
    merge_map=UNE_MERGE_V1_MAP,
    base_types=(),
    standalone_types=tuple(STROKE_TYPES_14_UNE_MERGE_V1) + ('unknown',),
    unknown_first=False,
)

TAXONOMY_RAW_35 = Taxonomy(
    name='raw_35',
    merge_map=None,
    base_types=tuple(STROKE_TYPES_17_RAW),
    standalone_types=('unknown',),
    unknown_first=False,
)

# Collapsed variant of une_merge_v1_nosides: 12 stroke types + 'unknown'.
# Merges biomechanically-similar pairs (wrist_smash→smash, passive_drop→drop)
# pre-emptively, motivated by the technical similarity of the swings rather
# than by post-hoc confusion-matrix tuning.
TAXONOMY_UNE_COLLAPSED_V1_NOSIDES = Taxonomy(
    name='une_collapsed_v1_nosides',
    merge_map=UNE_COLLAPSED_V1_MAP,
    base_types=(),
    standalone_types=tuple(STROKE_TYPES_12_MERGED) + ('unknown',),
    unknown_first=False,
)

DEFAULT_TAXONOMY = 'une_merge_v1_nosides'

TAXONOMIES: dict[str, Taxonomy] = {
    'merged_25':                TAXONOMY_MERGED_25,
    'une_merge_v1':             TAXONOMY_UNE_MERGE_V1,
    'une_merge_v1_nosides':     TAXONOMY_UNE_MERGE_V1_NOSIDES,
    'une_collapsed_v1_nosides': TAXONOMY_UNE_COLLAPSED_V1_NOSIDES,
    'raw_35':                   TAXONOMY_RAW_35,
}


