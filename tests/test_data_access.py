"""Tests for pipeline.data_access -- CSV-driven filtering of clips/shuttle/mmpose.

Fake filesystem layout (matches post-Phase-2 reality):
  clips_dir/{split}/{folder_name}/{clip_stem}.mp4   -- still nested.
  shuttle_npy_dir/{clip_stem}.npy                    -- flat (Phase 2.2).
  mmpose_npy_dir/{clip_stem}_joints.npy              -- flat (Phase 2.1).
  mmpose_npy_dir/{clip_stem}_pos.npy                 -- flat (Phase 2.1).

Split + taxonomy-class assignment comes from a synthetic clips_master.csv
fixture rather than the folder structure. Each test builds its own fake CSV
+ tree via ``_make_fake_dataset``.
"""
import csv
import tempfile
from pathlib import Path

import pytest

from pipeline.config import (
    TAXONOMIES,
    TAXONOMY_BST_24,
    TAXONOMY_BST_25,
    TAXONOMY_UNE_V1_14,
    resolve_taxonomy,
)
from pipeline.data_access import (
    ClipRecord,
    DataPaths,
    _derive_class_label,
    get_clip_records,
    interactive,
    summarise,
    _menu,
)


# ---------------------------------------------------------------------------
# Fake dataset + CSV builders
# ---------------------------------------------------------------------------

CSV_COLUMNS = (
    'clip_stem', 'raw_type_en', 'player_side',
    'split_bst_baseline', 'split_v2',
)


def _write_clips_csv(csv_path: Path, rows: list[dict]) -> None:
    """Write a synthetic clips_master.csv at ``csv_path``.

    :param csv_path: Destination path.
    :param rows: Each dict must contain every column in ``CSV_COLUMNS``.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, '') for c in CSV_COLUMNS})


def _make_fake_dataset(
    tmp: Path,
    rows: list[dict],
    taxonomy_name: str = 'bst_25',
    with_shuttle: bool = True,
    with_mmpose: bool = False,
    with_clips: bool = True,
) -> DataPaths:
    """Build a fake on-disk dataset + synthetic clips_master.csv.

    The clips tree mirrors the real nested layout so
    ``build_clip_path_index`` can walk it; shuttle and mmpose dirs are flat.

    :param tmp: Base temp directory.
    :param rows: Per-clip dicts with CSV_COLUMNS fields. The folder name in
        the nested clips tree is derived via the taxonomy's label rules so
        the fixture matches what the live pipeline would lay down.
    :param taxonomy_name: Used to derive the folder_name for nested clip
        placement.
    :param with_shuttle: Create matching flat shuttle .npy files.
    :param with_mmpose: Create matching flat mmpose _joints.npy + _pos.npy.
    :param with_clips: Create the .mp4 stub files; disable to test the
        missing-clip codepath.
    :return: DataPaths wired to the fake tree.
    """
    clips_dir = tmp / 'clips'
    shuttle_dir = tmp / 'shuttle_npy_flat'
    mmpose_dir = tmp / 'mmpose_npy_flat' if with_mmpose else None
    csv_path = tmp / 'clips_master.csv'

    taxonomy = resolve_taxonomy(taxonomy_name)
    if with_clips:
        clips_dir.mkdir(parents=True, exist_ok=True)
    if with_shuttle:
        shuttle_dir.mkdir(parents=True, exist_ok=True)
    if with_mmpose:
        mmpose_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        stem = row['clip_stem']
        if with_clips:
            folder_name = _derive_class_label(
                row['raw_type_en'], row['player_side'], taxonomy,
            )
            nested = clips_dir / row['split_bst_baseline'] / folder_name
            nested.mkdir(parents=True, exist_ok=True)
            (nested / f'{stem}.mp4').touch()
        if with_shuttle:
            (shuttle_dir / f'{stem}.npy').touch()
        if with_mmpose:
            (mmpose_dir / f'{stem}_joints.npy').touch()
            (mmpose_dir / f'{stem}_pos.npy').touch()

    _write_clips_csv(csv_path, rows)

    return DataPaths(
        clips_dir=clips_dir,
        shuttle_npy_dir=shuttle_dir,
        mmpose_npy_dir=mmpose_dir,
        clips_csv=csv_path,
    )


# Shared fixture rows. Seven clips across three splits plus a couple of
# rows where split_v2 disagrees with split_bst_baseline, so the split_column
# switching tests have something to bite on.
SIMPLE_ROWS = [
    {'clip_stem': '1_1_1_1', 'raw_type_en': 'smash',
     'player_side': 'Top', 'split_bst_baseline': 'train', 'split_v2': 'train'},
    {'clip_stem': '1_1_2_1', 'raw_type_en': 'smash',
     'player_side': 'Top', 'split_bst_baseline': 'train', 'split_v2': 'val'},
    {'clip_stem': '1_1_3_1', 'raw_type_en': 'smash',
     'player_side': 'Bottom', 'split_bst_baseline': 'train', 'split_v2': 'train'},
    {'clip_stem': '1_2_1_1', 'raw_type_en': 'unknown',
     'player_side': 'Top', 'split_bst_baseline': 'train', 'split_v2': 'train'},
    {'clip_stem': '35_1_1_1', 'raw_type_en': 'smash',
     'player_side': 'Top', 'split_bst_baseline': 'val', 'split_v2': 'val'},
    {'clip_stem': '35_1_2_1', 'raw_type_en': 'lob',
     'player_side': 'Bottom', 'split_bst_baseline': 'val', 'split_v2': 'val'},
    {'clip_stem': '39_1_1_1', 'raw_type_en': 'smash',
     'player_side': 'Top', 'split_bst_baseline': 'test', 'split_v2': 'test'},
]


# ---------------------------------------------------------------------------
# _derive_class_label
# ---------------------------------------------------------------------------

def test_derive_class_label_prefixes_with_player_side():
    """Sided taxonomy: smash maps to Top_/Bottom_ prefixes."""
    assert _derive_class_label('smash', 'Top', TAXONOMY_BST_25) == 'Top_smash'
    assert _derive_class_label('smash', 'Bottom', TAXONOMY_BST_25) == 'Bottom_smash'


def test_derive_class_label_applies_merge_map():
    """bst_25 merges back_court_drive -> drive (per MERGE_MAP_25)."""
    assert _derive_class_label(
        'back_court_drive', 'Top', TAXONOMY_BST_25,
    ) == 'Top_drive'


def test_derive_class_label_applies_bst_25_driven_flight_fix():
    """driven_flight -> drive on bst_25 (the MERGE_MAP_25 paper-faithful fix
    vs the legacy buggy driven_flight -> unknown convention).
    """
    assert _derive_class_label('driven_flight', 'Top', TAXONOMY_BST_25) == 'Top_drive'
    assert _derive_class_label('driven_flight', 'Bottom', TAXONOMY_BST_25) == 'Bottom_drive'


def test_derive_class_label_standalone_is_unprefixed():
    """Side-agnostic types (e.g. unknown) come through unprefixed."""
    assert _derive_class_label('unknown', 'Top', TAXONOMY_BST_25) == 'unknown'


def test_derive_class_label_excluded_returns_none():
    """raw_type in excluded_base_stroke_types yields None (filtered out)."""
    assert _derive_class_label('unknown', 'Top', TAXONOMY_BST_24) is None
    assert _derive_class_label('unknown', 'Top', TAXONOMY_UNE_V1_14) is None


# ---------------------------------------------------------------------------
# get_clip_records -- basic filtering
# ---------------------------------------------------------------------------

def test_no_filter_returns_all_rows():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        records = get_clip_records(paths)
    assert len(records) == len(SIMPLE_ROWS)
    assert all(isinstance(r, ClipRecord) for r in records)


def test_split_filter_restricts_to_split():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        records = get_clip_records(paths, split='val')
    assert all(r.split == 'val' for r in records)
    assert len(records) == 2


def test_class_filter_restricts_to_class():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        records = get_clip_records(paths, taxonomy_class='Top_smash')
    assert all(r.taxonomy_class == 'Top_smash' for r in records)
    # Top+smash occurrences across the fixture: 1_1_1_1, 1_1_2_1, 35_1_1_1, 39_1_1_1.
    assert len(records) == 4


def test_split_and_class_filter_combined():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        records = get_clip_records(
            paths, split='train', taxonomy_class='Top_smash',
        )
    assert len(records) == 2
    assert all(r.split == 'train' for r in records)
    assert all(r.taxonomy_class == 'Top_smash' for r in records)


def test_filter_returns_empty_when_no_match():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        records = get_clip_records(
            paths, split='test', taxonomy_class='Bottom_lob',
        )
    assert records == []


def test_split_column_switch_changes_assignment():
    # 1_1_2_1 is 'train' under split_bst_baseline but 'val' under split_v2.
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        baseline_train = get_clip_records(
            paths, split='train', split_column='split_bst_baseline',
        )
        v2_val = get_clip_records(
            paths, split='val', split_column='split_v2',
        )
    baseline_stems = {r.clip_stem for r in baseline_train}
    v2_val_stems = {r.clip_stem for r in v2_val}
    assert '1_1_2_1' in baseline_stems
    assert '1_1_2_1' in v2_val_stems


def test_taxonomy_with_excluded_unknown_drops_unknown_rows():
    """Under the contractual taxonomy, picking bst_24 (excludes unknown via
    excluded_base_stroke_types) drops unknown rows; picking bst_25 (keeps
    unknown) keeps them. No separate drop_unknown flag any more.
    """
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS, taxonomy_name='bst_25')
        with_unknown = get_clip_records(paths, taxonomy_name='bst_25')
        without_unknown = get_clip_records(paths, taxonomy_name='bst_24')
    assert any(r.taxonomy_class == 'unknown' for r in with_unknown)
    assert not any(r.taxonomy_class == 'unknown' for r in without_unknown)
    assert len(without_unknown) == len(with_unknown) - 1


# ---------------------------------------------------------------------------
# get_clip_records -- record contents
# ---------------------------------------------------------------------------

def test_clip_path_resolved_from_nested_tree():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        records = get_clip_records(
            paths, split='train', taxonomy_class='Top_smash',
        )
        assert all(r.clip is not None and r.clip.exists() for r in records)
        assert all(r.clip.suffix == '.mp4' for r in records)
        assert all(r.clip.stem == r.clip_stem for r in records)


def test_clip_path_is_none_when_mp4_missing():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS, with_clips=False)
        records = get_clip_records(paths)
    assert all(r.clip is None for r in records)


def test_shuttle_npy_resolved_flat_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS, with_shuttle=True)
        records = get_clip_records(
            paths, split='train', taxonomy_class='Top_smash',
        )
        assert all(r.shuttle_npy is not None for r in records)
        assert all(r.shuttle_npy.exists() for r in records)
        # Flat layout: shuttle sits directly under shuttle_npy_dir.
        assert all(r.shuttle_npy.parent == paths.shuttle_npy_dir for r in records)


def test_shuttle_npy_is_none_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS, with_shuttle=False)
        records = get_clip_records(
            paths, split='train', taxonomy_class='Top_smash',
        )
    assert all(r.shuttle_npy is None for r in records)


def test_mmpose_none_when_dir_not_set():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS, with_mmpose=False)
        assert paths.mmpose_npy_dir is None
        records = get_clip_records(paths)
    assert all(r.mmpose_joints is None for r in records)
    assert all(r.mmpose_pos is None for r in records)


def test_mmpose_resolved_flat_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS, with_mmpose=True)
        records = get_clip_records(
            paths, split='train', taxonomy_class='Top_smash',
        )
        assert all(r.mmpose_joints is not None for r in records)
        assert all(r.mmpose_pos is not None for r in records)
        assert all(r.mmpose_joints.exists() for r in records)
        assert all(
            r.mmpose_joints.parent == paths.mmpose_npy_dir for r in records
        )


def test_record_stem_matches_clip_and_shuttle():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        records = get_clip_records(
            paths, split='train', taxonomy_class='Top_smash',
        )
    for r in records:
        assert r.clip.stem == r.clip_stem
        assert r.shuttle_npy.stem == r.clip_stem


# ---------------------------------------------------------------------------
# get_clip_records -- validation errors
# ---------------------------------------------------------------------------

def test_invalid_split_raises():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        with pytest.raises(ValueError, match='split must be one of'):
            get_clip_records(paths, split='holdout')


def test_invalid_taxonomy_class_raises():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        with pytest.raises(ValueError, match='not a class in taxonomy'):
            get_clip_records(
                paths,
                taxonomy_class='Top_nonexistent_stroke',
                taxonomy_name='une_merge_v1',
            )


def test_invalid_taxonomy_name_raises():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        with pytest.raises(KeyError, match='not registered and not aliased'):
            get_clip_records(paths, taxonomy_name='does_not_exist')


def test_missing_split_column_raises():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        with pytest.raises(KeyError, match='split_column'):
            get_clip_records(paths, split_column='split_not_in_csv')


# ---------------------------------------------------------------------------
# summarise -- smoke tests
# ---------------------------------------------------------------------------

def test_summarise_runs_without_error(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        summarise(paths)
    captured = capsys.readouterr()
    assert 'train' in captured.out
    assert 'clips=' in captured.out


def test_summarise_split_filter(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        summarise(paths, split='val')
    captured = capsys.readouterr()
    assert 'val' in captured.out
    assert 'train' not in captured.out


def test_summarise_shows_mmpose_column_when_dir_set(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS, with_mmpose=True)
        summarise(paths)
    captured = capsys.readouterr()
    assert 'mmpose=' in captured.out


def test_summarise_hides_mmpose_column_when_dir_not_set(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS, with_mmpose=False)
        summarise(paths)
    captured = capsys.readouterr()
    assert 'mmpose=' not in captured.out


# ---------------------------------------------------------------------------
# _menu and interactive
# ---------------------------------------------------------------------------

def test_menu_returns_selected_option(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _: '2')
    result = _menu('Pick one:', ['alpha', 'beta', 'gamma'])
    assert result == 'beta'


def test_menu_rejects_out_of_range_then_accepts(monkeypatch, capsys):
    responses = iter(['0', '99', '1'])
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    result = _menu('Pick one:', ['only'])
    assert result == 'only'
    assert 'Enter a number' in capsys.readouterr().out


def test_interactive_summary(monkeypatch, capsys):
    # split_column=split_bst_baseline(1), taxonomy=bst_25(1)[first],
    # split=all(1), class=all(1), output=summary(1). No drop_unknown prompt
    # post-refactor: the taxonomy carries the unknown-exclude rule.
    responses = iter(['1', '1', '1', '1', '1'])
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        interactive(paths)
    assert 'clips=' in capsys.readouterr().out


def test_interactive_file_paths(monkeypatch, capsys):
    # split_column=split_bst_baseline(1), taxonomy=bst_25(1),
    # split=train(2), class=all(1), output=file paths(2).
    responses = iter(['1', '1', '2', '1', '2'])
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_ROWS)
        interactive(paths)
    out = capsys.readouterr().out
    assert 'train' in out
    assert '.mp4' in out
