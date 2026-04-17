"""Tests for pipeline.data_access — filtering clips/shuttle/mmpose by split and class.

Tests use a temporary fake filesystem that mirrors the real on-disk layout:
  clips_dir/{split}/{class}/{match}_{set}_{rally}_{ball}.mp4
  shuttle_npy_dir/{split}/{class}/{match}_{set}_{rally}_{ball}.npy
  mmpose_npy_dir/{split}/{class}/{match}_{set}_{rally}_{ball}_joints.npy  (optional)
  mmpose_npy_dir/{split}/{class}/{match}_{set}_{rally}_{ball}_pos.npy     (optional)
"""
import pytest
import tempfile
from pathlib import Path

from pipeline.data_access import (
    DataPaths,
    ClipRecord,
    get_clip_records,
    summarise,
    _class_dirs,
)
from pipeline.config import TAXONOMY_UNE_MERGE_V1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_dataset(
    tmp: Path,
    structure: dict[str, dict[str, list[str]]],
    with_shuttle: bool = True,
    with_mmpose: bool = False,
) -> DataPaths:
    """Create a fake data directory tree and return DataPaths pointing to it.

    :param tmp: Base temp directory.
    :param structure: {split: {class_name: [clip_stem, ...]}}
    :param with_shuttle: Create matching shuttle .npy files.
    :param with_mmpose: Create matching mmpose _joints.npy and _pos.npy files.
    :return: DataPaths configured to the fake tree.
    """
    clips_dir = tmp / 'clips'
    shuttle_dir = tmp / 'shuttle_npy'
    mmpose_dir = tmp / 'mmpose_npy' if with_mmpose else None

    for split, classes in structure.items():
        for cls, stems in classes.items():
            (clips_dir / split / cls).mkdir(parents=True, exist_ok=True)
            if with_shuttle:
                (shuttle_dir / split / cls).mkdir(parents=True, exist_ok=True)
            if with_mmpose:
                (mmpose_dir / split / cls).mkdir(parents=True, exist_ok=True)

            for stem in stems:
                (clips_dir / split / cls / f'{stem}.mp4').touch()
                if with_shuttle:
                    (shuttle_dir / split / cls / f'{stem}.npy').touch()
                if with_mmpose:
                    (mmpose_dir / split / cls / f'{stem}_joints.npy').touch()
                    (mmpose_dir / split / cls / f'{stem}_pos.npy').touch()

    return DataPaths(
        clips_dir=clips_dir,
        shuttle_npy_dir=shuttle_dir,
        mmpose_npy_dir=mmpose_dir,
    )


SIMPLE_STRUCTURE = {
    'train': {
        'Top_smash': ['1_1_1_1', '1_1_2_1'],
        'Bottom_smash': ['1_1_3_1'],
        'unknown': ['1_2_1_1'],
    },
    'val': {
        'Top_smash': ['35_1_1_1'],
        'Bottom_lob': ['35_1_2_1'],
    },
    'test': {
        'Top_smash': ['39_1_1_1'],
    },
}


# ---------------------------------------------------------------------------
# _class_dirs
# ---------------------------------------------------------------------------

def test_class_dirs_returns_sorted_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / 'b_class').mkdir()
        (base / 'a_class').mkdir()
        (base / 'not_a_dir.txt').touch()
        names = [d.name for d in _class_dirs(base)]
    assert names == ['a_class', 'b_class']


def test_class_dirs_missing_dir_yields_nothing():
    result = list(_class_dirs(Path('/nonexistent/path/xyz')))
    assert result == []


# ---------------------------------------------------------------------------
# get_clip_records — basic filtering
# ---------------------------------------------------------------------------

def test_no_filter_returns_all_clips():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        records = get_clip_records(paths)
    # 2 + 1 + 1 + 1 + 1 + 1 = 7 clips total
    assert len(records) == 7
    assert all(isinstance(r, ClipRecord) for r in records)


def test_split_filter_restricts_to_split():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        records = get_clip_records(paths, split='val')
    assert all(r.split == 'val' for r in records)
    assert len(records) == 2


def test_class_filter_restricts_to_class():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        records = get_clip_records(paths, taxonomy_class='Top_smash')
    assert all(r.taxonomy_class == 'Top_smash' for r in records)
    # train=2, val=1, test=1
    assert len(records) == 4


def test_split_and_class_filter_combined():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        records = get_clip_records(paths, split='train', taxonomy_class='Top_smash')
    assert len(records) == 2
    assert all(r.split == 'train' for r in records)
    assert all(r.taxonomy_class == 'Top_smash' for r in records)


def test_filter_returns_empty_when_no_match():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        records = get_clip_records(paths, split='test', taxonomy_class='Bottom_lob')
    assert records == []


# ---------------------------------------------------------------------------
# get_clip_records — record contents
# ---------------------------------------------------------------------------

def test_clip_path_exists():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        records = get_clip_records(paths, split='train', taxonomy_class='Top_smash')
        assert all(r.clip.exists() for r in records)
        assert all(r.clip.suffix == '.mp4' for r in records)


def test_shuttle_npy_resolved_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE, with_shuttle=True)
        records = get_clip_records(paths, split='train', taxonomy_class='Top_smash')
        assert all(r.shuttle_npy is not None for r in records)
        assert all(r.shuttle_npy.exists() for r in records)


def test_shuttle_npy_is_none_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE, with_shuttle=False)
        records = get_clip_records(paths, split='train', taxonomy_class='Top_smash')
    assert all(r.shuttle_npy is None for r in records)


def test_mmpose_none_when_dir_not_set():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE, with_mmpose=False)
        assert paths.mmpose_npy_dir is None
        records = get_clip_records(paths)
    assert all(r.mmpose_joints is None for r in records)
    assert all(r.mmpose_pos is None for r in records)


def test_mmpose_resolved_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE, with_mmpose=True)
        records = get_clip_records(paths, split='train', taxonomy_class='Top_smash')
        assert all(r.mmpose_joints is not None for r in records)
        assert all(r.mmpose_pos is not None for r in records)
        assert all(r.mmpose_joints.exists() for r in records)


def test_record_stem_matches_across_clip_and_shuttle():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        records = get_clip_records(paths, split='train', taxonomy_class='Top_smash')
    for r in records:
        assert r.clip.stem == r.shuttle_npy.stem


# ---------------------------------------------------------------------------
# get_clip_records — validation errors
# ---------------------------------------------------------------------------

def test_invalid_split_raises():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        with pytest.raises(ValueError, match='split must be one of'):
            get_clip_records(paths, split='holdout')


def test_invalid_taxonomy_class_raises_when_taxonomy_provided():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        with pytest.raises(ValueError, match='not a class in taxonomy'):
            get_clip_records(
                paths,
                taxonomy_class='Top_nonexistent_stroke',
                taxonomy=TAXONOMY_UNE_MERGE_V1,
            )


def test_unknown_class_without_taxonomy_does_not_raise():
    """taxonomy_class is not validated if no Taxonomy object is passed."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        # Should not raise — just return empty results
        records = get_clip_records(paths, taxonomy_class='Top_nonexistent_stroke')
    assert records == []


# ---------------------------------------------------------------------------
# summarise — smoke test (just checks it runs without error)
# ---------------------------------------------------------------------------

def test_summarise_runs_without_error(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        summarise(paths)
    captured = capsys.readouterr()
    assert 'train' in captured.out
    assert 'clips=' in captured.out


def test_summarise_split_filter(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        summarise(paths, split='val')
    captured = capsys.readouterr()
    assert 'val' in captured.out
    assert 'train' not in captured.out


def test_summarise_shows_mmpose_column_when_dir_set(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE, with_mmpose=True)
        summarise(paths)
    captured = capsys.readouterr()
    assert 'mmpose=' in captured.out


def test_summarise_hides_mmpose_column_when_dir_not_set(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE, with_mmpose=False)
        summarise(paths)
    captured = capsys.readouterr()
    assert 'mmpose=' not in captured.out


# ---------------------------------------------------------------------------
# _menu and interactive
# ---------------------------------------------------------------------------

from pipeline.data_access import _menu, interactive  # noqa: E402


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
    # Choices: split=all(1), class=all(1), output=summary table(1)
    responses = iter(['1', '1', '1'])
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        interactive(paths)
    assert 'clips=' in capsys.readouterr().out


def test_interactive_file_paths(monkeypatch, capsys):
    # Choices: split=train(2), class=all(1), output=file paths(2)
    responses = iter(['2', '1', '2'])
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_dataset(Path(tmp), SIMPLE_STRUCTURE)
        interactive(paths)
    out = capsys.readouterr().out
    assert 'train' in out
    assert '.mp4' in out
