"""Tests for hparam_sweep.py.

Covers config parsing, kill rules, verdict computation, top-3 movers,
reference promotion, requires evaluation, state IO with atomic write,
search log rendering, resume reconciliation, and end-to-end cell flow
via a monkeypatched bst_train shim.

Run from repo root::

    PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification \\
        pytest tests/test_hparam_sweep.py -v
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from main_on_shuttleset import hparam_sweep as hs


# ==========================================================================
# Fixtures
# ==========================================================================

CLASSES = [
    'net_shot', 'return_net', 'smash', 'wrist_smash', 'lob', 'clear', 'drive',
    'drop', 'passive_drop', 'push', 'rush', 'cross_court_net_shot',
    'short_service', 'long_service',
]


def make_metrics(macro: float, min_f1: float, accuracy: float = 0.76,
                 top2: float = 0.94, per_class: dict | None = None) -> dict:
    """Build a metrics dict mirroring what bst_train writes to manifest.yaml."""
    if per_class is None:
        # Default: spread classes around macro. Min class gets min_f1.
        per_class = {cls: macro for cls in CLASSES}
        per_class['smash'] = min_f1
    return {
        'macro_f1': macro,
        'min_f1': min_f1,
        'accuracy': accuracy,
        'top2_accuracy': top2,
        'num_strokes': 4202,
        'per_class_f1': per_class,
    }


def make_serial(serial_no: int, macro: float, min_f1: float, **kwargs) -> dict:
    """Build a serial entry as track_serial would write."""
    return {
        'serial_no': serial_no,
        'weights_path': f'weights/serial_{serial_no}.pt',
        'tb_dir': f'tb/serial_{serial_no}',
        'metrics': make_metrics(macro=macro, min_f1=min_f1, **kwargs),
        'recorded_at': '2026-05-06T12:00:00',
    }


def write_fake_run(experiments_dir: Path, run_id: str,
                   serials: list[dict]) -> Path:
    """Write a fake experiments/<run_id>/manifest.yaml + dirs."""
    run_dir = experiments_dir / run_id
    (run_dir / 'weights').mkdir(parents=True, exist_ok=True)
    (run_dir / 'tb').mkdir(parents=True, exist_ok=True)
    manifest = {
        'run_id': run_id,
        'started_at': '2026-05-06T11:00:00',
        'config': {},
        'serials': serials,
    }
    with open(run_dir / 'manifest.yaml', 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return run_dir


@pytest.fixture
def tmp_experiments(tmp_path, monkeypatch):
    """Redirect hparam_sweep's experiments / sweeps / test_logs dirs to tmp."""
    experiments_dir = tmp_path / 'experiments'
    sweeps_dir = experiments_dir / 'aug_hparam_sweep'
    test_logs_dir = tmp_path / 'test_logs'
    experiments_dir.mkdir()
    sweeps_dir.mkdir()
    test_logs_dir.mkdir()
    monkeypatch.setattr(hs, 'EXPERIMENTS_DIR', experiments_dir)
    monkeypatch.setattr(hs, 'SWEEPS_DIR', sweeps_dir)
    monkeypatch.setattr(hs, 'TEST_LOGS_DIR', test_logs_dir)
    return experiments_dir


@pytest.fixture
def baseline_runs(tmp_experiments):
    """Lay down the two reference runs the session config points at."""
    cb = [make_serial(i, 0.7447 + 0.005 * (i - 3), 0.4779) for i in range(1, 6)]
    write_fake_run(tmp_experiments, 'run_current_best', cb)
    wd = [make_serial(i, 0.7481 + 0.003 * (i - 3), 0.4742) for i in range(1, 6)]
    write_fake_run(tmp_experiments, 'run_wipe_drop_best', wd)
    return tmp_experiments


def make_session_config(name: str = 'test_session', cells: list | None = None) -> dict:
    if cells is None:
        cells = [
            {'name': 'p_flip_25', 'augmentation': {'p_flip': 0.25}},
            {'name': 'cap_bump', 'augmentation': {'cap_y': 0.075, 'cap_x': 0.15}},
        ]
    return {
        'session_name': name,
        'reference': {
            'current_best_run': 'run_current_best',
            'wipe_drop_best_run': 'run_wipe_drop_best',
        },
        'base_config': {
            'augmentation': {
                'p_flip': 0.5, 'p_jitter': 0.3,
                'cap_y': 0.05, 'cap_x': 0.10, 'eps': 0.15,
            },
        },
        'cells': cells,
    }


def make_session_dir(tmp_path: Path, config: dict, name: str = 'sess1') -> Path:
    session_dir = tmp_path / 'experiments' / 'aug_hparam_sweep' / f'sweep_{name}'
    session_dir.mkdir(parents=True, exist_ok=True)
    with open(session_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return session_dir


# ==========================================================================
# Config validation
# ==========================================================================

class TestConfigValidation:
    def test_valid_config_passes(self):
        hs.validate_config(make_session_config())

    def test_missing_top_level_raises(self):
        cfg = make_session_config()
        del cfg['cells']
        with pytest.raises(ValueError, match='cells'):
            hs.validate_config(cfg)

    def test_missing_reference_field_raises(self):
        cfg = make_session_config()
        del cfg['reference']['wipe_drop_best_run']
        with pytest.raises(ValueError, match='wipe_drop_best_run'):
            hs.validate_config(cfg)

    def test_missing_base_aug_key_raises(self):
        cfg = make_session_config()
        del cfg['base_config']['augmentation']['cap_y']
        with pytest.raises(ValueError, match='cap_y'):
            hs.validate_config(cfg)

    def test_duplicate_cell_name_raises(self):
        cells = [
            {'name': 'a', 'augmentation': {'p_flip': 0.25}},
            {'name': 'a', 'augmentation': {'cap_y': 0.075}},
        ]
        cfg = make_session_config(cells=cells)
        with pytest.raises(ValueError, match='duplicate'):
            hs.validate_config(cfg)

    def test_requires_unknown_cell_raises(self):
        cells = [
            {'name': 'a', 'augmentation': {'p_flip': 0.25}},
            {'name': 'b', 'requires': 'unknown_cell == WIN'},
        ]
        cfg = make_session_config(cells=cells)
        with pytest.raises(ValueError, match='unknown'):
            hs.validate_config(cfg)

    def test_requires_later_cell_raises(self):
        cells = [
            {'name': 'a', 'requires': 'b == WIN'},
            {'name': 'b'},
        ]
        cfg = make_session_config(cells=cells)
        with pytest.raises(ValueError, match='later'):
            hs.validate_config(cfg)

    def test_empty_cells_raises(self):
        cfg = make_session_config(cells=[])
        with pytest.raises(ValueError, match='empty'):
            hs.validate_config(cfg)


# ==========================================================================
# Requires evaluation
# ==========================================================================

class TestRequires:
    def test_simple_eq(self):
        assert hs.evaluate_requires('a == WIN', {'a': 'WIN'})
        assert not hs.evaluate_requires('a == WIN', {'a': 'LOSE'})

    def test_neq(self):
        assert hs.evaluate_requires('a != LOSE', {'a': 'TIE'})
        assert hs.evaluate_requires('a != LOSE', {'a': 'WIN'})
        assert not hs.evaluate_requires('a != LOSE', {'a': 'LOSE'})

    def test_and(self):
        assert hs.evaluate_requires('a == WIN and b != LOSE',
                                    {'a': 'WIN', 'b': 'TIE'})
        assert not hs.evaluate_requires('a == WIN and b != LOSE',
                                        {'a': 'WIN', 'b': 'LOSE'})

    def test_or(self):
        assert hs.evaluate_requires('a == WIN or b == WIN',
                                    {'a': 'LOSE', 'b': 'WIN'})
        assert not hs.evaluate_requires('a == WIN or b == WIN',
                                        {'a': 'LOSE', 'b': 'LOSE'})

    def test_parens(self):
        assert hs.evaluate_requires('(a == WIN or b == WIN) and c != LOSE',
                                    {'a': 'WIN', 'b': 'LOSE', 'c': 'TIE'})

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            hs.evaluate_requires('a == ?', {'a': 'WIN'})


# ==========================================================================
# Kill rules
# ==========================================================================

class TestKillRules:
    @pytest.fixture
    def tunables(self):
        return copy.deepcopy(hs.DEFAULT_TUNABLES)

    def test_s1_min_f1_floor_at_038(self, tunables):
        # S1 floor = 0.38. Below kills, at/above passes.
        cum = {'macro_f1': 0.74, 'min_f1': 0.42, 'accuracy': 0.76, 'top2_accuracy': 0.94}
        latest = make_serial(1, 0.74, 0.379)
        kill, reason = hs.check_kill(1, cum, latest, 0.7447, tunables)
        assert kill, reason
        assert 'min F1 floor' in reason

        latest = make_serial(1, 0.74, 0.38)
        kill, _ = hs.check_kill(1, cum, latest, 0.7447, tunables)
        assert not kill  # 0.38 == threshold, not below

        latest = make_serial(1, 0.74, 0.50)
        kill, _ = hs.check_kill(1, cum, latest, 0.7447, tunables)
        assert not kill

    def test_s2_min_f1_floor_at_040(self, tunables):
        cum = {'macro_f1': 0.74, 'min_f1': 0.45, 'accuracy': 0.76, 'top2_accuracy': 0.94}
        latest = make_serial(2, 0.74, 0.399)
        kill, reason = hs.check_kill(2, cum, latest, 0.7447, tunables)
        assert kill, reason

        latest = make_serial(2, 0.74, 0.40)
        kill, _ = hs.check_kill(2, cum, latest, 0.7447, tunables)
        assert not kill

    def test_s1_macro_never_kills(self, tunables):
        # No s1 entry in tolerance map. Even big deficit shouldn't kill on macro.
        cum = {'macro_f1': 0.50, 'min_f1': 0.45, 'accuracy': 0.55, 'top2_accuracy': 0.85}
        latest = make_serial(1, 0.50, 0.45)
        kill, _ = hs.check_kill(1, cum, latest, 0.7447, tunables)
        assert not kill

    def test_s2_macro_tolerance_25pp(self, tunables):
        # ref - 0.025 trips kill, ref - 0.024 doesn't.
        ref = 0.7447
        cum = {'macro_f1': ref - 0.0251, 'min_f1': 0.45, 'accuracy': 0.76, 'top2_accuracy': 0.94}
        latest = make_serial(2, cum['macro_f1'], cum['min_f1'])
        kill, reason = hs.check_kill(2, cum, latest, ref, tunables)
        assert kill, reason
        assert 'macro tolerance' in reason

        cum['macro_f1'] = ref - 0.024
        latest = make_serial(2, cum['macro_f1'], cum['min_f1'])
        kill, _ = hs.check_kill(2, cum, latest, ref, tunables)
        assert not kill

    def test_s3_macro_tolerance_15pp(self, tunables):
        ref = 0.7447
        cum = {'macro_f1': ref - 0.0151, 'min_f1': 0.45, 'accuracy': 0.76, 'top2_accuracy': 0.94}
        latest = make_serial(3, cum['macro_f1'], cum['min_f1'])
        kill, _ = hs.check_kill(3, cum, latest, ref, tunables)
        assert kill

        cum['macro_f1'] = ref - 0.014
        latest = make_serial(3, cum['macro_f1'], cum['min_f1'])
        kill, _ = hs.check_kill(3, cum, latest, ref, tunables)
        assert not kill

    def test_s4_macro_tolerance_07pp(self, tunables):
        ref = 0.7447
        cum = {'macro_f1': ref - 0.0071, 'min_f1': 0.45, 'accuracy': 0.76, 'top2_accuracy': 0.94}
        latest = make_serial(4, cum['macro_f1'], cum['min_f1'])
        kill, _ = hs.check_kill(4, cum, latest, ref, tunables)
        assert kill

        cum['macro_f1'] = ref - 0.006
        latest = make_serial(4, cum['macro_f1'], cum['min_f1'])
        kill, _ = hs.check_kill(4, cum, latest, ref, tunables)
        assert not kill

    def test_s5_no_macro_kill(self, tunables):
        ref = 0.7447
        cum = {'macro_f1': ref - 0.05, 'min_f1': 0.45, 'accuracy': 0.76, 'top2_accuracy': 0.94}
        latest = make_serial(5, cum['macro_f1'], cum['min_f1'])
        kill, _ = hs.check_kill(5, cum, latest, ref, tunables)
        assert not kill  # only min F1 floor still applies on S5


# ==========================================================================
# Verdict computation
# ==========================================================================

class TestVerdict:
    @pytest.fixture
    def tunables(self):
        return copy.deepcopy(hs.DEFAULT_TUNABLES)

    def test_win(self, tunables):
        mean = {'macro_f1': 0.7500, 'min_f1': 0.4779}
        verdict = hs.compute_verdict(False, 5, mean, 0.7447, 0.4779, tunables)
        assert verdict == 'WIN'

    def test_tie_when_within_05pp(self, tunables):
        mean = {'macro_f1': 0.7449, 'min_f1': 0.4779}
        verdict = hs.compute_verdict(False, 5, mean, 0.7447, 0.4779, tunables)
        assert verdict == 'TIE'

    def test_lose_when_macro_drops_more_than_05pp(self, tunables):
        mean = {'macro_f1': 0.7390, 'min_f1': 0.4779}
        verdict = hs.compute_verdict(False, 5, mean, 0.7447, 0.4779, tunables)
        assert verdict == 'LOSE'

    def test_lose_when_min_collapses(self, tunables):
        # Macro just barely WIN, but min drops 0.6%; WIN min_delta is -0.005,
        # so this should fall through to TIE then LOSE.
        mean = {'macro_f1': 0.7497, 'min_f1': 0.4719}  # min_delta = -0.006
        verdict = hs.compute_verdict(False, 5, mean, 0.7447, 0.4779, tunables)
        # macro_delta = 0.005 (right at WIN threshold), min_delta = -0.006
        # min_delta -0.006 < win_min_delta -0.005, so not WIN
        # |macro_delta| = 0.005 < tie_macro_delta 0.005 is False (not strictly less)
        # So it's LOSE. Boundary case.
        assert verdict == 'LOSE'

    def test_killed_always_lose(self, tunables):
        mean = {'macro_f1': 0.80, 'min_f1': 0.55}  # would otherwise WIN
        assert hs.compute_verdict(True, 3, mean, 0.7447, 0.4779, tunables) == 'LOSE'

    def test_partial_serials_lose(self, tunables):
        mean = {'macro_f1': 0.80, 'min_f1': 0.55}
        assert hs.compute_verdict(False, 3, mean, 0.7447, 0.4779, tunables) == 'LOSE'

    def test_none_mean_lose(self, tunables):
        assert hs.compute_verdict(False, 0, None, 0.7447, 0.4779, tunables) == 'LOSE'


# ==========================================================================
# Reductions: cumulative_mean, per_class_mean, per_seed_stdev, top_movers
# ==========================================================================

class TestReductions:
    def test_cumulative_mean(self):
        serials = [
            make_serial(1, 0.74, 0.45),
            make_serial(2, 0.76, 0.49),
        ]
        mean = hs.cumulative_mean(serials)
        assert mean['macro_f1'] == pytest.approx(0.75)
        assert mean['min_f1'] == pytest.approx(0.47)

    def test_cumulative_mean_empty(self):
        assert hs.cumulative_mean([]) is None

    def test_per_class_mean(self):
        s1 = make_serial(1, 0.74, 0.45)
        s2 = make_serial(2, 0.76, 0.49)
        s2['metrics']['per_class_f1']['smash'] = 0.51
        # s1 smash = 0.45 (set as min), s2 smash = 0.51, mean = 0.48
        mean = hs.per_class_mean([s1, s2])
        assert mean['smash'] == pytest.approx(0.48)

    def test_per_seed_stdev(self):
        serials = [
            make_serial(1, 0.74, 0.45),
            make_serial(2, 0.76, 0.49),
            make_serial(3, 0.75, 0.47),
        ]
        std = hs.per_seed_stdev(serials, 'macro_f1')
        # vals = [0.74, 0.76, 0.75], mean = 0.75, var = (0.0001 + 0.0001) / 2 = 0.0001
        # std = 0.01
        assert std == pytest.approx(0.01)

    def test_per_seed_stdev_too_few(self):
        assert hs.per_seed_stdev([make_serial(1, 0.74, 0.45)], 'macro_f1') == 0.0
        assert hs.per_seed_stdev([], 'macro_f1') == 0.0

    def test_top_movers(self):
        # Use ref of all zeros so deltas equal cell values exactly: avoids
        # float-subtraction noise that otherwise breaks abs() tie-equality.
        cell_pc = {'a': 0.10, 'b': 0.05, 'c': 0.05, 'd': 0.05}
        ref_pc = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}
        movers = hs.top_movers(cell_pc, ref_pc, n=3)
        # 'a' wins on magnitude; b/c/d tied at +0.05 are broken alphabetically.
        assert movers[0][0] == 'a'
        assert movers[0][1] == pytest.approx(0.10)
        assert [m[0] for m in movers[1:]] == ['b', 'c']

    def test_pick_best_serial(self):
        # Highest min F1 wins. Macro tiebreaker.
        serials = [
            make_serial(1, 0.75, 0.45),
            make_serial(2, 0.74, 0.50),  # winner: highest min
            make_serial(3, 0.76, 0.49),
        ]
        best = hs.pick_best_serial(serials)
        assert best['serial_no'] == 2

    def test_pick_best_serial_tie_on_min(self):
        # Two with same min, macro breaks the tie.
        serials = [
            make_serial(1, 0.75, 0.50),
            make_serial(2, 0.78, 0.50),  # winner: same min, higher macro
        ]
        best = hs.pick_best_serial(serials)
        assert best['serial_no'] == 2


# ==========================================================================
# State IO
# ==========================================================================

class TestStateIO:
    def test_roundtrip(self, tmp_path):
        state = {'session_name': 'x', 'cells': {'a': {'status': 'pending'}}}
        hs.save_state(tmp_path, state)
        assert (tmp_path / 'state.json').exists()
        assert not (tmp_path / 'state.json.tmp').exists()
        loaded = hs.load_state(tmp_path)
        assert loaded == state

    def test_load_missing_returns_none(self, tmp_path):
        assert hs.load_state(tmp_path) is None


# ==========================================================================
# Search log rendering
# ==========================================================================

class TestSearchLog:
    def test_renders_summary_and_cell_section(self, tmp_path, baseline_runs):
        config = make_session_config()
        session_dir = make_session_dir(tmp_path, config)
        state = hs.init_state(config, session_dir)
        # Mark first cell complete with WIN.
        cs = state['cells']['p_flip_25']
        cs['status'] = 'complete'
        cs['run_id'] = 'run_p_flip_25_test'
        cs['augmentation'] = {'p_flip': 0.25, 'p_jitter': 0.3, 'cap_y': 0.05,
                              'cap_x': 0.10, 'eps': 0.15}
        cs['serials_done'] = 5
        cs['serials'] = [make_serial(i, 0.75 + 0.001 * i, 0.49) for i in range(1, 6)]
        cs['cumulative_mean'] = hs.cumulative_mean(cs['serials'])
        cs['mean'] = cs['cumulative_mean']
        cs['per_class_mean'] = hs.per_class_mean(cs['serials'])
        cs['kill_ref_macro'] = state['current_best_mean']['macro_f1']
        cs['verdict_ref_macro'] = state['current_best_mean']['macro_f1']
        cs['verdict_ref_min'] = state['current_best_mean']['min_f1']
        cs['verdict_ref_per_class'] = dict(state['current_best_per_class'])
        cs['top_movers'] = hs.top_movers(cs['per_class_mean'], cs['verdict_ref_per_class'])
        cs['best_serial'] = 5
        cs['verdict'] = 'WIN'
        cs['macro_stdev'] = 0.005

        hs.write_search_log(session_dir, state, hs.DEFAULT_TUNABLES)
        body = (session_dir / 'manifest.md').read_text()
        assert '# Hparam search' in body
        assert 'p_flip_25' in body
        assert 'WIN' in body
        assert 'PICK: S5' in body
        assert 'Top movers' in body
        # Pending cell shouldn't get a section.
        assert '## Cell: cap_bump' not in body


# ==========================================================================
# init_state
# ==========================================================================

class TestInitState:
    def test_init_state_pulls_reference_means(self, tmp_path, baseline_runs):
        config = make_session_config()
        session_dir = make_session_dir(tmp_path, config)
        state = hs.init_state(config, session_dir)
        assert state['session_name'] == 'test_session'
        assert state['current_best_run'] == 'run_current_best'
        # Means computed from the fake manifest's 5 serials.
        assert state['current_best_mean']['macro_f1'] == pytest.approx(0.7447, abs=1e-4)
        assert 'cells' in state
        assert state['cells']['p_flip_25']['status'] == 'pending'
        assert state['cells']['p_flip_25']['serials_done'] == 0


# ==========================================================================
# Reference promotion
# ==========================================================================

class TestPromotion:
    def test_promotion_only_when_complete(self, tmp_path, baseline_runs, monkeypatch):
        """Killed cells with high partial means must NOT promote."""
        # Stub invoke_bst_train so the cell appears to run two serials at very
        # high means, then trip the min F1 floor.
        config = make_session_config(cells=[
            {'name': 'cell_a', 'augmentation': {'p_flip': 0.25}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # Two serials @ macro 0.8 (way above ref) but min F1 low enough to trip.
        responses = [
            (1, 0.80, 0.39),  # min < 0.40 still passes S1 (0.38 floor)
            (2, 0.80, 0.39),  # min < 0.40 trips S2 floor
        ]

        def fake_invoke(serial_no, run_id, log_path, augmentation):
            metrics_macro, metrics_min = next(
                (m, mn) for s, m, mn in responses if s == serial_no
            )
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            manifest_path = run_dir / 'manifest.yaml'
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f) or {}
            else:
                manifest = {'run_id': run_id, 'serials': []}
            manifest['serials'].append(make_serial(serial_no, metrics_macro, metrics_min))
            with open(manifest_path, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        # Cell killed, so current_best should NOT have promoted despite the
        # 0.80 macro on two serials.
        assert state['cells']['cell_a']['status'] == 'killed'
        assert state['current_best_run'] == 'run_current_best'

    def test_complete_cell_with_higher_macro_promotes(self, tmp_path,
                                                      baseline_runs, monkeypatch):
        """Even TIE verdicts promote if macro mean exceeds current best."""
        config = make_session_config(cells=[
            {'name': 'tie_winner', 'augmentation': {'p_flip': 0.25}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # 5 serials yielding mean macro just above ref but within TIE band.
        # Ref is 0.7447. Mean macro 0.7449 → TIE but should promote.
        macro_vals = [0.7449] * 5
        min_vals = [0.4779] * 5

        def fake_invoke(serial_no, run_id, log_path, augmentation):
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            manifest_path = run_dir / 'manifest.yaml'
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f) or {}
            else:
                manifest = {'run_id': run_id, 'serials': []}
            manifest['serials'].append(
                make_serial(serial_no, macro_vals[serial_no - 1], min_vals[serial_no - 1])
            )
            with open(manifest_path, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        cell = state['cells']['tie_winner']
        assert cell['status'] == 'complete'
        assert cell['serials_done'] == 5
        assert cell['verdict'] == 'TIE'
        # Promoted despite TIE because macro 0.7449 > ref 0.7447.
        assert state['current_best_run'] == cell['run_id']


# ==========================================================================
# Snapshot semantics: cell-start ref doesn't move during cell
# ==========================================================================

class TestSnapshot:
    def test_cell_start_ref_locked(self, tmp_path, baseline_runs, monkeypatch):
        """Cell N's kill_ref must be set at cell start and not change even if
        a previous cell promoted. Test by running cell A (which promotes),
        then checking cell B's kill_ref reflects the post-A current_best.
        """
        config = make_session_config(cells=[
            {'name': 'cell_a', 'augmentation': {'p_flip': 0.25}},
            {'name': 'cell_b', 'augmentation': {'cap_y': 0.075}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # cell_a promotes (clear WIN above ref); cell_b is the test subject.
        a_macro = 0.7600
        b_macro = 0.7500

        def fake_invoke(serial_no, run_id, log_path, augmentation):
            # cell_a override sets p_flip=0.25; cell_b inherits base p_flip=0.5.
            macro = a_macro if augmentation['p_flip'] == 0.25 else b_macro
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            manifest_path = run_dir / 'manifest.yaml'
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f) or {}
            else:
                manifest = {'run_id': run_id, 'serials': []}
            manifest['serials'].append(make_serial(serial_no, macro, 0.4800))
            with open(manifest_path, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        cell_a = state['cells']['cell_a']
        cell_b = state['cells']['cell_b']

        # cell_a promoted (macro 0.7600 > 0.7447).
        assert state['current_best_run'] == cell_a['run_id']

        # cell_b's kill_ref is the post-cell_a current_best (0.7600), not the
        # session-start 0.7447.
        assert cell_b['kill_ref_macro'] == pytest.approx(0.7600, abs=1e-4)


# ==========================================================================
# Resume reconciliation
# ==========================================================================

class TestResume:
    def test_resumes_partial_cell_from_manifest(self, tmp_path, baseline_runs,
                                                monkeypatch):
        """If state.json says cell ran 0 serials but manifest has 3, wrapper
        picks up at S4."""
        config = make_session_config(cells=[
            {'name': 'cell_a', 'augmentation': {'p_flip': 0.25}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # Pre-populate state with cell_a 'running' but only 3 serials in
        # manifest (simulating a death between S3 finish and state save).
        state = hs.init_state(config, session_dir)
        cell_state = state['cells']['cell_a']
        run_id = 'run_resume_test'
        cell_state['run_id'] = run_id
        cell_state['log_path'] = str(tmp_path / 'log.log')
        cell_state['augmentation'] = {'p_flip': 0.25, 'p_jitter': 0.3,
                                       'cap_y': 0.05, 'cap_x': 0.10, 'eps': 0.15}
        cell_state['status'] = 'running'
        # state says 0 serials done; manifest says 3.
        cell_state['serials_done'] = 0
        cell_state['kill_ref_macro'] = state['current_best_mean']['macro_f1']
        cell_state['verdict_ref_macro'] = state['current_best_mean']['macro_f1']
        cell_state['verdict_ref_min'] = state['current_best_mean']['min_f1']
        cell_state['verdict_ref_per_class'] = dict(state['current_best_per_class'])
        hs.save_state(session_dir, state)

        # Write 3 fake serials into the cell's manifest.
        write_fake_run(baseline_runs, run_id,
                       [make_serial(i, 0.75, 0.50) for i in range(1, 4)])

        # Track the serial numbers fake_invoke is called with.
        invocations = []
        def fake_invoke(serial_no, run_id, log_path, augmentation):
            invocations.append(serial_no)
            run_dir = baseline_runs / run_id
            with open(run_dir / 'manifest.yaml') as f:
                manifest = yaml.safe_load(f)
            manifest['serials'].append(make_serial(serial_no, 0.75, 0.50))
            with open(run_dir / 'manifest.yaml', 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        # Wrapper should have invoked S4 and S5 only, not S1-3.
        assert invocations == [4, 5]


# ==========================================================================
# Conditional skipping
# ==========================================================================

class TestConditionalSkipping:
    def test_child_skipped_when_parent_loses(self, tmp_path, baseline_runs,
                                             monkeypatch):
        config = make_session_config(cells=[
            {'name': 'parent', 'augmentation': {'p_flip': 0.25}},
            {'name': 'child', 'requires': 'parent != LOSE',
             'augmentation': {'cap_y': 0.075}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # parent gets killed at S1 by min floor 0.38.
        def fake_invoke(serial_no, run_id, log_path, augmentation):
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            manifest_path = run_dir / 'manifest.yaml'
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f) or {}
            else:
                manifest = {'run_id': run_id, 'serials': []}
            manifest['serials'].append(make_serial(serial_no, 0.74, 0.30))  # min 0.30 < 0.38 S1 floor
            with open(manifest_path, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        assert state['cells']['parent']['status'] == 'killed'
        assert state['cells']['parent']['verdict'] == 'LOSE'
        assert state['cells']['child']['status'] == 'skipped'

    def test_child_runs_when_parent_ties(self, tmp_path, baseline_runs,
                                         monkeypatch):
        config = make_session_config(cells=[
            {'name': 'parent', 'augmentation': {'p_flip': 0.25}},
            {'name': 'child', 'requires': 'parent != LOSE',
             'augmentation': {'cap_y': 0.075}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # parent gets a TIE (mean macro very close to ref). child should run.
        def fake_invoke(serial_no, run_id, log_path, augmentation):
            macro = 0.7448 if augmentation['p_flip'] == 0.25 else 0.7449
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            manifest_path = run_dir / 'manifest.yaml'
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f) or {}
            else:
                manifest = {'run_id': run_id, 'serials': []}
            manifest['serials'].append(make_serial(serial_no, macro, 0.4779))
            with open(manifest_path, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        assert state['cells']['parent']['verdict'] == 'TIE'
        assert state['cells']['child']['status'] == 'complete'


# ==========================================================================
# Robustness: malformed state.json, manifest, config drift
# ==========================================================================

class TestRobustness:
    def test_corrupt_state_json_fails_loudly(self, tmp_path):
        """Malformed state.json triggers a clear RuntimeError, not a stack trace."""
        (tmp_path / 'state.json').write_text('{not json')
        with pytest.raises(RuntimeError, match='malformed'):
            hs.load_state(tmp_path)

    def test_missing_metrics_key_in_manifest_fails_loudly(self, tmp_experiments):
        """A serial entry missing macro_f1 should error with a clear message."""
        bad = [{
            'serial_no': 1,
            'metrics': {'min_f1': 0.5, 'accuracy': 0.7, 'top2_accuracy': 0.9,
                        'per_class_f1': {}},  # macro_f1 missing
        }]
        write_fake_run(tmp_experiments, 'run_bad', bad)
        with pytest.raises(RuntimeError, match='missing metrics keys'):
            hs.read_cell_serials('run_bad')

    def test_added_cell_in_config_after_session_start(self, tmp_path, baseline_runs,
                                                      monkeypatch):
        """Editing config.yaml to add a cell mid-session should refuse."""
        config = make_session_config(cells=[
            {'name': 'a', 'augmentation': {'p_flip': 0.25}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # First pass: complete cell 'a'.
        def fake_invoke(serial_no, run_id, log_path, augmentation):
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            mp = run_dir / 'manifest.yaml'
            manifest = {'serials': []} if not mp.exists() else (yaml.safe_load(mp.read_text()) or {'serials': []})
            manifest['serials'].append(make_serial(serial_no, 0.7480, 0.4800))
            with open(mp, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)
        assert hs.load_state(session_dir)['cells']['a']['status'] == 'complete'

        # Now edit config to add a cell. Re-running should refuse.
        config['cells'].append({'name': 'b', 'augmentation': {'cap_y': 0.075}})
        with open(session_dir / 'config.yaml', 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        with pytest.raises(RuntimeError, match='cells added'):
            hs.run_session(session_dir)

    def test_bst_train_nonzero_marks_cell_failed_advances(self, tmp_path,
                                                          baseline_runs,
                                                          monkeypatch):
        """A non-zero bst_train rc should fail this cell, not the session."""
        config = make_session_config(cells=[
            {'name': 'crash_cell', 'augmentation': {'p_flip': 0.25}},
            {'name': 'next_cell', 'augmentation': {'cap_y': 0.075}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        def fake_invoke(serial_no, run_id, log_path, augmentation):
            # crash_cell's first serial returns non-zero, simulating CUDA OOM.
            if augmentation['p_flip'] == 0.25:
                return 1
            # next_cell runs cleanly.
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            mp = run_dir / 'manifest.yaml'
            manifest = {'serials': []} if not mp.exists() else (yaml.safe_load(mp.read_text()) or {'serials': []})
            manifest['serials'].append(make_serial(serial_no, 0.7480, 0.4800))
            with open(mp, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        assert state['cells']['crash_cell']['status'] == 'failed'
        assert state['cells']['crash_cell']['verdict'] == 'LOSE'
        assert 'exited with code 1' in state['cells']['crash_cell']['failed_reason']
        assert state['cells']['next_cell']['status'] == 'complete'

    def test_session_lock_blocks_concurrent_run(self, tmp_path, baseline_runs):
        config = make_session_config()
        session_dir = make_session_dir(tmp_path, config)
        # Pre-create lock file with a live PID (our own).
        (session_dir / '.lock').write_text(str(__import__('os').getpid()))
        with pytest.raises(RuntimeError, match='already running'):
            hs.run_session(session_dir)

    def test_session_lock_clears_on_stale_pid(self, tmp_path, baseline_runs,
                                              monkeypatch):
        config = make_session_config(cells=[{'name': 'a', 'augmentation': {'p_flip': 0.25}}])
        session_dir = make_session_dir(tmp_path, config)
        # Stale pid: 99999 unlikely to be alive.
        (session_dir / '.lock').write_text('99999')

        def fake_invoke(serial_no, run_id, log_path, augmentation):
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            mp = run_dir / 'manifest.yaml'
            manifest = {'serials': []} if not mp.exists() else (yaml.safe_load(mp.read_text()) or {'serials': []})
            manifest['serials'].append(make_serial(serial_no, 0.7480, 0.4800))
            with open(mp, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)
        assert hs.load_state(session_dir)['cells']['a']['status'] == 'complete'
        # Lock should be released on clean exit.
        assert not (session_dir / '.lock').exists()

    def test_missing_reference_run_raises_friendly_error(self, tmp_path,
                                                         tmp_experiments):
        """Typo in current_best_run should produce a clear error, not a stack trace."""
        config = make_session_config()
        config['reference']['current_best_run'] = 'run_typo_does_not_exist'
        session_dir = make_session_dir(tmp_path, config)
        with pytest.raises(FileNotFoundError, match='current_best_run'):
            hs.init_state(config, session_dir)

    def test_skipped_parent_appears_in_requires_namespace(self, tmp_path,
                                                          baseline_runs,
                                                          monkeypatch):
        """A child whose requires references a skipped parent should evaluate
        without NameError."""
        config = make_session_config(cells=[
            {'name': 'a', 'augmentation': {'p_flip': 0.25}},
            {'name': 'b', 'requires': 'a == WIN', 'augmentation': {'cap_y': 0.075}},
            {'name': 'c', 'requires': 'b != LOSE', 'augmentation': {'cap_x': 0.15}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # 'a' completes with TIE → 'b' skipped (requires a == WIN fails).
        # 'c' requires b != LOSE; b's status is skipped, so namespace has
        # b='SKIPPED', != LOSE evaluates True, c runs.
        def fake_invoke(serial_no, run_id, log_path, augmentation):
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            mp = run_dir / 'manifest.yaml'
            manifest = {'serials': []} if not mp.exists() else (yaml.safe_load(mp.read_text()) or {'serials': []})
            macro = 0.7449  # TIE on every cell
            manifest['serials'].append(make_serial(serial_no, macro, 0.4779))
            with open(mp, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        assert state['cells']['a']['verdict'] == 'TIE'
        assert state['cells']['b']['status'] == 'skipped'
        # c's requires references skipped b; should resolve cleanly to 'b != LOSE' True
        # and run the cell to completion.
        assert state['cells']['c']['status'] == 'complete'

    def test_removed_cell_in_config_after_session_start(self, tmp_path, baseline_runs):
        """Removing a cell from config mid-session should refuse."""
        config = make_session_config(cells=[
            {'name': 'a', 'augmentation': {'p_flip': 0.25}},
            {'name': 'b', 'augmentation': {'cap_y': 0.075}},
        ])
        session_dir = make_session_dir(tmp_path, config)
        # Initialise state.
        state = hs.init_state(config, session_dir)
        hs.save_state(session_dir, state)
        # Edit config: remove 'b'.
        config['cells'] = [config['cells'][0]]
        with open(session_dir / 'config.yaml', 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        with pytest.raises(RuntimeError, match='cells removed'):
            hs.run_session(session_dir)


# ==========================================================================
# Verdict boundary cases (T6)
# ==========================================================================

class TestVerdictBoundaries:
    @pytest.fixture
    def tunables(self):
        return copy.deepcopy(hs.DEFAULT_TUNABLES)

    def test_macro_delta_exactly_at_win_threshold(self, tunables):
        """+0.005 macro with min held → WIN (>= boundary)."""
        mean = {'macro_f1': 0.7497, 'min_f1': 0.4779}
        verdict = hs.compute_verdict(False, 5, mean, 0.7447, 0.4779, tunables)
        assert verdict == 'WIN'

    def test_macro_just_below_win_threshold_with_clean_min(self, tunables):
        """+0.0049 macro → not WIN, |delta| < 0.005 → TIE."""
        mean = {'macro_f1': 0.7496, 'min_f1': 0.4779}
        verdict = hs.compute_verdict(False, 5, mean, 0.7447, 0.4779, tunables)
        assert verdict == 'TIE'

    def test_win_macro_with_min_at_minus_006_falls_to_lose(self, tunables):
        """+0.006 macro WIN-eligible, but min -0.006 fails WIN min guard;
        |macro_delta| 0.006 > tie_macro_delta 0.005 fails TIE → LOSE."""
        mean = {'macro_f1': 0.7507, 'min_f1': 0.4719}
        verdict = hs.compute_verdict(False, 5, mean, 0.7447, 0.4779, tunables)
        assert verdict == 'LOSE'


# ==========================================================================
# Multi-cell promotion chain (T4)
# ==========================================================================

class TestPromotionChain:
    def test_three_cells_promote_in_sequence(self, tmp_path, baseline_runs,
                                             monkeypatch):
        """A→B→C, each lifts the bar. C's kill_ref should reflect B's mean."""
        config = make_session_config(cells=[
            {'name': 'a', 'augmentation': {'p_flip': 0.30}},
            {'name': 'b', 'augmentation': {'p_flip': 0.25}},
            {'name': 'c', 'augmentation': {'p_flip': 0.20}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # Each cell promotes a bit higher.
        macro_by_aug = {0.30: 0.7500, 0.25: 0.7600, 0.20: 0.7700}

        def fake_invoke(serial_no, run_id, log_path, augmentation):
            macro = macro_by_aug[augmentation['p_flip']]
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            mp = run_dir / 'manifest.yaml'
            manifest = {'serials': []} if not mp.exists() else (yaml.safe_load(mp.read_text()) or {'serials': []})
            manifest['serials'].append(make_serial(serial_no, macro, 0.50))
            with open(mp, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        assert state['cells']['a']['kill_ref_macro'] == pytest.approx(0.7447, abs=1e-4)
        assert state['cells']['b']['kill_ref_macro'] == pytest.approx(0.7500, abs=1e-4)
        assert state['cells']['c']['kill_ref_macro'] == pytest.approx(0.7600, abs=1e-4)
        # current_best ends up pointing at c.
        assert state['current_best_run'] == state['cells']['c']['run_id']


# ==========================================================================
# Variance-warn flag (T2)
# ==========================================================================

class TestVarianceWarning:
    def test_high_variance_cell_flags_in_state(self, tmp_path, baseline_runs,
                                                monkeypatch):
        """A cell with macro stdev > 0.010 should record macro_stdev > threshold."""
        config = make_session_config(cells=[
            {'name': 'noisy', 'augmentation': {'p_flip': 0.25}},
        ])
        session_dir = make_session_dir(tmp_path, config)

        # Macros with stdev ~0.02 (well above 0.010 threshold).
        macros_per_serial = {1: 0.72, 2: 0.76, 3: 0.74, 4: 0.78, 5: 0.73}

        def fake_invoke(serial_no, run_id, log_path, augmentation):
            macro = macros_per_serial[serial_no]
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            mp = run_dir / 'manifest.yaml'
            manifest = {'serials': []} if not mp.exists() else (yaml.safe_load(mp.read_text()) or {'serials': []})
            manifest['serials'].append(make_serial(serial_no, macro, 0.50))
            with open(mp, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        cell = state['cells']['noisy']
        assert cell['macro_stdev'] > 0.010
        # Search log includes the advisory note.
        log_text = (session_dir / 'manifest.md').read_text()
        assert 'verdict is advisory' in log_text


# ==========================================================================
# End-to-end happy path
# ==========================================================================

class TestEndToEnd:
    def test_two_cells_complete(self, tmp_path, baseline_runs, monkeypatch):
        config = make_session_config()
        session_dir = make_session_dir(tmp_path, config)

        def fake_invoke(serial_no, run_id, log_path, augmentation):
            macro = 0.7480
            run_dir = baseline_runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'weights').mkdir(exist_ok=True)
            manifest_path = run_dir / 'manifest.yaml'
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f) or {}
            else:
                manifest = {'run_id': run_id, 'serials': []}
            manifest['serials'].append(make_serial(serial_no, macro, 0.4800))
            with open(manifest_path, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            return 0

        monkeypatch.setattr(hs, 'invoke_bst_train', fake_invoke)
        hs.run_session(session_dir)

        state = hs.load_state(session_dir)
        for cell_name in ['p_flip_25', 'cap_bump']:
            cell = state['cells'][cell_name]
            assert cell['status'] == 'complete'
            assert cell['serials_done'] == 5
            assert cell['verdict'] in ('WIN', 'TIE', 'LOSE')

        # Search log written.
        assert (session_dir / 'manifest.md').exists()
        # State.json written.
        assert (session_dir / 'state.json').exists()
