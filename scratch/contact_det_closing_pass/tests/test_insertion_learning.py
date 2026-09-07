"""Group-isolated local insertion score caches."""

from types import SimpleNamespace

import numpy as np
import pytest

from scratch.contact_det_closing_pass.scripts import insertion_learning as learning


def test_build_local_cache_trains_only_allowed_rows_and_predicts_outer_group(monkeypatch) -> None:
    fits = []
    predictions = []

    def fake_fit(values, answers):
        fits.append((values.copy(), answers.copy()))
        return SimpleNamespace(name="model"), 0.25

    def fake_scores(model, values):
        predictions.append(values.copy())
        return np.arange(1, len(values) + 1, dtype=np.float64) / 10.0

    monkeypatch.setattr(learning, "fit_whole_model", fake_fit)
    monkeypatch.setattr(learning, "_positive_scores", fake_scores)
    matrix = np.arange(20, dtype=np.float64).reshape(5, 4)
    targets = np.asarray([0, 1, -1, 0, 1], dtype=np.int8)
    groups = np.asarray(["A", "A", "B", "C", "C"])

    cache, models, records = learning.build_local_cache(
        matrix, targets, groups, (frozenset({"B", "C"}),),
    )

    np.testing.assert_array_equal(fits[0][0], matrix[[2, 3, 4]])
    np.testing.assert_array_equal(fits[0][1], targets[[2, 3, 4]])
    np.testing.assert_array_equal(predictions[0], matrix[[0, 1]])
    np.testing.assert_equal(cache[frozenset({"B", "C"})][[0, 1]], [0.1, 0.2])
    assert np.isnan(cache[frozenset({"B", "C"})][[2, 3, 4]]).all()
    assert set(models) == {frozenset({"B", "C"})}
    assert records == [{
        "training_groups": ["B", "C"],
        "predicted_groups": ["A"],
        "training_rows": 3,
        "known_training_rows": 2,
        "predicted_rows": 2,
        "fit_seconds": 0.25,
    }]


def test_build_local_cache_pair_excludes_pair_from_prediction(monkeypatch) -> None:
    fits = []
    predictions = []

    monkeypatch.setattr(
        learning,
        "fit_whole_model",
        lambda values, answers: (fits.append((values.copy(), answers.copy())) or (object(), 1.0)),
    )
    monkeypatch.setattr(
        learning,
        "_positive_scores",
        lambda model, values: predictions.append(values.copy()) or np.ones(len(values)),
    )
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    targets = np.asarray([0, 1, -1, 1], dtype=np.int8)
    groups = np.asarray(["A", "B", "C", "D"])

    learning.build_local_cache(matrix, targets, groups, (frozenset({"A", "B"}),))

    np.testing.assert_array_equal(fits[0][0], matrix[[0, 1]])
    np.testing.assert_array_equal(fits[0][1], targets[[0, 1]])
    np.testing.assert_array_equal(predictions[0], matrix[[2, 3]])


def test_local_training_scores_uses_complementary_caches_and_leaves_outer_nan() -> None:
    groups = np.asarray(["A", "A", "B", "C"])
    cache = {
        frozenset({"B"}): np.asarray([0.4, 0.5, np.nan, np.nan]),
        frozenset({"A"}): np.asarray([np.nan, np.nan, 0.6, np.nan]),
    }

    scores = learning.local_training_scores(groups, cache, frozenset({"A", "B"}))

    np.testing.assert_allclose(scores[:3], [0.4, 0.5, 0.6])
    assert np.isnan(scores[3])


def test_local_training_scores_fails_on_missing_or_nonfinite_requested_score() -> None:
    groups = np.asarray(["A", "B"])
    with pytest.raises(KeyError, match="missing group set"):
        learning.local_training_scores(groups, {}, frozenset({"A"}))
    with pytest.raises(ValueError, match="nonfinite"):
        learning.local_training_scores(
            groups,
            {frozenset(): np.asarray([np.nan, np.nan])},
            frozenset({"A"}),
        )
