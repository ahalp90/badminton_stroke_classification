"""Stacked opening scores must exclude both the row and outer groups."""

from types import SimpleNamespace

import numpy as np

from scratch.contact_det_closing_pass.scripts import whole_rally_learning as learning
from scratch.contact_det_closing_pass.scripts.targets import EditTarget


def row(group: str, index: int) -> SimpleNamespace:
    return SimpleNamespace(identity=(f"fixture_{group}", 0, index, "add"), candidate=SimpleNamespace(group=group))


def test_opening_fit_only_reads_allowed_included_training_rows(monkeypatch) -> None:
    fits = []

    class Model:
        def fit(self, values, answers):
            fits.append((values.copy(), answers.copy()))

    monkeypatch.setattr(learning.start, "make_action_model", lambda spec: Model())
    rows = [row("A", 1), row("A", 2), row("B", 3), row("A", 4)]
    matrix = np.zeros((4, 12))
    matrix[:, 0] = [1, 2, 300, 400]
    targets = {item.identity: EditTarget(index != 3, "test", index == 0, False, 0, 0)
               for index, item in enumerate(rows)}
    learning.fit_opening_models(rows, matrix, targets, frozenset({"A"}), None)
    assert len(fits) == 2
    for values, answers in fits:
        assert values[:, 0].tolist() == [1, 2]
        assert answers.tolist() == [1, 0]
    assert fits[0][0].shape == (2, 10)
    assert fits[1][0].shape == (2, 12)


def test_upper_training_scores_exclude_row_group_and_outer_group() -> None:
    rows = [row(group, index) for index, group in enumerate("ABCD")]
    cache = {}
    for excluded in "BCD":
        trained = frozenset("BCD") - {excluded}
        item = next(item for item in rows if item.candidate.group == excluded)
        cache[trained] = {item.identity: (0.1, 0.2)}
    scores = learning.training_opening_scores(rows, cache, frozenset("BCD"))
    assert set(scores) == {item.identity for item in rows if item.candidate.group != "A"}
