"""Check that acceptance evidence comes from discarded automatic candidates."""

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.later_acceptance_features import (
    acceptance_features,
)
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction


def test_selected_contact_is_excluded_from_discarded_gap_evidence() -> None:
    first = FixedEvent("fixture", 20, .8, "Top")
    later = FixedEvent("fixture", 80, .8, "Top")
    inserted = FixedEvent("fixture", 50, .9, "Bot")
    discarded = FixedEvent("fixture", 110, .6, "Bot")
    before = FixedSpan("fixture", 0, 0, 130, (first, later))
    after = FixedSpan("fixture", 0, 0, 130, (first, inserted, later))
    base = CombinedAction("keep", None, None, before)
    options = (LaterOption(base, None, before), LaterOption(base, inserted, after))
    measurements = PhysicalMeasurements(("measurement",), {("fixture", 110): np.array([2.])}, {})
    matrix, names, identities = acceptance_features(
        options, np.array([.7, .9]), {("fixture", 0): options[1]},
        {("fixture", 0): (inserted, discarded)}, {"fixture": 30}, measurements,
    )
    values = dict(zip(names, matrix[0], strict=True))
    assert identities == [("fixture", 0)]
    assert values["discarded_candidate_count"] == 1
    assert values["strongest_discarded_score"] == .6
    assert values["strongest_discarded_left_gap_seconds"] == 1.
    assert values["strongest_discarded__measurement"] == 2.
    assert values["raw_top_vote"] == 1.
    assert np.isclose(values["advantage_over_next_output"], .2)
