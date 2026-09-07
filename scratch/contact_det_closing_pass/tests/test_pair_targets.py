"""Compare the pair eligibility shortcut with the existing whole-rally target."""

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedSpan, RallyReference
from scratch.contact_det_closing_pass.scripts.later_options import build_later_options
from scratch.contact_det_closing_pass.scripts.pair_targets import pair_targets
from scratch.contact_det_closing_pass.scripts.whole_rally_options import whole_targets
from scratch.contact_det_closing_pass.tests.test_later_options import event
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels


def test_pair_target_shortcut_matches_existing_evaluator() -> None:
    span = FixedSpan("fixture", 0, 0, 120, (event(20), event(100, "Bot")))
    base = CombinedAction("keep", None, None, span)
    options = build_later_options((base,), {("fixture", 0): (event(45, "Bot"), event(75), event(60))},
                                 {"fixture": 60}, max_insertions=2)
    singles = [option for option in options if option.second_inserted is None]
    pairs = [option for option in options if option.second_inserted is not None]
    for frames in ((20, 45, 75, 100), (20, 100), (), (20, 90, 110, 150)):
        labels = HumanLabels(
            {"fixture": () if not frames else (RallyReference("fixture", 0, "one", frames),)},
            {("fixture", frame): "Top" if index % 2 == 0 else "Bot" for index, frame in enumerate(frames)},
        )
        before, _ = whole_targets([option.proxy for option in singles], (span,), labels, {"fixture": 60})
        expected, _ = whole_targets([option.proxy for option in pairs], (span,), labels, {"fixture": 60})
        actual = pair_targets(pairs, singles, before, labels, {"fixture": 60})
        np.testing.assert_array_equal(actual, expected)
