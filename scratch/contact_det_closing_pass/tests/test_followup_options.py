"""Check pair contexts and reconstruction of saved inserted references."""

from scratch.contact_det.scripts.score_contact_rallies import FixedSpan
from scratch.contact_det_closing_pass.scripts.followup_options import (
    contextual_insertions,
    restore_choices,
)
from scratch.contact_det_closing_pass.scripts.later_options import (
    build_later_options,
    option_record,
)
from scratch.contact_det_closing_pass.tests.test_later_options import event
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction


def test_pair_contexts_keep_the_other_insertion_and_restore_saved_output() -> None:
    span = FixedSpan("fixture", 0, 0, 120, (event(20), event(100, "Bot")))
    options = build_later_options(
        (CombinedAction("keep", None, None, span),),
        {("fixture", 0): (event(45, "Bot"), event(75))}, {"fixture": 30}, max_insertions=2,
    )
    pair = options[-1]
    first, second = contextual_insertions((pair,))
    assert [contact.frame for contact in first.base.span.events] == [20, 75, 100]
    assert [contact.frame for contact in second.base.span.events] == [20, 45, 100]
    assert first.span == second.span == pair.span
    assert (first.inserted.frame, second.inserted.frame) == (45, 75)
    for original in (options[1], pair):
        restored = restore_choices(options, [option_record(original)])
        assert restored[("fixture", 0)] == original
