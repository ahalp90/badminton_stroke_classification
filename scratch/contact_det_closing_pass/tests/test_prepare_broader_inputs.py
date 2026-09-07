"""Check that contacts outside detected sections remain in the full output."""

import numpy as np

from scratch.contact_det_closing_pass.scripts.prepare_broader_inputs import (
    _validate_contacts,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import SCORE_DTYPE


def test_kept_contact_outside_sections_is_preserved() -> None:
    rows = np.zeros(2, dtype=SCORE_DTYPE)
    rows["frame"] = [12, 25]
    rows["kept"] = True
    rows["contact_score"] = 0.75
    contacts = [
        {"frame": 12, "contact_score": 0.75, "predicted_side": "Top", "span_id": 0},
        {"frame": 25, "contact_score": 0.75, "predicted_side": "Bot", "span_id": None},
    ]
    spans = [{"span_id": 0, "start_frame": 10, "end_frame": 20}]
    actual = _validate_contacts("8", contacts, rows, spans, {12: "Top", 25: "Bot"})
    assert [row["frame"] for row in actual] == [12, 25]
    assert [row["span_id"] for row in actual] == [0, None]
