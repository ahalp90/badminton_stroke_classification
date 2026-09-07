from scratch.contact_det_closing_pass.scripts.score_vlm_acceptance import visual_gate


def test_visual_gate_scales_once_and_converts_clip_index_to_source_frame() -> None:
    assert visual_gate({"serve_state": "visible", "contact_frame": 40}, 120, 100, 60) == (
        True, "timing_agrees", 140,
    )
    assert visual_gate({"serve_state": "visible", "contact_frame": 41}, 120, 100, 60) == (
        False, "timing_disagrees", 141,
    )


def test_uncertain_and_missing_replies_preserve_the_tree_decision() -> None:
    assert visual_gate(None, 100, 20, 30)[0] is True
    assert visual_gate({"serve_state": "unclear", "contact_frame": None}, 100, 20, 30)[0] is True
    assert visual_gate({"serve_state": "off_frame", "contact_frame": None}, 100, 20, 30)[0] is False
