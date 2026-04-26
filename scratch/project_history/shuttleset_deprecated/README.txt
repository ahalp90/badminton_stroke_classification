DEPRECATED (2026-04-08)
======================

These files and directories are from the original BST repository and have
been superseded by the refactored pipeline/ modules.  They are kept for
reference only.


Files
-----
gen_my_dataset.py
    Original manual clip generation script (one player/split at a time).
    Superseded by: pipeline/clip_generator.py

get_each_class_total.py
    Aggregates per-class stroke counts from CSV annotations into Excel.
    Superseded by: pipeline/player_mapping.py + pipeline/config.py

utils.py
    Frame/time conversion helpers used by gen_my_dataset.py.
    Superseded by: pipeline/clip_generator.py (contains _frame_to_time())

class_total.xlsx
    Reference spreadsheet with per-class stroke counts from the original
    BST authors.  Was used to verify clip generation and compute class
    weights.  No active code reads this file.

class_total_gen.xlsx
    Output of get_each_class_total.py.  Working copy of per-player stroke
    counts.  No active code reads this file.


Directories
-----------
shuttle_set/
    Empty placeholder output directory for gen_my_dataset.py when run with
    the default clip window.

shuttle_set_between_2_hits_with_max_limits/
    Empty placeholder output directory for gen_my_dataset.py when run with
    the between_2_hits_with_max_limits clip window.

Both directories have been replaced by the pipeline's CLIPS_OUTPUT_DIR
(ShuttleSet/clips/) defined in pipeline/config.py.


Reinstating
-----------
If any of these assets are needed again, restore them to ShuttleSet/ and
ensure the following directory structure exists:

    ShuttleSet/
      shuttle_set/                                  # Only if using gen_my_dataset.py
        {Top,Bottom}_{stroke_type_zh}/*.mp4         #   with default clip window
      shuttle_set_between_2_hits_with_max_limits/   # Only if using gen_my_dataset.py
        {Top,Bottom}_{stroke_type_zh}/*.mp4         #   with max-limits clip window
      class_total.xlsx                              # Reference counts
      class_total_gen.xlsx                          # Generated counts

Note that gen_my_dataset.py uses Chinese stroke-type folder names and
requires additional alignment work documented in its deprecation header.
The current pipeline uses English folder names exclusively.
