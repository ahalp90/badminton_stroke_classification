# Fix court corrections that make the outline worse

Published as [issue #148](https://github.com/ahalp90/badminton_cv_annotator/issues/148).

Two court errors in ShuttleSet22 are making the annotator miss usable play. The pictures below show
what happens before and after each correction. The numbers say which corner each point
is meant to mark: 0 top-left, 1 top-right, 2 bottom-right and 3 bottom-left. A bad
correction can put that point somewhere else.

## Video 53: OpenCV moves a corner to the wrong end of the court

The neural net puts the top-right corner near the right place, but gives it low
confidence. OpenCV replaces that corner with a point near the bottom-right of the
picture. The result is an almost triangular court.

![Video 53: numbered neural-net corners before and after OpenCV fallback.](https://raw.githubusercontent.com/ahalp90/badminton_cv_annotator/5ad74c4525731d09a23069e409e5418ed85528d8/scratch/annotator_wrapup_evaluation/figures/video53_nn_to_opencv.png)

The two-player check uses this broken outline and rejects the scene. Rejected scenes
never reach the later shared-outline correction, so that correction cannot rescue it.

## Video 17: the shared outline replaces a better one

The neural net finds a reasonable court outline for this camera view. All four corners
are confident, so OpenCV is not used for this scene.

The scene passes the two-player check. The next step replaces its outline with the
shared outline from accepted scenes across the video. That outline is too small for
this view, and the player picker loses
the far player. Changing back to the original outline restores the player pick in
a check that changes only the court outline.

![Video 17: numbered neural-net corners replaced by an undersized shared outline.](https://raw.githubusercontent.com/ahalp90/badminton_cv_annotator/5ad74c4525731d09a23069e409e5418ed85528d8/scratch/annotator_wrapup_evaluation/figures/video17_nn_to_shared.png)

## What needs fixing

OpenCV should not return a broken court. A usable scene should get a chance to recover
from a bad first outline. The shared outline should not overwrite a better fit for
the current camera view.

Use video 53, scene 334, and video 17, scene 0, to check the fixes. Then check whether
they improve player picks, contacts and complete rallies across the videos. These
examples are confirmed; we do not yet know how many other failures have the same cause.

<details>
<summary>How the pictures were checked</summary>

For video 53, the left panel shows the median neural-net positions across the ten
sampled frames, before the confidence check. The neural net was rerun with the original
weights. The rerun OpenCV fallback matches the saved fallback outline within
0.02 pixels. Corner 1 still falls below the confidence threshold.
The right panel uses the saved OpenCV fallback
outline.

Both video 17 outlines come directly from the saved run. The player check kept the
same incoming tracker state and changed only the outline.

</details>
