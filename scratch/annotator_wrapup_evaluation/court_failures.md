# Court failures: usable play is being blocked upstream

The court checks asked whether bad outlines were blocking usable play or losing visible players. Replays changed only the outline to see whether the court or player decision changed.

Outside bad-label video 15, the learned annotator misses 3,633 labelled contacts. **2,374 of them are in scenes the court stage rejected before normal contact scoring.** Two different geometry failures are confirmed: one rejects usable play at the scene gate; the other passes the scene, then replaces a good outline with a worse one and loses a visible player.

**Contents**  
[How the gate works](#how-the-gate-works)  
[How much error is upstream](#how-much-error-is-upstream)  
[Video 53: OpenCV breaks the outline](#video-53-opencv-breaks-the-outline)  
[Video 17: the shared outline loses a player](#video-17-the-shared-outline-loses-a-player)  
[How the shared outline works](#how-the-shared-outline-works)  
[What these checks prove](#what-these-checks-prove)  
[What to test in #148](#what-to-test-in-148)  
[Evidence](#evidence)

## How the gate works

For each scene, the annotator estimates a court outline. The saved files call this the **raw** outline, meaning only “before the shared-outline correction”; OpenCV may already have filled some corners.

That outline is used to count frames with exactly two detected people on or near the court. A scene passes when at least half its frames pass. Only then does it reach the video-wide shared-outline correction, normal player tracking and contact search.

So a bad first outline can remove a whole scene before the contact model sees it.

## How much error is upstream

The 46 videos outside video 15 contain 37,184 cleaned contact labels. The learned output matches 33,551 and misses 3,633.

| State at the labelled frame | Missed contacts |
|---|---:|
| Court rejected | **2,374** |
| Court accepted; at least one player missing | 96 |
| Court accepted; both players present | 1,163 |

At 2,277 of the 2,374 court-rejected misses, no scored candidate exists within ±10 frames. A later sequence model cannot choose a contact that never entered scoring.

All 1,259 misses in accepted scenes have saved features and nearby scored candidates. Ninety-six are missing a player pick; 1,163 have both players. Those are real downstream residuals, but they are the minority of current misses.

A filled shuttle coordinate is not evidence that the track is right. It exists at 3,514 of the 3,633 missed times outside video 15; video 15's opening graphics also have a filled coordinate.

The court problem is not just video 53. Remove video 53 as well and 1,640 of 2,891 remaining misses are still in rejected scenes.

## Video 53: OpenCV breaks the outline

In one rejected video 53 scene, the neural net puts the top-right corner in roughly the right place but with low confidence. OpenCV replaces it with a point near the bottom-right of the image. The court becomes almost triangular, the two-player vote fails, and the scene is rejected before the later shared-outline correction can help.

The numbered comparison is in [issue #148](https://github.com/ahalp90/badminton_cv_annotator/issues/148).

The investigation also replayed the original two-player vote in four rejected scenes and four successful controls, then changed only the outline to the existing same-video shared outline:

| Rejected scene | Original outline | Shared outline |
|---|---:|---:|
| Video 12 | 1/377 (0.3%) | 368/377 (97.6%) |
| Video 20 | 0/1,029 (0.0%) | 906/1,029 (88.0%) |
| Video 21 | 716/4,031 (17.8%) | 1,973/4,031 (48.9%) |
| Video 53 | 0/319 (0.0%) | 315/319 (98.7%) |

The four successful controls did not change. Three rejected scenes would cross the existing 50% threshold; video 21 would still fail. Geometry is clearly causing some false rejection, but one substitution rule does not fix everything.

Video 53 should stay in the evaluation. It has 742 missed labels, 734 in court-rejected scenes. In accepted scenes, 195 of 203 labels have a timing match and 194 have the right player. Nine game/score checks and four direct hit/player checks support its labels. This is a hard pipeline case, not another video 15.

## Video 17: the shared outline loses a player

Video 17 fails in the opposite direction. The checked scene has four confident neural-net corners and passes the two-player gate. OpenCV is not involved.

The later shared-outline step then replaces that scene outline with a smaller video-wide outline; the checked outline shift is about 164 image pixels. The player picker projects each detected person's feet into court coordinates; with the shared outline, the visible far player lands beyond the allowed distance.

| Contact frame | Shared outline | Original scene outline | Maximum allowed |
|---|---:|---:|---:|
| 47,276 | 0.706 | 0.280 | 0.600 |
| 46,045 | 0.654 | 0.240 | 0.600 |

Changing only the outline restores the far-player pick in both cases. The check keeps the real incoming tracker state and does not feed the alternative result into later frames, so it isolates the geometry change rather than pretending to repair the rally.

Video 17 accounts for 80 of the 96 accepted-scene misses with a missing player pick outside video 15. These two examples explain a real mechanism, not the rest of the video.

The full replay used the original 38 tracking segments and resets across 213,154 frames. Both player-validity fields matched all 91,970 saved feature rows, so the sampled losses are reproducible from the original logic.

## How the shared outline works

Video 17 has 657 saved scene records:

- 173 outlines use at least one OpenCV-filled corner;
- 107 use neural-net corners only;
- 377 have no outline;
- 38 scenes pass the two-player gate;
- 11 of those accepted outlines are replaced by the shared-outline step.

The shared reference is the median position of each corner across accepted scenes. If any scene corner differs from that reference by more than 55 pixels, the whole scene outline is replaced. If half or more of the accepted scenes disagree, the correction is abandoned.

There is no clustering by camera view. A large shift can therefore mean either “bad outline” or “different valid view”. The checked video 17 scene is the latter.

## What these checks prove

They show that:

- usable live play can be rejected before contact scoring;
- bad per-scene geometry can cause that rejection;
- the shared outline can fit another camera view badly enough to lose a visible player;
- changing only the outline changes the people/player decision in the expected direction on the checked examples.

They do not tell us how many of the 2,374 rejected-scene misses #148 will recover, how much non-play a looser gate might admit, or how many complete rallies will improve. Only an end-to-end rerun can answer those questions.

## What to test in #148

Use video 53 scene 334 and video 17 scene 0 as regression cases. Then rerun the collection and compare:

- fully correct rallies;
- contact timing P/R/F1;
- player-aware contact P/R/F1;
- serves;
- selected correct/wrong/unjudgeable clips;
- newly accepted scenes, including a small human check for both rescued live play and newly admitted non-play.

A fix that makes the two pictures prettier but does not improve annotation—or damages the review queue enough to cancel the gains—is not enough.

Live work: [promising_leads.md](promising_leads.md).

## Evidence

- `results/court_vote_check.csv.gz` — original and substituted two-player votes
- `results/replay_player_sample.csv.gz` / `.json.gz` — video 17 replay and geometry checks
- `results/visual_geometry.json.gz` — inspected outlines in native image coordinates
- `results/video17_court_summary.json.gz` — video 17 court-source summary
- `results/video53_nn_replay.json.gz` — rerun neural-net corners and fallback comparison
- [issue #148](https://github.com/ahalp90/badminton_cv_annotator/issues/148) — published before/after evidence

Commands: [evaluation_reproduction.md](evaluation_reproduction.md).
