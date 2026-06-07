# T2 Plan: One-pager

Drafted 2026-06-04. Companion to `where_to_next_tri.md`.


## Where we are

BST-X is at 0.84/0.83 macro F1 on the original BST taxonomy (without/with `unknown`) and ~0.75 on the harder 14-class no-sides taxonomy. `wrist_smash` cleared 0.5 min-F1 in T1. The smash-wrist smash confusion is the wall: 2D keypoints don't carry the information to distinguish a wrist-flick from a full-arm swing, and 3D keypoints (per the BST paper) did no better. Loss, capacity, and augmentation tweaks in T1 didn't move it.

BRIC (the RGB alternative) reaches comparable performance at ~10× BST-X's parameter count. T2 builds on BST-X, unless an incoming team-mate wants to take on the BRIC optimisation challenge.


## What we're building

1. **X3D-S wrist-crop fusion.** BST-X's first new input channel: a small RGB stream on the player's wrist. Targeted discrimination on features invisible to keypoints (e.g., forearm fortation). Stage 2.
2. **Classifier that handles amateur footage.** Self-supervised pretrain on amateur skeletons, fine-tune over the expanded pro training set. Requires Hunter Badminton Association annotated clips to assess real-world generalisation. Stage 3.
3. **CrossTrainer-style stroke-quality PoC.** Ego-Exo4D-pretrained multimodal LM head, badminton vocabulary fine-tune, zero-shot to per-stroke clips. Skill-attributes + actionable feedback + proficiency. Stage 4.

Rally-level analysis depends on a second ML team-member joining.


## Deliverables

- rtmlib pose extract migration (no MMPose dependency brittleness)
- X3D-S wrist-crop fusion for BST-X
- Expanded pro training set: ShuttleSet + ShuttleSet22 + VideoBadminton (~74k strokes)
- Amateur video collection from YouTube
- Self-supervised BST-X pretrain on amateur skeletons
- Amateur-video classification fine-tune
- CrossTrainer-style stroke-quality PoC: skill-attributes + actionable feedback + proficiency
- Live inference on user-uploaded videos (FE/integration)
- Hunter Badminton Association annotated amateur clips (val + test for generalisation, from Arthur)
- Final report + handover docs for next year's team
- Stretch/alternative: badminton expert-commentary dataset pipeline (parked)

The above is the full set if all four team-mates arrive on day 1 with strong ML and FE coverage. Actual delivery depends on headcount and strengths; fallbacks in the full doc.


## Timeline

| Weeks | Ariel | Curtis | New team-mates |
|-------|-------|--------|----------------|
| 1 | rtmlib + parity test | Data expansion (SS22, VideoBadminton, scouting) | Onboarding |
| 1-3 | X3D-S sweep on ShuttleSet | Amateur video scraper | See full doc |
| 3-4 | Final X3D-S + BST-X training run | Stage 4 setup (encoder bridge + training scripts) | See full doc |
| 4-6 | SSL pretrain + classification fine-tune | Live inference on user-uploaded videos starts | See full doc |
| 5-10 | Stage 4 quality stack | Live inference + connecting ML outputs to the UI | See full doc |
| 10-12 | Loose ends + model docs + writeup | Live inference wrap + demo prep | FE polish + docs (FE); rally or Stage 4 wrap (ML) |


## Conclusion

If the three main features are finished by week 10, T2 has built the core for the full autograder. The major deliverable for next year would be extending analysis across the whole rally, and extending the UI to match.
