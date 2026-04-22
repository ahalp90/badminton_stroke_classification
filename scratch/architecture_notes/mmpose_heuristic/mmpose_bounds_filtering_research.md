# Heuristic player filtering in racket-sport CV: four-question evidence review

This report answers the user's four questions using only direct quotes from papers that were actually fetched, or from source code that was actually read. Where no evidence was found, that is stated plainly. No cross-paper patterns or "standard practice" claims are asserted.

## Question 1: Published ablations that compare filter strategies

**No paper was found that ablates different player-filtering or missing-pose-handling strategies against downstream stroke/action classification accuracy.** The closest the examined set gets is that two papers *state* a chosen rule as fixed preprocessing; neither varies it.

**TemPose — Ibh et al., CVPRW 2023** (fetched at openaccess.thecvf.com). Section 3.1 ("Extraction of the skeleton, player position, and shuttlecock data") states verbatim:

> "However, irrelevant individuals, such as spectators in the crowd, can limit the quality of the skeleton data. To address this issue in badminton, we calculate a homography using the court's known dimensions and map the feet of the detected individuals to the ground plane. By doing so, we only consider skeletons within the court and can identify each sequence's top and bottom player. In cases where a whole skeleton is missing, we replace it with the pose from the previous frame."

A separate shuttlecock-only rule is also stated: "We only consider predictions with a confidence score above 0.75; we pad failed predictions with zeros." TemPose's Component Studies tables (Section 4.2, Tables 2–6) vary model version, embedding dim, transformer depth, and joints-vs-joints+bone; **no row varies the player-filter or missing-pose strategy.**

**BST — Chang, arXiv 2502.21085** (fetched at arxiv.org/html/2502.21085v2). The supplementary/appendix states:

> "The court information is provided in the dataset, so we don't need to detect it by ourselves. If there were less than two people in the court, we cleared the information (poses and shuttlecock trajectory) of that frame to zero, since that frame was definitely not in a standard camera perspective."

BST's ablations compare model variants (BST-0, BST-CG, BST-AP, BST-CG-AP) against TemPose/ST-GCN/MS-G3D and 2D vs 3D joints; **no row varies the person-filter or missing-pose strategy.** Important caveat: the **eps=0.01** court-tolerance value in `check_pos_in_court` in `prepare_train_on_shuttleset.py` is a code-only parameter; it does not appear in the written paper. The subagent was unable to fetch that specific script directly during this pass, so the code-level detail cannot be independently quoted here — only the paper-level rule above can.

**ShuttleSet (Wang et al., KDD 2023), ShuttleSet22, BadmintonDB, CoachAI family.** Dataset/tactical-forecasting works operating on human-annotated stroke-level tabular records, not on raw pose pipelines. **No player-filter/missing-pose preprocessing ablation applies or is present.**

**VideoBadminton — Li et al., 2024** (arxiv 2403.12385). The accessible sections (dataset construction, benchmarking) contain no discussion of player-filter policy or missing-pose handling, and no ablation row varies such a choice.

**TenniSet (Faulkner & Dick, DICTA 2017); TTNet (Voeikov et al., CVPRW 2020).** No player-pose-selection ablation found. TTNet's ablations concern ball-detection sequence length, not person filtering.

**MonoTrack (Liu & Wang, CVPRW 2022).** Describes its filtering rule (see Question 4) and handles jumping explicitly, but **does not ablate alternative filter strategies against downstream classification accuracy.**

**Bottom line for Q1:** No direct evidence found, across TemPose, BST, ShuttleSet, ShuttleSet22, VideoBadminton, TenniSet, TTNet, MonoTrack, BadmintonDB, or the CoachAI papers surveyed, of an explicit ablation comparing zero-fill vs. carry-forward vs. closest-to-line vs. per-half-court vs. confidence-gated strategies with classification-accuracy numbers per strategy. **One paper (TemPose) commits to carry-forward; one paper (BST) commits to zero-fill;** neither reports a comparison.

## Question 2: The doubles-badminton paper's MOT pipeline

Paper: **"Bridging the Gap: Doubles Badminton Analysis with Singles-Trained Models"** — Baek & Yun, arXiv 2508.13507 (HTML retrieved through search snippets and the arXiv HTML page). Section labels in the HTML are paragraph-level; exact locations given below.

**(a) Inputs to the tracker.** Person-class bounding boxes from YOLOv11x; the custom re-association step uses bounding-box **center coordinates**. From the Methods section ("ID Preserving Motion Tracking" paragraph):

> "First, we employ a pre-trained YOLOv11x model to detect objects classified as 'person'."
> "In the detection and tracking stage, we employ YOLOv11x combined with BoT-SORT for temporal association."
> "Predictive objects estimate the expected position of a player in the next frame by calculating direction and velocity based on the center coordinates of the object in the previous and current frames"

Pose keypoints (ViT-Pose) are extracted **after** tracking, for the downstream ST-GCN classifier — not as tracker input.

**(b) Cost/association function.** Initial association is whatever BoT-SORT uses internally (the paper does not elaborate). The paper's own custom re-association uses **Euclidean distance in pixels between bounding-box centers**, with a **200-pixel cutoff**:

> "When an object reappears within a 200-pixel radius of the predicted object's position, the ID of the predicted object is reassigned to the detected object, effectively mitigating the ID-switching problem."
> "we calculate the Euclidean distances between actual and predicted objects in doubles matches from non-missing consecutive frames."

No IoU, no appearance embedding, no keypoint-distance cost is described in the paper.

**(c) Handling of missing detections.** Constant-velocity linear extrapolation of the center position from the last two observed frames:

> "S(t_s + t) = S(t_s) + [S(t_s) − S(t_s − 1)] × t"
> "t_s is the time measured in frame numbers of the last successful object detection for a focal player, and … measured in pixels per frame."
> "We empirically determine a 200-pixel cutoff threshold. First, we find that the missing frame duration is typically less than 10 frames … We find that the prediction error is typically less than 200 pixels, except for a few outlier cases with long-missing frames."

The paper frames missing-detection causes as **"rapid, overlapping player movements"** and **"frequent player occlusions … due to overlapping"** — i.e., player-player occlusion, not airborne-out-of-court cases. **The paper does not address airborne players, jumping, or smashes as a detection/filtering edge case.** The only smash-related mention is in Future Work: "The backbone can be extended to recognize different shot types (e.g., smash, drop, clear, drive)…".

**(d) Code release.** Yes:

> "All training code, model architectures, and pre-trained weights for both the ST-GCN backbone and the Transformer-based shot classifier are available at GitHub (https://github.com/100-heon/badminton_double_analysis)."

The release sentence explicitly names the ST-GCN backbone and the Transformer shot classifier; it does not in that sentence explicitly name the custom tracking code, so the user should inspect the repo directly to confirm the tracking is included.

**(e) Court filter before the tracker.** A **court-polygon ROI**, not a homography ankle-projection test. A second YOLOv11x model detects four corner boxes; within each box, the "farthest corner" point is selected to form an ROI quadrilateral:

> "Since only two or four individuals are players in singles and doubles matches, respectively, we detect the court area from broadcast overhead camera footage. To extract the Region of Interest (ROI) of [the court] …"

Homography is listed only as future work:

> "When combined with homography transformation, our tracking system can also determine shot positions on the court, providing spatial context for tactical analysis."

**The paper does not describe projecting ankles through a homography; it does not mention airborne-player filtering at all.**

## Question 3: Airborne/above-ground homography projection error in racket sports

**Three papers were found that directly address this geometric issue**, with quotable content. One quantifies the geometry; one explicitly defers the case; one acknowledges it as a known failure mode.

**Javadiha, Andujar, Lacasa, Ric, Susin — "Estimating Player Positions from Padel High-Angle Videos: Accuracy Comparison of Recent Computer Vision Methods", Sensors 21(10):3368, 2021** (fetched at mdpi.com/1424-8220/21/10/3368). This is the padel paper the user wanted verified. **The previous research pass's specific accuracy numbers for this paper were fabricated; the verbatim text of what the paper actually reports is as follows.**

Section 2.3 ("Court-Space Positions from Image-Space Joints") explicitly derives the above-ground projection error:

> "We use this matrix to map points from image-space to court-space, but this is only exact for body parts on the floor. Figure 2b illustrates this problem. Multiple body parts (e.g., the hip and the left knee) are projected onto the same image-space pixel h. Since no reliable depth estimation is available, applying the perspective transform to h leads to an offset between the estimated position (black sphere) and the true one (blue sphere). The offset can be computed as H_z tan(θ), being thus 0 when either the body part is on the floor (H_z=0) or the camera angle θ equals 90 (zenithal camera)."

The paper then **quantifies the error-amplification factor** for the standard padel camera geometry:

> "Within the court, θ ranges from arctan(C/D) ≈ 26.3° at the bottom edge of the court, to arctan(C/(D+20)) ≈ 12.5° at the top edge. … if the feet of the player is (or is estimated at) some height c above the floor, the approximate offset (error) in the vertical Y coordinate of its estimated image 2D position will be 2.0c, 3.3c or 4.64c, depending on whether the player is located near the bottom edge, the net, or the top edge of the court."

Section 2.4 describes the authors' **non-ankle anchor choice — hip for X, feet for Y — with justification**:

> "From the image-space positions of these joints, we estimated the image-space position of the player as p = (p_x, p_y) with p_x = H_x and p_y = (L_y + R_y)/2 … That is, we use the hip to estimate the image-space horizontal coordinate and the average of the feet to estimate the vertical coordinate (near the bottom part of the image-space projection of the player)."

The authors **explicitly exclude airborne frames from their accuracy analysis** (from the Discussion, via the MDPI/PMC abstract-adjacent passage):

> "Since the perspective correction assumes that the player's feet are on the ground (height ≈ 0), we have not considered the error for players with both feet in the air, for example during a smash jump. For the standard camera setting, such vertical displacements result in an offset on the predicted court-space Y coordinate. We believe though that these displacements should have a negligible impact on tactical analyses of padel matches."

**What Javadiha et al. do NOT provide:** a numeric correction (e.g., a learned foot offset, a per-player height estimate, or a head/hip double-plane back-projection). Their approach is to use (hip_x, mean-ankle_y) as the image anchor and simply accept the residual error for jumps. The overall RMSE they report is "below 5 and 12 cm for horizontal/vertical court-space coordinates" for the best top-down pose method — but this is **on non-airborne frames**, since airborne frames were excluded.

**Hsu, Yu, Cheng — "Enhancing Badminton Game Analysis: An Approach to Shot Refinement via a Fusion of Shuttlecock Tracking and Hit Detection from Monocular Camera", Sensors 24(13):4372, 2024** (fetched at pmc.ncbi.nlm.nih.gov/articles/PMC11244353/). This badminton paper uses ankle projection through a perspective transform and explicitly acknowledges the jump-smash failure:

> "After extracting the ankle coordinates of the players in each frame, these coordinates are projected onto an aerial view by utilizing perspective transformation … Although the presence of non-players, such as referees and spectators, can also be detected by the DensePose model, their information can be ignored because the positioning of their ankle points is outside the court."

> "Nevertheless, because only a monocular camera is utilized as the input, the proposed system is constrained by a single perspective, making it difficult to correctly project 2D image coordinates back to 3D real coordinates. This may potentially affect the accuracy of ankle coordinates and lead to misjudgments of the player's position, thereby impacting the analysis of shot types and player movement direction. For example, when the player jumps and performs a power strike, the results of the inferred ankle position are incorrect. However, this scenario does not occur very often, hence, the approach still achieves satisfactory results in terms of the shot-type classification."

**No correction is proposed and no error is quantified numerically for this case.**

**Javadiha, Andujar, Calvanese, Lacasa, Moyés, Pontón, Susín, Wang — "PADELVIC: Multicamera videos and motion capture data of an amateur padel match", Padel Scientific Journal Vol. II (2024), pp. 89–106** (search snippets from researchgate). The paper contains a figure titled **"Figure 8. Potential positional errors due to vertical displacements of the players. Vertical displacements of the players (while running and jumping) severely influence the [...]"** (text truncated in snippet). It also describes a "system to predict … the center of mass of the players projected onto the court plane, from a single match video" using synthetic data from motion-capture-driven avatars. The snippet alone is not enough to verbatim-quote a concrete correction method; this paper is noted as a candidate for follow-up if the user wants deeper evidence.

**MonoTrack — Liu & Wang, CVPRW 2022** (fetched). Section 4.2 directly addresses the airborne case (see Q4 below): it relaxes the court boundary and falls back to the closest-to-last-in-court-pose on that side. It does **not** quantify the projection error and does not propose a learned or geometric correction.

**Other candidates checked:** Pose2Trajectory (arXiv 2411.04501, tennis), PadelTracker100, Decorte et al. "Multi-Modal Hit Detection and Positional Analysis in Padel Competitions" (CVPRW 2024), Novillo et al. "Padel Two-Dimensional Tracking Extraction from Monocular Video Recordings" (IDEAL 2024), and various TrackNet variants. **None of these, in the text accessible during this search, directly discuss the above-ground-projection geometric error or propose a correction.**

**Bottom line for Q3:** Two papers explicitly name the geometric issue with verbatim text (Javadiha et al. 2021 padel; Hsu et al. 2024 badminton); one of them (Javadiha et al.) also quantifies the error-amplification factor (2.0c / 3.3c / 4.64c per unit above-ground height, over the padel court). Neither proposes a principled correction — they either exclude airborne frames from evaluation or accept the residual error. **No paper was found that proposes a learned foot-offset, per-player-height, or head/hip double-plane correction specifically for an airborne racket-sport player.**

## Question 4: MonoTrack's player-filtering approach

Paper: **MonoTrack: Shuttle trajectory reconstruction from monocular badminton video — Liu & Wang, CVPRW 2022, arXiv 2204.01899** (PDF fetched at arxiv.org/pdf/2204.01899; repo at github.com/jhwang7628/monotrack).

**(a) Player filtering / identification (Section 4.2, "Pose estimation"):**

> "We perform pose estimation using a top-down HRNet model to compute per-frame poses through the mmpose framework. To track poses, instead of using methods developed for unstructured environments or recurrent network-based methods, we simply leverage the detected court as a strong cue. We filter all detected poses that do not have feet in the court, and identify near and far players based on their distance to the camera. This strategy is effective due to the fact that no one other than the players can step onto the court, and that players do not switch side during a point."

**(b) Airborne players — yes, MonoTrack explicitly handles this case, and is the only paper in this report that describes a concrete heuristic for it** (Section 4.2):

> "To accommodate jumping motions, which would misplace the player to a deeper position than they actually are, we make two modifications to increase robustness. Firstly, we relax the court boundary slightly. Secondly, if a side of the court has no pose within it, we find the pose closest to the last in-court pose recorded on that side. For all of our videos, this simple approach identifies the two players on every frame."

MonoTrack also flags jumping as a motivating difficulty in the introduction: "the frequent jumping of badminton players, render many of these prior methods infeasible." **The paper does not quantify the projection error for airborne players, and its "identifies the two players on every frame" claim is qualitative — no per-frame player-identification accuracy metric is reported.**

**(c) Commentary on court-based filtering failures.** The paper discusses failure modes of the **prior** Farin-style court detector as motivation for their graph-based improvement (Section 4.1):

> "Unfortunately, this algorithm fails in about 24% of videos in our dataset. We found the main culprit to be the hard angle constraints set when partitioning results in line misclassification, which ultimately break the algorithm."
> "On our dataset, our proposed approach increases the success rate of court detection from 73.9% to 85.5%, and decreased the average detection time by a factor of 40 while achieving an higher average IoU of 0.97 (vs. 0.96 from the original method)."

**The paper does not analyze the cascade of consequences for player filtering when court detection itself fails** on their own remaining ~14.5% of failures. It does not discuss homography-projection error for above-ground keypoints.

**Pose estimator / detector:** Top-down HRNet via MMPose; the repo README adds MMDet for person detection ("We use MMPose and MMDet for pose detection"), though MMDet is not named in the paper itself.

**Homography usage.** MonoTrack uses 6 known 3D↔2D correspondences (4 court corners + 2 net-pole tips) with DLT to obtain camera parameters (Section 4.5): "given 3D trajectory estimates and camera parameters, we can project x(t)∈R³ to image space to obtain 2D trajectory estimates … This requires 6 known 3D coordinates, which we have via the 4 boundary court corners detected in §4.1 plus the 2 tip points on the net poles." The detected court polygon is also used as the spatial mask for player filtering (quoted above) and as a feature input to HitNet.

**Code:** Released at **https://github.com/jhwang7628/monotrack** (MIT-style license; entry point `ai_badminton.pipeline_clean`).

## Sources searched but not directly accessible or not found

The following were searched for during this pass but could not be fetched in full (rate-limits on arXiv direct fetch, behind paywalls, or search returned only snippet-level content), and so their specific claims could not be independently quoted:

- BST source file `prepare_train_on_shuttleset.py` in the BST repo (the specific `check_pos_in_court` / eps=0.01 code path was not directly fetched this pass; only the paper-level rule above is quotable here).
- Ibh 2025 PhD thesis (en.itu.dk/.../Magnus-Ibh.pdf) — referenced as containing a chapter on "Quality of Pose Estimation on Badminton videos" but not paginated in full during this pass; **this is the single most likely place a filter-strategy ablation might exist in this line of work, and is worth a targeted follow-up.**
- RallyTemPose / "A Stroke of Genius" (Ibh et al., CVPRW 2024) — accessible abstracts/summaries only; no filter-strategy ablation found in accessible portions.
- PADELVIC (Javadiha et al., Padel Scientific Journal II, 2024) — search snippets only, Figure 8 caption about "vertical displacements … while running and jumping severely influence…" was truncated; full paper not fetched.
- Decorte et al., "Multi-Modal Hit Detection and Positional Analysis in Padel Competitions" (CVPRW 2024) — retrieved abstract only.
- PadelTracker100 (ScienceDirect Data in Brief) — retrieved abstract/structure only.
- Novillo et al., "Padel Two-Dimensional Tracking Extraction from Monocular Video Recordings" (Springer IDEAL 2024) — retrieved citation only.
- TenniSet DICTA 2017 paper — retrieved citing-paper summaries and GitHub README; full paper text was not fetched.
- BadmintonDB (MMSports @ ACM MM 2022) — retrieved abstract and repo description; full paper text was not fetched.
- ShuttleSet22 extended discussion beyond the arxiv HTML excerpts.

None of the above contain evidence contradicting the per-question findings above, but they represent remaining blind spots in the search perimeter.