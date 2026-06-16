#!/usr/bin/env python3
"""Combined overlay: detection rank/on-court story + sticky_anchor TOP/BOT picks.

For each frame of a sampled clip, draws:

- The homography's doubles-court rectangle (cyan).
- Every raw MMPose detection's bbox, coloured by rank-and-court status:
    - **green** : top-K by bbox_score AND on-court (likely captured player).
    - **red**   : top-K by bbox_score AND off-court (non-player in pool).
    - **blue**  : below top-K AND on-court (DISPLACED player, the failure mode).
    - **grey**  : below top-K AND off-court (ignore).
- The sticky_anchor TOP and BOTTOM picks: same colour as their rank-status
  outline above, but drawn extra-thick, with a ``TOP score`` / ``BOT score``
  tag and pink 17-joint keypoints overlaid. Anchor picks usually sit on green
  (top-K on-court); a thick blue pick tells you the anchor reached past a
  pool-capture failure to grab a displaced player.
- A red ``FAILED`` header naming which slot(s) zeroed for that frame.

Two input modes:

1. ``--clip-stems-file PATH``  -- a plain text list of clip stems, one per line.
   Headers show stem only.
2. ``--fe-json PATH``  -- a run's FE json (e.g. ``fe_jsons/test.json.gz``).
   Reads its ``class_list`` + per-clip ``y_true`` / ``y_pred`` and stratified-
   samples ``--samples-per-class`` clips per class (deterministic with
   ``--seed``). Headers gain a ``true: NAME`` line and a ``pred: NAME`` line
   coloured green if correct, red if wrong. Per-clip output dirs prefix the
   true-class name so sorting groups by class.

Pick matching: each ``pos`` value is matched back to the raw detection whose
bbox bottom-centre projects closest to it in normalised court coords (the
same projection sticky_anchor itself uses, so the match is within float
precision).

Stitches each clip's PNGs into an mp4 by default; pass ``--no-encode-mp4`` to skip.

Usage (per-class sampling):

    python src/bst_x/validation_scripts/mmpose_heuristic_investigation/render_anchor_and_dets_overlay.py \\
        --clips-dir /scratch/comp320a/ShuttleSet/clips \\
        --fe-json /path/to/run_XXX/fe_jsons/test.json.gz \\
        --samples-per-class 3 \\
        --raw-dir /scratch/comp320a/ShuttleSet_keypoints_raw \\
        --heuristic-dir /scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor \\
        --out-dir /scratch/comp320a/anchor_and_dets_overlays

The raw + heuristic dirs are the canonical taxonomy-agnostic pose extracts
(``BST_X_MMPOSE_NPY_DIR`` points at the heuristic one). They live at
``/scratch/comp320a/`` directly, not nested under any ``ShuttleSet_data_*``
collation tree -- the same pose extract feeds every collation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# File lives at src/bst_x/validation_scripts/mmpose_heuristic_investigation/<this>,
# so the repo root is four parents up.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "bst_x"))

from pipeline.config import HOMOGRAPHY_RESOLUTION  # noqa: E402
from pipeline.court_utils import get_court_info  # noqa: E402


# One palette for both stories: bbox colour conveys rank-and-court status,
# anchor picks are distinguished by thickness + tag + pink keypoints (not by
# colour) so the rank-status meaning still reads through on the picks.
# Saturated blue is the alarm colour (DISPLACED) so it's unambiguous against
# the green/red/cyan under protan-leaning vision.
COLOURS = {
    "topk_on":  (0, 220, 80, 255),     # green: top-K, on-court (captured player)
    "topk_off": (230, 50, 50, 255),    # red: top-K, off-court (non-player in pool)
    "rest_on":  (40, 90, 255, 255),    # blue: below top-K, on-court (DISPLACED)
    "rest_off": (160, 160, 160, 170),  # grey: below top-K, off-court (ignore)
    "court":    (0, 255, 255, 255),    # cyan: doubles court rectangle
    "joint":    (255, 105, 180, 235),  # hot pink: keypoint dots on picks
    "failed":   (230, 50, 50, 255),    # red: FAILED header line
    "pred_ok":  (0, 220, 80, 255),     # green: pred matches true
    "pred_bad": (230, 50, 50, 255),    # red: pred wrong
}

SLOT_TOP = 0
SLOT_BOTTOM = 1


def extract_all_frames(clip_path: Path, out_dir: Path) -> list[Path]:
    """Extract every frame of ``clip_path`` via ffmpeg into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(clip_path),
        "-vsync", "0", "-start_number", "0",
        str(out_dir / "frame_%03d.png"),
    ], check=True)
    return sorted(out_dir.glob("frame_*.png"))


def project_bottom_centres_norm(
    bboxes: np.ndarray, ndet: int, H: np.ndarray,
    src_w: int, src_h: int, court: dict,
) -> np.ndarray:
    """Project (ndet, 4) pixel bboxes' bottom-centres to normalised court coords."""
    if ndet == 0:
        return np.zeros((0, 2), dtype=np.float64)
    aim_w, aim_h = HOMOGRAPHY_RESOLUTION
    bx = (bboxes[:ndet, 0] + bboxes[:ndet, 2]) / 2.0 * (aim_w / src_w)
    by = bboxes[:ndet, 3] * (aim_h / src_h)
    pts = np.stack([bx, by, np.ones_like(bx)], axis=0)
    proj = H @ pts
    proj = proj[:2] / proj[2]
    x_n = (proj[0] - court["border_L"]) / (court["border_R"] - court["border_L"])
    y_n = (proj[1] - court["border_U"]) / (court["border_D"] - court["border_U"])
    return np.stack([x_n, y_n], axis=1)


def court_corner_pixels(
    homo_row: pd.Series, src_w: int, src_h: int,
) -> list[tuple]:
    """Return the 4 annotated court corners in clip-resolution pixel space.

    homography.csv stores corners at HOMOGRAPHY_RESOLUTION; scale to the
    clip's native resolution for drawing.
    """
    aim_w, aim_h = HOMOGRAPHY_RESOLUTION
    sx, sy = src_w / aim_w, src_h / aim_h
    return [
        (float(homo_row[f"{lbl}_x"]) * sx, float(homo_row[f"{lbl}_y"]) * sy)
        for lbl in ("upleft", "upright", "downright", "downleft")
    ]


def match_pick_to_raw(
    target_norm: np.ndarray, projected_norm: np.ndarray,
    match_tol: float = 1e-3,
) -> int | None:
    """Index of the raw detection closest to ``target_norm`` in projected space.

    Returns None if no raw detection projects within ``match_tol`` of the
    target. Signals an upstream inconsistency: a pick value with no matching
    raw detection visible to this script.
    """
    if projected_norm.shape[0] == 0:
        return None
    diff = projected_norm - target_norm[None, :]
    dists = np.linalg.norm(diff, axis=1)
    best = int(np.argmin(dists))
    if dists[best] > match_tol:
        return None
    return best


def load_fe_clips(fe_json_path: Path) -> tuple[list[str], list[dict]]:
    """Load (class_list, clips[]) from an FE json (gzipped if the suffix is .gz)."""
    opener = gzip.open if fe_json_path.suffix == ".gz" else open
    with opener(fe_json_path, "rt") as fh:
        d = json.load(fh)
    return d["class_list"], d["clips"]


def sample_per_class(
    clips: list[dict], n_classes: int, n_per_class: int, seed: int,
) -> list[dict]:
    """Pick up to ``n_per_class`` clips from each class index, deterministic with ``seed``.

    Classes with fewer clips than ``n_per_class`` contribute everything they
    have. Empty classes are silently skipped (caller logs the per-class count).
    Output order: class index ascending, then sampled order within each class.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[dict]] = {c: [] for c in range(n_classes)}
    for clip in clips:
        by_class[int(clip["y_true"])].append(clip)
    out: list[dict] = []
    for c in range(n_classes):
        bucket = by_class[c]
        if not bucket:
            continue
        if len(bucket) <= n_per_class:
            out.extend(bucket)
        else:
            idx = rng.choice(len(bucket), size=n_per_class, replace=False)
            # Sorting keeps the picks in their original list order so two
            # invocations with the same seed but a re-ordered FE json don't
            # produce different on-disk file orderings.
            out.extend(bucket[int(i)] for i in sorted(idx))
    return out


def build_work_items_from_fe_json(
    fe_json: Path, n_per_class: int, seed: int,
) -> list[dict]:
    """Read an FE json, stratified-sample N per class, log the per-class counts.

    Returns work-item dicts with keys: ``stem``, ``true_name``, ``pred_name``,
    ``is_correct``.
    """
    class_list, all_clips = load_fe_clips(fe_json)
    print(f"FE json: {fe_json}")
    print(f"  classes: {len(class_list)}   total clips: {len(all_clips)}")
    sampled = sample_per_class(all_clips, len(class_list), n_per_class, seed)

    # Per-class counts so the user can eyeball the stratification.
    by_class: dict[int, int] = {}
    for clip in all_clips:
        by_class[int(clip["y_true"])] = by_class.get(int(clip["y_true"]), 0) + 1
    print(f"  per-class availability vs sample (cap {n_per_class}):")
    for c, name in enumerate(class_list):
        avail = by_class.get(c, 0)
        taken = min(avail, n_per_class)
        flag = "  (under-cap)" if avail < n_per_class and avail > 0 else ""
        empty = "  (empty)" if avail == 0 else ""
        print(f"    {c:2d} {name:<24s} avail={avail:4d}  take={taken}{flag}{empty}")

    return [
        {
            "stem": clip["clip_stem"],
            "true_name": class_list[int(clip["y_true"])],
            "pred_name": class_list[int(clip["y_pred"])],
            "is_correct": bool(int(clip["y_true"]) == int(clip["y_pred"])),
        }
        for clip in sampled
    ]


def build_work_items_from_stems(stems_file: Path) -> list[dict]:
    """Read a stems file (one stem per line) into bare work-item dicts."""
    with stems_file.open() as fh:
        stems = [line.strip() for line in fh if line.strip()]
    return [
        {"stem": s, "true_name": None, "pred_name": None, "is_correct": None}
        for s in stems
    ]


def draw_header(
    draw: ImageDraw.ImageDraw,
    *,
    clip_stem: str,
    frame_idx: int,
    ndet: int,
    true_name: str | None,
    pred_name: str | None,
    is_correct: bool | None,
    failed_line: str | None,
) -> None:
    """Render the per-frame status header in the top-left of the image.

    Header lines are stacked top-to-bottom and the background rectangle is
    sized to the line count. Pred line is green if correct, red if wrong.
    """
    lines: list[tuple[str, tuple[int, int, int, int] | str]] = []
    lines.append((f"clip {clip_stem}  f{frame_idx:03d}  ndet={ndet}", "white"))
    if true_name is not None:
        lines.append((f"true: {true_name}", "white"))
        pred_colour = COLOURS["pred_ok"] if is_correct else COLOURS["pred_bad"]
        lines.append((f"pred: {pred_name}", pred_colour))
    lines.append((
        "bbox: green=topK on  red=topK off  blue=displaced  grey=ignore",
        "white",
    ))
    lines.append((
        "thick + TOP/BOT tag = anchor pick   pink = picked joints",
        "white",
    ))
    if failed_line is not None:
        lines.append((failed_line, COLOURS["failed"]))

    header_x0, header_y0 = 10, 10
    header_w = 330
    line_h, top_pad, bot_pad = 14, 6, 8
    header_h = top_pad + len(lines) * line_h + bot_pad

    draw.rectangle(
        [header_x0, header_y0, header_x0 + header_w, header_y0 + header_h],
        fill=(0, 0, 0, 215),
    )
    for i, (text, colour) in enumerate(lines):
        draw.text(
            (header_x0 + 8, header_y0 + top_pad + i * line_h),
            text, fill=colour,
        )


def render_one_clip(
    *,
    clip_path: Path,
    clip_stem: str,
    raw_dir: Path,
    heuristic_dir: Path,
    out_dir: Path,
    homo_df: pd.DataFrame,
    res_df: pd.DataFrame,
    frames_spec: str,
    top_k: int,
    margin: float,
    joint_radius: int,
    encode_mp4: bool,
    mp4_fps: int,
    true_name: str | None = None,
    pred_name: str | None = None,
    is_correct: bool | None = None,
) -> None:
    vid = int(clip_stem.split("_", 1)[0])
    homo_row = homo_df.loc[vid]
    court = get_court_info(homo_df, vid)
    H = court["H"]

    src_w = int(res_df.loc[vid, "width"])
    src_h = int(res_df.loc[vid, "height"])

    bboxes_all = np.load(raw_dir / f"{clip_stem}_raw_bboxes.npy")
    scores_all = np.load(raw_dir / f"{clip_stem}_raw_scores.npy")
    kps_all = np.load(raw_dir / f"{clip_stem}_raw_kps.npy")
    ndet_all = np.load(raw_dir / f"{clip_stem}_raw_ndet.npy")
    pos_all = np.load(heuristic_dir / f"{clip_stem}_pos.npy")
    F = int(bboxes_all.shape[0])

    out_dir.mkdir(parents=True, exist_ok=True)

    inconsistent_top = 0
    inconsistent_bot = 0

    work = Path(tempfile.mkdtemp(prefix="render_anchor_and_dets_"))
    try:
        raw_frames = extract_all_frames(clip_path, work / "raw")
        if len(raw_frames) != F:
            print(f"  WARN: ffmpeg gave {len(raw_frames)} frames, raw has {F}; using min")
        F = min(F, len(raw_frames))

        corners = court_corner_pixels(homo_row, src_w, src_h)

        if frames_spec == "all":
            frames_range = range(F)
        else:
            a, b = frames_spec.split(":")
            frames_range = range(int(a), int(b))

        for f in frames_range:
            img = Image.open(raw_frames[f]).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")

            draw.line([*corners, corners[0]], fill=COLOURS["court"], width=3)

            ndet = int(ndet_all[f])

            # Per-slot zeroed state from pos directly so the header can name
            # which slot failed; equivalent to failed.npy by construction but
            # avoids loading an extra sidecar.
            top_zeroed = not pos_all[f, SLOT_TOP].any()
            bot_zeroed = not pos_all[f, SLOT_BOTTOM].any()

            top_idx: int | None = None
            bot_idx: int | None = None

            if ndet > 0:
                bboxes_f = bboxes_all[f, :ndet]
                scores_f = scores_all[f, :ndet]
                kps_f = kps_all[f, :ndet]

                court_coords = project_bottom_centres_norm(
                    bboxes_f, ndet, H, src_w, src_h, court,
                )
                # On-court test mirrors render_detection_overlays.py: a single
                # axis-aligned tolerance band around the normalised [0, 1] court.
                on_court = (
                    (court_coords[:, 0] > -margin)
                    & (court_coords[:, 0] < 1 + margin)
                    & (court_coords[:, 1] > -margin)
                    & (court_coords[:, 1] < 1 + margin)
                )
                # rank_order: indices sorted by score descending.
                # rank_of[i]: the rank position of detection i (0 = highest).
                rank_order = np.argsort(-scores_f)  # (ndet,)
                rank_of = np.empty(ndet, dtype=int)
                rank_of[rank_order] = np.arange(ndet)

                if not top_zeroed:
                    top_idx = match_pick_to_raw(pos_all[f, SLOT_TOP], court_coords)
                    if top_idx is None:
                        inconsistent_top += 1
                if not bot_zeroed:
                    bot_idx = match_pick_to_raw(pos_all[f, SLOT_BOTTOM], court_coords)
                    if bot_idx is None:
                        inconsistent_bot += 1

                for i in range(ndet):
                    rank_i = int(rank_of[i])
                    in_pool = rank_i < top_k
                    is_on = bool(on_court[i])
                    colour_key = (
                        "topk_on" if in_pool and is_on else
                        "topk_off" if in_pool else
                        "rest_on" if is_on else
                        "rest_off"
                    )
                    colour = COLOURS[colour_key]

                    is_pick = (i == top_idx) or (i == bot_idx)
                    width = 5 if is_pick else 1
                    x1, y1, x2, y2 = bboxes_f[i]
                    draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)

                    if is_pick:
                        tag = "TOP" if i == top_idx else "BOT"
                        label = f"{tag} {float(scores_f[i]):.2f}"
                        tx, ty = x1 + 2, y1 + 2
                        draw.rectangle(
                            [tx - 1, ty - 1, tx + 70, ty + 13],
                            fill=(0, 0, 0, 210),
                        )
                        draw.text((tx, ty), label, fill=colour)

                        r = joint_radius
                        for jx, jy in kps_f[i]:
                            if np.isnan(jx) or np.isnan(jy):
                                continue
                            draw.ellipse(
                                [jx - r, jy - r, jx + r, jy + r],
                                fill=COLOURS["joint"],
                                outline=(0, 0, 0, 200), width=1,
                            )

            failed_names = []
            if top_zeroed:
                failed_names.append("TOP")
            if bot_zeroed:
                failed_names.append("BOT")
            failed_line = (
                f"FAILED ({', '.join(failed_names)} zeroed)" if failed_names else None
            )

            draw_header(
                draw,
                clip_stem=clip_stem,
                frame_idx=f,
                ndet=ndet,
                true_name=true_name,
                pred_name=pred_name,
                is_correct=is_correct,
                failed_line=failed_line,
            )

            out_path = out_dir / f"overlay_{clip_stem}_f{f:03d}.png"
            img.save(out_path)

        print(f"  wrote {len(list(frames_range))} overlays -> {out_dir}")
        if inconsistent_top or inconsistent_bot:
            print(
                f"  WARN: {inconsistent_top} TOP / {inconsistent_bot} BOT frames "
                f"had a non-zero pos but no raw det within match_tol"
            )

        if encode_mp4:
            mp4_path = out_dir.parent / f"{out_dir.name}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(mp4_fps),
                "-i", str(out_dir / f"overlay_{clip_stem}_f%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(mp4_path),
            ], check=True)
            print(f"  encoded -> {mp4_path}")

    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_stem_to_mp4(clips_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in clips_dir.glob("**/*.mp4")}


def subdir_name_for_item(item: dict) -> str:
    """Per-clip output dir name. Prefixed with true class when known so that
    `ls -1 out_dir` groups clips by class.
    """
    if item.get("true_name") is not None:
        return f"{item['true_name']}__{item['stem']}"
    return item["stem"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips-dir", type=Path, required=True)

    input_mode = parser.add_mutually_exclusive_group(required=True)
    input_mode.add_argument(
        "--clip-stems-file", type=Path,
        help="Plain text list of clip stems (one per line). Bare-stems mode.",
    )
    input_mode.add_argument(
        "--fe-json", type=Path,
        help="FE json (e.g. fe_jsons/test.json.gz). Enables per-class sampling "
             "and true/pred header.",
    )

    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--heuristic-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--homography-csv", type=Path,
        default=REPO_ROOT / "data" / "shuttleset" / "set" / "homography.csv",
    )
    parser.add_argument(
        "--resolution-csv", type=Path,
        default=REPO_ROOT / "data" / "shuttleset" / "video_metadata.csv",
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=3,
        help="Clips per class when --fe-json is given (default 3).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for per-class sampling (default 42).",
    )
    parser.add_argument(
        "--frames", type=str, default="all",
        help='"all" or a range like "20:40".',
    )
    parser.add_argument(
        "--top-k", type=int, default=8,
        help="Pool size for the top-K / on-court colour story (default 8).",
    )
    parser.add_argument(
        "--margin", type=float, default=0.15,
        help='Normalised tolerance for "on court" (default 0.15).',
    )
    parser.add_argument(
        "--joint-radius", type=int, default=5,
        help="Pink-dot radius in pixels for picked-player keypoints (default 5).",
    )
    parser.add_argument(
        "--encode-mp4", action=argparse.BooleanOptionalAction, default=True,
        help="Stitch per-clip PNGs into an mp4 via ffmpeg (default: on).",
    )
    parser.add_argument(
        "--mp4-fps", type=int, default=25,
        help="Framerate for the stitched mp4 (default 25).",
    )
    args = parser.parse_args()

    homo_df = pd.read_csv(args.homography_csv).set_index("id")
    res_df = pd.read_csv(args.resolution_csv).set_index("id")

    # Fail-loud guards: the per-clip loop silently skips on missing mp4s,
    # which masks a wholesale misconfig (wrong dir, /scratch not staged on
    # this host, dir empty) as "72 polite skips". Catch the wholesale case
    # up front. raw-dir / heuristic-dir misconfigs surface naturally as
    # FileNotFoundError on the first np.load.
    if not args.clips_dir.exists():
        sys.exit(f"ERROR: --clips-dir does not exist: {args.clips_dir}")
    stem_to_mp4 = build_stem_to_mp4(args.clips_dir)
    if not stem_to_mp4:
        sys.exit(
            f"ERROR: no mp4 files found under {args.clips_dir} "
            "(recursive glob). Check the path on this host; /scratch is "
            "host-local so a path that works on bourbaki/engelbart may be "
            "empty here."
        )

    if args.fe_json is not None:
        work_items = build_work_items_from_fe_json(
            args.fe_json, args.samples_per_class, args.seed,
        )
    else:
        work_items = build_work_items_from_stems(args.clip_stems_file)

    print(f"\nRendering {len(work_items)} clips into {args.out_dir}\n")

    for idx, item in enumerate(work_items, 1):
        stem = item["stem"]
        clip_path = stem_to_mp4.get(stem)
        subdir = args.out_dir / subdir_name_for_item(item)
        tag = f" [{item['true_name']} -> {item['pred_name']}]" if item['true_name'] else ""
        if clip_path is None:
            print(f"[{idx}/{len(work_items)}] {stem}{tag}: mp4 missing under --clips-dir; skipping")
            continue
        print(f"[{idx}/{len(work_items)}] {stem}{tag}")
        render_one_clip(
            clip_path=clip_path,
            clip_stem=stem,
            raw_dir=args.raw_dir,
            heuristic_dir=args.heuristic_dir,
            out_dir=subdir,
            homo_df=homo_df,
            res_df=res_df,
            frames_spec=args.frames,
            top_k=args.top_k,
            margin=args.margin,
            joint_radius=args.joint_radius,
            encode_mp4=args.encode_mp4,
            mp4_fps=args.mp4_fps,
            true_name=item.get("true_name"),
            pred_name=item.get("pred_name"),
            is_correct=item.get("is_correct"),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
