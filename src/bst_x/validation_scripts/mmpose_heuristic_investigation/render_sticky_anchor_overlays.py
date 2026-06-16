#!/usr/bin/env python3
"""Render per-frame overlays showing sticky_anchor's Top/Bottom picks.

For each frame in a sampled clip, draws:

- The homography's doubles-court rectangle (cyan).
- All raw MMPose detections' bboxes in grey.
- The bbox sticky_anchor assigned to the Top slot in GREEN.
- The bbox sticky_anchor assigned to the Bottom slot in BLUE.
- Pink keypoint dots on the Top and Bottom picks.
- A label per picked bbox: ``TOP score`` or ``BOT score``.
- A "FAILED" header on frames where either slot was zeroed.

Pick identification: each _pos value is matched back to the raw bbox whose
bottom-centre projects closest to it in normalised court coords. sticky_anchor
computed that projection when choosing the pick, so the match is essentially
exact (within float precision).

Runs over a list of clip stems so a random sample can be rendered in one
invocation. Optionally stitches PNGs into an mp4 per clip via ffmpeg for
easier review.

Usage:

    python src/bst_x/validation_scripts/mmpose_heuristic_investigation/render_sticky_anchor_overlays.py \\
        --clips-dir /scratch/comp320a/ShuttleSet/clips \\
        --clip-stems-file /tmp/sample.txt \\
        --raw-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1 \\
        --heuristic-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_h_sticky_anchor \\
        --out-dir /scratch/comp320a/sticky_anchor_inspection \\
        --encode-mp4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "bst_x"))

from pipeline.config import HOMOGRAPHY_RESOLUTION  # noqa: E402
from pipeline.court_utils import get_court_info  # noqa: E402


COLOURS = {
    "top":     (0, 220, 80, 255),     # green: sticky_anchor Top pick
    "bottom":  (40, 90, 255, 255),    # blue: sticky_anchor Bottom pick
    "other":   (160, 160, 160, 180),  # grey: unpicked raw detections
    "court":   (0, 255, 255, 255),    # cyan: doubles court rectangle
    "joint":   (255, 105, 180, 235),  # hot pink: keypoint dots on picks
    "failed":  (230, 50, 50, 255),    # red: FAILED frame header
}

SLOT_TOP = 0
SLOT_BOTTOM = 1


def extract_all_frames(clip_path: Path, out_dir: Path) -> list[Path]:
    """Extract every frame via ffmpeg into ``out_dir``; return frame paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(clip_path),
        "-vsync", "0", "-start_number", "0",
        str(out_dir / "frame_%03d.png"),
    ], check=True)
    return sorted(out_dir.glob("frame_*.png"))


def project_bottom_centres_norm(
    bboxes: np.ndarray, n: int, H: np.ndarray,
    src_w: int, src_h: int, court: dict,
) -> np.ndarray:
    """Project (n, 4) pixel bboxes' bottom-centres to normalised court coords."""
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    aim_w, aim_h = HOMOGRAPHY_RESOLUTION
    bx = (bboxes[:n, 0] + bboxes[:n, 2]) / 2.0 * (aim_w / src_w)
    by = bboxes[:n, 3] * (aim_h / src_h)
    pts = np.stack([bx, by, np.ones_like(bx)], axis=0)
    proj = H @ pts
    proj = proj[:2] / proj[2]
    x_n = (proj[0] - court["border_L"]) / (court["border_R"] - court["border_L"])
    y_n = (proj[1] - court["border_U"]) / (court["border_D"] - court["border_U"])
    return np.stack([x_n, y_n], axis=1)


def court_corner_pixels(
    homo_row: pd.Series, src_w: int, src_h: int,
) -> list[tuple]:
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
    """Return the index of the raw detection whose projection is closest to ``target_norm``.

    Returns None if the closest distance exceeds ``match_tol``; that signals
    an upstream inconsistency (picked position doesn't correspond to any raw
    detection we can see), which shouldn't happen but is worth flagging.
    """
    if projected_norm.shape[0] == 0:
        return None
    diff = projected_norm - target_norm[None, :]
    dists = np.linalg.norm(diff, axis=1)
    best = int(np.argmin(dists))
    if dists[best] > match_tol:
        return None
    return best


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
    joint_radius: int,
    encode_mp4: bool,
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

    pos = np.load(heuristic_dir / f"{clip_stem}_pos.npy")
    failed = np.load(heuristic_dir / f"{clip_stem}_failed.npy")
    F = int(bboxes_all.shape[0])

    out_dir.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="render_sticky_"))
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

            n = int(ndet_all[f])
            bboxes_f = bboxes_all[f, :n]
            scores_f = scores_all[f, :n]
            kps_f = kps_all[f, :n]

            projected = project_bottom_centres_norm(
                bboxes_f, n, H, src_w, src_h, court,
            )

            # Test per-slot, not frame-level `failed`: a partial success (one
            # slot picked, one zeroed) should still show the valid pick in
            # its colour rather than falling back to grey for both.
            top_idx = (
                match_pick_to_raw(pos[f, SLOT_TOP], projected)
                if pos[f, SLOT_TOP].any() else None
            )
            bot_idx = (
                match_pick_to_raw(pos[f, SLOT_BOTTOM], projected)
                if pos[f, SLOT_BOTTOM].any() else None
            )

            for i in range(n):
                if i == top_idx:
                    colour, width, tag = COLOURS["top"], 4, "TOP"
                elif i == bot_idx:
                    colour, width, tag = COLOURS["bottom"], 4, "BOT"
                else:
                    colour, width, tag = COLOURS["other"], 1, None
                x1, y1, x2, y2 = bboxes_f[i]
                draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)
                if tag is not None:
                    label = f"{tag} {float(scores_f[i]):.2f}"
                    tx, ty = x1 + 2, y1 + 2
                    draw.rectangle(
                        [tx - 1, ty - 1, tx + 70, ty + 13],
                        fill=(0, 0, 0, 210),
                    )
                    draw.text((tx, ty), label, fill=colour)

            for pick_idx in (top_idx, bot_idx):
                if pick_idx is None:
                    continue
                for jx, jy in kps_f[pick_idx]:
                    if np.isnan(jx) or np.isnan(jy):
                        continue
                    r = joint_radius
                    draw.ellipse(
                        [jx - r, jy - r, jx + r, jy + r],
                        fill=COLOURS["joint"], outline=(0, 0, 0, 200), width=1,
                    )

            header_h = 76
            draw.rectangle([10, 10, 260, header_h], fill=(0, 0, 0, 210))
            draw.text((18, 16), f"{clip_stem}  f{f:03d}  ndet={n}", fill="white")
            if failed[f]:
                draw.text((18, 30), "FAILED (slot zeroed)", fill=COLOURS["failed"])
            else:
                draw.text((18, 30), "green=Top  blue=Bottom  grey=other", fill="white")
            draw.text((18, 44), "pink=picked keypoints", fill=COLOURS["joint"])

            out_path = out_dir / f"overlay_{clip_stem}_f{f:03d}.png"
            img.save(out_path)

        print(f"  wrote {len(list(frames_range))} overlays -> {out_dir}")

        if encode_mp4:
            mp4_path = out_dir.parent / f"{clip_stem}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", "25",
                "-i", str(out_dir / f"overlay_{clip_stem}_f%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(mp4_path),
            ], check=True)
            print(f"  encoded -> {mp4_path}")

    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_stem_to_mp4(clips_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in clips_dir.glob("**/*.mp4")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--clip-stems-file", type=Path, required=True)
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
    parser.add_argument("--frames", type=str, default="all",
                        help='"all" or a range like "20:40".')
    parser.add_argument("--joint-radius", type=int, default=5)
    parser.add_argument("--encode-mp4", action="store_true",
                        help="Also stitch the per-clip PNGs into an mp4 via ffmpeg.")
    args = parser.parse_args()

    homo_df = pd.read_csv(args.homography_csv).set_index("id")
    res_df = pd.read_csv(args.resolution_csv).set_index("id")

    stem_to_mp4 = build_stem_to_mp4(args.clips_dir)

    with args.clip_stems_file.open() as fh:
        stems = [line.strip() for line in fh if line.strip()]
    print(f"Rendering {len(stems)} clips into {args.out_dir}")

    for i, stem in enumerate(stems, 1):
        clip_path = stem_to_mp4.get(stem)
        if clip_path is None:
            print(f"[{i}/{len(stems)}] {stem}: mp4 missing under --clips-dir; skipping")
            continue
        print(f"[{i}/{len(stems)}] {stem}")
        render_one_clip(
            clip_path=clip_path,
            clip_stem=stem,
            raw_dir=args.raw_dir,
            heuristic_dir=args.heuristic_dir,
            out_dir=args.out_dir / stem,
            homo_df=homo_df,
            res_df=res_df,
            frames_spec=args.frames,
            joint_radius=args.joint_radius,
            encode_mp4=args.encode_mp4,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
