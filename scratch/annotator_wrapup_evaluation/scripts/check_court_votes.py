"""Check whether saved court outlines explain rejected scenes in the visual sample."""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from annotator.court_evidence import (
    _as_ref_corners,
    build_keep_vote,
    detected_court_info,
)

ROOT = Path(__file__).resolve().parents[1]


def load_array(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        return np.load(source, allow_pickle=False)


def run(prepared: Path) -> None:
    started = perf_counter()
    sample = pd.read_csv(ROOT / "results/visual_sample.csv.gz")
    sample = sample[sample.pair_id <= 4]
    records = []
    for fixture, group in sample.groupby("fixture"):
        folder, = prepared.glob(f"{int(fixture):02d} *")
        with gzip.open(folder / "court_evidence.json.gz", "rt") as source:
            court = json.load(source)
        resolution = tuple(court["inputs"]["resolution"])
        bboxes = load_array(folder / "pose_bboxes.npy.xz")
        scores = load_array(folder / "pose_scores.npy.xz")
        ndet = load_array(folder / "pose_ndet.npy.xz")
        saved_vote = load_array(folder / "court_keep_vote.npy.xz")
        consensus = np.asarray(court["consensus"]["consensus_quad"])
        replacement_info = detected_court_info(_as_ref_corners(consensus, resolution))
        for row in group.itertuples(index=False):
            scene = next(scene for scene in court["scene_records"]
                         if scene["start_frame"] <= row.source_frame < scene["end_frame"])
            start, end = scene["start_frame"], scene["end_frame"]
            native = scene["raw_corners_px"]
            original_info = None if native is None else detected_court_info(
                _as_ref_corners(np.asarray(native), resolution),
            )
            original = build_keep_vote(bboxes[start:end], scores[start:end], ndet[start:end], resolution,
                                       [(0, end - start)], [original_info])
            assert np.array_equal(original, saved_vote[start:end]), (fixture, scene["scene_index"])
            revised = build_keep_vote(bboxes[start:end], scores[start:end], ndet[start:end], resolution,
                                      [(0, end - start)], [replacement_info])
            records.append({"sample_id": row.sample_id, "fixture": int(fixture), "scene_index": scene["scene_index"],
                            "scene_start": start, "scene_end": end, "scene_frames": end - start,
                            "raw_source": scene["raw_source"], "saved_scene_valid": scene["scene_valid"],
                            "saved_two_person_frames": int(original.sum()),
                            "replacement_two_person_frames": int(revised.sum()),
                            "saved_fraction": float(original.mean()), "replacement_fraction": float(revised.mean())})
            print(row.sample_id, original.mean(), "->", revised.mean(), flush=True)
    pd.DataFrame(records).to_csv(ROOT / "results/court_vote_check.csv.gz", index=False)
    print("Complete", round(perf_counter() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    run(parser.parse_args().prepared_root)
