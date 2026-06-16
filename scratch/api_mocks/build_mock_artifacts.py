"""Emit mock Tier 1 sidecar JSONs for run_20260505_154907.

Writes five files under <run_dir>/fe_jsons/, each carrying `_mock_data: true`:
  - val.json + test.json                     (per-clip preds + softmax)
  - perclass_stats_val.json + ..._test.json
  - clip_index.json

Run from repo root:
    python3 scratch/api_mocks/build_mock_artifacts.py

The numbers are seeded for reproducibility. Class list + per-class F1
are taken from run_20260505_154907/manifest.yaml serial 5.
"""
import json
import random
from pathlib import Path

RUN_DIR = Path(
    "src/bst_x/stroke_classification/main_on_shuttleset/"
    "experiments/run_20260505_154907"
)
FE_DIR = RUN_DIR / "fe_jsons"

RUN_ID = "run_20260505_154907"
SERIAL = 5
TEMPERATURE = 1.0
CLIPS_PER_CLASS = 2

# Active 14-class no-sides taxonomy (manifest.yaml -> arch.active_class_list).
CLASS_LIST = [
    "net_shot", "return_net", "smash", "wrist_smash", "lob", "clear",
    "drive", "drop", "passive_drop", "push", "rush",
    "cross_court_net_shot", "short_service", "long_service",
]

# Serial 5 per-class F1 from manifest.
PER_CLASS_F1 = {
    "net_shot": 0.8924, "return_net": 0.8184, "smash": 0.5147, "wrist_smash": 0.5186,
    "lob": 0.7846, "clear": 0.9465, "drive": 0.6628, "drop": 0.6821,
    "passive_drop": 0.6765, "push": 0.6546, "rush": 0.7742,
    "cross_court_net_shot": 0.6130, "short_service": 0.9801, "long_service": 0.9517,
}

# Plausible confusion neighbours per class. Drives the mock softmax shape.
CONFUSIONS = {
    "smash": ["wrist_smash", "drive"],
    "wrist_smash": ["smash", "drop"],
    "drop": ["passive_drop", "wrist_smash"],
    "passive_drop": ["drop", "push"],
    "net_shot": ["return_net", "cross_court_net_shot"],
    "return_net": ["net_shot", "rush"],
    "cross_court_net_shot": ["net_shot", "return_net"],
    "lob": ["clear", "drive"],
    "clear": ["lob", "drive"],
    "drive": ["push", "smash"],
    "push": ["drive", "drop"],
    "rush": ["return_net", "push"],
    "short_service": ["push", "long_service"],
    "long_service": ["clear", "short_service"],
}

MATCH_BANK = [
    "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals",
    "Viktor_AXELSEN_Anthony_Sinisuka_GINTING_All_England_2022_Finals",
    "Akane_YAMAGUCHI_TAI_Tzu_Ying_All_England_2022_Semis",
    "An_Se_young_Carolina_MARIN_BWF_World_Tour_Finals_2023",
    "LOH_Kean_Yew_LEE_Zii_Jia_World_Championships_2021_Finals",
]
SIDES = ["Bottom", "Top"]


def softmax_for_clip(rng: random.Random, y_true_idx: int, y_true_name: str,
                     target_f1: float) -> tuple[list[float], int]:
    """Build a plausible calibrated softmax. Returns (probs, argmax_idx)."""
    probs = [rng.uniform(0.005, 0.025) for _ in CLASS_LIST]
    correct = rng.random() < target_f1
    if correct:
        peak = y_true_idx
        probs[peak] = rng.uniform(0.45, 0.85)
        for nbr in CONFUSIONS.get(y_true_name, []):
            probs[CLASS_LIST.index(nbr)] = rng.uniform(0.05, 0.18)
    else:
        nbrs = CONFUSIONS.get(y_true_name, [c for c in CLASS_LIST if c != y_true_name])
        wrong = rng.choice(nbrs)
        peak = CLASS_LIST.index(wrong)
        probs[peak] = rng.uniform(0.32, 0.55)
        probs[y_true_idx] = rng.uniform(0.15, 0.32)
        for nbr in CONFUSIONS.get(y_true_name, []):
            if nbr != wrong:
                probs[CLASS_LIST.index(nbr)] = rng.uniform(0.04, 0.12)
    total = sum(probs)
    probs = [p / total for p in probs]
    return probs, peak


def build_clip_records(rng: random.Random, split: str,
                       used_stems: set[str]) -> tuple[list[dict], dict]:
    """Generate per-clip prediction records + matching clip_index entries."""
    clip_records = []
    clip_index = {}
    for c in CLASS_LIST:
        for _ in range(CLIPS_PER_CLASS):
            while True:
                vid = rng.randint(1, 50)
                set_idx = rng.randint(1, 3)
                rally = rng.randint(1, 30)
                ball_round = rng.randint(1, 40)
                stem = f"{vid}_{set_idx}_{rally}_{ball_round}"
                if stem not in used_stems:
                    used_stems.add(stem)
                    break

            y_true_idx = CLASS_LIST.index(c)
            probs, peak = softmax_for_clip(rng, y_true_idx, c, PER_CLASS_F1[c])
            ranked = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)[:5]
            top_k_idx = [i for i, _ in ranked]
            top_k_prob = [round(p, 4) for _, p in ranked]

            clip_records.append({
                "clip_stem": stem,
                "y_true": y_true_idx,
                "y_pred": peak,
                "softmax_calibrated": [round(p, 4) for p in probs],
                "top_k_idx": top_k_idx,
                "top_k_prob": top_k_prob,
            })

            side = rng.choice(SIDES)
            clip_index[stem] = {
                "video_path": f"{split}/{side}_{c}/{stem}.mp4",
                "match": rng.choice(MATCH_BANK),
                "set_id": f"set{set_idx}",
                "rally": rally,
                "ball_round": ball_round,
                "split": split,
                "raw_type_en": c,
                "player_side": side,
            }
    return clip_records, clip_index


def build_perclass_stats(split: str, n_clips: int) -> dict:
    """Per-class precision/recall/f1 + top5_when_true/pred from F1 + confusions."""
    rng = random.Random(f"perclass_{split}")
    per_class = {}
    n_per_class = n_clips // len(CLASS_LIST)
    for c in CLASS_LIST:
        f1 = PER_CLASS_F1[c]
        precision = max(0.0, min(1.0, f1 + rng.uniform(-0.05, 0.05)))
        recall = max(0.0, min(1.0, f1 + rng.uniform(-0.05, 0.05)))
        support_true = max(20, n_per_class + rng.randint(-2, 2))
        support_pred = max(20, int(support_true * (precision / max(recall, 1e-3))))

        nbrs = CONFUSIONS.get(c, [])

        def build_top5(self_share: float) -> list[list]:
            entries = [[c, round(self_share, 2)]]
            residual = 1.0 - self_share
            for nbr in nbrs[:2]:
                share = round(residual * rng.uniform(0.35, 0.55), 2)
                entries.append([nbr, share])
                residual -= share
            others = [x for x in CLASS_LIST if x not in {e[0] for e in entries}]
            while len(entries) < 5:
                pick = rng.choice(others)
                others.remove(pick)
                entries.append([pick, round(max(0.0, residual) / 3.0, 2)])
            return entries

        per_class[c] = {
            "support_true": support_true,
            "support_pred": support_pred,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "top5_when_true": build_top5(recall),
            "top5_when_pred": build_top5(precision),
        }
    return {
        "_mock_data": True,
        "split": split,
        "n_clips": n_clips,
        "class_list": CLASS_LIST,
        "per_class": per_class,
    }


def main() -> None:
    FE_DIR.mkdir(parents=True, exist_ok=True)

    used_stems: set[str] = set()
    full_clip_index = {}

    for split in ("val", "test"):
        rng = random.Random(f"clips_{split}")
        records, clip_index = build_clip_records(rng, split, used_stems)
        full_clip_index.update(clip_index)

        preds = {
            "_mock_data": True,
            "run_id": RUN_ID,
            "serial_no": SERIAL,
            "split": split,
            "active_class_list": CLASS_LIST,
            "temperature": TEMPERATURE,
            "clips": records,
        }
        (FE_DIR / f"{split}.json").write_text(json.dumps(preds, indent=2))
        print(f"wrote {FE_DIR / f'{split}.json'} ({len(records)} clips)")

        stats = build_perclass_stats(split, len(records))
        (FE_DIR / f"perclass_stats_{split}.json").write_text(json.dumps(stats, indent=2))
        print(f"wrote {FE_DIR / f'perclass_stats_{split}.json'}")

    clip_index_out = {"_mock_data": True, "clips": full_clip_index}
    (FE_DIR / "clip_index.json").write_text(json.dumps(clip_index_out, indent=2))
    print(f"wrote {FE_DIR / 'clip_index.json'} ({len(full_clip_index)} entries)")


if __name__ == "__main__":
    main()
