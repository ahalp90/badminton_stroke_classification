""" BRIC evaluation script."""

import argparse
import json
import torch
import yaml
from datetime import datetime
from pathlib import Path

from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy, multiclass_confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from bric.dataset import ShuttleSetDataset, collate_strokes
from bric.network import BRICNetwork
from bric.train import _resolve_taxonomy, _move_batch, _forward_for_variant
from shared.eval_plots import plot_confusion_matrix

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS = _REPO_ROOT / 'training' / 'bric' / 'experiments'

_ARCHITECTURE = 'bric'
_CHECKPOINT_FILENAME = 'best.pt'

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
         '--run-id', required=True, help='The run_id of the model to evaluate.'
    )
    return p.parse_args(argv)

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def _top_n_confusions(cm: torch.Tensor, classes: list[str], n: int = 10) -> list[dict]:
    """N largest off-diagonal confusion-matrix entries, sorted desc."""
    entries = [
        {'true': classes[i], 'predicted': classes[j], 'count': int(cm[i][j])}
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i][j] > 0
    ]
    entries.sort(key=lambda e: e['count'], reverse=True)
    return entries[:n]

def main():
    device = _select_device()
    args = _parse_args()
    run_dir = _EXPERIMENTS / args.run_id
    manifest = yaml.safe_load((run_dir / 'manifest.yaml').read_text())
    variant = manifest['config']['variant']
    taxonomy_name = manifest['config']['taxonomy']
    taxonomy = _resolve_taxonomy(taxonomy_name)
    classes = manifest['config']['classes']
    n_classes = len(classes)
    use_shuttle = manifest['config']['use_shuttle']
    use_court = manifest['config']['use_court']
    shuttle_encoder = manifest['training']['hparams'].get('shuttle_encoder') or 'mean'
    court_encoder = manifest['training']['hparams'].get('court_encoder') or 'snapshot'
    shuttle_window = manifest['training']['hparams'].get('shuttle_window') or 'between_hits'
    court_window = manifest['training']['hparams'].get('court_window') or 'between_hits'

    ds: ShuttleSetDataset = ShuttleSetDataset(
    split='test', taxonomy=taxonomy, rgb_transform=None,
    shuttle_window=shuttle_window,
    court_window=court_window,
    )

    model = BRICNetwork(
        taxonomy=taxonomy, pretrained=False,
        use_shuttle=use_shuttle, use_court=use_court,
        shuttle_encoder=shuttle_encoder,
        court_encoder=court_encoder,
    ).to(device)
    
    state = torch.load(run_dir / 'best.pt', weights_only=False, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    loader = DataLoader(
        ds, batch_size=32, pin_memory=(device.type == 'cuda'), 
        collate_fn=collate_strokes
    )

    preds, labels, probs, stems = [], [], [], [] 
    with torch.no_grad():
        for batch in tqdm(loader, desc='test', leave=False):
            batch = _move_batch(batch, device)
            logits = _forward_for_variant(model, batch)
            preds.append(logits.argmax(1).cpu())
            probs.append(torch.softmax(logits, dim=1).cpu())
            labels.append(batch['label'].cpu())
            stems.extend(batch['clip_stem']) 

    preds = torch.cat(preds); labels = torch.cat(labels); probs = torch.cat(probs)

    per_class_f1_tensor = multiclass_f1_score(preds, labels, num_classes=n_classes, average=None)
    supports_tensor = torch.bincount(labels, minlength=n_classes)
    present_idx = (supports_tensor > 0).nonzero(as_tuple=True)[0].tolist()
    cm_tensor = multiclass_confusion_matrix(preds, labels, num_classes=n_classes)

    test_summary = {
        'architecture': _ARCHITECTURE,
        'run_id': args.run_id,
        'num_strokes': len(labels),
        'evaluated_at': datetime.now().isoformat(timespec='seconds'),
        'config': {
            'variant': variant,
            'use_shuttle': use_shuttle,
            'shuttle_encoder': shuttle_encoder,
            'use_court': use_court,
            'court_encoder': court_encoder,
            'taxonomy': taxonomy_name,
            'classes': classes,
        },
        'metrics': {
            'macro_f1': float(multiclass_f1_score(preds, labels, num_classes=n_classes, average='macro')),
            'min_f1': float(per_class_f1_tensor[present_idx].min()) if present_idx else 0.0,
            'accuracy': float(multiclass_accuracy(preds, labels)),
            'top2_accuracy': float(multiclass_accuracy(probs, labels, k=2)),
            'per_class_f1': {classes[i]: float(per_class_f1_tensor[i]) for i in range(n_classes)},
            'top_confusions': _top_n_confusions(cm_tensor, classes, n=10)
        }
    }

    summary_path = run_dir / 'eval' / 'test_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(test_summary, indent=2))
    print(json.dumps(test_summary, indent=2))

    top5_probs, top5_idx = torch.topk(probs, k=5, dim=1)

    test_predictions = {
        'architecture': _ARCHITECTURE,
        'run_id': args.run_id,
        'num_strokes': len(labels),
        'evaluated_at': test_summary['evaluated_at'],
        'config': test_summary['config'],
        'predictions': [
            {
                'clip_stem': stems[i],
                'true': classes[int(labels[i])],
                'predicted': classes[int(preds[i])],
                'top_probs':  [
                    {'class': classes[int(top5_idx[i, k])], 'p': float(top5_probs[i, k])}
                    for k in range(5)
                ],
            }
            for i in range(len(labels))
            ]
    }

    predictions_path = run_dir / 'eval' / 'test_predictions.json'
    predictions_path.write_text(json.dumps(test_predictions, indent=2))

    test_predictions_v1_api_compatible = {
        'architecture': _ARCHITECTURE,
        '_real_stems': True,
        'run_id': args.run_id,
        'split': 'test',
        'evaluated_at': test_summary['evaluated_at'],
        'num_strokes': len(labels),
        'config': test_summary['config'],
        'active_class_list': classes,
        'temperature': 1.0,
        'clips': [
            {
                'clip_stem': stems[i],
                'y_true': int(labels[i]),
                'y_pred': int(preds[i]),
                'softmax_calibrated': 'None',
                'top_k_idx': [int(top5_idx[i, k]) for k in range(5)],
                'top_k_prob': [float(top5_probs[i, k]) for k in range(5)]
            }
            for i in range(len(labels))
            ]
    }

    predictions_v1_api_compatible_path = run_dir / 'predictions' / 'test.json'
    predictions_v1_api_compatible_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_v1_api_compatible_path.write_text(json.dumps(test_predictions_v1_api_compatible, indent=2))

    plot_confusion_matrix(
        y_true=labels.numpy(), y_pred=preds.numpy(),
        class_names=classes, model_name=args.run_id,
        save_name=str(run_dir / 'eval' / 'test'),
    )

if __name__ == '__main__':
    main()