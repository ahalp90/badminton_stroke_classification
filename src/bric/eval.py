""" BRIC evaluation script."""

import argparse
import json
import torch
import yaml
from pathlib import Path
from typing import Any

from torcheval.metrics.functional import multiclass_f1_score
from torch.utils.data import DataLoader

from bric.dataset import ShuttleSetDataset, collate_strokes
from bric.network import BRICNetwork
from bric.train import _resolve_taxonomy

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

def _forward_for_variant(model: BRICNetwork, batch: dict[str, Any]) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if model.use_shuttle:
        kwargs['shuttle'] = batch['shuttle']
        kwargs['shuttle_length'] = batch['shuttle_length']
    if model.use_court:
        kwargs['court_snapshot'] = batch['court_snapshot']
        kwargs['court_sequence'] = batch['court_sequence']
        kwargs['court_sequence_length'] = batch['court_sequence_length']
    return model(batch['rgb'], **kwargs)

def main():
    device = _select_device()
    args = _parse_args()
    run_dir = _EXPERIMENTS / args.run_id
    manifest = yaml.safe_load((run_dir / 'manifest.yaml').read_text())
    taxonomy_name = manifest['config']['taxonomy']
    taxonomy = _resolve_taxonomy(taxonomy_name)
    n_classes = len(manifest['config']['classes'])
    use_shuttle = manifest['config']['use_shuttle']
    use_court = manifest['config']['use_court']
    shuttle_encoder = manifest['training']['hparams'].get('shuttle_encoder', 'mean')
    court_encoder = manifest['training']['hparams'].get('court_encoder', 'null')
    shuttle_window = manifest['training']['hparams'].get('shuttle_window', 'between_hits')
    
    ds: ShuttleSetDataset = ShuttleSetDataset(
    split='test', taxonomy=taxonomy, rgb_transform=None,
    shuttle_window=shuttle_window,
)
    model = BRICNetwork(
    taxonomy=taxonomy, pretrained=False,
    use_shuttle=use_shuttle, use_court=use_court,
    shuttle_encoder=shuttle_encoder,
    court_encoder=court_encoder,
).to(device)
    model.load_state_dict(torch.load(run_dir / 'best.pt', weights_only=False, map_location=device)['model_state_dict'])
    model.eval()

    loader = DataLoader(
    ds, batch_size=32, pin_memory=(device.type == 'cuda'), collate_fn=collate_strokes)

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = _forward_for_variant(model, batch)
            preds.append(logits.argmax(1).cpu())
            labels.append(batch['label'].cpu())
    preds = torch.cat(preds); labels = torch.cat(labels)

    out = {
        'run_id': args.run_id,
        'num_strokes': len(labels),
        'macro_f1': float(multiclass_f1_score(preds, labels, num_classes=n_classes, average='macro')),
        'accuracy': float((preds == labels).float().mean()),
    }
    out_path = run_dir / 'eval' / 'test_summary.json'
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)) 

if __name__ == '__main__':
      main()