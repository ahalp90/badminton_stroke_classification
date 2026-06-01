/**
 * toModelCard function is a registry adapter
 * 
 * /api/registry returns architecture-agnostic model entries. Function maps each
 * into the visual model-card shape the interface uses, so adding a new model 
 * (e.g. Architecture 2) later needs no card changes.
 * 
 * @param {} entry 
 * @returns 
 */
const ARCH_LABELS = { 'bst-x': 'BST-X', 'bric': 'BRIC' };

// Short, architecture-level blurb for the card. The per-run provenance
// (taxonomy / split) already lives in the display_name title and the tags, so
// the card body describes what the *architecture* is, not the run. Falls back
// to the registry entry's own description for any arch not listed here.
const ARCH_NOTES = {
  'bst-x': 'Skeleton keypoint and shuttle trajectory transformer. No pre-training. Built on BST-CG-AP. Adds: player detection, CDB-F1 loss, scheduling and augmentations. 1.85M trainable parameters, fed by a frozen perception stack (RTMPose-L pose, RTMDet-nano detector, TrackNetV3 shuttle; ~40M params).',
  'bric': 'R(2+1)D-18 RGB backbone (Kinetics-400 pretrained, fine-tuned end-to-end) fused with a shuttle-trajectory TCN. 31.3M trainable parameters, plus a frozen perception stack (YOLO11n player crops, TrackNetV3 shuttle; ~14.5M params).',
};

export function toModelCard(entry) {
  const macro = entry.test_metrics?.macro_f1;
  const min   = entry.test_metrics?.min_f1;
  const acc   = entry.test_metrics?.accuracy;
  const arch  = entry.architecture ?? 'bst-x';
  return {
    id:       entry.id,
    name:     entry.display_name,
    // Carried through (not just rendered) so the Configure screen can group by
    // architecture and pick the headline card without re-fetching the registry.
    architecture: arch,
    isDefault:    entry.is_default === true,
    // Single non-title fact kept as a tag: the class count (varies across
    // variants, and isn't in the title). Architecture and taxonomy are already
    // in the display_name title, so they're not repeated as tags here.
    tags: [
      { label: `${entry.num_classes}-class`, color: 'muted' },
    ],
    description: ARCH_NOTES[arch] ?? entry.description,
    stats: [
      ...(macro != null ? [{ label: 'Macro F1', value: macro.toFixed(3) }] : []),
      ...(min   != null ? [{ label: 'Min F1',   value: min.toFixed(3)   }] : []),
      ...(acc   != null ? [{ label: 'Accuracy', value: acc.toFixed(3)   }] : []),
    ],
  };
}