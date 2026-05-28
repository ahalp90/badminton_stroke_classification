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

export function toModelCard(entry) {
  const macro = entry.test_metrics?.macro_f1;
  const min   = entry.test_metrics?.min_f1;
  const acc   = entry.test_metrics?.accuracy;
  const arch  = entry.architecture ?? 'bst-x';
  return {
    id:       entry.id,
    name:     entry.display_name,
    subtitle: `${entry.taxonomy} · ${entry.ablation_id}`,
    tags: [
      { label: ARCH_LABELS[arch] ?? arch.toUpperCase(), color: 'blue' },
      { label: entry.taxonomy,                   color: 'pine' },
      { label: `${entry.num_classes}-class`,     color: 'muted' },
    ],
    description: entry.description,
    stats: [
      ...(macro != null ? [{ label: 'Macro F1', value: macro.toFixed(3) }] : []),
      ...(min   != null ? [{ label: 'Min F1',   value: min.toFixed(3)   }] : []),
      ...(acc   != null ? [{ label: 'Accuracy', value: acc.toFixed(3)   }] : []),
    ],
  };
}