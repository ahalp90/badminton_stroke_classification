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
    // Provenance subtitle from whatever the entry actually carries. ablation_id
    // is null for non-ablation runs (and for BRIC), so filter falsy parts out
    // rather than rendering "taxonomy · null". split is included so models that
    // differ only by split (e.g. the two bst_24 cells) stay distinguishable.
    subtitle: [entry.taxonomy, entry.split_column, entry.ablation_id].filter(Boolean).join(' · '),
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