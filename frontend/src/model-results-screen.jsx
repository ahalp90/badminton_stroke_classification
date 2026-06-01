import { useState, useEffect, useMemo } from 'react';
import { useTheme, SectionHeader } from './shared';
import { ModelEvaluationPanel } from './components/ModelEvaluationPanel';
import { ModelComingSoonCard } from './components/ModelComingSoonCard';

/* ─── Full, always-expanded model section (default BST-X, BRIC) ─── */
function ModelSection({ model }) {
  const { t } = useTheme();
  return (
    <section>
      <h3 style={{ fontSize: 18, fontWeight: 700, color: t.text, margin: '0 0 12px' }}>
        {model.display_name}
      </h3>
      {model.status === 'pending'
        ? <ModelComingSoonCard model={model} />
        : <ModelEvaluationPanel modelId={model.id} model={model} />}
    </section>
  );
}

/* ─── Collapsed-by-default accordion row (the alternate BST-X variants) ───
   Mounts the heavy ModelEvaluationPanel (which fires a /clips fetch via
   Tier1ClipBrowser) only once opened, so the page doesn't fan out a fetch
   per variant on load. The header stays useful while collapsed: name +
   taxonomy/split + headline macro F1. */
function CollapsibleModelSection({ model }) {
  const { t } = useTheme();
  const [open, setOpen] = useState(false);
  const macro = model.test_metrics?.macro_f1;
  const min = model.test_metrics?.min_f1;
  return (
    <section style={{ border: `1px solid ${t.border}`, borderRadius: 10, overflow: 'hidden' }}>
      <button
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14,
          padding: '14px 18px', background: open ? t.surface2 : t.surface, border: 'none',
          cursor: 'pointer', textAlign: 'left', color: t.text,
          fontFamily: "'Space Grotesk', sans-serif",
        }}
      >
        <span style={{ display: 'flex', alignItems: 'baseline', gap: 12, minWidth: 0 }}>
          <span style={{ fontSize: 15, fontWeight: 600, color: t.text }}>{model.display_name}</span>
          <span style={{ fontSize: 11, color: t.muted, fontFamily: "'JetBrains Mono',monospace", whiteSpace: 'nowrap' }}>
            {model.num_classes}-class · {model.split_column?.replace(/^split_/, '') || '—'} split
          </span>
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0 }}>
          {typeof macro === 'number' && (
            <span style={{ fontSize: 12, color: t.muted, fontFamily: "'JetBrains Mono',monospace" }}>
              macro F1 <span style={{ color: t.text, fontWeight: 600 }}>{macro.toFixed(3)}</span>
              {typeof min === 'number' && (
                <> / min F1 <span style={{ color: t.text, fontWeight: 600 }}>{min.toFixed(3)}</span></>
              )}
            </span>
          )}
          <span style={{
            fontSize: 12, color: t.muted, transition: 'transform 0.15s',
            transform: open ? 'rotate(90deg)' : 'none',
          }}>▶</span>
        </span>
      </button>
      {open && (
        <div style={{ padding: '16px 18px 4px', borderTop: `1px solid ${t.border}` }}>
          {model.status === 'pending'
            ? <ModelComingSoonCard model={model} />
            : <ModelEvaluationPanel modelId={model.id} model={model} />}
        </div>
      )}
    </section>
  );
}

/* Model Results: registry-driven showcase of each trained model's held-out
   evaluation. The default BST-X cell and any non-BST-X models (BRIC) get full
   expanded panels; the alternate BST-X variants are collapsed into accordion
   rows so the page leads with one BST-X result and BRIC, not six. No
   placeholder/fabricated content. */
export function ModelResultsScreen() {
  const { t } = useTheme();
  const [models, setModels] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/registry')
      .then(r => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(data => setModels(data.models || []))
      .catch(err => setError(err.message));
  }, []);

  // ──── Group: default BST-X (lead), alternate BST-X (collapsed), others (BRIC) ────
  const { bstDefault, bstOthers, others } = useMemo(() => {
    const bst = models.filter(m => m.architecture === 'bst-x');
    const def = bst.find(m => m.is_default) ?? bst[0] ?? null;
    return {
      bstDefault: def,
      bstOthers: bst.filter(m => m.id !== def?.id),
      others: models.filter(m => m.architecture !== 'bst-x'),
    };
  }, [models]);

  return (
    <div style={{
      maxWidth: 1100, margin: '0 auto', padding: 32,
      display: 'flex', flexDirection: 'column', gap: 32,
    }}>
      <SectionHeader title="Model Results" subtitle="Held-out evaluation for each trained model" />

      {error && (
        <div style={{ fontSize: 13, color: t.danger }}>
          Couldn&apos;t load models: {error}
        </div>
      )}
      {!error && models.length === 0 && (
        <div style={{ fontSize: 13, color: t.muted }}>Loading models…</div>
      )}

      {/* Lead BST-X result, expanded. */}
      {bstDefault && <ModelSection model={bstDefault} />}

      {/* Other architectures (BRIC), expanded — kept alongside the lead result. */}
      {others.map(model => (
        <ModelSection key={model.id} model={model} />
      ))}

      {/* Alternate BST-X variants, collapsed, below the headline results. */}
      {bstOthers.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{
            fontSize: 11, color: t.muted, textTransform: 'uppercase',
            letterSpacing: '0.06em', fontWeight: 600,
          }}>
            Other BST-X variants ({bstOthers.length})
          </div>
          {bstOthers.map(model => (
            <CollapsibleModelSection key={model.id} model={model} />
          ))}
        </div>
      )}
    </div>
  );
}
