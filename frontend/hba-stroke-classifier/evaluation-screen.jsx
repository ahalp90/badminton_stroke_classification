import { useState, useEffect } from 'react';
import { useTheme, SectionHeader } from './shared';
import { ModelEvaluationPanel } from './components/ModelEvaluationPanel';

/* ─── Standalone model-evaluation page (#87 / #83) ───────────────────
 * This page is deliberately OUTSIDE the upload wizard. It shows how each
 * registered model scores across the held-out val / test sets — the same
 * numbers for every visitor, describing the model, not any uploaded clip.
 * Keeping this off the wizard's Results screen is the fix for the demo
 * confusion where assessors read whole-set eval as live inference on the
 * video they had just selected.
 *
 * Scaffold scope (build-order steps 1-2): registry fetch + a single panel
 * with a basic model selector. Filters (shot type / match, client-side)
 * and side-by-side model compare are deliberately NOT here yet.
 */
export function EvaluationScreen() {
  const { t } = useTheme();
  const [models, setModels] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [error, setError] = useState(null);

  // Registry fetch moved here out of ResultsScreen: this page, not the
  // wizard, is what depends on /api/registry now.
  useEffect(() => {
    let alive = true;
    fetch('/api/registry')
      .then(response => response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`)))
      .then(data => {
        if (!alive) return;
        const list = data.models || [];
        setModels(list);
        setSelectedId(prev => prev ?? list[0]?.id ?? null);
      })
      .catch(err => { if (alive) setError(err.message); });
    return () => { alive = false; };
  }, []);

  const activeModel = models.find(m => m.id === selectedId) || models[0] || null;

  return (
    <div style={{ maxWidth: 1120, margin: '0 auto', padding: 32 }}>
      <SectionHeader
        title="Model Results"
        subtitle="How each model scores across the held-out val / test sets. These numbers describe the model — they are not inference on any uploaded clip."
      />

      {error && (
        <div style={{ fontSize: 13, color: t.danger, padding: '10px 14px', background: t.dangerDim, borderRadius: 8, marginBottom: 20 }}>
          Couldn&apos;t load the model registry: {error}
        </div>
      )}

      {/* Pick which single model to view. Side-by-side compare is a later step. */}
      {models.length > 1 && (
        <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
          {models.map(m => {
            const active = m.id === activeModel?.id;
            return (
              <button
                key={m.id}
                onClick={() => setSelectedId(m.id)}
                style={{
                  background: active ? t.blue : t.surface2,
                  color: active ? '#fff' : t.muted,
                  border: `1px solid ${active ? t.blue : t.border}`,
                  padding: '6px 14px', borderRadius: 7,
                  fontSize: 13, fontWeight: 600,
                  cursor: 'pointer',
                  fontFamily: "'Space Grotesk', sans-serif",
                }}
              >
                {m.display_name}
              </button>
            );
          })}
        </div>
      )}

      {activeModel ? (
        <ModelEvaluationPanel modelId={activeModel.id} model={activeModel} />
      ) : !error && (
        <div style={{ fontSize: 13, color: t.muted }}>Loading registry…</div>
      )}
    </div>
  );
}
