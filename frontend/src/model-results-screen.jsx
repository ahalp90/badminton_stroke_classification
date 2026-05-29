import { useState, useEffect } from 'react';
import { useTheme, SectionHeader } from './shared';
import { ModelEvaluationPanel } from './components/ModelEvaluationPanel';
import { ModelComingSoonCard } from './components/ModelComingSoonCard';

/* Model Results: registry-driven showcase of each trained model's held-out
   evaluation. One panel per model; pending models (no metrics yet) get a
   coming-soon card. No placeholder/fabricated content. */
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

      {models.map(model => (
        <section key={model.id}>
          <h3 style={{ fontSize: 18, fontWeight: 700, color: t.text, margin: '0 0 12px' }}>
            {model.display_name}
          </h3>
          {model.status === 'pending'
            ? <ModelComingSoonCard model={model} />
            : <ModelEvaluationPanel modelId={model.id} model={model} />}
        </section>
      ))}
    </div>
  );
}
