import { useState, useEffect } from 'react';
import { useTheme, Btn, Card, Badge, SectionHeader } from './shared';
import { toModelCard } from './utils/adapters';
import { ModelCard } from './components/ModelCard';

/* ─── Param slider ───────────────────────────────────────────────── */
function ParamSlider({ label, hint, value, min, max, step, onChange, fmt }) {
  const { t } = useTheme();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <div>
          <span style={{ fontSize: 13, color: t.text }}>{label}</span>
          {hint && <span style={{ fontSize: 11, color: t.muted, marginLeft: 6 }}>{hint}</span>}
        </div>
        <span style={{ fontSize: 13, fontWeight: 700, color: t.blue, fontFamily: "'JetBrains Mono', monospace" }}>
          {fmt ? fmt(value) : value}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: t.blue }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: t.muted }}>
        <span>{fmt ? fmt(min) : min}</span><span>{fmt ? fmt(max) : max}</span>
      </div>
    </div>
  );
}

/* ─── Configure Screen ───────────────────────────────────────────── */
export function ConfigureScreen({ markup, onSubmit, onBack }) {
  const { t } = useTheme();
  const [models,    setModels]    = useState([]);
  const [loadError, setLoadError] = useState(null);
  const [enabled,   setEnabled]   = useState({});
  const [taskName,  setTaskName]  = useState(
    `Analysis — ${markup?.video?.match?.split(' vs ')[0] ?? 'Video'} — ${new Date().toLocaleDateString('en-AU')}`
  );

  useEffect(() => {
    let alive = true;
    fetch('/api/registry')
      .then(response => response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`)))
      .then(data => {
        if (!alive) return;
        const items = (data.models || []).map(toModelCard);
        setModels(items);
        // Default: first model on, rest off. Tier 1 ships with one anyway.
        const init = {};
        items.forEach((m, i) => { init[m.id] = i === 0; });
        setEnabled(init);
      })
      .catch(err => { if (alive) setLoadError(err.message); });
    return () => { alive = false; };
  }, []);

  const anyEnabled = Object.values(enabled).some(Boolean);
  const toggle = id => setEnabled(prev => ({ ...prev, [id]: !prev[id] }));

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: 32 }}>
      <SectionHeader
        title="Configure Analysis"
        subtitle="Select models and tune parameters before submitting the classification job."
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 24, alignItems: 'start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Classification Models
          </div>
          {loadError && (
            <div style={{ fontSize: 12, color: t.danger, padding: '8px 12px', background: t.dangerDim, borderRadius: 6 }}>
              Couldn't load model registry: {loadError}
            </div>
          )}
          {!loadError && models.length === 0 && (
            <div style={{ fontSize: 12, color: t.muted, padding: '8px 12px' }}>
              Loading models…
            </div>
          )}
          {models.map(m => (
            <ModelCard
              key={m.id}
              model={m}
              enabled={enabled[m.id]}
              disabled={false}
              onToggle={() => toggle(m.id)}
            />
          ))}
          {models.length > 0 && (
            <div style={{ fontSize: 11, color: t.muted, lineHeight: 1.5 }}>
              Models come from <code>docs/models_registry.yaml</code>. Add a new entry there to surface it here.
            </div>
          )}
          {!anyEnabled && models.length > 0 && (
            <div style={{ fontSize: 12, color: t.danger, padding: '8px 12px', background: t.dangerDim, borderRadius: 6 }}>
              Select at least one model to continue.
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Card style={{ padding: 18 }}>
            <div style={{ fontSize: 12, color: t.muted, marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Task Label</div>
            <input
              value={taskName}
              onChange={e => setTaskName(e.target.value)}
              style={{
                width: '100%', background: t.surface2, border: `1px solid ${t.border}`,
                borderRadius: 6, padding: '8px 10px', color: t.text, fontSize: 13,
                fontFamily: "'Space Grotesk', sans-serif", outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </Card>

          <Card style={{ padding: 18 }}>
            <div style={{ fontSize: 12, color: t.muted, marginBottom: 16, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Parameters</div>
            <div style={{ fontSize: 11, color: t.muted, lineHeight: 1.5 }}>
              Using default model parameters.
            </div>
          </Card>

          <Card style={{ padding: 16 }}>
            <div style={{ fontSize: 12, color: t.muted, marginBottom: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Input Summary</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: t.text, marginBottom: 2 }}>{markup?.video?.match}</div>
            <div style={{ fontSize: 11, color: t.muted, marginBottom: 10 }}>{markup?.video?.tournament}</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <Badge color={markup?.video?.annotated ? 'green' : 'muted'}>
                {markup?.video?.annotated ? 'Annotated' : 'Unannotated'}
              </Badge>
              <Badge color="blue">Player {markup?.player ?? '—'}</Badge>
              {markup?.timeframe && (
                <Badge color="pine">{markup.timeframe.duration}s segment</Badge>
              )}
            </div>
          </Card>

          <Btn disabled={!anyEnabled} onClick={() => onSubmit({ markup, enabled, taskName, models })}>
            Submit for Analysis →
          </Btn>
          <Btn variant="secondary" onClick={onBack}>← Back</Btn>
        </div>
      </div>
    </div>
  );
}