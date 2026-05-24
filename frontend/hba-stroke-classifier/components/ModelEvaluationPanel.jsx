import { useState } from 'react';
import { useTheme, Card } from '../shared';
import { Tier1ClipBrowser } from "./Tier1ClipBrowser";

/* ─── Model-level reference: aggregate metrics (from /api/registry) ─ */
// Renamed and re-framed (Fix 3): these numbers describe the *model*,
// not the analysis the user just ran. The card lives below the
// per-clip browser and the uploaded-inference card so "your result"
// reads first.
function TestEvalCard({ model, split }) {
  const { t } = useTheme();
  if (!model) return null;
  const Mono = ({ children }) => (
    <span style={{ fontFamily: "'JetBrains Mono',monospace", color: t.text }}>{children}</span>
  );
  const heading = `Model performance — ${split} set`;
  const subtitle = `How ${model.display_name} performed across the full ${split} set. Same numbers for every analysis — they describe the model, not this video.`;
  const m = model[`${split}_metrics`];
  const hasMetrics = m && typeof m.macro_f1 === 'number';
  const pct = (x) => (x * 100).toFixed(1) + '%';
  return (
    <Card style={{ padding: 22, marginBottom: 22 }}>
      <div style={{ fontSize: 11, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        {heading}
      </div>
      <div style={{ fontSize: 12, color: t.muted, marginBottom: 4, lineHeight: 1.6 }}>
        {subtitle}
      </div>
      <div style={{ fontSize: 11, color: t.muted, marginBottom: 16, fontFamily: "'JetBrains Mono',monospace" }}>
        {model.num_classes} classes · taxonomy <Mono>{model.taxonomy}</Mono>
      </div>
      {hasMetrics ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
          {[
            { label: 'Macro F1',       value: m.macro_f1.toFixed(3),    color: t.blue },
            { label: 'Accuracy',       value: pct(m.accuracy),          color: t.text },
            { label: 'Top-2 accuracy', value: pct(m.top2_accuracy),     color: t.text },
            { label: 'Min F1',         value: m.min_f1.toFixed(3),      color: t.text },
          ].map((s) => (
            <div key={s.label} style={{ background: t.surface2, borderRadius: 7, padding: 14 }}>
              <div style={{ fontSize: 10, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{s.label}</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono',monospace" }}>{s.value}</div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ fontSize: 13, color: t.muted, padding: '8px 0' }}>
          No {split} metrics available for this model.
        </div>
      )}
    </Card>
  );
}

/* ─── Model-level reference: per-class F1 (from /api/registry) ─── */
// Same framing as TestEvalCard: describes the model's strengths/weaknesses
// across the dataset, NOT a per-stroke output of this analysis (Fix 3).
function PerClassF1Card({ model, split }) {
  const { t } = useTheme();
  if (!model) return null;
  const heading = `Model per-class F1 — ${split} set`;
  const subtitle = `Per-class strength of the model. Lower bars indicate classes it confuses more often.`;
  const perClass = model[`${split}_metrics`]?.per_class_f1;
  const hasData = perClass && Object.keys(perClass).length > 0;
  // Sort desc so the strongest classes read top-down; bar widths scale to
  // the max so small differences between weak classes are still visible.
  const entries = hasData
    ? Object.entries(perClass).sort(([,a],[,b]) => b - a)
    : [];
  const max = hasData ? Math.max(...entries.map(([,v]) => v)): 1;
  return (
    <Card style={{ padding: 22 }}>
      <div style={{ fontSize: 11, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        {heading}
      </div>
      <div style={{ fontSize: 12, color: t.muted, marginBottom: 14, lineHeight: 1.6 }}>
        {subtitle}
      </div>
      {hasData ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {entries.map(([cls, f1]) => (
            <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{
                width: 170, fontSize: 12, color: t.text,
                fontFamily: "'JetBrains Mono',monospace", textAlign: 'right', flexShrink: 0,
              }}>
                {cls}
              </div>
              <div style={{ flex: 1, height: 16, background: t.surface2, borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  height: '100%', width: `${(f1 / max) * 100}%`,
                  background: t.blue, borderRadius: 3,
                }} />
              </div>
              <div style={{
                width: 48, fontSize: 12, fontWeight: 600,
                fontFamily: "'JetBrains Mono',monospace", color: t.text, textAlign: 'right',
              }}>
                {f1.toFixed(2)}
              </div>
            </div>
          ))}
          </div>
      ) : (
        <div style={{ fontSize: 13, color: t.muted, padding: '8px 0' }}>
          No {split} per-class F1 available for this model.
        </div>
      )}
      </Card>
  );
}

export function ModelEvaluationPanel({ modelId, model }) {
    const [split, setSplit] = useState('test');
    return (
        <>
          <Tier1ClipBrowser modelId={modelId} split={split} onSplitChange={setSplit} />
          <TestEvalCard model={model} split={split}/>
          <PerClassF1Card model={model} split={split}/>
        </>
    );
}