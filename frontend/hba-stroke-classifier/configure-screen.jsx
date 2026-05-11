import { useState, useEffect, useRef } from 'react';
import { useTheme, Btn, Card, Badge, SectionHeader } from './shared';

/* ─── Model definitions ─────────────────────────────────────────── */
const MODELS = [
  {
    id: 'A',
    name: 'Model A — Spatio-Temporal',
    subtitle: 'Object detection → 3D-CNN + ShuttleTracker → Deep 1D-CNN',
    tags: [{ label: 'Best Performance', color: 'blue' }, { label: 'GPU Required', color: 'pine' }],
    description: 'Processes raw video frames through 3D convolutional layers to capture spatiotemporal motion patterns. Combined with ShuttleTracker for shuttlecock trajectory analysis. Targets 80–85% classification accuracy.',
    stats: [
      { label: 'Target Acc.',  value: '80–85%' },
      { label: 'Inference',    value: '~2.4s / stroke' },
      { label: 'Architecture', value: '3D-CNN' },
    ],
  },
  {
    id: 'B',
    name: 'Model B — Keypoint Graph',
    subtitle: 'Object detection → MediaPipe keypoints + ShuttleTracker → TCN',
    tags: [{ label: 'Lightweight', color: 'green' }, { label: 'Near Real-Time', color: 'green' }],
    description: 'Extracts body-keypoint graphs via MediaPipe, eliminating raw pixel processing. A Temporal Convolutional Network processes skeleton sequences. Lower compute cost, suitable for near-real-time inference. Targets 70–80%.',
    stats: [
      { label: 'Target Acc.',  value: '70–80%' },
      { label: 'Inference',    value: '~0.3s / stroke' },
      { label: 'Architecture', value: 'TCN' },
    ],
    disabled: true,
  },
];

/* ─── Model card ─────────────────────────────────────────────────── */
function ModelCard({ model, enabled, disabled, onToggle }) {
  const { t } = useTheme();
  const [hov, setHov] = useState(false);
  return (
    <div
      onClick={onToggle}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        border: `1.5px solid ${enabled ? t.blue : hov ? t.border : t.border}`,
        borderRadius: 10, padding: 20,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.25 : 1,
        background: enabled ? t.blueDim : hov ? t.surface2 : t.surface2,
        transition: 'all 0.15s',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: t.text, marginBottom: 2 }}>{model.name}</div>
          <div style={{ fontSize: 11, color: t.muted, fontFamily: "'JetBrains Mono', monospace" }}>{model.subtitle}</div>
        </div>
        <div style={{
          width: 22, height: 22, borderRadius: 5, flexShrink: 0,
          background: enabled ? t.blue : t.surface,
          border: `1.5px solid ${enabled ? t.blue : t.border}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: disabled ? t.muted : '#fff', fontSize: 12, transition: 'all 0.15s',
        }}>
          {enabled && '✓'}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
        {model.tags.map(tag => <Badge key={tag.label} color={tag.color}>{tag.label}</Badge>)}
      </div>

      <div style={{ fontSize: 12, color: t.muted, lineHeight: 1.6, marginBottom: 14 }}>{model.description}</div>

      <div style={{ paddingTop: 12, borderTop: `1px solid ${t.border}`, display: 'flex', gap: 20 }}>
        {model.stats.map(s => (
          <div key={s.label}>
            <div style={{ fontSize: 10, color: t.muted, marginBottom: 2, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{s.label}</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: t.text, fontFamily: "'JetBrains Mono', monospace" }}>{s.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

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
  const [enabled,    setEnabled]    = useState({ A: true, B: false });
  const [taskName,   setTaskName]   = useState(
    `Analysis — ${markup?.video?.match?.split(' vs ')[0] ?? 'Video'} — ${new Date().toLocaleDateString('en-AU')}`
  );

  const anyEnabled = enabled.A || enabled.B;

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
          {MODELS.map(m => (
            <ModelCard
              key={m.id}
              model={m}
              enabled={enabled[m.id]}
              disabled={m.disabled}
              onToggle={() => {}}
            />
          ))}
          <div style={{ fontSize: 11, color: t.muted, lineHeight: 1.5 }}>
            Only Model A is currently available for inference. Model B shown for reference.
          </div>
          {!anyEnabled && (
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

          <Btn disabled={!anyEnabled} onClick={() => onSubmit({ markup, enabled, taskName })}>
            Submit for Analysis →
          </Btn>
          <Btn variant="secondary" onClick={onBack}>← Back</Btn>
        </div>
      </div>
    </div>
  );
}

/* ─── Progress Screen ────────────────────────────────────────────── */
const PIPELINE_STAGES = [
  { label: 'Preprocessing',      desc: 'Extracting frames · normalising court perspective' },
  { label: 'Feature Extraction', desc: 'Object detection · keypoint graphs · shuttle tracking' },
  { label: 'Model Inference',    desc: 'Running selected classification models' },
  { label: 'Postprocessing',     desc: 'Aggregating results · computing evaluation metrics' },
];

const LOG_EVENTS = [
  { at:  0, msg: 'Job submitted — loading video segment…' },
  { at:  4, msg: 'Applying court boundary homography transform' },
  { at:  9, msg: 'Frame extraction: 847 frames identified' },
  { at: 16, msg: 'Player bounding boxes confirmed (det: 0.97, 0.94)' },
  { at: 24, msg: 'ShuttleTracker: shuttlecock trajectory computed' },
  { at: 32, msg: 'MediaPipe: skeleton keypoint sequences extracted' },
  { at: 41, msg: 'Model A (3D-CNN): inference started' },
  { at: 64, msg: 'Model A: inference complete (847 strokes)' },
  { at: 82, msg: 'Computing evaluation metrics against ground truth' },
  { at: 90, msg: 'Generating class activation maps' },
  { at: 96, msg: 'Writing results to database' },
  { at: 100, msg: '✓ Analysis complete' },
];

export function ProgressScreen({ task, onComplete }) {
  const { t } = useTheme();
  const [pct,    setPct]    = useState(0);
  const [stage,  setStage]  = useState(0);
  const [log,    setLog]    = useState([]);
  const logRef = useRef(null);

  useEffect(() => {
    let current = 0;
    const iv = setInterval(() => {
      const increment = Math.random() * 2.8 + 0.6;
      current = Math.min(current + increment, 100);
      setPct(current);

      const newEvents = LOG_EVENTS.filter(e => e.at <= current && e.at > current - increment - 0.1);
      if (newEvents.length > 0) {
        setLog(l => [...l, ...newEvents.map(e => ({ ...e, time: new Date().toLocaleTimeString('en-AU', { hour12: false }) }))]);
      }

      if (current < 25)      setStage(0);
      else if (current < 52) setStage(1);
      else if (current < 80) setStage(2);
      else                   setStage(3);

      if (current >= 100) {
        clearInterval(iv);
        setTimeout(onComplete, 1400);
      }
    }, 280);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const done = pct >= 100;

  return (
    <div style={{ maxWidth: 820, margin: '0 auto', padding: 32 }}>
      <SectionHeader
        title="Analysis in Progress"
        subtitle={task?.taskName}
      />

      <Card style={{ padding: 26, marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: t.text }}>{PIPELINE_STAGES[stage].label}</div>
            <div style={{ fontSize: 12, color: t.muted, marginTop: 2 }}>{PIPELINE_STAGES[stage].desc}</div>
          </div>
          <div style={{ fontSize: 28, fontWeight: 700, color: done ? t.success : t.blue, fontFamily: "'JetBrains Mono', monospace" }}>
            {Math.round(pct)}%
          </div>
        </div>
        <div style={{ height: 8, background: t.surface2, borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            height: '100%', borderRadius: 4,
            width: `${pct}%`,
            background: done
              ? t.success
              : `linear-gradient(90deg, ${t.blue}, ${t.blueLight})`,
            transition: 'width 0.28s ease',
          }} />
        </div>
        {done && (
          <div style={{ marginTop: 12, fontSize: 13, color: t.success, fontWeight: 600 }}>
            ✓ Analysis complete — loading results…
          </div>
        )}
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        <Card style={{ padding: 20 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 16 }}>
            Pipeline
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {PIPELINE_STAGES.map((s, i) => {
              const isDone   = i < stage || done;
              const isActive = i === stage && !done;
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                  <div style={{
                    width: 24, height: 24, borderRadius: '50%', flexShrink: 0, marginTop: 1,
                    background: isDone ? t.success : isActive ? t.blue : 'transparent',
                    border: `1.5px solid ${isDone ? t.success : isActive ? t.blue : t.border}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 10, color: isDone || isActive ? '#fff' : t.muted,
                    fontWeight: 700,
                  }}>
                    {isDone ? '✓' : isActive ? '⟳' : i + 1}
                  </div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: isActive ? 600 : 400, color: isDone ? t.muted : isActive ? t.text : t.muted }}>
                      {s.label}
                    </div>
                    <div style={{ fontSize: 11, color: t.muted, marginTop: 1 }}>{s.desc}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        <Card style={{ padding: 20, display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 16 }}>
            Activity Log
          </div>
          <div
            ref={logRef}
            style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 5, maxHeight: 200 }}
          >
            {log.length === 0 && (
              <div style={{ fontSize: 12, color: t.muted }}>Waiting for pipeline…</div>
            )}
            {log.map((entry, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, fontSize: 11, alignItems: 'flex-start' }}>
                <span style={{ color: t.muted, fontFamily: "'JetBrains Mono', monospace", flexShrink: 0 }}>
                  {entry.time}
                </span>
                <span style={{ color: entry.at >= 100 ? t.success : t.text, lineHeight: 1.4 }}>
                  {entry.msg}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {MODELS.filter(m => !m.disabled).map(m => {
          const modelPct = m.id === 'A' ? Math.max(0, Math.min(100, (pct - 41) / 0.23)) : Math.max(0, Math.min(100, (pct - 52) / 0.22));
          const active   = pct > (m.id === 'A' ? 41 : 52);
          const complete = pct > (m.id === 'A' ? 64 : 74);
          return (
            <Card key={m.id} style={{ padding: 18, gridColumn: '1 / -1' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: t.text }}>Model {m.id}</div>
                  <div style={{ fontSize: 11, color: t.muted }}>{m.stats[2].value}</div>
                </div>
                <Badge color={complete ? 'green' : active ? 'blue' : 'muted'}>
                  {complete ? 'Done' : active ? 'Running' : 'Queued'}
                </Badge>
              </div>
              <div style={{ height: 5, background: t.surface2, borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  height: '100%', borderRadius: 3,
                  width: `${complete ? 100 : modelPct}%`,
                  background: complete ? t.success : t.blue,
                  transition: 'width 0.28s ease',
                }} />
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
