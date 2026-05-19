import { useState, useEffect, useRef } from 'react';
import { useTheme, Card, Badge, SectionHeader } from './shared';

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
  { at:  9, msg: 'Frame extraction in progress' },
  { at: 16, msg: 'Player detection: bounding boxes extracted' },
  { at: 24, msg: 'TrackNetV3: shuttlecock trajectory computed' },
  { at: 32, msg: 'MMPose: skeleton keypoint sequences extracted' },
  { at: 52, msg: 'Model A (BST): inference started' },
  { at: 76, msg: 'Model A: inference complete' },
  { at: 82, msg: 'Writing results' },
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

        {(task?.models || []).filter(m => task?.enabled?.[m.id]).map((m, i) => {
          // Model inference sits in stage 2 (52-80% of overall pipeline pct).
          // Stagger each enabled model a bit so multi-model jobs look distinct.
          const startAt    = 52 + i * 4;
          const completeAt = 78 - i * 2;
          const span       = Math.max(1, completeAt - startAt);
          const modelPct   = Math.max(0, Math.min(100, ((pct - startAt) / span) * 100));
          const active     = pct > startAt;
          const complete   = pct > completeAt;
          return (
            <Card key={m.id} style={{ padding: 18, gridColumn: '1 / -1' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: t.text }}>{m.name}</div>
                  <div style={{ fontSize: 11, color: t.muted }}>{m.subtitle}</div>
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