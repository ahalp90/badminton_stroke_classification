import { useState, useEffect } from 'react';
import { useTheme } from '../../shared';
import { UPLOAD_STAGES } from './uploadStages';

export function UploadingPanel({ filename, onDone }) {
  const { t } = useTheme();
  const [stage, setStage] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (stage >= UPLOAD_STAGES.length) {
      const id = setTimeout(onDone, 200);
      return () => clearTimeout(id);
    }
    setProgress(0);
    const start = Date.now();
    const dur = UPLOAD_STAGES[stage].ms;
    const tick = setInterval(() => {
      const f = Math.min(1, (Date.now() - start) / dur);
      setProgress(f);
      if (f >= 1) {
        clearInterval(tick);
        setStage(s => s + 1);
      }
    }, 50);
    return () => clearInterval(tick);
  }, [stage, onDone]);

  return (
    <div style={{
      background: t.surface, border: `1px solid ${t.border}`, borderRadius: 10,
      padding: '24px 28px',
    }}>
      <div style={{
        display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 18,
      }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: t.text }}>{filename}</div>
        <div style={{ fontSize: 12, color: t.muted, fontFamily: "'JetBrains Mono', monospace" }}>
          {Math.min(stage, UPLOAD_STAGES.length - 1) + 1} / {UPLOAD_STAGES.length}
        </div>
      </div>

      {UPLOAD_STAGES.map((s, i) => {
        const done = i < stage;
        const active = i === stage;
        const pct = active ? progress : done ? 1 : 0;
        return (
          <div key={s.id} style={{ marginBottom: 12 }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 10, marginBottom: 5,
              fontSize: 12, color: done ? t.success : active ? t.text : t.muted,
            }}>
              <span style={{
                display: 'inline-flex', width: 16, height: 16, borderRadius: '50%',
                alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                background: done ? t.success : active ? t.blue : 'transparent',
                border: `1.5px solid ${done ? t.success : active ? t.blue : t.border}`,
                color: '#fff', fontSize: 9, fontWeight: 700,
              }}>
                {done ? '✓' : active ? '' : ''}
              </span>
              <span style={{ flex: 1, fontWeight: active ? 600 : 400 }}>{s.label}</span>
              <span style={{
                fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                color: t.muted, opacity: done || active ? 1 : 0.5,
              }}>
                {Math.round(pct * 100)}%
              </span>
            </div>
            <div style={{ height: 3, background: t.surface2, borderRadius: 2, overflow: 'hidden' }}>
              <div style={{
                height: '100%', width: `${pct * 100}%`,
                background: done ? t.success : t.blue,
                transition: active ? 'width 0.05s linear' : 'width 0.2s ease',
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}