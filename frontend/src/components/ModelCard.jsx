import { useState } from 'react'
import { useTheme, Badge } from '../shared'

/**
 * ModelCard function
 * 
 * @param {} param0 
 * @returns 
 */
export function ModelCard({ model, enabled, disabled, onToggle }) {
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