import { useState, useMemo } from 'react';
import { useTheme } from '../shared';

export function BrowseAllModal({ items, onSelect, onClose }) {
  const { t } = useTheme();
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(v =>
      v.match.toLowerCase().includes(q) ||
      v.tournament.toLowerCase().includes(q)
    );
  }, [query, items]);

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.65)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 100, padding: 32,
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: t.surface, border: `1px solid ${t.border}`,
          borderRadius: 12, width: 'min(640px, 100%)', maxHeight: '80vh',
          display: 'flex', flexDirection: 'column',
          boxShadow: '0 24px 60px rgba(0,0,0,0.55)',
        }}
      >
        <div style={{
          padding: '18px 20px', borderBottom: `1px solid ${t.border}`,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14,
        }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: t.text }}>Match Library</div>
            <div style={{ fontSize: 11, color: t.muted, marginTop: 2 }}>
              {filtered.length} of {items.length} matches
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none', border: 'none', color: t.muted,
              fontSize: 22, cursor: 'pointer', padding: 4, lineHeight: 1,
            }}
            aria-label="Close"
          >×</button>
        </div>

        <div style={{ padding: '14px 20px', borderBottom: `1px solid ${t.border}` }}>
          <input
            autoFocus
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search by player or tournament…"
            style={{
              width: '100%', padding: '10px 12px',
              background: t.surface2, border: `1px solid ${t.border}`,
              borderRadius: 7, color: t.text, fontSize: 13,
              fontFamily: "'Space Grotesk', sans-serif", outline: 'none',
            }}
          />
        </div>

        <div style={{ overflowY: 'auto', padding: '8px 0' }}>
          {filtered.length === 0 && (
            <div style={{ padding: '32px 20px', textAlign: 'center', color: t.muted, fontSize: 13 }}>
              No matches found.
            </div>
          )}
          {filtered.map(v => (
            <button
              key={v.id}
              onClick={() => onSelect(v)}
              style={{
                width: '100%', padding: '10px 20px',
                background: 'none', border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 14,
                textAlign: 'left', color: t.text,
                fontFamily: "'Space Grotesk', sans-serif",
              }}
              onMouseEnter={e => e.currentTarget.style.background = t.surface2}
              onMouseLeave={e => e.currentTarget.style.background = 'none'}
            >
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 500, color: t.text, marginBottom: 2 }}>
                  {v.match}
                </div>
                <div style={{ fontSize: 11, color: t.muted }}>{v.tournament}</div>
              </div>
              <div style={{
                fontSize: 11, color: t.muted, whiteSpace: 'nowrap',
                fontFamily: "'JetBrains Mono', monospace",
              }}>
                {v.strokes} strokes
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}