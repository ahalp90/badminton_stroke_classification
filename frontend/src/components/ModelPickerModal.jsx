import { useState, useMemo } from 'react';
import { useTheme } from '../shared';
import { ModelCard } from './ModelCard';

/**
 * ModelPickerModal
 *
 * Lets the user swap the active BST-X variant for a different one without
 * cluttering the Configure screen with every cell as a flat card. Mirrors the
 * BrowseAllModal shell (backdrop click closes, stop-propagation on the panel)
 * but renders the shared ModelCard so the variants look identical to the
 * headline card. Picking a card calls onPick(id) and closes.
 *
 * @param {Object[]} models   BST-X variant cards (already adapted via toModelCard,
 *                            each carrying the raw `architecture`/`is_default` too).
 * @param {string}   activeId Currently selected variant id (highlighted).
 * @param {Function} onPick   (id) => void; caller swaps the active variant.
 * @param {Function} onClose  () => void.
 */
export function ModelPickerModal({ models, activeId, onPick, onClose }) {
  const { t } = useTheme();
  const [query, setQuery] = useState('');

  // ──── Search filter ──────────────────────────────────────────────────────────────────────────
  // Match on the human-facing name, taxonomy, and split so users can type
  // "baseline", "bst_24", etc. Falsy parts are skipped to avoid join noise.
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return models;
    return models.filter(m =>
      [m.name, m.subtitle, ...m.tags.map(tag => tag.label)]
        .filter(Boolean)
        .some(s => s.toLowerCase().includes(q))
    );
  }, [query, models]);

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
          borderRadius: 12, width: 'min(720px, 100%)', maxHeight: '82vh',
          display: 'flex', flexDirection: 'column',
          boxShadow: '0 24px 60px rgba(0,0,0,0.55)',
        }}
      >
        <div style={{
          padding: '18px 20px', borderBottom: `1px solid ${t.border}`,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14,
        }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: t.text }}>Choose BST-X variant</div>
            <div style={{ fontSize: 11, color: t.muted, marginTop: 2 }}>
              Same architecture, different taxonomy and/or split. Compare the test metrics below.
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

        <div style={{ overflowY: 'auto', padding: '14px 20px 18px' }}>
          <input
            autoFocus
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Filter by taxonomy or split…"
            style={{
              width: '100%', padding: '10px 12px', marginBottom: 14,
              background: t.surface2, border: `1px solid ${t.border}`,
              borderRadius: 7, color: t.text, fontSize: 13,
              fontFamily: "'Space Grotesk', sans-serif", outline: 'none',
              boxSizing: 'border-box',
            }}
          />

          {filtered.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: t.muted, fontSize: 13 }}>
              No variants match that filter.
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {filtered.map(m => (
              <ModelCard
                key={m.id}
                model={m}
                enabled={m.id === activeId}
                disabled={false}
                onToggle={() => onPick(m.id)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
