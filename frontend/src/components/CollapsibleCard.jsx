import { useState } from 'react';
import { useTheme, Card } from '../shared';

/**
 * CollapsibleCard
 *
 * Card with a clickable title header that expands/collapses its body. Used to
 * break the model evaluation panel into independently collapsible sections
 * (per-clip predictions, performance, per-class F1, training notes) so a page
 * with several models stays navigable.
 *
 * @param {string}   title       Uppercase section label shown in the header.
 * @param {boolean}  defaultOpen Whether the body starts expanded (default true).
 * @param {Node}     children    The section body.
 */
export function CollapsibleCard({ title, defaultOpen = true, children }) {
  const { t } = useTheme();
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card style={{ padding: 0, marginBottom: 22 }}>
      <button
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14,
          padding: '16px 22px', background: 'none', border: 'none', cursor: 'pointer',
          textAlign: 'left', fontFamily: "'Space Grotesk', sans-serif",
        }}
      >
        <span style={{ fontSize: 11, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
          {title}
        </span>
        <span style={{
          fontSize: 12, color: t.muted, flexShrink: 0,
          transition: 'transform 0.15s', transform: open ? 'rotate(90deg)' : 'none',
        }}>▶</span>
      </button>
      {open && <div style={{ padding: '0 22px 22px' }}>{children}</div>}
    </Card>
  );
}
