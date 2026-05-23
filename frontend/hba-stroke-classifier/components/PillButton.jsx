import { useState } from "react";

export function PillButton({ label, badge, badgeColor, isActive, onSelect, onDelete, onContextMenu, t }) {
  const [hover, setHover] = useState(false);
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onContextMenu={onContextMenu}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 6,
        padding: '5px 10px', borderRadius: 6,
        background: isActive ? t.blue : t.surface,
        color: isActive ? '#fff' : t.text,
        border: `1px solid ${isActive ? t.blue : t.border}`,
        fontSize: 12, fontWeight: 600,
        cursor: 'pointer',
        position: 'relative',
        transition: 'all 0.12s',
      }}
      onClick={onSelect}
    >
      <span style={{ color: isActive ? '#fff' : badgeColor }}>{badge}</span>
      <span>{label}</span>
      {isActive && (
        <span style={{ fontSize: 10, opacity: 0.85, marginLeft: 2 }}>
          (editing)
        </span>
      )}
      {(hover || isActive) && (
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          title="Delete stroke"
          style={{
            background: isActive ? 'rgba(255,255,255,0.25)' : 'transparent',
            color: isActive ? '#fff' : t.muted,
            border: 'none', cursor: 'pointer',
            width: 16, height: 16, borderRadius: 3,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 12, fontWeight: 700, lineHeight: 1,
            marginLeft: 2,
          }}
        >
          ×
        </button>
      )}
    </div>
  );
}