import { PillButton } from "./PillButton";

/* ─── Stroke pill strip (multi-stroke selector / add / delete) ───── */
export function StrokePillStrip({
  annotations, activeId, onSelect, onDelete, onAdd,
  pendingDeleteId, onConfirmDelete, onCancelDelete,
  isAnnotationComplete, isAnnotationValid, ready, t,
}) {
  return (
    <div style={{
      display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8,
      padding: '8px 10px', background: t.surface2, borderRadius: 8,
    }}>
      <span style={{
        fontSize: 11, color: t.muted, textTransform: 'uppercase',
        letterSpacing: '0.05em', marginRight: 2,
      }}>
        Strokes
      </span>
      {annotations.map((a, i) => {
        const isActive = a.id === activeId;
        const complete = isAnnotationComplete(a);
        const valid = isAnnotationValid(a);
        const empty = !complete && a.startSec === null && a.targetSec === null && a.endSec === null;
        let badge;
        if (empty) badge = '·';
        else if (!complete) badge = '⚠';
        else if (!valid) badge = '⚠';
        else badge = '✓';
        const badgeColor = !complete || !valid ? t.warning : t.success;
        const isPending = pendingDeleteId === a.id;
        if (isPending) {
          return (
            <div key={a.id} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '4px 8px', borderRadius: 6,
              background: t.dangerDim, color: t.danger,
              border: `1px solid ${t.danger}`, fontSize: 11,
            }}>
              Delete stroke {i + 1}?
              <button
                onClick={() => onConfirmDelete(a.id)}
                style={{
                  background: t.danger, color: '#fff', border: 'none',
                  padding: '3px 8px', borderRadius: 4, fontSize: 11,
                  fontWeight: 600, cursor: 'pointer',
                }}
              >Delete</button>
              <button
                onClick={onCancelDelete}
                style={{
                  background: 'transparent', color: t.text,
                  border: `1px solid ${t.border}`,
                  padding: '3px 8px', borderRadius: 4, fontSize: 11,
                  cursor: 'pointer',
                }}
              >Cancel</button>
            </div>
          );
        }
        return (
          <PillButton
            key={a.id}
            label={`Stroke ${i + 1}`}
            badge={badge}
            badgeColor={badgeColor}
            isActive={isActive}
            onSelect={() => onSelect(a.id)}
            onDelete={() => onDelete(a.id)}
            onContextMenu={(e) => { e.preventDefault(); onDelete(a.id); }}
            t={t}
          />
        );
      })}
      <button
        onClick={onAdd}
        disabled={!ready}
        title={ready ? 'Add a new stroke at the current playhead' : 'Waiting for video to load'}
        style={{
          background: 'transparent', color: t.blue,
          border: `1px dashed ${t.blue}`,
          padding: '6px 12px', borderRadius: 6,
          fontSize: 12, fontWeight: 600,
          cursor: ready ? 'pointer' : 'not-allowed',
          opacity: ready ? 1 : 0.5,
        }}
      >
        + Add stroke
      </button>
    </div>
  );
}