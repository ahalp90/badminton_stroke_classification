import { useRef } from 'react'
import { useTheme } from '../../shared'
import { bindFileToUpload, deleteUpload, fmtSize, getSessionFile, listStoredUploads, toUploadVideo, useStoredUploads } from '../../utils/uploadStorage';

/* ─── My Uploads section (used inside BrowseAllModal) ───────────── */
export function MyUploadsList({ onSelect }) {
  const { t } = useTheme();
  const items = useStoredUploads();
  const rebindInputRef = useRef(null);
  const pendingIdRef = useRef(null);

  if (items.length === 0) {
    return (
      <div style={{ padding: '14px 20px', fontSize: 12, color: t.muted }}>
        No uploads yet. Drop a video on the <em>Upload Video</em> tab to add one.
      </div>
    );
  }

  const handleRebind = (id) => {
    pendingIdRef.current = id;
    rebindInputRef.current?.click();
  };

  const onRebindFileChosen = (e) => {
    const file = e.target.files?.[0];
    const id = pendingIdRef.current;
    e.target.value = ''; // reset so the same file can be picked twice
    pendingIdRef.current = null;
    if (!file || !id) return;
    bindFileToUpload(id, file);
    const entry = listStoredUploads().find(x => x.id === id);
    if (entry) onSelect(toUploadVideo(entry));
  };

  return (
    <div>
      <input
        ref={rebindInputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/*"
        style={{ display: 'none' }}
        onChange={onRebindFileChosen}
      />
      {items.map(entry => {
        const live = !!getSessionFile(entry.id);
        return (
          <div
            key={entry.id}
            style={{
              padding: '10px 20px',
              display: 'flex', alignItems: 'center', gap: 14,
              borderBottom: `1px solid ${t.border}`,
            }}
          >
            <div style={{
              width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
              background: live ? t.success : t.muted,
              boxShadow: live ? `0 0 0 3px ${t.success}22` : 'none',
            }} title={live ? 'Ready to use' : 'Re-upload required'} />
            <div style={{ minWidth: 0, flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: t.text, marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {entry.filename}
              </div>
              <div style={{ fontSize: 11, color: t.muted, fontFamily: "'JetBrains Mono', monospace" }}>
                {fmtSize(entry.size)} · {new Date(entry.uploaded_at).toLocaleString('en-AU')}
                {!live && <span style={{ color: t.warning, marginLeft: 8 }}>— file not in session</span>}
              </div>
            </div>
            <button
              onClick={() => live ? onSelect(toUploadVideo(entry)) : handleRebind(entry.id)}
              style={{
                background: live ? t.blue : t.surface2,
                color: live ? '#fff' : t.text,
                border: `1px solid ${live ? t.blue : t.border}`,
                borderRadius: 5, padding: '5px 12px', fontSize: 12, fontWeight: 600,
                cursor: 'pointer', fontFamily: "'Space Grotesk', sans-serif",
              }}
            >
              {live ? 'Use' : 'Re-upload'}
            </button>
            <button
              onClick={() => {
                if (confirm(`Delete "${entry.filename}" from your uploads list?`)) deleteUpload(entry.id);
              }}
              style={{
                background: 'none', border: 'none',
                color: t.muted, fontSize: 16, cursor: 'pointer',
                padding: '2px 6px', lineHeight: 1,
              }}
              aria-label="Delete upload"
              title="Delete from list"
            >
              ×
            </button>
          </div>
        );
      })}
    </div>
  );
}