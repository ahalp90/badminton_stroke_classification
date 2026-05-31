import { useState, useRef } from 'react';
import { useTheme } from '../../shared';
import { recordUpload, useStoredUploads, toUploadVideo } from '../../utils/uploadStorage';

import { ACCEPTED_VIDEO_TYPES } from '../../utils/uploadStorage';

/** File upload tab. Accepts drag-and-drop or click-to-browse.
 * Shows a count of past uploads stored on this device with instructions to access them. */
export function UploadTab({ onUpload }) {
  const { t } = useTheme();

  // ──── State ────────────────────────────────────────────────────────────────────────────────────
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const stored = useStoredUploads();

  // ──── Actions ──────────────────────────────────────────────────────────────────────────────────
  /** Stages a file for upload - saves file metadata to IndexedDB and advances the wizard. */
  const acceptFile = (file) => {
    if (!file) return;
    setDragOver(false);
    // recordUpload saves metadata to IndexedDB (for the My Uploads list) and returns an
    // entry used to create the video object for the wizard - the actual backend upload
    // happens later in ProgressScreen on Submit.
    const entry = recordUpload(file);
    onUpload(toUploadVideo(entry));
  };

  // ──── Render ───────────────────────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_VIDEO_TYPES}
        style={{ display: 'none' }}
        onChange={e => {
          const file = e.target.files?.[0];
          // Reset so re-picking the same file later still fires onChange.
          e.target.value = '';
          acceptFile(file);
        }}
      />
      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); acceptFile(e.dataTransfer.files?.[0]); }}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? t.blue : t.border}`,
          borderRadius: 12, padding: '52px 32px', textAlign: 'center',
          background: dragOver ? t.blueDim : t.surface2,
          transition: 'all 0.2s', cursor: 'pointer',
        }}
      >
        <div style={{ fontSize: 32, marginBottom: 12, opacity: 0.7 }}>⬆</div>
        <div style={{ fontSize: 15, fontWeight: 600, color: t.text, marginBottom: 6 }}>
          Drop video here, or click to browse
        </div>
        <div style={{ fontSize: 12, color: t.muted }}>MP4, MOV, AVI · up to 10 GB</div>
      </div>

      <div style={{ display: 'flex', gap: 12, padding: '14px 16px', background: t.surface2, borderRadius: 8, border: `1px solid ${t.border}` }}>
        <div style={{ fontSize: 20 }}>ℹ</div>
        <div style={{ fontSize: 12, color: t.muted, lineHeight: 1.6 }}>
          Your upload stays on this device — the file is sent to the
          backend only when you click <em>Submit for Analysis</em> on the
          Configure screen. Past uploads appear under <em>My Uploads </em>
          in the <em>Browse all matches</em> dialog; if you've refreshed,
          re-pick the file to continue using it.
        </div>
      </div>

      {stored.length > 0 && (
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`,
          borderRadius: 8, padding: '14px 16px',
        }}>
          <div style={{ fontSize: 11, color: t.muted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
            {stored.length} past upload{stored.length === 1 ? '' : 's'} on this device
          </div>
          <div style={{ fontSize: 12, color: t.muted, lineHeight: 1.6 }}>
            Open the <strong style={{ color: t.text }}>Match Library</strong> tab → <strong style={{ color: t.text }}>Browse all</strong> to pick one of your earlier uploads.
          </div>
        </div>
      )}
    </div>
  );
}