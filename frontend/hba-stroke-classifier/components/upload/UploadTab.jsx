import { useState } from 'react';
import { useTheme } from '../../shared';
import { UploadingPanel } from './UploadingPanel';

export function UploadTab({ videos, onUpload }) { // TODO: Remove videos param after un-mocking the upload
  const { t } = useTheme();
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(null);

  const startMockUpload = () => {
    setDragOver(false);
    const random = videos[Math.floor(Math.random() * videos.length)];
    const filename = `match_${Date.now()}.mp4`;
    setUploading({ filename, video: { ...random, id: 'upload_' + Date.now(), uploadedAs: random.match } });
  };

  if (uploading) {
    return (
      <UploadingPanel
        filename={uploading.filename}
        onDone={() => onUpload(uploading.video)}
      />
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); startMockUpload(); }}
        onClick={startMockUpload}
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
          Demo mode — uploaded videos are stand-ins for matches in the library.
          The classifier will run against an annotated match so validation metrics stay meaningful.
        </div>
      </div>
    </div>
  );
}