import { useState } from 'react';
import { useTheme, Btn, Badge, SectionHeader } from './shared';

const SHUTTLESET_VIDEOS = [
  { id: 'ss001', match: 'Chen Long vs Lee Chong Wei', tournament: 'BWF World Championships 2019', date: '2019-08-25', duration: '1:23:42', strokes: 847, annotated: true },
  { id: 'ss002', match: 'Viktor Axelsen vs Kento Momota', tournament: 'All England Open 2020', date: '2020-03-15', duration: '0:58:17', strokes: 623, annotated: true },
  { id: 'ss003', match: 'Tai Tzu-ying vs P.V. Sindhu', tournament: 'Tokyo Olympics 2021', date: '2021-07-28', duration: '1:12:05', strokes: 734, annotated: true },
  { id: 'ss004', match: 'Anders Antonsen vs Jonatan Christie', tournament: 'Thomas Cup 2022', date: '2022-05-14', duration: '0:45:33', strokes: 512, annotated: true },
  { id: 'ss005', match: 'An Se-young vs Akane Yamaguchi', tournament: 'Korea Open 2023', date: '2023-09-09', duration: '1:05:19', strokes: 698, annotated: true },
  { id: 'ss006', match: 'Loh Kean Yew vs Chou Tien Chen', tournament: 'Singapore Open 2023', date: '2023-06-07', duration: '0:52:44', strokes: 581, annotated: false },
];

function CourtThumbnail({ seed, selected }) {
  const { t } = useTheme();
  const hue = 28 + (seed * 7) % 12;
  return (
    <div style={{
      height: 96, borderRadius: 7, overflow: 'hidden', position: 'relative',
      background: `linear-gradient(135deg, hsl(${hue},55%,32%), hsl(${hue+5},48%,40%), hsl(${hue-3},52%,28%))`,
    }}>
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: 'repeating-linear-gradient(91deg, rgba(0,0,0,0.06) 0px, transparent 1px, transparent 16px)',
        pointerEvents: 'none',
      }} />
      <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} viewBox="0 0 200 96" preserveAspectRatio="none">
        <polygon points="22,8 178,8 186,88 14,88" fill="rgba(37,99,235,0.12)" stroke="rgba(255,255,255,0.55)" strokeWidth="1.2" />
        <line x1="100" y1="8" x2="100" y2="88" stroke="rgba(255,255,255,0.3)" strokeWidth="0.8" />
        <line x1="18" y1="48" x2="182" y2="48" stroke="rgba(255,255,255,0.6)" strokeWidth="1.4" />
        <line x1="44" y1="8" x2="40" y2="88" stroke="rgba(255,255,255,0.2)" strokeWidth="0.7" />
        <line x1="156" y1="8" x2="160" y2="88" stroke="rgba(255,255,255,0.2)" strokeWidth="0.7" />
        <rect x="60" y="8" width="80" height="40" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="0.7" />
        <rect x="60" y="48" width="80" height="40" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="0.7" />
      </svg>
      {selected && (
        <div style={{
          position: 'absolute', top: 6, right: 6,
          width: 22, height: 22, borderRadius: '50%',
          background: '#22C55E', display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#fff', fontSize: 12, fontWeight: 700,
        }}>✓</div>
      )}
    </div>
  );
}

function VideoCard({ video, selected, onSelect, index }) {
  const { t } = useTheme();
  const [hov, setHov] = useState(false);
  return (
    <div
      onClick={() => onSelect(video)}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: selected ? t.blueDim : hov ? t.surface2 : t.surface,
        border: `1.5px solid ${selected ? t.blue : hov ? t.blue + '55' : t.border}`,
        borderRadius: 10, padding: 14, cursor: 'pointer',
        transition: 'all 0.15s',
        display: 'flex', flexDirection: 'column', gap: 10,
      }}
    >
      <CourtThumbnail seed={index} selected={selected} />
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: t.text, marginBottom: 2, lineHeight: 1.35, textWrap: 'pretty' }}>
          {video.match}
        </div>
        <div style={{ fontSize: 11, color: t.muted }}>{video.tournament}</div>
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
        <Badge color={video.annotated ? 'green' : 'muted'}>
          {video.annotated ? 'Annotated' : 'Unannotated'}
        </Badge>
        <Badge color="blue">{video.strokes} strokes</Badge>
        <span style={{ fontSize: 11, color: t.muted, marginLeft: 'auto', fontFamily: "'JetBrains Mono', monospace" }}>
          {video.duration}
        </span>
      </div>
    </div>
  );
}

function UploadTab({ onUpload }) {
  const { t } = useTheme();
  const [dragOver, setDragOver] = useState(false);
  const [uploadType, setUploadType] = useState('annotated');

  const handleDrop = () => {
    setDragOver(false);
    onUpload({
      id: 'upload_' + Date.now(),
      match: 'Uploaded Video',
      tournament: 'Custom Upload',
      duration: '—',
      strokes: 0,
      annotated: uploadType === 'annotated',
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <div style={{ fontSize: 13, color: t.muted, marginBottom: 10 }}>Video type</div>
        <div style={{ display: 'flex', gap: 8 }}>
          {[
            { id: 'annotated',   label: 'Annotated', desc: 'Includes stroke labels — enables validation metrics' },
            { id: 'unannotated', label: 'Unannotated', desc: 'Classification only, no ground-truth comparison' },
          ].map(opt => (
            <div
              key={opt.id}
              onClick={() => setUploadType(opt.id)}
              style={{
                flex: 1, padding: '12px 16px', borderRadius: 8, cursor: 'pointer',
                border: `1.5px solid ${uploadType === opt.id ? t.blue : t.border}`,
                background: uploadType === opt.id ? t.blueDim : t.surface2,
                transition: 'all 0.15s',
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 600, color: uploadType === opt.id ? t.blue : t.text, marginBottom: 3 }}>
                {opt.label}
              </div>
              <div style={{ fontSize: 11, color: t.muted }}>{opt.desc}</div>
            </div>
          ))}
        </div>
      </div>

      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); handleDrop(); }}
        onClick={handleDrop}
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
        {uploadType === 'unannotated' && (
          <div style={{
            marginTop: 14, display: 'inline-block',
            background: t.pineDim, color: t.pine,
            fontSize: 12, padding: '6px 14px', borderRadius: 6,
          }}>
            No ground-truth labels — validation metrics unavailable
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: 12, padding: '14px 16px', background: t.surface2, borderRadius: 8, border: `1px solid ${t.border}` }}>
        <div style={{ fontSize: 20 }}>ℹ</div>
        <div style={{ fontSize: 12, color: t.muted, lineHeight: 1.6 }}>
          To add an annotated video to the ShuttleSet validation set, upload it here with its corresponding label file.
          Unannotated videos are classified immediately without accuracy evaluation.
        </div>
      </div>
    </div>
  );
}

export function LibraryScreen({ onNext }) {
  const { t } = useTheme();
  const [tab, setTab] = useState('shuttleset');
  const [selected, setSelected] = useState(null);

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto', padding: 32 }}>
      <SectionHeader
        title="Select Match Video"
        subtitle="Choose from the ShuttleSet validation set or upload your own footage."
      />

      <div style={{ display: 'flex', borderBottom: `1px solid ${t.border}`, marginBottom: 24 }}>
        {[['shuttleset', 'ShuttleSet Validation Set'], ['upload', 'Upload Video']].map(([id, label]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            style={{
              padding: '10px 20px', background: 'none', border: 'none', marginBottom: -1,
              borderBottom: tab === id ? `2px solid ${t.blue}` : '2px solid transparent',
              color: tab === id ? t.blue : t.muted,
              fontSize: 14, fontWeight: tab === id ? 600 : 400,
              cursor: 'pointer', fontFamily: "'Space Grotesk', sans-serif",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'shuttleset' ? (
        <>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
            gap: 14, marginBottom: 24,
          }}>
            {SHUTTLESET_VIDEOS.map((v, i) => (
              <VideoCard
                key={v.id}
                video={v}
                index={i}
                selected={selected?.id === v.id}
                onSelect={setSelected}
              />
            ))}
          </div>

          {selected && (
            <div style={{
              position: 'sticky', bottom: 24,
              background: t.surface, border: `1px solid ${t.border}`,
              borderRadius: 10, padding: '14px 20px',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              boxShadow: '0 8px 32px rgba(0,0,0,0.35)',
            }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: t.text }}>{selected.match}</div>
                <div style={{ fontSize: 12, color: t.muted }}>
                  {selected.tournament} · {selected.strokes} annotated strokes · {selected.duration}
                </div>
              </div>
              <Btn onClick={() => onNext(selected)}>Begin Markup →</Btn>
            </div>
          )}
        </>
      ) : (
        <UploadTab onUpload={v => { onNext(v); }} />
      )}
    </div>
  );
}
