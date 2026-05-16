import { useState } from 'react';
import { useTheme, Btn, Badge, SectionHeader } from './shared';
import { toVideo } from './utils/videoTransforms';
import { BrowseAllModal } from './components/BrowseAllModal';
import { UploadTab } from './components/upload/UploadTab';
import matchesData from './data/matches.json';

const frameModules = import.meta.glob('./data/frames/*.jpg', { eager: true, import: 'default' });
const frameUrl = (id) => frameModules[`./data/frames/${id}.jpg`];

const CURATED = matchesData.filter(m => m.curated).map(toVideo);
const ALL = matchesData.map(toVideo);

function VideoCard({ video, selected, onSelect }) {
  const { t } = useTheme();
  const [hov, setHov] = useState(false);
  const src = frameUrl(video.youtubeId);
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
      <div style={{
        height: 110, borderRadius: 7, overflow: 'hidden',
        position: 'relative', background: '#000',
      }}>
        {src && (
          <img
            src={src}
            alt=""
            loading="lazy"
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
          />
        )}
        {selected && (
          <div style={{
            position: 'absolute', top: 6, right: 6,
            width: 22, height: 22, borderRadius: '50%',
            background: '#22C55E', display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#fff', fontSize: 12, fontWeight: 700,
          }}>✓</div>
        )}
      </div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: t.text, marginBottom: 2, lineHeight: 1.35, textWrap: 'pretty' }}>
          {video.match}
        </div>
        <div style={{ fontSize: 11, color: t.muted }}>{video.tournament}</div>
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
        <Badge color="green">Annotated</Badge>
        <Badge color="blue">{video.strokes} strokes</Badge>
      </div>
    </div>
  );
}

export function LibraryScreen({ onNext }) {
  const { t } = useTheme();
  const [tab, setTab] = useState('library');
  const [selected, setSelected] = useState(null);
  const [browsing, setBrowsing] = useState(false);

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto', padding: 32 }}>
      <SectionHeader
        title="Select Match Video"
        subtitle="Choose from our match library or upload your own footage."
      />

      <div style={{ display: 'flex', borderBottom: `1px solid ${t.border}`, marginBottom: 24 }}>
        {[['library', 'Match Library'], ['upload', 'Upload Video']].map(([id, label]) => (
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

      {tab === 'library' ? (
        <>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
            gap: 14, marginBottom: 16,
          }}>
            {CURATED.map(v => (
              <VideoCard
                key={v.id}
                video={v}
                selected={selected?.id === v.id}
                onSelect={setSelected}
              />
            ))}
          </div>

          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 24 }}>
            <button
              onClick={() => setBrowsing(true)}
              style={{
                background: 'none', border: `1px solid ${t.border}`,
                color: t.muted, padding: '10px 18px', borderRadius: 7,
                fontSize: 13, cursor: 'pointer',
                fontFamily: "'Space Grotesk', sans-serif",
                transition: 'all 0.15s',
              }}
              onMouseEnter={e => { e.currentTarget.style.color = t.text; e.currentTarget.style.borderColor = t.blue; }}
              onMouseLeave={e => { e.currentTarget.style.color = t.muted; e.currentTarget.style.borderColor = t.border; }}
            >
              Browse all {ALL.length} matches →
            </button>
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
                  {selected.tournament} · {selected.strokes} annotated strokes
                </div>
              </div>
              <Btn onClick={() => onNext(selected)}>Begin Markup →</Btn>
            </div>
          )}

          {browsing && (
            <BrowseAllModal
              items={ALL}
              onSelect={v => { setSelected(v); setBrowsing(false); }}
              onClose={() => setBrowsing(false)}
            />
          )}
        </>
      ) : (
        <UploadTab onUpload={v => onNext(v)} />
      )}
    </div>
  );
}
