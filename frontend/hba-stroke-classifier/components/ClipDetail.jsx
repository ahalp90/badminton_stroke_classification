import { useState, useEffect } from 'react';
import { useTheme } from '../shared';

export function ClipDetail({ detail }) {
    const { t } = useTheme();
    const [videoError, setVideoError] = useState(false);
    
    // Reset the video error gate when the clip changes; otherwise a missing
    // mock clip would poison subsequent picks that might actually be on disk.
    useEffect(() => { setVideoError(false); }, [detail.clip_stem]);
    
    // Bar widths scale to the strongest class, so the runner-up reads as
    // "almost as confident" when the model is genuinely torn.
    const maxConf = Math.max(...detail.top_k.map(k => k.confidence));
    
    return (
    <div>
      <div style={{
        marginBottom: 14, background: '#000', borderRadius: 6, overflow: 'hidden',
        aspectRatio: '16 / 9', display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {videoError ? (
            <div style={{ color: t.muted, fontSize: 12, padding: 16, textAlign: 'center', lineHeight: 1.5 }}>
                Video available for this clip on the current host.<br/>
                The backend serves clips from <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>BST_CLIPS_DIR</span>;
                only clips whose mp4 is present there will play.
            </div>
            ) : (
            <video
              src={detail.video_url}
              controls
              preload="metadata"
              onError={() => setVideoError(true)}
              style={{ width: '100%', height: '100%', display: 'block' }}
            />
          )
        }
      </div>

      <div style={{ fontSize: 11, color: t.muted, marginBottom: 6, fontFamily: "'JetBrains Mono',monospace" }}>
        {detail.match} · {detail.set_id} · rally {detail.rally} · ball round {detail.ball_round}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 6 }}>
        <span style={{ fontSize: 24, fontWeight: 700, color: t.text }}>{detail.predicted_class}</span>
        <span style={{ fontSize: 13, color: t.muted }}>predicted</span>
        <span style={{
          fontSize: 24, fontWeight: 700, color: t.blue,
          fontFamily: "'JetBrains Mono',monospace", marginLeft: 'auto',
        }}>
          {detail.confidence_pct}%
        </span>
      </div>

      <div style={{ fontSize: 13, color: t.muted, marginBottom: 16 }}>
        true: <span style={{ color: t.text, fontFamily: "'JetBrains Mono',monospace" }}>{detail.true_class}</span>
        {' · '}
        <span style={{ color: detail.is_correct ? t.success : t.danger, fontWeight: 600 }}>
          {detail.is_correct ? '✓ correct' : '✗ wrong'}
        </span>
      </div>

      <div style={{ fontSize: 11, color: t.muted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        Top-{detail.top_k.length}
      </div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {detail.top_k.map((entry, i) => (
          <div key={entry.class} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 170, fontSize: 12, color: t.text,
              fontFamily: "'JetBrains Mono',monospace", textAlign: 'right', flexShrink: 0,
            }}>
              {entry.class}
            </div>
            <div style={{ flex: 1, height: 16, background: t.surface2, borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${(entry.confidence / maxConf) * 100}%`,
                background: i === 0 ? t.blue : t.pine,
                borderRadius: 3,
              }} />
            </div>
            <div style={{
              width: 56, fontSize: 12, fontWeight: 600,
              fontFamily: "'JetBrains Mono',monospace", color: t.text, textAlign: 'right',
            }}>
              {(entry.confidence * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>

    </div>
  );
}