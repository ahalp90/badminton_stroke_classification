import { useState, useEffect } from 'react';
import { useTheme } from '../shared';

const MONO = "'JetBrains Mono',monospace";

export function ClipDetail({ detail }) {
    const { t } = useTheme();
    const [videoError, setVideoError] = useState(false);

    // Reset the video error gate when the clip changes; otherwise a missing
    // clip would poison subsequent picks that might actually be on disk.
    useEffect(() => { setVideoError(false); }, [detail.clip_stem]);

    // A prediction is only "real" when the backend ran a live forward pass.
    // The committed predictions JSON is placeholder data (y_pred == y_true at
    // 100%), surfaced by the API as drawn_from === 'cached_predictions_json';
    // a real BST run sets 'live_forward_pass' (see src/api/bst_inference.py).
    // So when real per-clip inference flows (live BST on the deploy box) this
    // flips to true automatically — no change needed here. If real predictions
    // are ever shipped via the cached JSON instead, relax this single check.
    const predictionReal = detail.drawn_from === 'live_forward_pass';
    const topK = detail.top_k ?? [];
    const maxConf = topK.length ? Math.max(...topK.map(k => k.confidence)) : 1;

    return (
    <div>
      <div style={{
        marginBottom: 14, background: '#000', borderRadius: 6, overflow: 'hidden',
        aspectRatio: '16 / 9', display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {videoError ? (
            <div style={{ color: t.muted, fontSize: 12, padding: 16, textAlign: 'center', lineHeight: 1.6 }}>
                Clip not available locally.<br/>
                Drop <span style={{ fontFamily: MONO, color: t.text }}>{detail.clip_stem}.mp4</span> into{' '}
                <span style={{ fontFamily: MONO, color: t.text }}>clips_local/</span> to play it here
                {' '}— or set <span style={{ fontFamily: MONO }}>BST_CLIPS_DIR</span> on a host that has the dataset.
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

      <div style={{ fontSize: 11, color: t.muted, marginBottom: 12, fontFamily: MONO, wordBreak: 'break-all', overflow: 'hidden' }}>
        {detail.match} · {detail.set_id} · rally {detail.rally} · ball round {detail.ball_round}
      </div>

      {/* Ground truth — the real dataset label for this clip; always shown. */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 14 }}>
        <span style={{ fontSize: 11, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
          Actual stroke
        </span>
        <span style={{ fontSize: 22, fontWeight: 700, color: t.text }}>{detail.true_class}</span>
        <span style={{ fontSize: 11, color: t.muted }}>ground-truth label</span>
      </div>

      {/* Model prediction — only rendered as a real result when it came from a
          live forward pass; otherwise an explicit pending/placeholder state so
          placeholder data never reads as a real "100% ✓ correct" result. */}
      {predictionReal ? (
        <>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 6 }}>
            <span style={{ fontSize: 11, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
              Model predicted
            </span>
            <span style={{ fontSize: 22, fontWeight: 700, color: t.text }}>{detail.predicted_class}</span>
            <span style={{
              fontSize: 22, fontWeight: 700, color: t.blue,
              fontFamily: MONO, marginLeft: 'auto',
            }}>
              {detail.confidence_pct}%
            </span>
          </div>

          <div style={{ fontSize: 13, color: t.muted, marginBottom: 16 }}>
            <span style={{ color: detail.is_correct ? t.success : t.danger, fontWeight: 600 }}>
              {detail.is_correct ? '✓ matches actual' : '✗ differs from actual'}
            </span>
          </div>

          <div style={{ fontSize: 11, color: t.muted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
            Top-{topK.length}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {topK.map((entry, i) => (
              <div key={entry.class} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{
                  width: 170, fontSize: 12, color: t.text,
                  fontFamily: MONO, textAlign: 'right', flexShrink: 0,
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
                  fontFamily: MONO, color: t.text, textAlign: 'right',
                }}>
                  {(entry.confidence * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div style={{
          border: `1px dashed ${t.warning}66`,
          background: t.warning + '14',
          borderRadius: 8, padding: '12px 14px',
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: t.warning, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 5 }}>
            Model prediction — pending
          </div>
          <div style={{ fontSize: 12, color: t.muted, lineHeight: 1.6 }}>
            Not a real model output yet. The committed per-clip predictions are
            placeholder data (they mirror the ground-truth label at 100%). Real
            per-clip inference runs as a live BST forward pass on the deploy box;
            the predicted class and confidence will appear here automatically once
            that data is served.
          </div>
        </div>
      )}
    </div>
  );
}
