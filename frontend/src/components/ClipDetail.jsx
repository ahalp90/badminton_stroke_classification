import { useState, useEffect } from 'react';
import { useTheme } from '../shared';

const MONO = "'JetBrains Mono',monospace";

export function ClipDetail({ detail }) {
    const { t } = useTheme();
    const [videoError, setVideoError] = useState(false);

    // Reset the video error gate when the clip changes; otherwise a missing
    // clip would poison subsequent picks that might actually be on disk.
    useEffect(() => { setVideoError(false); }, [detail.clip_stem]);

    // A prediction is "real" unless it's the old placeholder data (y_pred ==
    // y_true at 100%). The backend signals provenance via drawn_from:
    //   'live_forward_pass'          — real, computed live on the deploy box
    //   'cached_predictions_json'    — real, precomputed offline over the split
    //   'placeholder_predictions_json' — mock; show the pending state instead
    // The registered models now ship real precomputed predictions, so the
    // cached path is real; only a flagged mock sidecar reads as pending.
    const predictionReal = detail.drawn_from === 'live_forward_pass'
                        || detail.drawn_from === 'cached_predictions_json';
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
                {' '}— or set <span style={{ fontFamily: MONO }}>BST_X_CLIPS_DIR</span> on a host that has the dataset.
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

      {/* Actual vs predicted — a lean label→value comparison (no heavy boxes;
          the Top-5 below already carries the predicted class + confidence).
          The prediction row renders only for a real result (live forward pass
          or real precomputed preds); placeholder data falls to the pending
          state below so it never reads as a real "100% ✓" result. The grid
          keeps labels aligned and lets the value column wrap on narrow panels. */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'auto minmax(0, 1fr)',
        columnGap: 12, rowGap: 6, alignItems: 'baseline', marginBottom: 16,
      }}>
        <span style={{ fontSize: 10, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
          Actual
        </span>
        <span style={{ fontSize: 17, fontWeight: 700, color: t.text, fontFamily: MONO, overflowWrap: 'anywhere', lineHeight: 1.25 }}>
          {detail.true_class}
        </span>

        {predictionReal && (
          <>
            <span style={{ fontSize: 10, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
              Predicted
            </span>
            <span style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 17, fontWeight: 700, color: t.text, fontFamily: MONO, overflowWrap: 'anywhere', lineHeight: 1.25 }}>
                {detail.predicted_class}
              </span>
              <span style={{ fontSize: 13, fontWeight: 700, color: t.blue, fontFamily: MONO }}>
                {detail.confidence_pct}%
              </span>
              <span style={{ fontSize: 11, fontWeight: 600, color: detail.is_correct ? t.success : t.danger }}>
                {detail.is_correct ? '✓ matches' : '✗ differs'}
              </span>
            </span>
          </>
        )}
      </div>

      {predictionReal ? (
        <>
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
                    background: i === 0 
                      ? (detail.is_correct ? t.success : t.danger)
                      : t.pine,
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
