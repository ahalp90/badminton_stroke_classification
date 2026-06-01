import { useState, useEffect } from 'react';
import { useTheme } from '../shared';
import { useClipList, useClipDetail } from '../hooks';
import { ClipDetail } from './ClipDetail';
import { SPLIT_LABELS } from '../utils/format';

const SPLITS = ['val', 'test'];

/* ─── Per-clip browser (Tier 1, from /api/registry) ── */
// Renders only when live per-clip predictions are available for the selected
// split (`livePredictions[split]`); the /clips list endpoint then serves real
// forward-pass predictions. When live inference isn't available the browser is
// replaced by an honest note — we never show placeholder predictions.

export function Tier1ClipBrowser({ modelId, split, onSplitChange, livePredictions }) {
  const { t } = useTheme();
  const live = !!livePredictions?.[split];
  const [selectedStem, setSelectedStem] = useState(null);
  const [errorsOnly,   setErrorsOnly]   = useState(false);
  const { clips, total, offset, setOffset, limit, error: listError } = useClipList({ modelId, split, errorsOnly, enabled: live });
  const { detail, loading: detailLoading, error: detailError }      = useClipDetail({ modelId, split, selectedStem });

  useEffect(() => {
    setSelectedStem(clips[0]?.clip_stem ?? null);
  }, [clips]);

  if (!modelId && !listError) {
    return (
      <div>
        <div style={{ fontSize: 12, color: t.muted }}>Loading registry…</div>
      </div>
    );
  }

  return (
    <div>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 14, gap: 12, flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ display: 'inline-flex', border: `1px solid ${t.border}`, borderRadius: 5, overflow: 'hidden' }}>
            {SPLITS.map(s => {
              const active = s === split;
              return (
                <button
                  key={s}
                  onClick={() => onSplitChange(s)}
                  style={{
                    minWidth: 92, textAlign: 'center',
                    background: active ? t.blue : t.surface2,
                    color: active ? '#fff' : t.muted,
                    border: 'none',
                    padding: '4px 12px',
                    fontSize: 11, fontWeight: 600,
                    fontFamily: "'Space Grotesk', sans-serif",
                    cursor: 'pointer',
                  }}
                >
                  {SPLIT_LABELS[s]}
                </button>
              );
            })}
          </div>
        </div>
        {/* The "Errors only" filter only acts on the live clip list, so hide it
            when live per-clip predictions aren't available — otherwise it sits
            above the "not available" note and silently does nothing. */}
        {live && (
          <label style={{ fontSize: 11, color: t.muted, display: 'flex', gap: 6, alignItems: 'center', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={errorsOnly}
              onChange={e => setErrorsOnly(e.target.checked)}
              style={{ accentColor: t.blue }}
            />
            Errors only
          </label>
        )}
      </div>

      <div style={{ fontSize: 11, color: t.muted, lineHeight: 1.5, marginBottom: 14 }}>
        ⓘ <strong style={{ color: t.text }}>Test</strong> = unseen players — the real-world score.{' '}
        <strong style={{ color: t.text }}>Validation</strong> is the tuning split; it shares players with
        training, so it reads a little higher.
      </div>

      {listError && (
        <div style={{ fontSize: 12, color: t.danger, padding: '8px 12px', background: t.dangerDim, borderRadius: 6 }}>
          Couldn't load clips: {listError}
        </div>
      )}

      {!listError && live && (
        <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 18 }}>
          {/* LEFT: list + pagination */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8}}>
            <div style={{ border: `1px solid ${t.border}`, borderRadius: 6, maxHeight: 480, overflowY: 'auto' }}>
              {clips.length === 0 && (
                <div style={{ padding: 12, fontSize: 12, color: t.muted }}>
                  {errorsOnly ? 'No mispredicted clips in this split.' : 'Loading clips…'}
                </div>
              )}
              {clips.map(c => {
                const sel = c.clip_stem === selectedStem;
                return (
                  <div
                  key={c.clip_stem}
                  onClick={() => setSelectedStem(c.clip_stem)}
                  style={{
                    padding: '8px 12px',
                    background: sel ? t.blueDim : 'transparent',
                    borderLeft: sel ? `3px solid ${t.blue}` : '3px solid transparent',
                    borderBottom: `1px solid ${t.border}`,
                    cursor: 'pointer',
                    fontSize: 12,
                  }}
                  >
                    <div style={{ fontFamily: "'JetBrains Mono',monospace", color: t.text, marginBottom: 3 }}>
                      {c.clip_stem}
                    </div>
                    <div style={{ color: t.muted, fontSize: 10 }}>
                      {c.true_class} → {c.predicted_class}
                      {' '}
                      <span style={{ color: c.is_correct ? t.success : t.danger, fontWeight: 600 }}>
                        {c.is_correct ? '✓' : '✗'} {c.confidence_pct}%
                      </span>
                    </div>
                  </div>
                  );
                  })}
            </div>
            {/* Pagination sits below the list, inside the left column */}
            <div style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              marginTop: 8, fontSize: 11, color: t.muted,
              }}>
              <button
              onClick={() => setOffset(o => Math.max(0, o - limit))}
              disabled={offset === 0}
              style={{
                background: 'none', border: `1px solid ${t.border}`, borderRadius: 4,
                padding: '3px 10px', fontSize: 11, color: t.text,
                cursor: offset === 0 ? 'not-allowed' : 'pointer',
                opacity: offset === 0 ? 0.4 : 1, 
              }}
              >
                ← Prev
              </button>
              <span>
                {total === 0 ? '-' : `${offset + 1}-${Math.min(offset + limit, total)} of ${total}`}
              </span>
              <button
              onClick={() => setOffset(o => o + limit)}
              disabled={offset + limit >= total}
              style={{
                background: 'none', border: `1px solid ${t.border}`, borderRadius: 4,
                padding: '3px 10px', fontSize: 11, color: t.text,
                cursor: offset + limit >= total ? 'not-allowed' : 'pointer',
                opacity: offset + limit >= total ? 0.4 : 1, 
              }}
              >
                Next →
              </button>
            </div>
          </div>

          {/* RIGHT: clip detail */}
          <div style={{ minWidth: 0 }}>
            {detailLoading && <div style={{ fontSize: 13, color: t.muted }}>Loading clip…</div>}
            {detailError && <div style={{ fontSize: 13, color: t.danger }}>Couldn't load clip: {detailError}</div>}
            {!detailLoading && !detailError && detail && <ClipDetail detail={detail} />}
            {!detailLoading && !detailError && !detail && (
              <div style={{ fontSize: 13, color: t.muted }}>Pick a clip on the left.</div>
            )}
          </div>
        </div>
      )}
      {!listError && !live && (
        <div style={{
          fontSize: 13, color: t.muted, lineHeight: 1.6,
          padding: '12px 14px', background: t.surface2, borderRadius: 6,
        }}>
          Live per-clip predictions for the <strong>{split}</strong> set run on
          the analysis server and aren&apos;t available in this environment.
          The metrics above are computed over the full {split} set.
        </div>
      )}
    </div>
  );
}