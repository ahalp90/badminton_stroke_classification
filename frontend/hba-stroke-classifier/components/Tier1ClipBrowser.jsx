import { useState, useEffect } from 'react';
import { useTheme, Card } from '../shared';
import { useClipList, useClipDetail } from '../hooks';
import { ClipDetail } from './ClipDetail';

const SPLITS = ['val', 'test'];

/* ─── Per-clip browser (Tier 1, from /api/registry sidecar JSONs) ── */
// Both val and test have mock predictions via build_mock_artifacts.py, so
// the split toggle below works against either. For real data: only `test`
// is emitted by the current eval_dump_predictions.py and only test_metrics
 // land in manifest.yaml. Train + val headline metrics could probably be
 // reconstructed from the per-epoch TensorBoard scalars (final val_macro_f1
 // etc.) rather than re-running eval, but that's a follow-up.

export function Tier1ClipBrowser({ modelId, split, onSplitChange }) {
  const { t } = useTheme();
  const [selectedStem, setSelectedStem] = useState(null);
  const [errorsOnly,   setErrorsOnly]   = useState(false);
  const { clips, total, offset, setOffset, limit, error: listError} = useClipList({ modelId, split, errorsOnly });
  const { detail, loading: detailLoading, error: detailError }      = useClipDetail({ modelId, split, selectedStem });

  useEffect(() => {
    setSelectedStem(clips[0]?.clip_stem ?? null);
  }, [clips]);

  if (!modelId && !listError) {
    return (
      <Card style={{ padding: 22, marginBottom: 22 }}>
        <div style={{ fontSize: 12, color: t.muted }}>Loading registry…</div>
      </Card>
    );
  }

  return (
    <Card style={{ padding: 22, marginBottom: 22 }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 14, gap: 12, flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 11, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
            Per-clip predictions
          </span>
          <div style={{ display: 'inline-flex', gap: 4 }}>
            {SPLITS.map(s => {
              const active = s === split;
              return (
                <button
                  key={s}
                  onClick={() => onSplitChange(s)}
                  style={{
                    background: active ? t.blue : t.surface2,
                    color: active ? '#fff' : t.muted,
                    border: `1px solid ${active ? t.blue : t.border}`,
                    padding: '3px 10px', borderRadius: 4,
                    fontSize: 11, fontWeight: 600,
                    fontFamily: "'JetBrains Mono', monospace",
                    cursor: 'pointer',
                  }}
                >
                  {s}
                </button>
              );
            })}
          </div>
        </div>
        <label style={{ fontSize: 11, color: t.muted, display: 'flex', gap: 6, alignItems: 'center', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={errorsOnly}
            onChange={e => setErrorsOnly(e.target.checked)}
            style={{ accentColor: t.blue }}
          />
          Errors only
        </label>
      </div>

      {listError && (
        <div style={{ fontSize: 12, color: t.danger, padding: '8px 12px', background: t.dangerDim, borderRadius: 6 }}>
          Couldn't load clips: {listError}
        </div>
      )}

      {!listError && (
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
          <div>
            {detailLoading && <div style={{ fontSize: 13, color: t.muted }}>Loading clip…</div>}
            {detailError && <div style={{ fontSize: 13, color: t.danger }}>Couldn't load clip: {detailError}</div>}
            {!detailLoading && !detailError && detail && <ClipDetail detail={detail} />}
            {!detailLoading && !detailError && !detail && (
              <div style={{ fontSize: 13, color: t.muted }}>Pick a clip on the left.</div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}