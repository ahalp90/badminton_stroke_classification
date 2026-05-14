import { useState, useEffect } from 'react';
import { useTheme, Btn, Card } from './shared';

// Real held-out test-set numbers for the picked BST_CG_AP serial.
// Source: src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/
//         run_20260505_154907/manifest.yaml (best_serials: [5]).
// Do not edit these by hand — re-export from the manifest if the run changes.
const TEST_EVAL = {
  runId: 'run_20260505_154907',
  serial: 5,
  variant: 'BST_CG_AP',
  taxonomy: 'une_merge_v1_nosides',
  numClasses: 14,
  numStrokes: 4202,
  macroF1: 0.7479,
  minF1: 0.5147,
  accuracy: 0.7675,
  top2Accuracy: 0.9407,
  perClassF1: [
    ['short_service',         0.9801],
    ['long_service',          0.9517],
    ['clear',                 0.9465],
    ['net_shot',              0.8924],
    ['return_net',            0.8184],
    ['lob',                   0.7846],
    ['rush',                  0.7742],
    ['drop',                  0.6821],
    ['passive_drop',          0.6765],
    ['drive',                 0.6628],
    ['push',                  0.6546],
    ['cross_court_net_shot',  0.6130],
    ['wrist_smash',           0.5186],
    ['smash',                 0.5147],
  ],
};

/* ─── Per-clip browser (Tier 1, from /api/registry sidecar JSONs) ── */
// Both val and test have mock predictions via build_mock_artifacts.py, so
// the split toggle below works against either. For real data: only `test`
// is emitted by the current eval_dump_predictions.py and only test_metrics
// land in manifest.yaml. Train + val headline metrics could probably be
// reconstructed from the per-epoch TensorBoard scalars (final val_macro_f1
// etc.) rather than re-running eval, but that's a follow-up.
const SPLITS = ['val', 'test'];

function Tier1ClipBrowser({ modelId, initialSplit = 'test' }) {
  const { t } = useTheme();
  const [resolvedId,   setResolvedId]   = useState(modelId ?? null);
  const [split,        setSplit]        = useState(initialSplit);
  const [clips,        setClips]        = useState([]);
  const [listError,    setListError]    = useState(null);
  const [selectedStem, setSelectedStem] = useState(null);
  const [detail,       setDetail]       = useState(null);
  const [detailError,  setDetailError]  = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [errorsOnly,   setErrorsOnly]   = useState(false);

  // Fallback: if no model was picked upstream (e.g. dev-jump straight to
  // Results), grab the first registered model so the browser still works.
  useEffect(() => {
    if (modelId) { setResolvedId(modelId); return; }
    fetch('/api/registry')
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
      .then(data => setResolvedId(data.models?.[0]?.id ?? null))
      .catch(err => setListError(err.message));
  }, [modelId]);

  // Pull the clip list whenever model / split / filter changes.
  useEffect(() => {
    if (!resolvedId) return;
    setListError(null);
    const params = errorsOnly ? '?errors_only=true' : '';
    fetch(`/api/registry/${resolvedId}/splits/${split}/clips${params}`)
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
      .then(data => {
        const items = data.clips || [];
        setClips(items);
        setSelectedStem(items[0]?.clip_stem ?? null);
      })
      .catch(err => setListError(err.message));
  }, [resolvedId, split, errorsOnly]);

  // Pull the selected clip's per-clip JSON.
  useEffect(() => {
    if (!resolvedId || !selectedStem) { setDetail(null); return; }
    setDetailLoading(true);
    setDetailError(null);
    fetch(`/api/registry/${resolvedId}/splits/${split}/clips/${selectedStem}`)
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
      .then(d => { setDetail(d); setDetailLoading(false); })
      .catch(err => { setDetailError(err.message); setDetailLoading(false); });
  }, [resolvedId, split, selectedStem]);

  if (!resolvedId && !listError) {
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
                  onClick={() => setSplit(s)}
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

function ClipDetail({ detail }) {
  const { t } = useTheme();
  // Bar widths scale to the strongest class, so the runner-up reads as
  // "almost as confident" when the model is genuinely torn.
  const maxConf = Math.max(...detail.top_k.map(k => k.confidence));
  return (
    <div>
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

      <div style={{ marginTop: 14, fontSize: 10, color: t.muted, fontFamily: "'JetBrains Mono',monospace" }}>
        video: {detail.video_url} (Tier 1: clip mp4s live on /scratch, endpoint stubbed for now)
      </div>
    </div>
  );
}

/* ─── Held-out test set evaluation (real numbers from manifest.yaml) ─ */
function TestEvalCard() {
  const { t } = useTheme();
  const pct = (x) => (x * 100).toFixed(1) + '%';
  const stats = [
    { label: 'Macro F1',       value: TEST_EVAL.macroF1.toFixed(3),    color: t.blue },
    { label: 'Accuracy',       value: pct(TEST_EVAL.accuracy),         color: t.text },
    { label: 'Top-2 accuracy', value: pct(TEST_EVAL.top2Accuracy),     color: t.text },
    { label: 'Min F1',         value: TEST_EVAL.minF1.toFixed(3),      color: t.text },
  ];
  const Mono = ({ children }) => (
    <span style={{ fontFamily: "'JetBrains Mono',monospace", color: t.text }}>{children}</span>
  );
  return (
    <Card style={{ padding: 22, marginBottom: 22 }}>
      <div style={{ fontSize: 11, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        Held-out test set evaluation
      </div>
      <div style={{ fontSize: 12, color: t.muted, marginBottom: 16, lineHeight: 1.6 }}>
        Run <Mono>{TEST_EVAL.runId}</Mono> · serial {TEST_EVAL.serial} · variant <Mono>{TEST_EVAL.variant}</Mono>
        {' '}· taxonomy <Mono>{TEST_EVAL.taxonomy}</Mono>
        {' '}· {TEST_EVAL.numClasses} classes · {TEST_EVAL.numStrokes.toLocaleString()} test strokes.
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
        {stats.map((s) => (
          <div key={s.label} style={{ background: t.surface2, borderRadius: 7, padding: 14 }}>
            <div style={{ fontSize: 10, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{s.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono',monospace" }}>{s.value}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ─── Per-class F1 on test set (real numbers from manifest.yaml) ──── */
function PerClassF1Card() {
  const { t } = useTheme();
  const max = Math.max(...TEST_EVAL.perClassF1.map(([, v]) => v));
  return (
    <Card style={{ padding: 22 }}>
      <div style={{ fontSize: 11, color: t.muted, marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        Per-class F1 — test set
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {TEST_EVAL.perClassF1.map(([cls, f1]) => (
          <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 170, fontSize: 12, color: t.text,
              fontFamily: "'JetBrains Mono',monospace", textAlign: 'right', flexShrink: 0,
            }}>
              {cls}
            </div>
            <div style={{ flex: 1, height: 16, background: t.surface2, borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                height: '100%', width: `${(f1 / max) * 100}%`,
                background: t.blue, borderRadius: 3,
              }} />
            </div>
            <div style={{
              width: 48, fontSize: 12, fontWeight: 600,
              fontFamily: "'JetBrains Mono',monospace", color: t.text, textAlign: 'right',
            }}>
              {f1.toFixed(2)}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ─── Results Shell ──────────────────────────────────────────────── */
export function ResultsScreen({ task, onNew }) {
  const { t } = useTheme();

  // Active model = first enabled in the task's models list. Tier 1 only
  // has one registered model so this collapses to "the one model". When
  // upstream task is empty (dev-jump), Tier1ClipBrowser falls back to the
  // first model in /api/registry.
  const activeModel = task?.models?.find(m => task?.enabled?.[m.id]);

  return (
    <div style={{ maxWidth: 1120, margin: '0 auto', padding: 32 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: t.text, marginBottom: 4 }}>Classification Results</h1>
          <p style={{ fontSize: 13, color: t.muted }}>
            {task?.taskName ?? 'Analysis'} · Completed {new Date().toLocaleString('en-AU')}
          </p>
        </div>
        <Btn variant="secondary" size="sm" onClick={onNew}>New Analysis</Btn>
      </div>

      <Tier1ClipBrowser modelId={activeModel?.id} />
      <TestEvalCard />
      <PerClassF1Card />
    </div>
  );
}

/* ============================================================================
 * BLOCKED — pending predictions.csv from the engelbart inference run.
 * ============================================================================
 *
 * The four tab components below were built against a fabricated `classify()`
 * helper that hashed stroke timestamps into invented predictions. They were
 * commented out (rather than deleted) so the ML team has a reference for the
 * UI shape when the real per-stroke predictions land.
 *
 *   - StrokesTab        — per-stroke results table (errors-only filter, window scope)
 *   - DistributionTab   — shot-class histogram (GT vs Predicted)
 *   - ComparisonTab     — per-class accuracy on the selected match
 *   - ExplainabilityTab — top-k probabilities + CAM placeholder
 *
 * To re-enable:
 *   1. Drop predictions.csv + taxonomy.json into ./data/ per
 *      scratch/real-predictions-data-format.md (Strategy A).
 *   2. Add a CSV loader that yields a Map<youtubeId, stroke[]> where each
 *      stroke is { id, time, gt, pred, conf, correct, split, rally, ball_round }.
 *   3. Rebuild a CLASSES palette from taxonomy.json (14 entries) instead of
 *      the 7-class palette this file used to ship with (which did not match
 *      the trained model's taxonomy).
 *   4. Uncomment the components below and wire them to the loader; consider
 *      visually distinguishing strokes whose `split == 'train'` so the model's
 *      predictions on its own training data are not mistaken for evaluation
 *      evidence.
 *
 * Removed permanently (do NOT bring back without a real apples-to-apples
 * comparison — different taxonomy and split):
 *   - "BST Baseline 80–85% · Chang 2025 (reference)" card. Chang 2025 reports
 *     BST_CG_AP at 0.8254 accuracy / 0.7983 macro-F1 on ShuttleSet 25-class
 *     with their split; our run uses 14 classes (une_merge_v1_nosides) and
 *     split_v2, so direct comparison overstates the relationship. If the team
 *     wants a published-baseline comparison, re-run our model on
 *     split_bst_baseline with the 25-class taxonomy first.
 *
 * Removed permanently (was wrong):
 *   - The "3D-CNN" badge in the explainability tab — the actual backbone is
 *     TCN → Transformer (see scratch/frontend-model-label-fixes.md).
 *   - The fake class-activation-map rendering (two CSS radial-gradient blobs).
 *     A real CAM tab requires CAM artefacts from the inference run, not just
 *     predictions.csv.
 *
 * ----------------------------------------------------------------------------
 * import { Badge } from './shared';
 *
 * const frameModules = import.meta.glob('./data/frames/*.jpg', { eager: true, import: 'default' });
 * const frameUrl = (id) => frameModules[`./data/frames/${id}.jpg`];
 *
 * // 14-class taxonomy palette — rebuild from taxonomy.json when predictions land.
 * const CLASSES = [
 *   // { label: '<class_name>', color: '<hex>' }, ...
 * ];
 *
 * function ConfBar({ value, correct }) {
 *   const pct = (value * 100).toFixed(0);
 *   return (
 *     <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 90 }}>
 *       <div style={{ flex: 1, height: 5, background: 'rgba(127,127,127,0.15)', borderRadius: 3 }}>
 *         <div style={{ height: '100%', borderRadius: 3, width: `${pct}%`, background: correct ? '#22C55E' : '#EF4444' }} />
 *       </div>
 *       <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace", color: correct ? '#22C55E' : '#EF4444', minWidth: 30 }}>
 *         {pct}%
 *       </span>
 *     </div>
 *   );
 * }
 *
 * function HBar({ value, max, color }) {
 *   return (
 *     <div style={{ flex: 1, height: 22, background: 'rgba(127,127,127,0.1)', borderRadius: 3, overflow: 'hidden' }}>
 *       <div style={{
 *         height: '100%', borderRadius: 3,
 *         width: `${(value / max) * 100}%`,
 *         background: color,
 *         display: 'flex', alignItems: 'center', paddingLeft: 8,
 *         transition: 'width 0.5s ease',
 *       }}>
 *         <span style={{ fontSize: 11, fontWeight: 600, color: '#fff', fontFamily: "'JetBrains Mono',monospace" }}>{value}</span>
 *       </div>
 *     </div>
 *   );
 * }
 *
 * function FocalStrokeCard({ focal, video, timeframe }) { ... see git history ... }
 * function StrokesTab({ classifications, focal, timeframe }) { ... }
 * function DistributionTab({ classifications, focal }) { ... }
 * function ComparisonTab({ classifications, focal }) { ... }  // omit the Chang 2025 card
 * function ExplainabilityTab({ classifications, focal }) { ... }  // omit fake CAM + 3D-CNN badge
 *
 * // Old per-stroke focal logic inside ResultsScreen:
 * // const focal = useMemo(() => {
 * //   if (!classifications.length || !timeframe?.targetSec) return null;
 * //   let best = classifications[0], bestDist = Infinity;
 * //   for (const s of classifications) {
 * //     const d = Math.abs(s.time - timeframe.targetSec);
 * //     if (d < bestDist) { best = s; bestDist = d; }
 * //   }
 * //   return best;
 * // }, [classifications, timeframe?.targetSec]);
 *
 * (Full original implementations are preserved in git history on branch
 *  feat/alt-frontend at 20a9efe — `git show 20a9efe:frontend/hba-stroke-classifier/results-screen.jsx`.)
 *
 * ============================================================================ */

