import { useState, useEffect, Fragment } from 'react';
import { useTheme, Btn, Card } from './shared';
import { Tier1ClipBrowser } from './components/Tier1ClipBrowser';

/* ─── Model-level reference: aggregate metrics (from /api/registry) ─ */
// Renamed and re-framed (Fix 3): these numbers describe the *model*,
// not the analysis the user just ran. The card lives below the
// per-clip browser and the uploaded-inference card so "your result"
// reads first.
function TestEvalCard({ model, split }) {
  const { t } = useTheme();
  if (!model) return null;
  const Mono = ({ children }) => (
    <span style={{ fontFamily: "'JetBrains Mono',monospace", color: t.text }}>{children}</span>
  );
  const heading = `Model performance — ${split} set`;
  const subtitle = `How ${model.display_name} performed across the full ${split} set. Same numbers for every analysis — they describe the model, not this video.`;
  const m = model[`${split}_metrics`];
  const hasMetrics = m && typeof m.macro_f1 === 'number';
  const pct = (x) => (x * 100).toFixed(1) + '%';
  return (
    <Card style={{ padding: 22, marginBottom: 22 }}>
      <div style={{ fontSize: 11, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        {heading}
      </div>
      <div style={{ fontSize: 12, color: t.muted, marginBottom: 4, lineHeight: 1.6 }}>
        {subtitle}
      </div>
      <div style={{ fontSize: 11, color: t.muted, marginBottom: 16, fontFamily: "'JetBrains Mono',monospace" }}>
        {model.num_classes} classes · taxonomy <Mono>{model.taxonomy}</Mono>
      </div>
      {hasMetrics ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
          {[
            { label: 'Macro F1',       value: m.macro_f1.toFixed(3),    color: t.blue },
            { label: 'Accuracy',       value: pct(m.accuracy),          color: t.text },
            { label: 'Top-2 accuracy', value: pct(m.top2_accuracy),     color: t.text },
            { label: 'Min F1',         value: m.min_f1.toFixed(3),      color: t.text },
          ].map((s) => (
            <div key={s.label} style={{ background: t.surface2, borderRadius: 7, padding: 14 }}>
              <div style={{ fontSize: 10, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{s.label}</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono',monospace" }}>{s.value}</div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ fontSize: 13, color: t.muted, padding: '8px 0' }}>
          No {split} metrics available for this model.
        </div>
      )}
    </Card>
  );
}

/* ─── Model-level reference: per-class F1 (from /api/registry) ─── */
// Same framing as TestEvalCard: describes the model's strengths/weaknesses
// across the dataset, NOT a per-stroke output of this analysis (Fix 3).
function PerClassF1Card({ model, split }) {
  const { t } = useTheme();
  if (!model) return null;
  const heading = `Model per-class F1 — ${split} set`;
  const subtitle = `Per-class strength of the model. Lower bars indicate classes it confuses more often.`;
  const perClass = model[`${split}_metrics`]?.per_class_f1;
  const hasData = perClass && Object.keys(perClass).length > 0;
  // Sort desc so the strongest classes read top-down; bar widths scale to
  // the max so small differences between weak classes are still visible.
  const entries = hasData
    ? Object.entries(perClass).sort(([,a],[,b]) => b - a)
    : [];
  const max = hasData ? Math.max(...entries.map(([,v]) => v)): 1;
  return (
    <Card style={{ padding: 22 }}>
      <div style={{ fontSize: 11, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        {heading}
      </div>
      <div style={{ fontSize: 12, color: t.muted, marginBottom: 14, lineHeight: 1.6 }}>
        {subtitle}
      </div>
      {hasData ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {entries.map(([cls, f1]) => (
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
      ) : (
        <div style={{ fontSize: 13, color: t.muted, padding: '8px 0' }}>
          No {split} per-class F1 available for this model.
        </div>
      )}
      </Card>
  );
}

/* ─── Inference echo card (Items 4 + Gaps 1/2: real /api/results payload) ── */
function UploadedInferenceCard({ task }) {
  const { t } = useTheme();
  const result = task?.uploadResult;
  if (!result) return null;
  const strokes = result.strokes || [];
  const rally = result.rally_summary || {};
  const v = task?.markup?.video;
  const echo = result.markup_echo;
  const isLibrary = result.source === 'library';
  const heading = isLibrary ? 'Inference on library clip' : 'Inference on your upload';
  const subjectLabel = isLibrary
    ? (result.clip_stem ? `clip ${result.clip_stem}` : 'library clip')
    : v?.filename;
  return (
    <Card style={{ padding: 22, marginBottom: 22, borderColor: t.success + '55' }}>
      <div style={{ fontSize: 11, color: t.success, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        {heading}
      </div>
      <div style={{ fontSize: 12, color: t.muted, marginBottom: 16, lineHeight: 1.6 }}>
        Live response from the backend's <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>/api/results</span> endpoint
        {subjectLabel && <> · <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>{subjectLabel}</span></>}
        <span style={{ color: t.warning, marginLeft: 8 }}>
          (stubbed inference — real BST/MMPose/TrackNet pipeline pending)
        </span>
      </div>

      {echo && (() => {
        const echoAnnos = echo.annotations || [];
        const strokeCount = strokes.length;
        const mismatch = echoAnnos.length > 0 && strokeCount < echoAnnos.length;
        return (
          <div style={{
            background: t.surface2, borderRadius: 7, padding: '12px 14px', marginBottom: 14,
            fontSize: 11, color: t.muted, lineHeight: 1.6,
            border: `1px solid ${t.success}33`,
          }}>
            <div style={{ fontWeight: 600, color: t.success, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              ✓ Markup received by server
            </div>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: t.text }}>
              architecture <span style={{ color: t.blue }}>{echo.architecture ?? 'null'}</span>
              {' · '}orientation <span style={{ color: t.blue }}>{echo.orientation}</span>
              {' · '}{echoAnnos.length} annotation{echoAnnos.length === 1 ? '' : 's'}
            </div>
            {echoAnnos.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ marginBottom: 4, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.05em', fontSize: 10, fontWeight: 600 }}>
                  Strokes sent
                </div>
                <div style={{
                  display: 'grid', gridTemplateColumns: 'max-content max-content',
                  columnGap: 16, rowGap: 3,
                  fontFamily: "'JetBrains Mono',monospace", fontSize: 11,
                }}>
                  {echoAnnos.map((a, i) => (
                    <Fragment key={i}>
                      <span style={{ color: t.muted }}>Stroke {i + 1}:</span>
                      <span style={{ color: t.text }}>
                        {a.region_start_frame}–{a.region_end_frame} (target {a.target_frame})
                        {a.player_side && <span style={{ color: t.muted }}> · {a.player_side}</span>}
                      </span>
                    </Fragment>
                  ))}
                </div>
              </div>
            )}
            {mismatch && (
              <div style={{
                marginTop: 8, color: t.warning, fontWeight: 600, fontSize: 11,
                background: 'transparent',
              }}>
                ⚠ Backend returned {strokeCount} stroke{strokeCount === 1 ? '' : 's'}
                {' '}but {echoAnnos.length} annotation{echoAnnos.length === 1 ? ' was' : 's were'} sent.
                Some marked strokes may not have been classified.
              </div>
            )}
            {echo.boundary && echo.boundary.length === 4 && (
              <div style={{ marginTop: 8, fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: t.text }}>
                boundary corners (xy, normalised):
                <div style={{ marginTop: 4, display: 'grid', gridTemplateColumns: 'repeat(2, max-content)', columnGap: 16, rowGap: 2 }}>
                  {echo.boundary.map((p, i) => (
                    <span key={i} style={{ color: t.muted }}>
                      [{i}] ({p.x.toFixed(3)}, {p.y.toFixed(3)})
                    </span>
                  ))}
                </div>
              </div>
            )}
            {!echo.boundary && (
              <div style={{ marginTop: 4, color: t.muted }}>
                no boundary sent (library_predict allows omission)
              </div>
            )}
          </div>
          );
      })()}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 18 }}>
        <div style={{ background: t.surface2, borderRadius: 7, padding: 14 }}>
          <div style={{ fontSize: 10, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Total strokes</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: t.text, fontFamily: "'JetBrains Mono',monospace" }}>
            {rally.total_strokes ?? strokes.length}
          </div>
        </div>
        <div style={{ background: t.surface2, borderRadius: 7, padding: 14 }}>
          <div style={{ fontSize: 10, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Rally length</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: t.text, fontFamily: "'JetBrains Mono',monospace" }}>
            {rally.rally_length_seconds != null ? `${rally.rally_length_seconds.toFixed(1)} s` : '—'}
          </div>
        </div>
      </div>
      {strokes.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: t.muted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
            Detected strokes
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {strokes.map((s, i) => (
              <div key={i} style={{
                display: 'grid', gridTemplateColumns: '70px 70px 1fr 80px',
                fontSize: 12, padding: '6px 10px', background: t.surface2, borderRadius: 5,
                alignItems: 'center', gap: 8,
              }}>
                <span style={{ fontFamily: "'JetBrains Mono',monospace", color: t.muted }}>
                  Stroke {(s.stroke_index ?? i) + 1}
                </span>
                <span style={{ fontFamily: "'JetBrains Mono',monospace", color: t.muted }}>
                  {s.timestamp_sec?.toFixed?.(2) ?? s.timestamp_sec}s
                </span>
                <span style={{ color: t.text, fontWeight: 500 }}>{s.stroke_type}</span>
                <span style={{
                  textAlign: 'right', fontFamily: "'JetBrains Mono',monospace",
                  color: t.blue, fontWeight: 600,
                }}>
                  {((s.confidence ?? 0) * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

/* ─── Results Shell ──────────────────────────────────────────────── */
export function ResultsScreen({ task, onNew }) {
  const { t } = useTheme();

  // Resolve active model from /api/registry, so TestEvalCard +
  // PerClassF1Card get full test_metrics (configure-screen's toModelCard
  // strips those for the picker). Prefer the model the user picked in
  // configure; if no task (dev-jump) or no picked id, fall back to first.
  const pickedId = task?.models?.find(m => task?.enabled?.[m.id])?.id ?? null;
  const [activeModel, setActiveModel] = useState(null);

  // Shared split state for all three Tier 1 panels (browser, eval card, per-class F1).
  const [split, setSplit] = useState('test');

  useEffect(() => {
    let alive = true;
    fetch('/api/registry')
      .then(response => response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`)))
      .then(data => {
        if (!alive) return;
        const models = data.models || [];
        const picked = (pickedId && models.find(m => m.id === pickedId)) || models[0];
        setActiveModel(picked || null);
      })
      .catch(() => { if (alive) setActiveModel(null); });
    return () => { alive = false; };
  }, [pickedId]);

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

      <UploadedInferenceCard task={task} />
      <Tier1ClipBrowser modelId={activeModel?.id} split={split} onSplitChange={setSplit} />
      <TestEvalCard model={activeModel} split={split}/>
      <PerClassF1Card model={activeModel} split={split}/>
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

