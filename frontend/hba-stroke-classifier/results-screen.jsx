import { useState, useEffect } from 'react';
import { useTheme, Btn, Card } from './shared';
import { Tier1ClipBrowser } from './components/Tier1ClipBrowser';

/* ─── Held-out test set evaluation (from /api/registry, no hardcoded numbers) ─ */
function TestEvalCard({ model }) {
  const { t } = useTheme();
  if (!model || !model.test_metrics) return null;
  const m = model.test_metrics;
  const pct = (x) => (x * 100).toFixed(1) + '%';
  const stats = [
    { label: 'Macro F1',       value: m.macro_f1.toFixed(3),    color: t.blue },
    { label: 'Accuracy',       value: pct(m.accuracy),          color: t.text },
    { label: 'Top-2 accuracy', value: pct(m.top2_accuracy),     color: t.text },
    { label: 'Min F1',         value: m.min_f1.toFixed(3),      color: t.text },
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
        {model.display_name} · {model.num_classes} classes · taxonomy <Mono>{model.taxonomy}</Mono>
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

/* ─── Per-class F1 on test set (from /api/registry per_class_f1 map) ── */
function PerClassF1Card({ model }) {
  const { t } = useTheme();
  if (!model?.test_metrics?.per_class_f1) return null;
  // Sort desc so the strongest classes read top-down; bar widths scale to
  // the max so small differences between weak classes are still visible.
  const entries = Object.entries(model.test_metrics.per_class_f1)
    .sort(([, a], [, b]) => b - a);
  const max = Math.max(...entries.map(([, v]) => v));
  return (
    <Card style={{ padding: 22 }}>
      <div style={{ fontSize: 11, color: t.muted, marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        Per-class F1 — test set
      </div>
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

      <Tier1ClipBrowser modelId={activeModel?.id} />
      <TestEvalCard model={activeModel} />
      <PerClassF1Card model={activeModel} />
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

