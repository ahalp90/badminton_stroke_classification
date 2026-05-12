import { useState, useMemo } from 'react';
import { useTheme, Btn, Card, Badge } from './shared';

const frameModules = import.meta.glob('./data/frames/*.jpg', { eager: true, import: 'default' });
const frameUrl = (id) => frameModules[`./data/frames/${id}.jpg`];

const CLASSES = [
  { label: 'Smash',    color: '#EF4444' },
  { label: 'Clear',    color: '#3B82F6' },
  { label: 'Drop',     color: '#8B5CF6' },
  { label: 'Drive',    color: '#F59E0B' },
  { label: 'Net Kill', color: '#22C55E' },
  { label: 'Lift',     color: '#06B6D4' },
  { label: 'Service',  color: '#D4A843' },
];

// Cheap deterministic hash for stable per-stroke pseudo-randomness.
function hash(seed) {
  let h = Math.imul(seed | 0, 2654435761);
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 0x100000000;
}

const fmtTime = (s) => {
  if (!isFinite(s)) return '–:––';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = String(m).padStart(2, '0');
  const ss = String(sec).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
};

/** Given the video's real stroke timestamps, fabricate stable synthetic
 * classifications: ground truth, predicted class, and confidence.
 * 12% error rate; correct = 0.7–0.99 confidence, wrong = 0.45–0.70. */
function classify(strokeTimes) {
  return strokeTimes.map((t, i) => {
    const seed = Math.round(t * 1000) + i * 7919;
    const gt = CLASSES[Math.floor(hash(seed) * CLASSES.length)];
    const isWrong = hash(seed + 1) < 0.12;
    const pred = isWrong
      ? CLASSES[(CLASSES.indexOf(gt) + 1 + Math.floor(hash(seed + 2) * (CLASSES.length - 1))) % CLASSES.length]
      : gt;
    const conf = isWrong
      ? 0.45 + hash(seed + 3) * 0.25
      : 0.70 + hash(seed + 4) * 0.29;
    return { id: i + 1, time: t, gt: gt.label, pred: pred.label, conf, correct: !isWrong };
  });
}

/* ─── Helpers ────────────────────────────────────────────────────── */
function ConfBar({ value, correct }) {
  const pct = (value * 100).toFixed(0);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 90 }}>
      <div style={{ flex: 1, height: 5, background: 'rgba(127,127,127,0.15)', borderRadius: 3 }}>
        <div style={{ height: '100%', borderRadius: 3, width: `${pct}%`, background: correct ? '#22C55E' : '#EF4444' }} />
      </div>
      <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace", color: correct ? '#22C55E' : '#EF4444', minWidth: 30 }}>
        {pct}%
      </span>
    </div>
  );
}

function HBar({ value, max, color }) {
  return (
    <div style={{ flex: 1, height: 22, background: 'rgba(127,127,127,0.1)', borderRadius: 3, overflow: 'hidden' }}>
      <div style={{
        height: '100%', borderRadius: 3,
        width: `${(value / max) * 100}%`,
        background: color,
        display: 'flex', alignItems: 'center', paddingLeft: 8,
        transition: 'width 0.5s ease',
      }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#fff', fontFamily: "'JetBrains Mono',monospace" }}>{value}</span>
      </div>
    </div>
  );
}

/* ─── Focal stroke card (always visible) ─────────────────────────── */
function FocalStrokeCard({ focal, video, timeframe }) {
  const { t } = useTheme();
  const src = frameUrl(video?.youtubeId);
  if (!focal) return null;

  return (
    <Card style={{ padding: 18, marginBottom: 22, borderColor: t.blue, borderWidth: 1.5 }}>
      <div style={{
        fontSize: 11, color: t.blue, marginBottom: 10,
        textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600,
      }}>
        Your selected stroke · target frame at {fmtTime(focal.time)}
      </div>
      <div style={{ display: 'flex', gap: 18, alignItems: 'center' }}>
        <div style={{
          width: 200, height: 112, borderRadius: 7, overflow: 'hidden',
          background: '#000', flexShrink: 0,
        }}>
          {src && <img src={src} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />}
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, flexWrap: 'wrap' }}>
            <div style={{
              fontSize: 28, fontWeight: 700, color: t.text,
              fontFamily: "'Space Grotesk', sans-serif",
            }}>
              {focal.pred}
            </div>
            <div style={{
              fontSize: 16, fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace",
              color: focal.correct ? t.success : t.danger,
            }}>
              {(focal.conf * 100).toFixed(1)}%
            </div>
            <Badge color={focal.correct ? 'green' : 'red'}>
              {focal.correct ? '✓ correct' : '✗ predicted ' + focal.pred + ', actual ' + focal.gt}
            </Badge>
          </div>
          <div style={{ fontSize: 12, color: t.muted, lineHeight: 1.6 }}>
            Ground truth: <span style={{ color: t.text, fontWeight: 600 }}>{focal.gt}</span>
            {timeframe && (
              <> · Within window {fmtTime(timeframe.startSec)} – {fmtTime(timeframe.endSec)}</>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ─── Tab: Per-stroke ────────────────────────────────────────────── */
function StrokesTab({ classifications, focal, timeframe }) {
  const { t } = useTheme();
  const [scope, setScope] = useState(timeframe ? 'window' : 'all');
  const [errorsOnly, setErrorsOnly] = useState(false);

  const inWindow = (s) => timeframe && s.time >= timeframe.startSec && s.time <= timeframe.endSec;

  const visible = classifications
    .filter(s => scope === 'all' || inWindow(s))
    .filter(s => !errorsOnly || !s.correct);

  const TH = ({ children }) => (
    <th style={{ padding: '7px 12px', textAlign: 'left', color: t.muted, fontWeight: 500, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap' }}>
      {children}
    </th>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        {timeframe && [['window', 'In your window'], ['all', 'Whole match']].map(([id, label]) => (
          <button key={id} onClick={() => setScope(id)} style={{
            padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: scope === id ? 600 : 400,
            border: `1px solid ${scope === id ? t.blue : t.border}`,
            background: scope === id ? t.blueDim : 'transparent',
            color: scope === id ? t.blue : t.muted,
            cursor: 'pointer', fontFamily: "'Space Grotesk',sans-serif",
          }}>{label}</button>
        ))}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: t.muted, cursor: 'pointer', marginLeft: 8 }}>
          <input
            type="checkbox"
            checked={errorsOnly}
            onChange={e => setErrorsOnly(e.target.checked)}
            style={{ accentColor: t.blue }}
          />
          Errors only
        </label>
        <span style={{ marginLeft: 'auto', fontSize: 12, color: t.muted }}>
          {visible.length} of {scope === 'all' ? classifications.length : classifications.filter(inWindow).length} strokes
        </span>
      </div>

      <div style={{ overflowX: 'auto', borderRadius: 8, border: `1px solid ${t.border}` }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead style={{ borderBottom: `2px solid ${t.border}`, background: t.surface2 }}>
            <tr>
              <TH></TH><TH>#</TH><TH>Time</TH><TH>Ground Truth</TH>
              <TH>Predicted</TH><TH>Confidence</TH>
            </tr>
          </thead>
          <tbody>
            {visible.map(s => {
              const isFocal = focal && s.id === focal.id;
              return (
                <tr key={s.id} style={{
                  borderBottom: `1px solid ${t.border}`,
                  background: isFocal ? t.blueDim : 'transparent',
                }}>
                  <td style={{ padding: '10px 12px', color: t.blue, fontSize: 14, width: 18 }}>
                    {isFocal ? '▶' : ''}
                  </td>
                  <td style={{ padding: '10px 12px', color: t.muted, fontFamily: "'JetBrains Mono',monospace", fontSize: 11 }}>
                    {String(s.id).padStart(3, '0')}
                  </td>
                  <td style={{ padding: '10px 12px', color: t.muted, fontFamily: "'JetBrains Mono',monospace", fontSize: 11 }}>
                    {fmtTime(s.time)}
                  </td>
                  <td style={{ padding: '10px 12px' }}><Badge color="blue">{s.gt}</Badge></td>
                  <td style={{ padding: '10px 12px', color: s.correct ? t.success : t.danger, fontWeight: s.correct ? 400 : 600 }}>
                    {!s.correct && '⚠ '}{s.pred}
                  </td>
                  <td style={{ padding: '10px 12px' }}><ConfBar value={s.conf} correct={s.correct} /></td>
                </tr>
              );
            })}
            {visible.length === 0 && (
              <tr><td colSpan={6} style={{ padding: '20px 12px', textAlign: 'center', color: t.muted, fontSize: 12 }}>
                No strokes match the current filter.
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ─── Tab: Distribution ──────────────────────────────────────────── */
function DistributionTab({ classifications, focal }) {
  const { t } = useTheme();
  const [series, setSeries] = useState('gt');

  // Aggregate counts per class.
  const counts = useMemo(() => {
    const acc = {};
    for (const cls of CLASSES) acc[cls.label] = { gt: 0, pred: 0, color: cls.color };
    for (const s of classifications) {
      acc[s.gt].gt += 1;
      acc[s.pred].pred += 1;
    }
    return CLASSES.map(c => ({ label: c.label, color: c.color, ...acc[c.label] }));
  }, [classifications]);

  const total = classifications.length;
  const maxVal = Math.max(...counts.map(d => Math.max(d.gt, d.pred)));
  const seriesMap = { gt: 'Ground Truth', pred: 'Model Prediction' };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', gap: 8 }}>
        {Object.entries(seriesMap).map(([id, label]) => (
          <button key={id} onClick={() => setSeries(id)} style={{
            padding: '7px 16px', borderRadius: 6, fontSize: 13, fontWeight: series === id ? 600 : 400,
            border: `1.5px solid ${series === id ? t.blue : t.border}`,
            background: series === id ? t.blueDim : 'transparent',
            color: series === id ? t.blue : t.muted,
            cursor: 'pointer', fontFamily: "'Space Grotesk',sans-serif",
          }}>{label}</button>
        ))}
      </div>

      <Card style={{ padding: 24 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: t.text, marginBottom: 18 }}>
          Shot Distribution — {seriesMap[series]} ({total} strokes)
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {counts.map(d => {
            const isFocalClass = focal && (series === 'gt' ? focal.gt : focal.pred) === d.label;
            return (
              <div key={d.label} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{
                  width: 78, fontSize: 13, color: isFocalClass ? t.blue : t.text,
                  textAlign: 'right', flexShrink: 0, fontWeight: isFocalClass ? 600 : 400,
                }}>
                  {d.label}
                </div>
                <HBar value={d[series]} max={maxVal} color={d.color} />
                <div style={{ fontSize: 11, color: t.muted, width: 38, textAlign: 'right', flexShrink: 0, fontFamily: "'JetBrains Mono',monospace" }}>
                  {total ? ((d[series] / total) * 100).toFixed(0) : 0}%
                </div>
                <div style={{ width: 84, fontSize: 11, color: t.blue, fontWeight: 600 }}>
                  {isFocalClass ? '← your stroke' : ''}
                </div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

/* ─── Tab: Model Comparison ──────────────────────────────────────── */
function ComparisonTab({ classifications, focal }) {
  const { t } = useTheme();
  const correct = classifications.filter(s => s.correct).length;
  const acc = classifications.length ? (correct / classifications.length) * 100 : 0;

  // Per-class accuracy across the match.
  const perClass = CLASSES.map(c => {
    const all = classifications.filter(s => s.gt === c.label);
    const ok = all.filter(s => s.correct).length;
    return { label: c.label, color: c.color, n: all.length, accPct: all.length ? (ok / all.length) * 100 : 0 };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {focal && (
        <Card style={{ padding: 18, borderColor: t.blue, borderWidth: 1.5 }}>
          <div style={{ fontSize: 11, color: t.blue, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
            On your selected stroke
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, flexWrap: 'wrap' }}>
            <div style={{ fontSize: 13, color: t.muted }}>Model A predicted</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: focal.correct ? t.success : t.danger, fontFamily: "'JetBrains Mono', monospace" }}>
              {focal.pred} · {(focal.conf * 100).toFixed(0)}%
            </div>
            <Badge color={focal.correct ? 'green' : 'red'}>
              {focal.correct ? '✓ matches ground truth' : `✗ actual: ${focal.gt}`}
            </Badge>
          </div>
        </Card>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 14 }}>
        {[
          { label: 'Model A Accuracy', value: `${acc.toFixed(1)}%`, sub: 'BST (TCN + Transformer)', color: t.blue },
          { label: 'BST Baseline',     value: '80–85%',              sub: 'Chang 2025 (reference)', color: t.pine },
        ].map(c => (
          <Card key={c.label} style={{ padding: 20, textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: t.muted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{c.label}</div>
            <div style={{ fontSize: 30, fontWeight: 700, color: c.color, fontFamily: "'JetBrains Mono',monospace", marginBottom: 4 }}>{c.value}</div>
            <div style={{ fontSize: 11, color: t.muted }}>{c.sub}</div>
          </Card>
        ))}
      </div>

      <Card style={{ padding: 22 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: t.text, marginBottom: 16 }}>Per-Class Accuracy (whole match)</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {perClass.map(d => (
            <div key={d.label} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{ width: 78, fontSize: 12, color: t.text, flexShrink: 0 }}>{d.label}</div>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ flex: 1, height: 6, background: t.surface2, borderRadius: 3 }}>
                  <div style={{ height: '100%', borderRadius: 3, width: `${d.accPct}%`, background: d.color }} />
                </div>
                <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace", color: t.text, width: 44, textAlign: 'right' }}>
                  {d.n ? `${d.accPct.toFixed(0)}%` : '—'}
                </span>
                <span style={{ fontSize: 10, color: t.muted, width: 44, textAlign: 'right' }}>
                  n={d.n}
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ─── Tab: Explainability ────────────────────────────────────────── */
function ExplainabilityTab({ classifications, focal }) {
  const { t } = useTheme();
  const initialIdx = focal ? classifications.findIndex(s => s.id === focal.id) : 0;
  const [selectedIdx, setSelectedIdx] = useState(initialIdx >= 0 ? initialIdx : 0);
  const s = classifications[selectedIdx];

  if (!s) return <div style={{ color: t.muted, fontSize: 13 }}>No strokes available.</div>;

  // Synthetic top-k probability sketch.
  const top = [
    { cls: s.pred, prob: s.conf, isPred: true,  isCorrect: s.correct },
    { cls: s.gt === s.pred ? CLASSES[(CLASSES.findIndex(c => c.label === s.pred) + 1) % CLASSES.length].label : s.gt,
      prob: s.conf * 0.32, isPred: false, isCorrect: s.gt !== s.pred ? false : true },
    { cls: CLASSES[(CLASSES.findIndex(c => c.label === s.pred) + 3) % CLASSES.length].label,
      prob: s.conf * 0.18, isPred: false, isCorrect: false },
  ];

  // Show focal first, then nearest 8 from window/match.
  const buttons = useMemo(() => {
    const focalIdx = focal ? classifications.findIndex(c => c.id === focal.id) : -1;
    const others = classifications.filter((_, i) => i !== focalIdx).slice(0, 8);
    return focalIdx >= 0 ? [classifications[focalIdx], ...others] : classifications.slice(0, 9);
  }, [classifications, focal]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ fontSize: 13, color: t.muted, lineHeight: 1.6 }}>
        Class activation map and top-class probabilities for individual stroke classifications.
        Defaults to your selected stroke; pick another below to inspect.
      </div>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {buttons.map((stroke) => {
          const idx = classifications.findIndex(c => c.id === stroke.id);
          const isFocal = focal && stroke.id === focal.id;
          return (
            <button key={stroke.id} onClick={() => setSelectedIdx(idx)} style={{
              padding: '5px 12px', borderRadius: 5, fontSize: 12,
              border: `1px solid ${selectedIdx === idx ? t.blue : isFocal ? t.blue + '88' : t.border}`,
              background: selectedIdx === idx ? t.blueDim : 'transparent',
              color: selectedIdx === idx ? t.blue : t.muted,
              cursor: 'pointer', fontFamily: "'Space Grotesk',sans-serif",
            }}>
              {isFocal && '★ '}#{stroke.id} {stroke.gt} · {fmtTime(stroke.time)}
            </button>
          );
        })}
      </div>

      <Card style={{ padding: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: t.text }}>
            Stroke #{s.id} · {fmtTime(s.time)}
          </div>
          <Badge color="blue">3D-CNN</Badge>
        </div>

        <div style={{
          height: 140, borderRadius: 7, marginBottom: 14, position: 'relative', overflow: 'hidden',
          background: t.surface2, border: `1px solid ${t.border}`,
        }}>
          <div style={{
            position: 'absolute', borderRadius: '50%',
            width: 110, height: 110, top: '5%', left: '40%',
            background: 'radial-gradient(circle, rgba(239,68,68,0.75) 0%, rgba(251,146,60,0.4) 45%, transparent 70%)',
            filter: 'blur(5px)',
          }} />
          <div style={{
            position: 'absolute', borderRadius: '50%',
            width: 70, height: 70, top: '35%', left: '15%',
            background: 'radial-gradient(circle, rgba(34,197,94,0.5) 0%, transparent 70%)',
            filter: 'blur(4px)',
          }} />
          <div style={{
            position: 'absolute', top: 6, left: 8, fontSize: 10,
            color: t.muted, background: t.surface2 + 'cc', padding: '2px 6px', borderRadius: 3,
          }}>Class Activation Map</div>
          <div style={{
            position: 'absolute', bottom: 6, right: 8,
            width: 60, height: 8, borderRadius: 3,
            background: 'linear-gradient(90deg, rgba(34,197,94,0.6), rgba(251,146,60,0.7), rgba(239,68,68,0.9))',
          }} />
          <div style={{
            position: 'absolute', bottom: 16, right: 8,
            fontSize: 9, color: t.muted, display: 'flex', justifyContent: 'space-between', width: 60,
          }}>
            <span>low</span><span>high</span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {top.map((c, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{
                width: 78, fontSize: 12,
                color: c.isCorrect ? t.success : c.isPred ? t.text : t.muted,
                fontWeight: c.isPred ? 600 : 400,
              }}>{c.cls}</div>
              <div style={{ flex: 1, height: 6, background: t.surface2, borderRadius: 3 }}>
                <div style={{
                  height: '100%', borderRadius: 3,
                  width: `${c.prob * 100}%`,
                  background: c.isCorrect ? t.success : c.isPred ? t.blue : t.muted + '88',
                }} />
              </div>
              <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace", color: t.muted, width: 38, textAlign: 'right' }}>
                {(c.prob * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>

        <div style={{
          marginTop: 14, padding: '8px 12px', borderRadius: 6,
          background: s.correct ? t.successDim : t.dangerDim,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{ fontSize: 12, color: t.muted }}>Prediction vs ground truth</span>
          <span style={{ fontSize: 12, fontWeight: 700, color: s.correct ? t.success : t.danger }}>
            {s.pred} {s.correct ? '✓' : `✗  (gt: ${s.gt})`}
          </span>
        </div>
      </Card>
    </div>
  );
}

/* ─── Results Shell ──────────────────────────────────────────────── */
export function ResultsScreen({ task, onNew }) {
  const { t } = useTheme();
  const [tab, setTab] = useState('strokes');

  const markup = task?.markup;
  const video = markup?.video;
  const timeframe = markup?.timeframe;
  const strokeTimes = video?.strokeTimes || [];

  const classifications = useMemo(() => classify(strokeTimes), [strokeTimes]);

  // Focal stroke = whichever annotated stroke is closest to the user's target frame.
  const focal = useMemo(() => {
    if (!classifications.length || !timeframe?.targetSec) return null;
    let best = classifications[0], bestDist = Infinity;
    for (const s of classifications) {
      const d = Math.abs(s.time - timeframe.targetSec);
      if (d < bestDist) { best = s; bestDist = d; }
    }
    return best;
  }, [classifications, timeframe?.targetSec]);

  const inWindow = (s) => timeframe && s.time >= timeframe.startSec && s.time <= timeframe.endSec;
  const windowStrokes = classifications.filter(inWindow);
  const windowCorrect = windowStrokes.filter(s => s.correct).length;
  const matchCorrect  = classifications.filter(s => s.correct).length;

  const TABS = [
    { id: 'strokes',        label: 'Per-Stroke Results' },
    { id: 'distribution',   label: 'Shot Distribution' },
    { id: 'comparison',     label: 'Model Comparison' },
    { id: 'explainability', label: 'Explainability' },
  ];

  const CONTENT = {
    strokes:        <StrokesTab classifications={classifications} focal={focal} timeframe={timeframe} />,
    distribution:   <DistributionTab classifications={classifications} focal={focal} />,
    comparison:     <ComparisonTab classifications={classifications} focal={focal} />,
    explainability: <ExplainabilityTab classifications={classifications} focal={focal} />,
  };

  const matchAcc  = classifications.length ? (matchCorrect  / classifications.length) * 100 : 0;
  const windowAcc = windowStrokes.length    ? (windowCorrect / windowStrokes.length)    * 100 : 0;
  const stats = timeframe && windowStrokes.length ? [
    { label: 'Strokes in your window', value: String(windowStrokes.length),                color: t.text },
    { label: 'Window accuracy',        value: `${windowAcc.toFixed(0)}%`,                  color: t.blue },
    { label: 'Match accuracy',         value: `${matchAcc.toFixed(0)}%`,                   color: t.pine },
  ] : [
    { label: 'Strokes classified',     value: String(classifications.length),              color: t.text },
    { label: 'Match accuracy',         value: `${matchAcc.toFixed(0)}%`,                   color: t.blue },
    { label: 'Conf. ≥ 70%',            value: `${classifications.length ? ((classifications.filter(s => s.conf >= 0.70).length / classifications.length) * 100).toFixed(0) : 0}%`, color: t.pine },
  ];

  return (
    <div style={{ maxWidth: 1120, margin: '0 auto', padding: 32 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: t.text, marginBottom: 4 }}>Classification Results</h1>
          <p style={{ fontSize: 13, color: t.muted }}>
            {task?.taskName ?? 'Analysis'} · Completed {new Date().toLocaleString('en-AU')}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <Btn variant="secondary" size="sm">Export JSON</Btn>
          <Btn variant="secondary" size="sm" onClick={onNew}>New Analysis</Btn>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12, marginBottom: 22 }}>
        {stats.map((s, i) => (
          <Card key={i} style={{ padding: 18 }}>
            <div style={{ fontSize: 10, color: t.muted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{s.label}</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono',monospace" }}>{s.value}</div>
          </Card>
        ))}
      </div>

      <FocalStrokeCard focal={focal} video={video} timeframe={timeframe} />

      <div style={{ display: 'flex', borderBottom: `1px solid ${t.border}`, marginBottom: 22 }}>
        {TABS.map(tb => (
          <button key={tb.id} onClick={() => setTab(tb.id)} style={{
            padding: '10px 20px', background: 'none', border: 'none', marginBottom: -1,
            borderBottom: tab === tb.id ? `2px solid ${t.blue}` : '2px solid transparent',
            color: tab === tb.id ? t.blue : t.muted,
            fontSize: 13, fontWeight: tab === tb.id ? 600 : 400,
            cursor: 'pointer', fontFamily: "'Space Grotesk',sans-serif",
            whiteSpace: 'nowrap',
          }}>
            {tb.label}
          </button>
        ))}
      </div>

      {CONTENT[tab]}
    </div>
  );
}
