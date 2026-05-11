import { useState } from 'react';
import { useTheme, Btn, Card, Badge, SectionHeader } from './shared';

/* ─── Fake data ──────────────────────────────────────────────────── */
const STROKES = [
  { id:1,  time:'0:00:26', gt:'Smash',    predA:'Smash',    cA:0.94, predB:'Smash',    cB:0.88 },
  { id:2,  time:'0:00:52', gt:'Clear',    predA:'Clear',    cA:0.87, predB:'Clear',    cB:0.79 },
  { id:3,  time:'0:01:21', gt:'Drop',     predA:'Drop',     cA:0.71, predB:'Net Kill', cB:0.62 },
  { id:4,  time:'0:01:44', gt:'Drive',    predA:'Drive',    cA:0.83, predB:'Drive',    cB:0.75 },
  { id:5,  time:'0:02:10', gt:'Net Kill', predA:'Net Kill', cA:0.92, predB:'Net Kill', cB:0.86 },
  { id:6,  time:'0:02:36', gt:'Lift',     predA:'Lift',     cA:0.68, predB:'Clear',    cB:0.57 },
  { id:7,  time:'0:03:03', gt:'Smash',    predA:'Smash',    cA:0.96, predB:'Smash',    cB:0.91 },
  { id:8,  time:'0:03:31', gt:'Drop',     predA:'Drive',    cA:0.55, predB:'Drop',     cB:0.69 },
  { id:9,  time:'0:03:57', gt:'Clear',    predA:'Clear',    cA:0.89, predB:'Clear',    cB:0.81 },
  { id:10, time:'0:04:24', gt:'Service',  predA:'Service',  cA:0.78, predB:'Service',  cB:0.74 },
  { id:11, time:'0:04:51', gt:'Smash',    predA:'Smash',    cA:0.93, predB:'Smash',    cB:0.84 },
  { id:12, time:'0:05:18', gt:'Clear',    predA:'Lift',     cA:0.52, predB:'Clear',    cB:0.73 },
];

const SHOT_DIST = [
  { label:'Clear',    gt:203, mA:197, mB:218, color:'#3B82F6' },
  { label:'Smash',    gt:187, mA:191, mB:182, color:'#EF4444' },
  { label:'Drop',     gt:142, mA:138, mB:149, color:'#8B5CF6' },
  { label:'Lift',     gt:89,  mA:84,  mB:77,  color:'#06B6D4' },
  { label:'Drive',    gt:98,  mA:101, mB:96,  color:'#F59E0B' },
  { label:'Net Kill', gt:76,  mA:79,  mB:71,  color:'#22C55E' },
  { label:'Service',  gt:52,  mA:57,  mB:54,  color:'#D4A843' },
];

const METRICS = [
  { label:'Overall Accuracy',  a:83.2, b:76.4, base:80.0,  fmt: v => `${v.toFixed(1)}%` },
  { label:'Macro Precision',   a:82.1, b:74.8, base:null,  fmt: v => `${v.toFixed(1)}%` },
  { label:'Macro Recall',      a:80.6, b:73.2, base:null,  fmt: v => `${v.toFixed(1)}%` },
  { label:'Macro F1',          a:0.813,b:0.740,base:null,  fmt: v => v.toFixed(3) },
  { label:'Smash Precision',   a:91.4, b:85.7, base:88.0,  fmt: v => `${v.toFixed(1)}%` },
  { label:'Drop Recall',       a:74.2, b:67.1, base:72.0,  fmt: v => `${v.toFixed(1)}%` },
  { label:'Inference / stroke',a:2.4,  b:0.31, base:null,  fmt: v => `${v}s`, lowerBetter:true },
];

/* ─── Helpers ────────────────────────────────────────────────────── */
function ConfBar({ value, correct }) {
  const pct = (value * 100).toFixed(0);
  return (
    <div style={{ display:'flex', alignItems:'center', gap:6, minWidth:90 }}>
      <div style={{ flex:1, height:5, background:'rgba(127,127,127,0.15)', borderRadius:3 }}>
        <div style={{ height:'100%', borderRadius:3, width:`${pct}%`, background: correct ? '#22C55E' : '#EF4444' }} />
      </div>
      <span style={{ fontSize:11, fontFamily:"'JetBrains Mono',monospace", color: correct?'#22C55E':'#EF4444', minWidth:30 }}>
        {pct}%
      </span>
    </div>
  );
}

function HBar({ value, max, color }) {
  return (
    <div style={{ flex:1, height:22, background:'rgba(127,127,127,0.1)', borderRadius:3, overflow:'hidden' }}>
      <div style={{
        height:'100%', borderRadius:3,
        width:`${(value/max)*100}%`,
        background: color,
        display:'flex', alignItems:'center', paddingLeft:8,
        transition:'width 0.5s ease',
      }}>
        <span style={{ fontSize:11, fontWeight:600, color:'#fff', fontFamily:"'JetBrains Mono',monospace" }}>{value}</span>
      </div>
    </div>
  );
}

/* ─── Tab: Per-stroke ────────────────────────────────────────────── */
function StrokesTab() {
  const { t } = useTheme();
  const [filter, setFilter] = useState('all');

  const filtered = filter === 'errors'
    ? STROKES.filter(s => s.predA !== s.gt)
    : STROKES;

  const TH = ({ children }) => (
    <th style={{ padding:'7px 12px', textAlign:'left', color:t.muted, fontWeight:500, fontSize:11, textTransform:'uppercase', letterSpacing:'0.05em', whiteSpace:'nowrap' }}>
      {children}
    </th>
  );

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
      <div style={{ display:'flex', gap:8, alignItems:'center' }}>
        {[['all','All strokes'],['errors','Errors only']].map(([id, label]) => (
          <button key={id} onClick={() => setFilter(id)} style={{
            padding:'6px 14px', borderRadius:6, fontSize:12, fontWeight: filter===id ? 600 : 400,
            border:`1px solid ${filter===id ? t.blue : t.border}`,
            background: filter===id ? t.blueDim : 'transparent',
            color: filter===id ? t.blue : t.muted,
            cursor:'pointer', fontFamily:"'Space Grotesk',sans-serif",
          }}>{label}</button>
        ))}
        <span style={{ marginLeft:'auto', fontSize:12, color:t.muted }}>
          {filtered.length} of {STROKES.length} strokes
        </span>
      </div>

      <div style={{ overflowX:'auto', borderRadius:8, border:`1px solid ${t.border}` }}>
        <table style={{ width:'100%', borderCollapse:'collapse', fontSize:13 }}>
          <thead style={{ borderBottom:`2px solid ${t.border}`, background:t.surface2 }}>
            <tr>
              <TH>#</TH><TH>Time</TH><TH>Ground Truth</TH>
              <TH>Model A</TH><TH>Conf. A</TH>
            </tr>
          </thead>
          <tbody>
            {filtered.map(s => {
            const aOk = s.predA === s.gt;
              return (
                <tr key={s.id} style={{ borderBottom:`1px solid ${t.border}` }}>
                  <td style={{ padding:'10px 12px', color:t.muted, fontFamily:"'JetBrains Mono',monospace", fontSize:11 }}>
                    {String(s.id).padStart(3,'0')}
                  </td>
                  <td style={{ padding:'10px 12px', color:t.muted, fontFamily:"'JetBrains Mono',monospace", fontSize:11 }}>{s.time}</td>
                  <td style={{ padding:'10px 12px' }}><Badge color="blue">{s.gt}</Badge></td>
                  <td style={{ padding:'10px 12px', color: aOk ? t.success : t.danger, fontWeight: aOk ? 400 : 600 }}>
                    {!aOk && '⚠ '}{s.predA}
                  </td>
                  <td style={{ padding:'10px 12px' }}><ConfBar value={s.cA} correct={aOk} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ─── Tab: Distribution ──────────────────────────────────────────── */
function DistributionTab() {
  const { t } = useTheme();
  const [series, setSeries] = useState('gt');
  const maxVal = Math.max(...SHOT_DIST.map(d => Math.max(d.gt, d.mA, d.mB)));

  const seriesMap = { gt:'Ground Truth', mA:'Model A' };

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
      <div style={{ display:'flex', gap:8 }}>
        {Object.entries(seriesMap).map(([id, label]) => (
          <button key={id} onClick={() => setSeries(id)} style={{
            padding:'7px 16px', borderRadius:6, fontSize:13, fontWeight: series===id ? 600 : 400,
            border:`1.5px solid ${series===id ? t.blue : t.border}`,
            background: series===id ? t.blueDim : 'transparent',
            color: series===id ? t.blue : t.muted,
            cursor:'pointer', fontFamily:"'Space Grotesk',sans-serif",
          }}>{label}</button>
        ))}
      </div>

      <Card style={{ padding:24 }}>
        <div style={{ fontSize:14, fontWeight:600, color:t.text, marginBottom:18 }}>
          Shot Distribution — {seriesMap[series]}
        </div>
        <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
          {SHOT_DIST.map(d => (
            <div key={d.label} style={{ display:'flex', alignItems:'center', gap:12 }}>
              <div style={{ width:68, fontSize:13, color:t.text, textAlign:'right', flexShrink:0 }}>{d.label}</div>
              <HBar value={d[series]} max={maxVal} color={d.color} />
              <div style={{ fontSize:11, color:t.muted, width:28, textAlign:'right', flexShrink:0, fontFamily:"'JetBrains Mono',monospace" }}>
                {((d[series] / SHOT_DIST.reduce((a,b) => a + b[series], 0)) * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
        {Object.entries(seriesMap).map(([id, label]) => (
          <Card key={id} style={{ padding:16 }}>
            <div style={{ fontSize:12, fontWeight:600, color:t.muted, marginBottom:12 }}>{label}</div>
            <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
              {SHOT_DIST.map(d => (
                <div key={d.label} style={{ display:'flex', alignItems:'center', gap:6 }}>
                  <div style={{ width:4, height:4, borderRadius:'50%', background:d.color, flexShrink:0 }} />
                  <div style={{ fontSize:11, color:t.muted, width:56, flexShrink:0 }}>{d.label}</div>
                  <div style={{ flex:1, height:3, background:t.surface2, borderRadius:2 }}>
                    <div style={{ height:'100%', borderRadius:2, width:`${(d[id]/maxVal)*100}%`, background:d.color }} />
                  </div>
                </div>
              ))}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

/* ─── Tab: Model Comparison ──────────────────────────────────────── */
function ComparisonTab() {
  const { t } = useTheme();

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
      <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:14 }}>
        {[
          { label:'Model A Accuracy', value:'83.2%', sub:'Spatio-Temporal 3D-CNN', color:t.blue },
          { label:'BST Baseline',     value:'80–85%', sub:'Chang 2025 (reference)', color:t.pine },
        ].map(c => (
          <Card key={c.label} style={{ padding:20, textAlign:'center' }}>
            <div style={{ fontSize:11, color:t.muted, marginBottom:8, textTransform:'uppercase', letterSpacing:'0.05em' }}>{c.label}</div>
            <div style={{ fontSize:30, fontWeight:700, color:c.color, fontFamily:"'JetBrains Mono',monospace", marginBottom:4 }}>{c.value}</div>
            <div style={{ fontSize:11, color:t.muted }}>{c.sub}</div>
          </Card>
        ))}
      </div>

      <Card style={{ padding:22 }}>
        <div style={{ fontSize:13, fontWeight:600, color:t.text, marginBottom:16 }}>Metric Breakdown</div>
        <table style={{ width:'100%', borderCollapse:'collapse', fontSize:13 }}>
          <thead>
            <tr style={{ borderBottom:`2px solid ${t.border}` }}>
              {['Metric','Model A','BST Baseline'].map(h => (
                <th key={h} style={{ padding:'6px 12px', textAlign:'left', color:t.muted, fontSize:11, fontWeight:500, textTransform:'uppercase', letterSpacing:'0.05em' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METRICS.map(m => {
              return (
                <tr key={m.label} style={{ borderBottom:`1px solid ${t.border}` }}>
                  <td style={{ padding:'10px 12px', color:t.text, fontWeight:500 }}>{m.label}</td>
                  <td style={{ padding:'10px 12px', fontFamily:"'JetBrains Mono',monospace", color:t.text }}>
                    {m.fmt(m.a)}
                  </td>
                  <td style={{ padding:'10px 12px', fontFamily:"'JetBrains Mono',monospace", color:t.muted }}>
                    {m.base ? m.fmt(m.base) : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      <Card style={{ padding:22 }}>
        <div style={{ fontSize:13, fontWeight:600, color:t.text, marginBottom:16 }}>Per-Class Accuracy</div>
        <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
          {SHOT_DIST.map(d => {
            const accA = 72 + (d.gt % 28);
            return (
              <div key={d.label} style={{ display:'flex', alignItems:'center', gap:12 }}>
                <div style={{ width:68, fontSize:12, color:t.text, flexShrink:0 }}>{d.label}</div>
                <div style={{ flex:1, display:'flex', flexDirection:'column', gap:3 }}>
                  <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                    <div style={{ width:52, fontSize:10, color:t.muted }}>Model A</div>
                    <div style={{ flex:1, height:5, background:t.surface2, borderRadius:3 }}>
                      <div style={{ height:'100%', borderRadius:3, width:`${accA}%`, background:t.blue }} />
                    </div>
                    <span style={{ fontSize:11, fontFamily:"'JetBrains Mono',monospace", color:t.blue, width:34 }}>{accA}%</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

/* ─── Tab: Explainability ────────────────────────────────────────── */
function ExplainabilityTab() {
  const { t } = useTheme();
  const [selected, setSelected] = useState(0);
  const s = STROKES[selected];

  const classProbs = [
    { cls: s.predA, prob: s.cA, model:'A', correct: s.predA === s.gt },
    { cls: 'Clear',  prob: s.cA * 0.31, model:'A', correct: false },
    { cls: 'Drop',   prob: s.cA * 0.18, model:'A', correct: false },
    { cls: s.predB, prob: s.cB, model:'B', correct: s.predB === s.gt },
    { cls: 'Smash',  prob: s.cB * 0.28, model:'B', correct: false },
    { cls: 'Lift',   prob: s.cB * 0.15, model:'B', correct: false },
  ];

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
      <div style={{ fontSize:13, color:t.muted, lineHeight:1.6 }}>
        Class activation maps and confidence distributions for individual stroke classifications.
        Select a stroke below to inspect model decision logic.
      </div>

      <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
        {STROKES.slice(0,8).map((stroke, i) => (
          <button key={i} onClick={() => setSelected(i)} style={{
            padding:'5px 12px', borderRadius:5, fontSize:12,
            border:`1px solid ${selected===i ? t.blue : t.border}`,
            background: selected===i ? t.blueDim : 'transparent',
            color: selected===i ? t.blue : t.muted,
            cursor:'pointer', fontFamily:"'Space Grotesk',sans-serif",
          }}>
            #{stroke.id} {stroke.gt}
          </button>
        ))}
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:18 }}>
        {['A','B'].map(model => (
          <Card key={model} style={{ padding:18 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:12 }}>
              <div style={{ fontSize:13, fontWeight:600, color:t.text }}>Model {model} — Stroke #{s.id}</div>
              <Badge color={model==='A' ? 'blue' : 'green'}>
                {model==='A' ? '3D-CNN' : 'TCN'}
              </Badge>
            </div>

            <div style={{
              height:100, borderRadius:7, marginBottom:14, position:'relative', overflow:'hidden',
              background: t.surface2, border:`1px solid ${t.border}`,
            }}>
              <div style={{
                position:'absolute', borderRadius:'50%',
                width:80, height:80, top:'5%', left: model==='A' ? '45%' : '30%',
                background:'radial-gradient(circle, rgba(239,68,68,0.75) 0%, rgba(251,146,60,0.4) 45%, transparent 70%)',
                filter:'blur(4px)',
              }} />
              <div style={{
                position:'absolute', borderRadius:'50%',
                width:50, height:50, top:'30%', left: model==='A' ? '20%' : '60%',
                background:'radial-gradient(circle, rgba(34,197,94,0.5) 0%, transparent 70%)',
                filter:'blur(3px)',
              }} />
              <div style={{
                position:'absolute', top:6, left:8, fontSize:10,
                color:t.muted, background: t.surface2 + 'cc', padding:'2px 6px', borderRadius:3,
              }}>Class Activation Map</div>
              <div style={{
                position:'absolute', bottom:6, right:8,
                width:60, height:8, borderRadius:3,
                background:'linear-gradient(90deg, rgba(34,197,94,0.6), rgba(251,146,60,0.7), rgba(239,68,68,0.9))',
              }} />
              <div style={{
                position:'absolute', bottom:16, right:8,
                fontSize:9, color:t.muted, display:'flex', justifyContent:'space-between', width:60,
              }}>
                <span>low</span><span>high</span>
              </div>
            </div>

            <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
              {classProbs.filter(c => c.model===model).map((c,i) => (
                <div key={i} style={{ display:'flex', alignItems:'center', gap:8 }}>
                  <div style={{ width:62, fontSize:12, color: c.correct ? t.success : t.text, fontWeight: i===0 ? 600 : 400 }}>{c.cls}</div>
                  <div style={{ flex:1, height:6, background:t.surface2, borderRadius:3 }}>
                    <div style={{
                      height:'100%', borderRadius:3,
                      width:`${c.prob*100}%`,
                      background: c.correct ? (model==='A' ? t.blue : t.success) : t.muted + '88',
                    }} />
                  </div>
                  <span style={{ fontSize:11, fontFamily:"'JetBrains Mono',monospace", color:t.muted, width:30, textAlign:'right' }}>
                    {(c.prob*100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>

            <div style={{
              marginTop:12, padding:'8px 12px', borderRadius:6,
              background: (model==='A' ? s.predA : s.predB) === s.gt ? t.successDim : t.dangerDim,
              display:'flex', alignItems:'center', justifyContent:'space-between',
            }}>
              <span style={{ fontSize:12, color:t.muted }}>Prediction</span>
              <span style={{ fontSize:12, fontWeight:700, color: (model==='A' ? s.predA : s.predB) === s.gt ? t.success : t.danger }}>
                {model==='A' ? s.predA : s.predB}
                {(model==='A' ? s.predA : s.predB) === s.gt ? ' ✓' : ' ✗'}
              </span>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

/* ─── Results Shell ──────────────────────────────────────────────── */
export function ResultsScreen({ task, onNew }) {
  const { t } = useTheme();
  const [tab, setTab] = useState('strokes');

  const TABS = [
    { id:'strokes',        label:'Per-Stroke Results' },
    { id:'distribution',   label:'Shot Distribution' },
    { id:'comparison',     label:'Model Comparison' },
    { id:'explainability', label:'Explainability' },
  ];

  const CONTENT = {
    strokes:        <StrokesTab />,
    distribution:   <DistributionTab />,
    comparison:     <ComparisonTab />,
    explainability: <ExplainabilityTab />,
  };

  return (
    <div style={{ maxWidth:1120, margin:'0 auto', padding:32 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:20 }}>
        <div>
          <h1 style={{ fontSize:22, fontWeight:700, color:t.text, marginBottom:4 }}>Classification Results</h1>
          <p style={{ fontSize:13, color:t.muted }}>
            {task?.taskName ?? 'Analysis'} · Completed {new Date().toLocaleString('en-AU')}
          </p>
        </div>
        <div style={{ display:'flex', gap:10 }}>
          <Btn variant="secondary" size="sm">Export JSON</Btn>
          <Btn variant="secondary" size="sm" onClick={onNew}>New Analysis</Btn>
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12, marginBottom:26 }}>
        {[
          { label:'Strokes Classified', value:'847',   color:t.text },
          { label:'Model A Accuracy',   value:'83.2%', color:t.blue },
          { label:'Conf. ≥ 65%',        value:'91.3%', color:t.pine },
        ].map(s => (
          <Card key={s.label} style={{ padding:18 }}>
            <div style={{ fontSize:10, color:t.muted, marginBottom:6, textTransform:'uppercase', letterSpacing:'0.06em' }}>{s.label}</div>
            <div style={{ fontSize:24, fontWeight:700, color:s.color, fontFamily:"'JetBrains Mono',monospace" }}>{s.value}</div>
          </Card>
        ))}
      </div>

      <div style={{ display:'flex', borderBottom:`1px solid ${t.border}`, marginBottom:22 }}>
        {TABS.map(tb => (
          <button key={tb.id} onClick={() => setTab(tb.id)} style={{
            padding:'10px 20px', background:'none', border:'none', marginBottom:-1,
            borderBottom: tab===tb.id ? `2px solid ${t.blue}` : '2px solid transparent',
            color: tab===tb.id ? t.blue : t.muted,
            fontSize:13, fontWeight: tab===tb.id ? 600 : 400,
            cursor:'pointer', fontFamily:"'Space Grotesk',sans-serif",
            whiteSpace:'nowrap',
          }}>
            {tb.label}
          </button>
        ))}
      </div>

      {CONTENT[tab]}
    </div>
  );
}
