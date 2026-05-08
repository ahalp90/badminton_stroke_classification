import { useState, useRef, useEffect, useCallback, Fragment } from 'react';
import { useTheme, Btn, Card } from './shared';

/* ─── Step 1: Court Boundary ─────────────────────────────────────── */
function CourtBoundaryStep({ onComplete }) {
  const { t } = useTheme();
  const canvasRef = useRef(null);
  const W = 620, H = 348;

  const [pts, setPts] = useState([
    { x: 0.17, y: 0.14 },
    { x: 0.83, y: 0.14 },
    { x: 0.89, y: 0.84 },
    { x: 0.11, y: 0.84 },
  ]);
  const [dragging, setDragging]   = useState(null);
  const [confirmed, setConfirmed] = useState(false);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const grad = ctx.createLinearGradient(0, 0, W, H);
    grad.addColorStop(0, '#C0882A');
    grad.addColorStop(0.45, '#D4A843');
    grad.addColorStop(1, '#B07030');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(0,0,0,0.07)';
    ctx.lineWidth = 1;
    for (let x = -H; x < W + H; x += 22) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x + H * 0.06, H);
      ctx.stroke();
    }

    const px = pts.map(p => ({ x: p.x * W, y: p.y * H }));

    ctx.beginPath();
    px.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.fillStyle = confirmed ? 'rgba(34,197,94,0.12)' : 'rgba(37,99,235,0.14)';
    ctx.fill();
    ctx.strokeStyle = confirmed ? '#22C55E' : '#3B82F6';
    ctx.lineWidth = 2;
    ctx.stroke();

    const netL = { x: (px[0].x + px[3].x) / 2, y: (px[0].y + px[3].y) / 2 };
    const netR = { x: (px[1].x + px[2].x) / 2, y: (px[1].y + px[2].y) / 2 };
    ctx.strokeStyle = confirmed ? 'rgba(34,197,94,0.5)' : 'rgba(255,255,255,0.45)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(netL.x, netL.y); ctx.lineTo(netR.x, netR.y); ctx.stroke();

    const lerp = (a, b, f) => a + (b - a) * f;
    const tL = { x: lerp(px[0].x, px[3].x, 0.38), y: lerp(px[0].y, px[3].y, 0.38) };
    const tR = { x: lerp(px[1].x, px[2].x, 0.38), y: lerp(px[1].y, px[2].y, 0.38) };
    const bL = { x: lerp(px[0].x, px[3].x, 0.62), y: lerp(px[0].y, px[3].y, 0.62) };
    const bR = { x: lerp(px[1].x, px[2].x, 0.62), y: lerp(px[1].y, px[2].y, 0.62) };
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(tL.x, tL.y); ctx.lineTo(tR.x, tR.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(bL.x, bL.y); ctx.lineTo(bR.x, bR.y); ctx.stroke();

    const midT = { x: (px[0].x + px[1].x) / 2, y: (px[0].y + px[1].y) / 2 };
    const midB = { x: (px[3].x + px[2].x) / 2, y: (px[3].y + px[2].y) / 2 };
    ctx.beginPath(); ctx.moveTo(midT.x, midT.y); ctx.lineTo(midB.x, midB.y); ctx.stroke();

    px.forEach((p, i) => {
      const radius = dragging === i ? 10 : 7;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = confirmed ? '#22C55E' : (dragging === i ? '#60A5FA' : '#2563EB');
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }, [pts, dragging, confirmed]);

  useEffect(() => { draw(); }, [draw]);

  const getCanvasPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (W / rect.width),
      y: (e.clientY - rect.top)  * (H / rect.height),
    };
  };

  const onMouseDown = e => {
    const pos = getCanvasPos(e);
    const hit = pts.findIndex(p =>
      Math.hypot(p.x * W - pos.x, p.y * H - pos.y) < 14
    );
    if (hit >= 0) { setConfirmed(false); setDragging(hit); }
  };

  const onMouseMove = e => {
    if (dragging === null) return;
    const pos = getCanvasPos(e);
    setPts(prev => prev.map((p, i) =>
      i === dragging
        ? { x: Math.max(0, Math.min(1, pos.x / W)), y: Math.max(0, Math.min(1, pos.y / H)) }
        : p
    ));
  };

  const onMouseUp = () => setDragging(null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <p style={{ fontSize: 13, color: t.muted, lineHeight: 1.6 }}>
        Drag the <span style={{ color: t.blue, fontWeight: 600 }}>four corner handles</span> to align the
        quadrilateral with the court boundary edges. This homography transform normalises inputs across varied camera angles.
      </p>

      <div style={{ position: 'relative' }}>
        <canvas
          ref={canvasRef}
          width={W} height={H}
          style={{
            borderRadius: 8, display: 'block', maxWidth: '100%',
            cursor: dragging !== null ? 'grabbing' : 'crosshair',
          }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        />
        <div style={{
          position: 'absolute', top: 8, left: 8,
          background: 'rgba(0,0,0,0.65)', color: '#fff',
          fontSize: 11, padding: '3px 9px', borderRadius: 4,
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          Frame 00:02:14 · drag handles to align
        </div>
        {confirmed && (
          <div style={{
            position: 'absolute', bottom: 8, left: 8,
            background: 'rgba(34,197,94,0.9)', color: '#fff',
            fontSize: 11, padding: '3px 9px', borderRadius: 4, fontWeight: 600,
          }}>
            ✓ Boundary confirmed
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: 10 }}>
        <Btn
          variant="secondary"
          onClick={() => { setConfirmed(false); setPts([{ x:0.17,y:0.14 },{ x:0.83,y:0.14 },{ x:0.89,y:0.84 },{ x:0.11,y:0.84 }]); }}
        >
          Reset
        </Btn>
        {!confirmed
          ? <Btn onClick={() => setConfirmed(true)}>Confirm Boundary</Btn>
          : <Btn onClick={() => onComplete(pts)}>Next: Select Player →</Btn>
        }
      </div>
    </div>
  );
}

/* ─── Step 2: Player Selection ───────────────────────────────────── */
function PlayerSelectionStep({ onComplete }) {
  const { t } = useTheme();
  const [selected, setSelected] = useState(null);

  const players = [
    { id: 'A', cx: '27%', cy: '38%', label: 'Player A', side: 'Near court', rotation: 0 },
    { id: 'B', cx: '65%', cy: '57%', label: 'Player B', side: 'Far court',  rotation: 12 },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <p style={{ fontSize: 13, color: t.muted, lineHeight: 1.6 }}>
        Click a player to <span style={{ color: t.blue, fontWeight: 600 }}>select the target subject</span> whose strokes will be classified.
        Object detection bounding boxes are shown automatically.
      </p>

      <div style={{
        position: 'relative', width: 620, maxWidth: '100%', height: 348,
        borderRadius: 8, overflow: 'hidden',
        background: 'linear-gradient(135deg, #C0882A, #D4A843 50%, #B07030)',
      }}>
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          backgroundImage: 'repeating-linear-gradient(91deg, rgba(0,0,0,0.06) 0px, transparent 1px, transparent 22px)',
        }} />

        <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} viewBox="0 0 620 348">
          <polygon points="105,49 515,49 553,299 67,299"
            fill="rgba(37,99,235,0.1)" stroke="rgba(255,255,255,0.5)" strokeWidth="1.5" />
          <line x1="310" y1="49" x2="310" y2="299" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
          <line x1="86" y1="174" x2="534" y2="174" stroke="rgba(255,255,255,0.6)" strokeWidth="2" />
          <line x1="141" y1="49" x2="133" y2="299" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
          <line x1="479" y1="49" x2="487" y2="299" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
          <rect x="200" y="49" width="220" height="125" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
          <rect x="200" y="174" width="220" height="125" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
        </svg>

        {players.map(p => (
          <div
            key={p.id}
            onClick={() => setSelected(p.id)}
            style={{
              position: 'absolute', left: p.cx, top: p.cy,
              transform: 'translate(-50%,-50%)',
              cursor: 'pointer',
            }}
          >
            <div style={{
              width: 58, height: 92,
              border: `2px solid ${selected === p.id ? '#22C55E' : '#3B82F6'}`,
              borderRadius: 4,
              background: selected === p.id ? 'rgba(34,197,94,0.18)' : 'rgba(37,99,235,0.14)',
              transition: 'all 0.15s',
              boxShadow: selected === p.id ? '0 0 18px rgba(34,197,94,0.45)' : '0 0 10px rgba(37,99,235,0.3)',
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            }}>
              <svg width="34" height="58" viewBox="0 0 34 58" style={{ transform: `rotate(${p.rotation}deg)`, opacity: 0.9 }}>
                <circle cx="17" cy="8"  r="6.5"  fill="white" />
                <line x1="17" y1="15" x2="17" y2="40" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
                <line x1="17" y1="23" x2="5"  y2="34" stroke="white" strokeWidth="2"   strokeLinecap="round" />
                <line x1="17" y1="23" x2="29" y2="19" stroke="white" strokeWidth="2"   strokeLinecap="round" />
                <line x1="17" y1="40" x2="11" y2="55" stroke="white" strokeWidth="2"   strokeLinecap="round" />
                <line x1="17" y1="40" x2="23" y2="55" stroke="white" strokeWidth="2"   strokeLinecap="round" />
              </svg>
            </div>
            <div style={{
              marginTop: 4, fontSize: 10, fontWeight: 700, textAlign: 'center',
              color: selected === p.id ? '#22C55E' : '#fff',
              textShadow: '0 1px 4px rgba(0,0,0,0.9)',
            }}>
              {p.label}
            </div>
            <div style={{
              position: 'absolute', top: -10, left: '50%', transform: 'translateX(-50%)',
              background: 'rgba(0,0,0,0.7)', color: '#fff',
              fontSize: 9, padding: '1px 5px', borderRadius: 3,
              fontFamily: "'JetBrains Mono', monospace", whiteSpace: 'nowrap',
            }}>
              det: {p.id === 'A' ? '0.97' : '0.94'}
            </div>
          </div>
        ))}

        <div style={{
          position: 'absolute', left: '46%', top: '28%',
          width: 10, height: 10, borderRadius: '50%',
          background: '#fff', border: '1.5px solid rgba(0,0,0,0.3)',
          boxShadow: '0 0 6px rgba(255,255,255,0.8)',
        }} />
        <div style={{
          position: 'absolute', left: 'calc(46% + 3px)', top: '29%',
          fontSize: 9, color: '#fff', fontFamily: 'JetBrains Mono, monospace',
          background: 'rgba(0,0,0,0.6)', padding: '1px 5px', borderRadius: 3,
        }}>shuttle</div>
      </div>

      {selected ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 13, color: t.success }}>
            ✓ {players.find(p => p.id === selected)?.label} selected ({players.find(p => p.id === selected)?.side})
          </span>
          <Btn onClick={() => onComplete(selected)}>Next: Set Timeframe →</Btn>
        </div>
      ) : (
        <p style={{ fontSize: 13, color: t.muted }}>Click a player above to select them.</p>
      )}
    </div>
  );
}

/* ─── Step 3: Timeframe ──────────────────────────────────────────── */
function TimeframeStep({ onComplete }) {
  const { t } = useTheme();
  const trackRef  = useRef(null);
  const [startPct, setStartPct] = useState(0.21);
  const [endPct,   setEndPct]   = useState(0.37);
  const [dragging, setDragging] = useState(null);
  const TOTAL = 4152;

  const fmt = s => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    return `${h}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
  };

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const applyMove = useCallback((e, which) => {
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const pct  = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    if (which === 'start') setStartPct(() => clamp(pct, 0, endPct - 0.02));
    else                   setEndPct(()   => clamp(pct, startPct + 0.02, 1));
  }, [startPct, endPct]);

  useEffect(() => {
    if (!dragging) return;
    const move = e => applyMove(e, dragging);
    const up   = () => setDragging(null);
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
  }, [dragging, applyMove]);

  const ANNOTATIONS = [
    { pct: 0.06, type: 'Smash'   },
    { pct: 0.13, type: 'Clear'   },
    { pct: 0.24, type: 'Drop'    },
    { pct: 0.31, type: 'Drive'   },
    { pct: 0.39, type: 'Net'     },
    { pct: 0.48, type: 'Lift'    },
    { pct: 0.57, type: 'Smash'   },
    { pct: 0.68, type: 'Clear'   },
    { pct: 0.76, type: 'Service' },
    { pct: 0.85, type: 'Drop'    },
    { pct: 0.92, type: 'Drive'   },
  ];

  const duration = ((endPct - startPct) * TOTAL).toFixed(1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <p style={{ fontSize: 13, color: t.muted, lineHeight: 1.6 }}>
        Drag the <span style={{ color: t.blue, fontWeight: 600 }}>start and end handles</span> to isolate the stroke segment.
        Gold markers indicate annotated stroke events from the ShuttleSet labels.
      </p>

      <div style={{ height: 72, borderRadius: 8, overflow: 'hidden', position: 'relative', border: `1px solid ${t.border}` }}>
        <div style={{ display: 'flex', height: '100%' }}>
          {Array.from({ length: 24 }).map((_, i) => (
            <div key={i} style={{
              flex: 1, borderRight: `1px solid rgba(0,0,0,0.15)`,
              background: `hsl(${28 + (i % 5) * 3}, ${46 + (i % 3) * 4}%, ${22 + Math.sin(i * 0.9) * 4}%)`,
            }} />
          ))}
        </div>
        <div style={{
          position: 'absolute', inset: 0,
          background: `linear-gradient(90deg,
            rgba(0,0,0,0.55) 0%,
            rgba(0,0,0,0.55) ${startPct * 100}%,
            transparent ${startPct * 100}%,
            transparent ${endPct * 100}%,
            rgba(0,0,0,0.55) ${endPct * 100}%,
            rgba(0,0,0,0.55) 100%)`,
          pointerEvents: 'none',
        }} />
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: `${startPct * 100}%`, width: `${(endPct - startPct) * 100}%`,
          border: `2px solid ${t.blue}`, borderRadius: 2, pointerEvents: 'none',
        }} />
      </div>

      <div
        ref={trackRef}
        style={{ position: 'relative', height: 44, userSelect: 'none', padding: '0 8px' }}
      >
        <div style={{
          position: 'absolute', top: '50%', left: 8, right: 8,
          height: 4, background: t.surface2, borderRadius: 2,
          transform: 'translateY(-50%)',
        }}>
          <div style={{
            position: 'absolute', top: 0, bottom: 0, borderRadius: 2,
            left: `${startPct * 100}%`,
            width: `${(endPct - startPct) * 100}%`,
            background: t.blue,
          }} />
        </div>

        {ANNOTATIONS.map((a, i) => (
          <div
            key={i}
            title={a.type}
            style={{
              position: 'absolute', top: '50%',
              left: `${a.pct * 100}%`,
              transform: 'translate(-50%, -50%)',
              width: 7, height: 7, borderRadius: '50%',
              background: t.pine,
              border: `1.5px solid ${t.bg}`,
              zIndex: 1,
            }}
          />
        ))}

        {['start', 'end'].map(which => (
          <div
            key={which}
            onMouseDown={() => setDragging(which)}
            style={{
              position: 'absolute', top: '50%',
              left: `${(which === 'start' ? startPct : endPct) * 100}%`,
              transform: 'translate(-50%, -50%)',
              width: 14, height: 30, borderRadius: 5,
              background: t.blue, cursor: 'ew-resize', zIndex: 3,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 2px 10px rgba(0,0,0,0.5)',
            }}
          >
            <div style={{ display: 'flex', gap: 2 }}>
              {[0,1].map(n => <div key={n} style={{ width: 1.5, height: 12, background: 'rgba(255,255,255,0.65)', borderRadius: 1 }} />)}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 12 }}>
        {[
          { label: 'Start',    value: fmt(startPct * TOTAL), color: t.text },
          { label: 'End',      value: fmt(endPct   * TOTAL), color: t.text },
          { label: 'Duration', value: `${duration}s`,        color: t.pine },
        ].map(s => (
          <div key={s.label} style={{ background: t.surface2, borderRadius: 7, padding: '9px 14px' }}>
            <div style={{ fontSize: 10, color: t.muted, marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{s.label}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono', monospace" }}>{s.value}</div>
          </div>
        ))}
        <div style={{ background: t.surface2, borderRadius: 7, padding: '9px 14px', marginLeft: 'auto' }}>
          <div style={{ fontSize: 10, color: t.muted, marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Strokes in range</div>
          <div style={{ fontSize: 14, fontWeight: 700, color: t.text, fontFamily: "'JetBrains Mono', monospace" }}>
            {ANNOTATIONS.filter(a => a.pct >= startPct && a.pct <= endPct).length}
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: t.muted }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: t.pine, flexShrink: 0 }} />
        Annotated stroke markers (ShuttleSet ground-truth labels)
      </div>

      <Btn onClick={() => onComplete({ startPct, endPct, duration: parseFloat(duration) })}>
        Confirm Timeframe →
      </Btn>
    </div>
  );
}

/* ─── Markup Shell ───────────────────────────────────────────────── */
export function MarkupScreen({ video, onNext, onBack }) {
  const { t } = useTheme();
  const [step, setStep]         = useState(0);
  const [boundary, setBoundary] = useState(null);
  const [player, setPlayer]     = useState(null);

  const STEPS = [
    { label: 'Court Boundary',   desc: 'Align perspective transform' },
    { label: 'Player Selection', desc: 'Identify target subject' },
    { label: 'Timeframe',        desc: 'Isolate stroke segment' },
  ];

  const content = [
    <CourtBoundaryStep onComplete={pts => { setBoundary(pts); setStep(1); }} />,
    <PlayerSelectionStep onComplete={p => { setPlayer(p); setStep(2); }} />,
    <TimeframeStep onComplete={tf => onNext({ video, boundary, player, timeframe: tf })} />,
  ];

  return (
    <div style={{ maxWidth: 780, margin: '0 auto', padding: 32 }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: t.text, marginBottom: 4 }}>Video Markup</h1>
        <p style={{ fontSize: 13, color: t.muted }}>{video?.match} · {video?.tournament}</p>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 28 }}>
        {STEPS.map((s, i) => {
          const done   = i < step;
          const active = i === step;
          return (
            // eslint-disable-next-line react/no-array-index-key
            <Fragment key={i}>
              <div
                onClick={() => i < step && setStep(i)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '8px 14px', borderRadius: 7, cursor: i < step ? 'pointer' : 'default',
                  background: active ? t.blueDim : 'transparent',
                  border: `1px solid ${active ? t.blue : done ? t.success + '60' : t.border}`,
                  color: active ? t.blue : done ? t.success : t.muted,
                  fontSize: 13, fontWeight: active ? 600 : 400,
                  transition: 'all 0.15s',
                }}
              >
                <span style={{
                  width: 18, height: 18, borderRadius: '50%', flexShrink: 0,
                  background: done ? t.success : active ? t.blue : 'transparent',
                  border: `1.5px solid ${done ? t.success : active ? t.blue : t.muted}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 9, fontWeight: 700, color: done || active ? '#fff' : t.muted,
                }}>
                  {done ? '✓' : i + 1}
                </span>
                <div>
                  <div style={{ fontSize: 12, lineHeight: 1.2 }}>{s.label}</div>
                  <div style={{ fontSize: 10, opacity: 0.7 }}>{s.desc}</div>
                </div>
              </div>
              {i < STEPS.length - 1 && (
                <div style={{ width: 20, height: 1, background: i < step ? t.success : t.border, flexShrink: 0 }} />
              )}
            </Fragment>
          );
        })}
      </div>

      <Card style={{ padding: 28 }}>
        {content[step]}
      </Card>

      <div style={{ marginTop: 16 }}>
        <Btn variant="secondary" onClick={step === 0 ? onBack : () => setStep(s => s - 1)}>
          ← Back
        </Btn>
      </div>
    </div>
  );
}
