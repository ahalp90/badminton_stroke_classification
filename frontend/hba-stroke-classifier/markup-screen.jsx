import { useState, useRef, useEffect, useCallback, Fragment } from 'react';
import { useTheme, Btn, Card } from './shared';
import { fmtTime } from './utils/format';
import { loadYouTubeAPI } from './utils/youtube'

const frameModules = import.meta.glob('./data/frames/*.jpg', { eager: true, import: 'default' });
const frameUrl = (id) => frameModules[`./data/frames/${id}.jpg`];

/* ─── Step 1: Court Boundary ─────────────────────────────────────── */
function CourtBoundaryStep({ video, onComplete }) {
  const { t } = useTheme();
  const canvasRef = useRef(null);
  const loupeRef = useRef(null);
  const W = 640, H = 360;
  const LOUPE_SIZE = 130;
  const LOUPE_ZOOM = 4;
  const imgRef = useRef(null);

  const [pts, setPts] = useState([
    { x: 0.17, y: 0.30 },
    { x: 0.83, y: 0.30 },
    { x: 0.93, y: 0.92 },
    { x: 0.07, y: 0.92 },
  ]);
  const [dragging, setDragging] = useState(null);
  const [cursor, setCursor] = useState(null); // {x,y} in canvas coords while dragging
  const [confirmed, setConfirmed] = useState(false);

  useEffect(() => {
    const src = frameUrl(video?.youtubeId);
    if (!src) return;
    const img = new Image();
    img.src = src;
    img.onload = () => { imgRef.current = img; draw(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video?.youtubeId]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    if (imgRef.current) {
      ctx.drawImage(imgRef.current, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#0E1422';
      ctx.fillRect(0, 0, W, H);
    }

    const px = pts.map(p => ({ x: p.x * W, y: p.y * H }));

    ctx.beginPath();
    px.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.fillStyle = confirmed ? 'rgba(34,197,94,0.18)' : 'rgba(37,99,235,0.18)';
    ctx.fill();
    ctx.strokeStyle = confirmed ? '#22C55E' : '#3B82F6';
    ctx.lineWidth = 2;
    ctx.stroke();

    px.forEach((p, i) => {
      const radius = dragging === i ? 11 : 8;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = confirmed ? '#22C55E' : (dragging === i ? '#60A5FA' : '#2563EB');
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    });
  }, [pts, dragging, confirmed]);

  useEffect(() => { draw(); }, [draw]);

  // Render the magnifier loupe whenever the dragged corner moves.
  useEffect(() => {
    const lc = loupeRef.current;
    if (!lc || dragging === null || !imgRef.current) return;
    const ctx = lc.getContext('2d');
    const img = imgRef.current;
    const p = pts[dragging];
    const cx = p.x * img.naturalWidth;
    const cy = p.y * img.naturalHeight;
    const cropSize = LOUPE_SIZE / LOUPE_ZOOM;
    ctx.clearRect(0, 0, LOUPE_SIZE, LOUPE_SIZE);
    ctx.save();
    ctx.beginPath();
    ctx.arc(LOUPE_SIZE / 2, LOUPE_SIZE / 2, LOUPE_SIZE / 2, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(
      img,
      cx - cropSize / 2, cy - cropSize / 2, cropSize, cropSize,
      0, 0, LOUPE_SIZE, LOUPE_SIZE
    );
    ctx.restore();
    // Crosshair
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(LOUPE_SIZE / 2, LOUPE_SIZE / 2 - 10);
    ctx.lineTo(LOUPE_SIZE / 2, LOUPE_SIZE / 2 + 10);
    ctx.moveTo(LOUPE_SIZE / 2 - 10, LOUPE_SIZE / 2);
    ctx.lineTo(LOUPE_SIZE / 2 + 10, LOUPE_SIZE / 2);
    ctx.stroke();
  }, [pts, dragging]);

  const getCanvasPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (W / rect.width),
      y: (e.clientY - rect.top) * (H / rect.height),
    };
  };

  const onMouseDown = e => {
    const pos = getCanvasPos(e);
    let nearest = 0, nearestDist = Infinity;
    pts.forEach((p, i) => {
      const d = Math.hypot(p.x * W - pos.x, p.y * H - pos.y);
      if (d < nearestDist) { nearest = i; nearestDist = d; }
    });
    setConfirmed(false);
    setDragging(nearest);
    setCursor(pos);
    if (nearestDist >= 16) {
      // Snap the nearest corner to the click point, then allow dragging.
      setPts(prev => prev.map((p, i) =>
        i === nearest
          ? { x: Math.max(0, Math.min(1, pos.x / W)), y: Math.max(0, Math.min(1, pos.y / H)) }
          : p
      ));
    }
  };

  const onMouseMove = e => {
    if (dragging === null) return;
    const pos = getCanvasPos(e);
    setCursor(pos);
    setPts(prev => prev.map((p, i) =>
      i === dragging
        ? { x: Math.max(0, Math.min(1, pos.x / W)), y: Math.max(0, Math.min(1, pos.y / H)) }
        : p
    ));
  };

  const onMouseUp = () => { setDragging(null); setCursor(null); };

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
            background: '#000',
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
          Reference frame · drag handles to align
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
        {dragging !== null && cursor && (() => {
          // Position loupe diagonally offset from the cursor; flip if near edges.
          const offset = LOUPE_SIZE / 2 + 12;
          const flipX = cursor.x > W - LOUPE_SIZE - 20;
          const flipY = cursor.y < LOUPE_SIZE + 20;
          const left = (cursor.x / W) * 100;
          const top = (cursor.y / H) * 100;
          return (
            <canvas
              ref={loupeRef}
              width={LOUPE_SIZE}
              height={LOUPE_SIZE}
              style={{
                position: 'absolute',
                left: `calc(${left}% + ${flipX ? -offset : offset}px)`,
                top: `calc(${top}% + ${flipY ? offset : -offset}px)`,
                transform: 'translate(-50%, -50%)',
                width: LOUPE_SIZE, height: LOUPE_SIZE,
                borderRadius: '50%',
                border: '2px solid rgba(255,255,255,0.9)',
                boxShadow: '0 4px 18px rgba(0,0,0,0.55)',
                pointerEvents: 'none',
                background: '#000',
              }}
            />
          );
        })()}
      </div>

      <div style={{ display: 'flex', gap: 10 }}>
        <Btn
          variant="secondary"
          onClick={() => {
            setConfirmed(false);
            setPts([
              { x: 0.17, y: 0.30 }, { x: 0.83, y: 0.30 },
              { x: 0.93, y: 0.92 }, { x: 0.07, y: 0.92 },
            ]);
          }}
        >
          Reset
        </Btn>
        {!confirmed
          ? <Btn onClick={() => setConfirmed(true)}>Confirm Boundary</Btn>
          : <Btn onClick={() => onComplete(pts)}>Next: Set Timeframe →</Btn>
        }
      </div>
    </div>
  );
}

/* ─── Scrubber: buffered + density-binned pips + click-drag seek ─── */
function Scrubber({
  duration, currentTime, loaded,
  startSec, endSec, targetSec,
  strokeTimes, showPips, onSeek, t,
}) {
  const trackRef = useRef(null);
  const scrollRef = useRef(null);
  const draggingRef = useRef(false);
  const [zoom, setZoom] = useState(1);
  const ZOOM_LEVELS = [1, 2, 5, 10, 25, 50];
  const N_BUCKETS = 200 * zoom;

  const buckets = (() => {
    if (!duration || !strokeTimes.length) return null;
    const arr = new Array(N_BUCKETS).fill(0);
    for (const s of strokeTimes) {
      const i = Math.min(N_BUCKETS - 1, Math.max(0, Math.floor((s / duration) * N_BUCKETS)));
      arr[i]++;
    }
    return arr;
  })();
  const bucketMax = buckets ? Math.max(1, ...buckets) : 1;

  const pct = (s) => duration > 0 ? (s / duration) * 100 : 0;

  const seekFromEvent = (e) => {
    const track = trackRef.current;
    if (!track || !duration) return;
    const rect = track.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(f * duration);
  };

  const onMouseDown = (e) => {
    draggingRef.current = true;
    seekFromEvent(e);
    const move = (ev) => { if (draggingRef.current) seekFromEvent(ev); };
    const up = () => {
      draggingRef.current = false;
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  };

  const TRACK_HEIGHT = 38;

  // Auto-scroll to keep the playhead visible when zoomed.
  useEffect(() => {
    const sc = scrollRef.current;
    if (!sc || !duration || draggingRef.current) return;
    const playheadPx = (currentTime / duration) * sc.scrollWidth;
    const visibleLeft = sc.scrollLeft;
    const visibleRight = visibleLeft + sc.clientWidth;
    const margin = sc.clientWidth * 0.1;
    if (playheadPx < visibleLeft + margin || playheadPx > visibleRight - margin) {
      sc.scrollLeft = playheadPx - sc.clientWidth / 2;
    }
  }, [currentTime, duration, zoom]);

  return (
    <div style={{ userSelect: 'none' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8,
        fontSize: 11, color: t.muted,
      }}>
        <span style={{ textTransform: 'uppercase', letterSpacing: '0.05em', marginRight: 2 }}>
          Zoom
        </span>
        {ZOOM_LEVELS.map(z => (
          <button
            key={z}
            onClick={() => setZoom(z)}
            style={{
              background: zoom === z ? t.blue : t.surface2,
              color: zoom === z ? '#fff' : t.text,
              border: `1px solid ${zoom === z ? t.blue : t.border}`,
              padding: '3px 9px', borderRadius: 4,
              fontSize: 11, fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace",
              cursor: 'pointer',
            }}
          >
            {z}×
          </button>
        ))}
      </div>

      <div
        ref={scrollRef}
        style={{
          overflowX: zoom > 1 ? 'auto' : 'hidden',
          overflowY: 'hidden',
          padding: '12px 8px 4px',
        }}
      >
        <div style={{ position: 'relative', width: `${zoom * 100}%` }}>
      {/* Density histogram (annotation pips, binned) */}
      {showPips && buckets && (
        <div
          aria-hidden
          style={{
            position: 'absolute', left: 0, right: 0, top: 0, height: 20,
            display: 'flex', alignItems: 'flex-end', pointerEvents: 'none',
          }}
        >
          {buckets.map((c, i) => (
            <div key={i} style={{
              flex: 1, height: c ? `${(c / bucketMax) * 100}%` : 0,
              background: t.pine, opacity: 0.55,
              minHeight: c ? 2 : 0,
            }} />
          ))}
        </div>
      )}

      {/* Track (click + drag to seek) */}
      <div
        ref={trackRef}
        onMouseDown={onMouseDown}
        style={{
          position: 'relative', height: TRACK_HEIGHT, marginTop: showPips ? 2 : 0,
          cursor: duration > 0 ? 'pointer' : 'default',
        }}
      >
        <div style={{
          position: 'absolute', top: '50%', left: 0, right: 0, height: 8,
          background: t.surface2, borderRadius: 4, transform: 'translateY(-50%)',
          overflow: 'hidden',
        }}>
          {loaded > 0 && (
            <div style={{
              position: 'absolute', top: 0, bottom: 0, left: 0,
              width: `${loaded * 100}%`, background: t.muted, opacity: 0.35,
            }} />
          )}
          {startSec !== null && endSec !== null && (
            <div style={{
              position: 'absolute', top: 0, bottom: 0, borderRadius: 2,
              left: `${pct(startSec)}%`, width: `${Math.max(0, pct(endSec) - pct(startSec))}%`,
              background: t.blue,
            }} />
          )}
        </div>

        {/* Playhead */}
        {duration > 0 && (
          <div style={{
            position: 'absolute', top: '50%', left: `${pct(currentTime)}%`,
            width: 2, height: 26, background: t.text,
            transform: 'translate(-50%, -50%)', pointerEvents: 'none',
            boxShadow: '0 0 4px rgba(0,0,0,0.6)',
          }} />
        )}

        {/* S / ◉ / E markers (click to seek) */}
        {[
          { val: startSec,  label: 'S', color: t.blue },
          { val: targetSec, label: '◉', color: t.warning },
          { val: endSec,    label: 'E', color: t.blue },
        ].map((h) => h.val !== null && (
          <div
            key={h.label}
            onClick={(e) => { e.stopPropagation(); onSeek(h.val); }}
            onMouseDown={(e) => e.stopPropagation()}
            title={`Seek to ${fmtTime(h.val)}`}
            style={{
              position: 'absolute', top: '50%', left: `${pct(h.val)}%`,
              transform: 'translate(-50%, -50%)',
              width: 16, height: 26, borderRadius: 4,
              background: h.color, color: '#fff',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 10, fontWeight: 700,
              boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
            }}
          >
            {h.label}
          </div>
        ))}
      </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Step 2: Timeframe ──────────────────────────────────────────── */
function TimeframeStep({ video, onComplete }) {
  const { t } = useTheme();
  const playerHostRef = useRef(null);
  const playerRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loaded, setLoaded] = useState(0);
  const [showPips, setShowPips] = useState(true);

  const [startSec, setStartSec] = useState(null);
  const [targetSec, setTargetSec] = useState(null);
  const [endSec, setEndSec] = useState(null);

  useEffect(() => {
    if (!video?.youtubeId) return;
    let player = null;
    let pollId = null;
    let cancelled = false;

    loadYouTubeAPI().then(YT => {
      if (cancelled || !YT || !playerHostRef.current) return;
      player = new YT.Player(playerHostRef.current, {
        videoId: video.youtubeId,
        playerVars: { rel: 0, modestbranding: 1, playsinline: 1 },
        events: {
          onReady: (e) => {
            if (cancelled) return;
            setDuration(e.target.getDuration());
            setReady(true);
            playerRef.current = e.target;
            pollId = setInterval(() => {
              if (playerRef.current && playerRef.current.getCurrentTime) {
                setCurrentTime(playerRef.current.getCurrentTime());
                if (playerRef.current.getVideoLoadedFraction) {
                  setLoaded(playerRef.current.getVideoLoadedFraction());
                }
              }
            }, 250);
          },
          onStateChange: (e) => {
            // 1 = playing, 2 = paused, 0 = ended, 3 = buffering
            if (e.data === 1) setPlaying(true);
            else if (e.data === 2 || e.data === 0) setPlaying(false);
          },
        },
      });
    });

    return () => {
      cancelled = true;
      if (pollId) clearInterval(pollId);
      try { player && player.destroy && player.destroy(); } catch { /* noop */ }
      playerRef.current = null;
    };
  }, [video?.youtubeId]);

  const seekTo = (s) => {
    if (playerRef.current && playerRef.current.seekTo) {
      playerRef.current.seekTo(s, true);
    }
  };

  const nudge = (delta) => {
    if (!playerRef.current) return;
    const now = playerRef.current.getCurrentTime?.() ?? 0;
    const next = Math.max(0, Math.min(duration || Infinity, now + delta));
    playerRef.current.seekTo(next, true);
    setCurrentTime(next);
  };

  const togglePlay = () => {
    if (!playerRef.current) return;
    if (playing) playerRef.current.pauseVideo?.();
    else playerRef.current.playVideo?.();
  };

  const setHandle = (which) => {
    const t = playerRef.current?.getCurrentTime?.() ?? 0;
    if (which === 'start') {
      setStartSec(t);
      if (targetSec !== null && targetSec < t) setTargetSec(t);
      if (endSec !== null && endSec < t) setEndSec(t);
    } else if (which === 'target') {
      const lo = startSec ?? 0;
      const hi = endSec ?? duration;
      setTargetSec(Math.max(lo, Math.min(hi, t)));
    } else if (which === 'end') {
      setEndSec(t);
      if (targetSec !== null && targetSec > t) setTargetSec(t);
      if (startSec !== null && startSec > t) setStartSec(t);
    }
  };

  const reset = () => { setStartSec(null); setTargetSec(null); setEndSec(null); };

  const allSet = startSec !== null && targetSec !== null && endSec !== null;
  const valid = allSet && startSec <= targetSec && targetSec <= endSec;

  // Track visualisation: start–end span + target marker
  const pct = (s) => duration > 0 ? (s / duration) * 100 : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <p style={{ fontSize: 13, color: t.muted, lineHeight: 1.6 }}>
        Scrub the video to the moment you want, then mark the
        <span style={{ color: t.blue, fontWeight: 600 }}> start</span>,
        <span style={{ color: t.warning, fontWeight: 600 }}> target hit frame</span>, and
        <span style={{ color: t.blue, fontWeight: 600 }}> end</span> of the stroke segment.
        The classifier will receive the window between start and end, with the target frame as the predicted hit moment.
      </p>

      <div style={{
        position: 'relative', width: '100%', aspectRatio: '16 / 9',
        background: '#000', borderRadius: 8, overflow: 'hidden',
      }}>
        <div ref={playerHostRef} style={{ width: '100%', height: '100%' }} />
        {!ready && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            color: t.muted, fontSize: 13,
          }}>
            Loading video…
          </div>
        )}
      </div>

      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        background: t.surface2, borderRadius: 8, padding: '6px 10px',
      }}>
        <button
          onClick={togglePlay}
          disabled={!ready}
          aria-label={playing ? 'Pause' : 'Play'}
          style={{
            background: t.blue, border: 'none',
            color: '#fff', width: 32, height: 28, borderRadius: 5,
            fontSize: 13, fontWeight: 700, cursor: ready ? 'pointer' : 'not-allowed',
            opacity: ready ? 1 : 0.4, marginRight: 6,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          {playing ? '❚❚' : '▶'}
        </button>
        <span style={{ fontSize: 11, color: t.muted, marginRight: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Nudge
        </span>
        {[
          { label: '−1s',   d: -1 },
          { label: '−0.1s', d: -0.1 },
          { label: '+0.1s', d: 0.1 },
          { label: '+1s',   d: 1 },
        ].map(b => (
          <button
            key={b.label}
            onClick={() => nudge(b.d)}
            disabled={!ready}
            style={{
              background: t.surface, border: `1px solid ${t.border}`,
              color: t.text, padding: '5px 10px', borderRadius: 5,
              fontSize: 12, fontWeight: 600, cursor: ready ? 'pointer' : 'not-allowed',
              opacity: ready ? 1 : 0.4,
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            {b.label}
          </button>
        ))}
        <label style={{
          marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6,
          fontSize: 11, color: t.muted, cursor: 'pointer',
        }}>
          <input
            type="checkbox"
            checked={showPips}
            onChange={e => setShowPips(e.target.checked)}
            style={{ accentColor: t.pine }}
          />
          Show annotation markers
        </label>
        <div style={{ fontSize: 12, color: t.muted, fontFamily: "'JetBrains Mono', monospace" }}>
          {fmtTime(currentTime)} / {fmtTime(duration)}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <Btn variant="secondary" size="sm" onClick={() => setHandle('start')} disabled={!ready}>
          ⟨ Set start
        </Btn>
        <Btn variant="secondary" size="sm" onClick={() => setHandle('target')} disabled={!ready}>
          ◉ Set target frame
        </Btn>
        <Btn variant="secondary" size="sm" onClick={() => setHandle('end')} disabled={!ready}>
          Set end ⟩
        </Btn>
        <Btn variant="ghost" size="sm" onClick={reset} disabled={!ready}>
          Reset
        </Btn>
      </div>

      <Scrubber
        duration={duration}
        currentTime={currentTime}
        loaded={loaded}
        startSec={startSec}
        endSec={endSec}
        targetSec={targetSec}
        strokeTimes={video?.strokeTimes || []}
        showPips={showPips}
        onSeek={seekTo}
        t={t}
      />

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {[
          { label: 'Start',  value: startSec  !== null ? fmtTime(startSec)  : '—', color: t.text },
          { label: 'Target', value: targetSec !== null ? fmtTime(targetSec) : '—', color: t.warning },
          { label: 'End',    value: endSec    !== null ? fmtTime(endSec)    : '—', color: t.text },
          {
            label: 'Window',
            value: allSet ? `${(endSec - startSec).toFixed(1)}s` : '—',
            color: t.pine,
          },
        ].map(s => (
          <div key={s.label} style={{ background: t.surface2, borderRadius: 7, padding: '9px 14px' }}>
            <div style={{ fontSize: 10, color: t.muted, marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{s.label}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono', monospace" }}>{s.value}</div>
          </div>
        ))}
      </div>

      {allSet && !valid && (
        <div style={{
          background: t.dangerDim, color: t.danger, border: `1px solid ${t.danger}`,
          padding: '8px 12px', borderRadius: 6, fontSize: 12,
        }}>
          Order must be start ≤ target ≤ end. Adjust handles before continuing.
        </div>
      )}

      <Btn
        disabled={!valid}
        onClick={() => onComplete({
          startSec, targetSec, endSec,
          duration: endSec - startSec,
        })}
      >
        Confirm Timeframe →
      </Btn>
    </div>
  );
}

/* ─── Markup Shell ───────────────────────────────────────────────── */
// `orientation` is fixed to 'portrait' for v1: every official badminton
// broadcast camera is portrait. See frontend_integration_handoff.md §
// "About corners" for the contract.
const ORIENTATION = 'portrait';

export function MarkupScreen({ video, onNext, onBack }) {
  const { t } = useTheme();
  const [step, setStep] = useState(0);
  const [boundary, setBoundary] = useState(null);

  const STEPS = [
    { label: 'Court Boundary', desc: 'Align perspective transform' },
    { label: 'Timeframe',      desc: 'Isolate stroke segment' },
  ];

  // Tier 3 contract: backend wants `corners` (4 normalised xy points) plus
  // an `orientation` flag. Click order doesn't matter — backend re-sorts.
  const buildMarkupPayload = (tf) => ({
    video,
    boundary,
    orientation: ORIENTATION,
    timeframe: tf,
  });

  const content = [
    <CourtBoundaryStep video={video} onComplete={pts => { setBoundary(pts); setStep(1); }} />,
    <TimeframeStep video={video} onComplete={tf => onNext(buildMarkupPayload(tf))} />,
  ];

  return (
    <div style={{ maxWidth: 780, margin: '0 auto', padding: 32 }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: t.text, marginBottom: 4 }}>Video Markup</h1>
        <p style={{ fontSize: 13, color: t.muted }}>{video?.match} · {video?.tournament}</p>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 28 }}>
        {STEPS.map((s, i) => {
          const done = i < step;
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

      {boundary && (
        <div style={{
          background: t.surface2, border: `1px solid ${t.border}`,
          borderRadius: 7, padding: '8px 12px', marginBottom: 12,
          fontSize: 11, color: t.muted,
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          captured · {boundary.length} corners · orientation {ORIENTATION}
        </div>
      )}

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
