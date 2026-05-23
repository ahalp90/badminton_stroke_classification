import { useState, useRef, useEffect, useCallback, Fragment } from 'react';
import { useTheme, Btn, Card } from './shared';
import { fmtTime } from './utils/format';
import { loadYouTubeAPI } from './utils/youtube'
import { Scrubber } from './components/Scrubber'
import { StrokePillStrip } from './components/StrokePillStrip';

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
  // sourceRef holds either an HTMLImageElement (library/youtube) or an
  // HTMLVideoElement (upload). drawImage accepts both, and we read intrinsic
  // dimensions via .naturalWidth/.videoWidth in the loupe code below.
  const sourceRef = useRef(null);
  const hiddenVideoRef = useRef(null);
  const isUpload = video?.source === 'upload' && !!video?.objectURL;

  const [pts, setPts] = useState([
    { x: 0.17, y: 0.30 },
    { x: 0.83, y: 0.30 },
    { x: 0.93, y: 0.92 },
    { x: 0.07, y: 0.92 },
  ]);
  const [dragging, setDragging] = useState(null);
  const [cursor, setCursor] = useState(null); // {x,y} in canvas coords while dragging
  const [confirmed, setConfirmed] = useState(false);
  const [sourceLabel, setSourceLabel] = useState('Reference frame · drag handles to align');

  useEffect(() => {
    if (isUpload) {
      const vid = hiddenVideoRef.current;
      if (!vid) return;
      const onReady = () => {
        sourceRef.current = vid;
        // Seek to a representative early frame (first frame can be black).
        try {
          const target = Math.min(0.5, (vid.duration || 1) / 2);
          if (Math.abs(vid.currentTime - target) > 0.01) vid.currentTime = target;
        } catch { /* noop */ }
        setSourceLabel(`Uploaded frame · ${video.filename || 'video'}`);
        draw();
      };
      const onSeeked = () => draw();
      if (vid.readyState >= 2) onReady();
      else vid.addEventListener('loadeddata', onReady, { once: true });
      vid.addEventListener('seeked', onSeeked);
      return () => {
        vid.removeEventListener('loadeddata', onReady);
        vid.removeEventListener('seeked', onSeeked);
      };
    }
    const src = frameUrl(video?.youtubeId);
    if (!src) return;
    const img = new Image();
    img.src = src;
    img.onload = () => { sourceRef.current = img; draw(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video?.youtubeId, video?.objectURL, isUpload]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    if (sourceRef.current) {
      try { ctx.drawImage(sourceRef.current, 0, 0, W, H); }
      catch { ctx.fillStyle = '#0E1422'; ctx.fillRect(0, 0, W, H); }
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
    if (!lc || dragging === null || !sourceRef.current) return;
    const ctx = lc.getContext('2d');
    const src = sourceRef.current;
    // Image uses naturalWidth/Height; Video uses videoWidth/Height.
    const intrW = src.naturalWidth || src.videoWidth || W;
    const intrH = src.naturalHeight || src.videoHeight || H;
    const p = pts[dragging];
    const cx = p.x * intrW;
    const cy = p.y * intrH;
    const cropSize = LOUPE_SIZE / LOUPE_ZOOM;
    ctx.clearRect(0, 0, LOUPE_SIZE, LOUPE_SIZE);
    ctx.save();
    ctx.beginPath();
    ctx.arc(LOUPE_SIZE / 2, LOUPE_SIZE / 2, LOUPE_SIZE / 2, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(
      src,
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
        {isUpload && (
          <video
            ref={hiddenVideoRef}
            src={video.objectURL}
            preload="auto"
            muted
            playsInline
            crossOrigin="anonymous"
            style={{ position: 'absolute', width: 1, height: 1, opacity: 0, pointerEvents: 'none' }}
          />
        )}
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
          maxWidth: 'calc(100% - 16px)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {sourceLabel}
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


/* ─── Step 2: Timeframe ──────────────────────────────────────────── */
// Default ±50-frame window at the implicit 30 fps demo rate (the markup
// contract doesn't carry fps, see configure-screen's buildMarkupPayload).
const DEFAULT_HALF_WINDOW_SEC = 50 / 30;

const newStrokeId = () => `a${Date.now()}${Math.floor(Math.random() * 1e4)}`;

function TimeframeStep({ video, onComplete }) {
  const { t } = useTheme();
  const isUpload = video?.source === 'upload' && !!video?.objectURL;
  const playerHostRef = useRef(null);
  const playerRef     = useRef(null);
  const videoElRef    = useRef(null);
  const [ready, setReady] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loaded, setLoaded] = useState(0);
  const [showPips, setShowPips] = useState(true);

  // Multi-stroke state: a list of annotations, each with seconds-based
  // start/target/end handles. The conversion to integer frames happens
  // later in configure-screen's buildMarkupPayload using the video's fps.
  // Initial id is a literal so both useState slots seed with the same
  // value without reading a ref during render (React 19 lint rule).
  const [annotations, setAnnotations] = useState(() => [
    { id: 'init', startSec: null, targetSec: null, endSec: null },
  ]);
  const [activeId, setActiveId] = useState('init');
  const [playerSide, setPlayerSide] = useState(null); // 'top' | 'bottom' | null
  const [pendingDeleteId, setPendingDeleteId] = useState(null);

  const active = annotations.find(a => a.id === activeId) || null;
  const startSec  = active?.startSec  ?? null;
  const targetSec = active?.targetSec ?? null;
  const endSec    = active?.endSec    ?? null;

  // HTML5 <video> path — uploaded files use the local objectURL.
  useEffect(() => {
    if (!isUpload) return;
    const vid = videoElRef.current;
    if (!vid) return;
    const onMeta = () => { setDuration(vid.duration || 0); setReady(true); };
    const onTime = () => setCurrentTime(vid.currentTime);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onProgress = () => {
      if (vid.buffered.length && vid.duration > 0) {
        setLoaded(vid.buffered.end(vid.buffered.length - 1) / vid.duration);
      }
    };
    vid.addEventListener('loadedmetadata', onMeta);
    vid.addEventListener('timeupdate', onTime);
    vid.addEventListener('play', onPlay);
    vid.addEventListener('pause', onPause);
    vid.addEventListener('ended', onPause);
    vid.addEventListener('progress', onProgress);
    // Smooth playhead between timeupdate events (which only fire ~4×/s).
    const pollId = setInterval(() => {
      if (videoElRef.current && !videoElRef.current.paused) {
        setCurrentTime(videoElRef.current.currentTime);
      }
    }, 100);
    if (vid.readyState >= 1) onMeta();
    return () => {
      clearInterval(pollId);
      vid.removeEventListener('loadedmetadata', onMeta);
      vid.removeEventListener('timeupdate', onTime);
      vid.removeEventListener('play', onPlay);
      vid.removeEventListener('pause', onPause);
      vid.removeEventListener('ended', onPause);
      vid.removeEventListener('progress', onProgress);
    };
  }, [isUpload, video?.objectURL]);

  // YouTube IFrame path — library matches keep the original behaviour.
  useEffect(() => {
    if (isUpload || !video?.youtubeId) return;
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
  }, [isUpload, video?.youtubeId]);

  const seekTo = (s) => {
    if (isUpload) {
      const vid = videoElRef.current;
      if (!vid) return;
      const clamped = Math.max(0, Math.min(duration || s, s));
      vid.currentTime = clamped;
      setCurrentTime(clamped);
    } else if (playerRef.current && playerRef.current.seekTo) {
      playerRef.current.seekTo(s, true);
    }
  };

  const nudge = (delta) => {
    if (isUpload) {
      const vid = videoElRef.current;
      if (!vid) return;
      const next = Math.max(0, Math.min(duration || Infinity, vid.currentTime + delta));
      vid.currentTime = next;
      setCurrentTime(next);
      return;
    }
    if (!playerRef.current) return;
    const now = playerRef.current.getCurrentTime?.() ?? 0;
    const next = Math.max(0, Math.min(duration || Infinity, now + delta));
    playerRef.current.seekTo(next, true);
    setCurrentTime(next);
  };

  const togglePlay = () => {
    if (isUpload) {
      const vid = videoElRef.current;
      if (!vid) return;
      if (vid.paused) vid.play().catch(() => { /* autoplay rejected — keep paused */ });
      else vid.pause();
      return;
    }
    if (!playerRef.current) return;
    if (playing) playerRef.current.pauseVideo?.();
    else playerRef.current.playVideo?.();
  };

  const getCurrentTimeNow = () =>
    isUpload
      ? (videoElRef.current?.currentTime ?? 0)
      : (playerRef.current?.getCurrentTime?.() ?? 0);

  const setHandle = (which) => {
    if (!active) return;
    const now = getCurrentTimeNow();
    setAnnotations(prev => prev.map(a => {
      if (a.id !== activeId) return a;
      let { startSec: s, targetSec: tg, endSec: e } = a;
      if (which === 'start') {
        s = now;
        if (tg !== null && tg < s) tg = s;
        if (e !== null && e < s) e = s;
      } else if (which === 'target') {
        const lo = s ?? 0;
        const hi = e ?? duration;
        tg = Math.max(lo, Math.min(hi, now));
      } else if (which === 'end') {
        e = now;
        if (tg !== null && tg > e) tg = e;
        if (s !== null && s > e) s = e;
      }
      return { ...a, startSec: s, targetSec: tg, endSec: e };
    }));
  };

  const reset = () => {
    setAnnotations(prev => prev.map(a =>
      a.id === activeId ? { ...a, startSec: null, targetSec: null, endSec: null } : a
    ));
  };

  const addStroke = () => {
    const cur = getCurrentTimeNow();
    const dur = duration || 0;
    const targetT = dur ? Math.max(0, Math.min(dur, cur)) : cur;
    const startT = Math.max(0, targetT - DEFAULT_HALF_WINDOW_SEC);
    const endT = dur ? Math.min(dur, targetT + DEFAULT_HALF_WINDOW_SEC) : targetT + DEFAULT_HALF_WINDOW_SEC;
    const id = newStrokeId();
    setAnnotations(prev => [...prev, { id, startSec: startT, targetSec: targetT, endSec: endT }]);
    setActiveId(id);
    setPendingDeleteId(null);
  };

  const requestDelete = (id) => {
    const a = annotations.find(x => x.id === id);
    if (!a) return;
    const populated = a.startSec !== null || a.targetSec !== null || a.endSec !== null;
    if (populated) {
      // Inline two-step confirm for non-empty strokes.
      setPendingDeleteId(id);
    } else {
      performDelete(id);
    }
  };

  const performDelete = (id) => {
    const remaining = annotations.filter(a => a.id !== id);
    // Always keep at least one slot available so the UI never hits an
    // empty list state with no add-button next to anything.
    if (remaining.length === 0) {
      const newId = newStrokeId();
      setAnnotations([{ id: newId, startSec: null, targetSec: null, endSec: null }]);
      setActiveId(newId);
    } else {
      setAnnotations(remaining);
      if (id === activeId) setActiveId(remaining[0].id);
    }
    setPendingDeleteId(null);
  };

  const isAnnotationComplete = (a) =>
    a.startSec !== null && a.targetSec !== null && a.endSec !== null;
  const isAnnotationValid = (a) =>
    isAnnotationComplete(a) && a.startSec <= a.targetSec && a.targetSec <= a.endSec;

  const populatedAnnotations = annotations.filter(a =>
    a.startSec !== null || a.targetSec !== null || a.endSec !== null
  );
  const allValid = populatedAnnotations.length >= 1
    && populatedAnnotations.every(isAnnotationValid);

  // Local convenience for the active-stroke summary panel below.
  const allSet = active ? isAnnotationComplete(active) : false;
  const valid  = active ? isAnnotationValid(active) : false;

  // Overlap warning: any two annotations whose [start, end] intervals overlap.
  const hasOverlap = (() => {
    const ranges = populatedAnnotations
      .filter(isAnnotationComplete)
      .map(a => [a.startSec, a.endSec])
      .sort((x, y) => x[0] - y[0]);
    for (let i = 1; i < ranges.length; i++) {
      if (ranges[i][0] < ranges[i - 1][1]) return true;
    }
    return false;
  })();

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
        {isUpload ? (
          <video
            ref={videoElRef}
            src={video.objectURL}
            preload="metadata"
            playsInline
            style={{ width: '100%', height: '100%', display: 'block', background: '#000' }}
          />
        ) : (
          <div ref={playerHostRef} style={{ width: '100%', height: '100%' }} />
        )}
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

      <StrokePillStrip
        annotations={annotations}
        activeId={activeId}
        onSelect={(id) => { setActiveId(id); setPendingDeleteId(null); }}
        onDelete={requestDelete}
        onAdd={addStroke}
        pendingDeleteId={pendingDeleteId}
        onConfirmDelete={performDelete}
        onCancelDelete={() => setPendingDeleteId(null)}
        isAnnotationComplete={isAnnotationComplete}
        isAnnotationValid={isAnnotationValid}
        ready={ready}
        t={t}
      />

      <Scrubber
        duration={duration}
        currentTime={currentTime}
        loaded={loaded}
        strokes={annotations}
        activeId={activeId}
        onSelectStroke={(id) => { setActiveId(id); setPendingDeleteId(null); }}
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
          Order must be start ≤ target ≤ end on the active stroke. Adjust handles before continuing.
        </div>
      )}

      {hasOverlap && (
        <div style={{
          background: t.surface2, color: t.warning, border: `1px solid ${t.warning}55`,
          padding: '8px 12px', borderRadius: 6, fontSize: 12,
        }}>
          ⚠ Two or more strokes overlap on the timeline. Allowed, but the
          backend will classify them independently.
        </div>
      )}

      {/* Player side — shared by all annotations for v1 (per-annotation
          player_side is a follow-up). */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        background: t.surface2, borderRadius: 8, padding: '8px 12px',
      }}>
        <span style={{ fontSize: 11, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Player side (applies to all strokes)
        </span>
        {[
          { v: 'top',    label: 'Top'    },
          { v: 'bottom', label: 'Bottom' },
        ].map(opt => {
          const sel = playerSide === opt.v;
          return (
            <button
              key={opt.v}
              onClick={() => setPlayerSide(sel ? null : opt.v)}
              style={{
                background: sel ? t.blue : t.surface,
                color: sel ? '#fff' : t.text,
                border: `1px solid ${sel ? t.blue : t.border}`,
                padding: '5px 12px', borderRadius: 5,
                fontSize: 12, fontWeight: 600, cursor: 'pointer',
                fontFamily: "'JetBrains Mono', monospace",
              }}
            >
              {opt.label}
            </button>
          );
        })}
        <span style={{ marginLeft: 'auto', fontSize: 11, color: t.muted }}>
          {playerSide ? `selected: ${playerSide}` : 'optional · leave unset to skip'}
        </span>
      </div>

      <Btn
        disabled={!allValid}
        onClick={() => onComplete({
          annotations: populatedAnnotations,
          playerSide,
        })}
      >
        Confirm {populatedAnnotations.length > 1
          ? `${populatedAnnotations.length} Strokes`
          : 'Timeframe'} →
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
  // `annotations` is the new shape: a list of {id, startSec, targetSec,
  // endSec}. Conversion to integer frames + player_side broadcast happens
  // in configure-screen's buildMarkupPayload right before the API call.
  const buildMarkupPayload = (out) => ({
    video,
    boundary,
    orientation: ORIENTATION,
    annotations: out.annotations,
    playerSide: out.playerSide,
  });

  const content = [
    <CourtBoundaryStep video={video} onComplete={pts => { setBoundary(pts); setStep(1); }} />,
    <TimeframeStep video={video} onComplete={out => onNext(buildMarkupPayload(out))} />,
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
