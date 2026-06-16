import { useState, useEffect, useRef } from 'react';
import { useTheme, Btn } from '../shared';
import { fmtTime } from '../utils/format';
import { loadYouTubeAPI } from '../utils/youtube';
import { Scrubber } from './Scrubber';
import { StrokePillStrip } from './StrokePillStrip';

// ──── Constants ──────────────────────────────────────────────────────────────────────────────────
// Default half-window each side of the target hit, in seconds. 1.5s matches the
// model's training clip window (between_2_hits_with_max_limits clamps to 1.5s per
// side), so auto-windowed inputs stay in-distribution. The markup contract carries
// seconds, not frames (see configure-screen's buildMarkupPayload).
const DEFAULT_HALF_WINDOW_SEC = 1.5;
// User-selectable half-window sizes (seconds each side of the target).
const WINDOW_OPTIONS = [1, 1.5, 2];

/** Generates a unique id for a new stroke annotation. */
const newStrokeId = () => `a${Date.now()}${Math.floor(Math.random() * 1e4)}`;

/** Step 2 of markup: video scrubber for marking stroke start, target and end times.
 *  Supports both uploaded videos (HTML5) and library matches (YouTube IFrame). */
export function TimeframeStep({ video, onComplete }) {
  const { t } = useTheme();
  const isUpload = video?.source === 'upload' && !!video?.objectURL;

  // ──── Refs ─────────────────────────────────────────────────────────────────────────────────────
  const playerHostRef = useRef(null);
  const playerRef     = useRef(null);
  const videoElRef    = useRef(null);

  // ──── Video player state ───────────────────────────────────────────────────────────────────────
  const [ready, setReady] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loaded, setLoaded] = useState(0);

  // ──── Annotation state ─────────────────────────────────────────────────────────────────────────
  // Multi-stroke state: a list of annotations, each with seconds-based start/target/end handles. 
  // The conversion to integer frames happens later in configure-screen's buildMarkupPayload using the 
  // video's fps. Initial id is a literal so both useState slots seed with the same value without 
  // reading a ref during render (React 19 lint rule).
  const [annotations, setAnnotations] = useState(() => [
    { id: 'init', startSec: null, targetSec: null, endSec: null },
  ]);
  const [activeId, setActiveId] = useState('init');
  const [playerSide, setPlayerSide] = useState(null); // 'top' | 'bottom' | null
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [zoom, setZoom] = useState(1);
  // Half-window (seconds each side of the target) used when auto-filling
  // start/end from the target hit, so the user mostly just marks the target.
  const [halfWindowSec, setHalfWindowSec] = useState(DEFAULT_HALF_WINDOW_SEC);

  const active = annotations.find(a => a.id === activeId) || null;
  const startSec  = active?.startSec  ?? null;
  const targetSec = active?.targetSec ?? null;
  const endSec    = active?.endSec    ?? null;

  // ──── Video setup ──────────────────────────────────────────────────────────────────────────────
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

  // ──── Video player controls ────────────────────────────────────────────────────────────────────
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

  // ──── Annotation handlers ──────────────────────────────────────────────────────────────────────
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
        if (s === null && e === null) {
          // Fresh stroke: the target defines the window. Auto-fill start/end to
          // ±halfWindowSec so one click gives a complete, valid window centered
          // on the hit (both handles stay adjustable afterwards).
          tg = Math.max(0, Math.min(duration || now, now));
          s = Math.max(0, tg - halfWindowSec);
          e = duration ? Math.min(duration, tg + halfWindowSec) : tg + halfWindowSec;
        } else {
          // Existing window: clamp the target inside [start, end].
          const lo = s ?? 0;
          const hi = e ?? duration;
          tg = Math.max(lo, Math.min(hi, now));
        }
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
    const startT = Math.max(0, targetT - halfWindowSec);
    const endT = dur ? Math.min(dur, targetT + halfWindowSec) : targetT + halfWindowSec;
    const id = newStrokeId();
    setAnnotations(prev => [...prev, { id, startSec: startT, targetSec: targetT, endSec: endT }]);
    setActiveId(id);
    setPendingDeleteId(null);
  };

  // Change the auto-window size and re-center the active stroke's start/end on
  // its target, so the new window is reflected immediately (no effect on
  // strokes without a target yet).
  const setWindow = (hw) => {
    setHalfWindowSec(hw);
    setAnnotations(prev => prev.map(a => {
      if (a.id !== activeId || a.targetSec === null) return a;
      const tg = a.targetSec;
      return {
        ...a,
        startSec: Math.max(0, tg - hw),
        endSec: duration ? Math.min(duration, tg + hw) : tg + hw,
      };
    }));
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

  // ──── Derived state ────────────────────────────────────────────────────────────────────────────
  const isAnnotationComplete = (a) =>
    a.startSec !== null && a.targetSec !== null && a.endSec !== null;
  const isAnnotationValid = (a) =>
    isAnnotationComplete(a) && a.startSec <= a.targetSec && a.targetSec <= a.endSec;

  const populatedAnnotations = annotations.filter(a =>
    a.startSec !== null || a.targetSec !== null || a.endSec !== null
  );
  const allValid = populatedAnnotations.length >= 1
    && populatedAnnotations.every(isAnnotationValid)
    && playerSide != null;

  // Local convenience for the active-stroke summary panel below.
  const allSet = active ? isAnnotationComplete(active) : false;
  const valid  = active ? isAnnotationValid(active) : false;

  // Display-only preview of the backend's window derivation: Start = previous
  // stroke's target, End = next stroke's target, falling back to ±0.5 s at the
  // rally's edges, clamped to [0, duration]. The backend's actual windows add
  // a small extension and ±1.5 s caps (compute_clip_bounds), so this is an
  // approximation for the summary cards, not the exact model input.
  const DEFAULT_HALF_WINDOW = 0.5;
  const sortedByTarget = (annotations || [])
    .filter(a => a.targetSec != null)
    .slice()
    .sort((a, b) => a.targetSec - b.targetSec);
  const activeIdx = sortedByTarget.findIndex(a => a.id === activeId);

  let derivedStart = null;
  let derivedEnd = null;
  if (active?.targetSec != null && activeIdx >= 0) {
    const rawStart = activeIdx > 0
      ? sortedByTarget[activeIdx - 1].targetSec
      : active.targetSec - DEFAULT_HALF_WINDOW;
    const rawEnd = activeIdx < sortedByTarget.length - 1
      ? sortedByTarget[activeIdx + 1].targetSec
      : active.targetSec + DEFAULT_HALF_WINDOW;
    derivedStart = Math.max(0, rawStart);
    derivedEnd = duration > 0 ? Math.min(duration, rawEnd) : rawEnd;
  }
  const derivedWindow = derivedStart != null && derivedEnd != null
    ? derivedEnd - derivedStart
    : null;

  // ──── Render ───────────────────────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <p style={{ fontSize: 13, color: t.muted, lineHeight: 1.6 }}>
        Scrub the video to the moment of each stroke, then mark the 
        <span style={{ color: t.warning, fontWeight: 600 }}> target shot frame</span>.
        The classifier processes each marked stroke independently; the window between
        consecutive markers defines what the model sees per stroke.
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
        <div style={{ fontSize: 12, color: t.muted, fontFamily: "'JetBrains Mono', monospace" }}>
          {fmtTime(currentTime)} / {fmtTime(duration)}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <Btn variant="secondary" size="sm" onClick={() => setHandle('target')} disabled={!ready}>
          ◉ Set target shot
        </Btn>
        <Btn variant="ghost" size="sm" onClick={reset} disabled={!ready}>
          Reset
        </Btn>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 11, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Zoom
          </span>
          {[1, 2, 5, 10, 25, 50].map(z => (
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
      </div>

      

      <Scrubber
        duration={duration}
        currentTime={currentTime}
        loaded={loaded}
        strokes={annotations}
        activeId={activeId}
        onSelectStroke={(id) => { setActiveId(id); setPendingDeleteId(null); }}
        strokeTimes={video?.strokeTimes || []}
        showPips={true}
        onSeek={seekTo}
        zoom={zoom}
        t={t}
      />


      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {[
          { label: 'Start',  value: derivedStart  !== null ? fmtTime(derivedStart)  : '—', color: t.text },
          { label: 'Target', value: targetSec     !== null ? fmtTime(targetSec)     : '—', color: t.warning },
          { label: 'End',    value: derivedEnd    !== null ? fmtTime(derivedEnd)    : '—', color: t.text },
          { label: 'Window', value: derivedWindow !== null ? `${derivedWindow.toFixed(2)}s` : '—', color: t.pine },
        ].map(s => (
          <div key={s.label} style={{ background: t.surface2, borderRadius: 7, padding: '9px 14px' }}>
            <div style={{ fontSize: 10, color: t.muted, marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{s.label}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono', monospace" }}>{s.value}</div>
          </div>
        ))}
      </div>

      <StrokePillStrip
        annotations={annotations}
        activeId={activeId}
        onSelect={(id) => { 
          setActiveId(id); 
          setPendingDeleteId(null);
          const stroke = annotations.find(a => a.id === id);
          if (stroke?.targetSec != null) seekTo(stroke.targetSec);
        }}
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

      {allSet && !valid && (
        <div style={{
          background: t.dangerDim, color: t.danger, border: `1px solid ${t.danger}`,
          padding: '8px 12px', borderRadius: 6, fontSize: 12,
        }}>
          Order must be start ≤ target ≤ end on the active stroke. Adjust handles before continuing.
        </div>
      )}

      {populatedAnnotations.length > 0 && !playerSide && (
        <div style={{
          background: t.surface2, color: t.warning, border: `1px solid ${t.warning}55`,
          padding: '8px 12px', borderRadius: 6, fontSize: 12,
        }}>
          Set the starting side before continuing — sides alternate per stroke.
        </div>
      )}
      
      <div style={{
          width: '100%',
          display: 'flex', alignItems: 'center', gap: 10,
          background: t.surface2, borderRadius: 8, padding: '8px 12px',
        }}>
          <span style={{ fontSize: 11, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Starting side (alternates per stroke)
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
            {playerSide ? `selected: ${playerSide}` : 'required: sides alternate from the first stroke'}
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