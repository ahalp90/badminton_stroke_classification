import { useState, useEffect, useRef } from 'react';
import { useTheme, Btn } from '../shared';
import { fmtTime } from '../utils/format';
import { loadYouTubeAPI } from '../utils/youtube';
import { Scrubber } from './Scrubber';
import { StrokePillStrip } from './StrokePillStrip';

// ──── Constants ──────────────────────────────────────────────────────────────────────────────────
// Default ±50-frame window at the implicit 30 fps demo rate (the markup contract does not carry 
// fps, see configure-screen's buildMarkupPayload).
const DEFAULT_HALF_WINDOW_SEC = 50 / 30;

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
  const [showPips, setShowPips] = useState(true);

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

  // ──── Derived state ────────────────────────────────────────────────────────────────────────────
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

  // ──── Render ───────────────────────────────────────────────────────────────────────────────────
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