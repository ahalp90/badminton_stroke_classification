import { useState, useEffect, useRef } from 'react';
import { useTheme } from '../shared';
import { fmtTime } from '../utils/format';

const ZOOM_LEVELS = [1, 2, 5, 10, 25, 50];
const TRACK_HEIGHT = 38;

/** Scrubber: buffered + density-binned pips + click-drag seek
* `strokes` is the list of user-marked annotations [{id, startSec, targetSec, endSec}]; 
* `activeId` picks which one renders handles + the saturated blue.
* `strokeTimes` is the unrelated dataset-level density overlay (kept). */
export function Scrubber({
  duration, currentTime, loaded,
  strokes, activeId, onSelectStroke,
  strokeTimes, showPips, onSeek,
}) {
  const { t } = useTheme();

  const trackRef = useRef(null);
  const scrollRef = useRef(null);
  const draggingRef = useRef(false);

  const [zoom, setZoom] = useState(ZOOM_LEVELS[0]);

  const N_BUCKETS = 200 * zoom; // defined inside component as depends on zoom state

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
          {/* Inactive stroke regions first (muted gray), so the active region
              paints over them when ranges overlap. */}
          {(strokes || []).filter(s => s.id !== activeId).map(s => (
            s.startSec !== null && s.endSec !== null && (
              <div
                key={s.id}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  onSelectStroke && onSelectStroke(s.id);
                }}
                title={`Stroke · click to edit`}
                style={{
                  position: 'absolute', top: 0, bottom: 0, borderRadius: 2,
                  left: `${pct(s.startSec)}%`,
                  width: `${Math.max(0, pct(s.endSec) - pct(s.startSec))}%`,
                  background: t.muted, opacity: 0.45,
                  cursor: 'pointer',
                }}
              />
            )
          ))}
          {(strokes || []).filter(s => s.id === activeId).map(s => (
            s.startSec !== null && s.endSec !== null && (
              <div
                key={s.id}
                style={{
                  position: 'absolute', top: 0, bottom: 0, borderRadius: 2,
                  left: `${pct(s.startSec)}%`,
                  width: `${Math.max(0, pct(s.endSec) - pct(s.startSec))}%`,
                  background: t.blue,
                  pointerEvents: 'none',
                }}
              />
            )
          ))}
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
        {/* S / ◉ / E markers (active stroke only) */}
        {(() => {
          const active = (strokes || []).find(s => s.id === activeId);
          if (!active) return null;
          return [
            { val: active.startSec,  label: 'S', color: t.blue },
            { val: active.targetSec, label: '◉', color: t.warning },
            { val: active.endSec,    label: 'E', color: t.blue },
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
          ));
        })()}
    </div>
        </div>
      </div>
    </div>
  );
}