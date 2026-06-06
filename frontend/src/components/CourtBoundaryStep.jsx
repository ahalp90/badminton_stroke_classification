import { useState, useEffect, useRef, useCallback } from 'react';
import { useTheme, Btn } from '../shared';
import { fmtTime } from '../utils/format';
import { Scrubber } from './Scrubber';

const frameModules = import.meta.glob('../data/frames/*.jpg', { eager: true, import: 'default' });
const frameUrl = (id) => frameModules[`../data/frames/${id}.jpg`];

const W = 640;
const H = 360;
const LOUPE_SIZE = 130;
const LOUPE_ZOOM = 4;
// Minimum side length (px) the model pipeline needs from the cropped region:
// X3D wants 224x224, pose needs comparable resolution. Mirrors the backend's
// MIN_MODEL_INPUT_PX; below this we warn rather than silently degrade.
const MIN_INPUT_PX = 224;
const DEFAULT_CORNERS = [
  { x: 0.17, y: 0.30 },
  { x: 0.83, y: 0.30 },
  { x: 0.93, y: 0.92 },
  { x: 0.07, y: 0.92 },
];

/** Step 1 of markup: interactive canvas for aligning the court boundary quadrilateral.
 * Supports both library (image frame) and uploaded (video frame) sources. */
export function CourtBoundaryStep({ video, onComplete }) {
  const { t } = useTheme();
  const canvasRef = useRef(null);
  const loupeRef = useRef(null);
  // sourceRef holds either an HTMLImageElement (library/youtube) or an
  // HTMLVideoElement (upload). drawImage accepts both, and we read intrinsic
  // dimensions via .naturalWidth/.videoWidth in the loupe code below.
  const sourceRef = useRef(null);
  const hiddenVideoRef = useRef(null);
  const isUpload = video?.source === 'upload' && !!video?.objectURL;

  const [pts, setPts] = useState(DEFAULT_CORNERS);
  const [dragging, setDragging] = useState(null);
  const [cursor, setCursor] = useState(null); // {x,y} in canvas coords while dragging
  const [confirmed, setConfirmed] = useState(false);
  const [sourceLabel, setSourceLabel] = useState('Reference frame · drag handles to align');
  // Intrinsic source resolution, captured once the frame loads. Drives the
  // low-resolution warning and is forwarded to the backend so it can run the
  // same check on the normalised boundary.
  const [srcDims, setSrcDims] = useState({ w: 0, h: 0 });
  // Scrubbing state (uploads only): lets the user seek to a frame where the
  // court boundary is clearly visible before aligning the quadrilateral.
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    if (isUpload) {
      const vid = hiddenVideoRef.current;
      if (!vid) return;
      const onReady = () => {
        sourceRef.current = vid;
        setSrcDims({ w: vid.videoWidth, h: vid.videoHeight });
        setDuration(vid.duration || 0);
        // Seek to a representative early frame (first frame can be black).
        try {
          const target = Math.min(0.5, (vid.duration || 1) / 2);
          if (Math.abs(vid.currentTime - target) > 0.01) vid.currentTime = target;
        } catch { /* noop */ }
        setCurrentTime(vid.currentTime);
        setSourceLabel(`Uploaded frame · ${video.filename || 'video'}`);
        draw();
      };
      const onSeeked = () => { setCurrentTime(vid.currentTime); draw(); };
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
    img.onload = () => {
      sourceRef.current = img;
      setSrcDims({ w: img.naturalWidth, h: img.naturalHeight });
      draw();
    };
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

  // Seek the hidden video to an absolute time (seconds). The 'seeked' listener
  // installed above redraws the canvas and syncs currentTime once the frame is ready.
  const seekTo = (s) => {
    const vid = hiddenVideoRef.current;
    if (!vid || !duration) return;
    const clamped = Math.max(0, Math.min(duration, s));
    setCurrentTime(clamped);
    try { vid.currentTime = clamped; } catch { /* noop */ }
  };

  const nudge = (delta) => {
    const vid = hiddenVideoRef.current;
    if (!vid) return;
    seekTo(vid.currentTime + delta);
  };

  // Bounding box of the four corners scaled back to source pixels. Null until
  // the source resolution is known; below MIN_INPUT_PX we warn (non-blocking).
  const xs = pts.map(p => p.x);
  const ys = pts.map(p => p.y);
  const bboxW = srcDims.w ? Math.round((Math.max(...xs) - Math.min(...xs)) * srcDims.w) : null;
  const bboxH = srcDims.h ? Math.round((Math.max(...ys) - Math.min(...ys)) * srcDims.h) : null;
  const lowRes = bboxW !== null && bboxH !== null && (bboxW < MIN_INPUT_PX || bboxH < MIN_INPUT_PX);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <p style={{ fontSize: 13, color: t.muted, lineHeight: 1.6 }}>
        Drag the <span style={{ color: t.blue, fontWeight: 600 }}>four corner handles</span> to
        align the quadrilateral with the court boundary edges. This homography transform
        normalises inputs across varied camera angles.
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

      {isUpload && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: t.surface2, borderRadius: 8, padding: '6px 10px',
          }}>
            <span style={{ fontSize: 11, color: t.muted, marginRight: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Scrub
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
                disabled={!duration}
                style={{
                  background: t.surface, border: `1px solid ${t.border}`,
                  color: t.text, padding: '5px 10px', borderRadius: 5,
                  fontSize: 12, fontWeight: 600, cursor: duration ? 'pointer' : 'not-allowed',
                  opacity: duration ? 1 : 0.4,
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                {b.label}
              </button>
            ))}
            <div style={{ marginLeft: 'auto', fontSize: 12, color: t.muted, fontFamily: "'JetBrains Mono', monospace" }}>
              {fmtTime(currentTime)} / {fmtTime(duration)}
            </div>
          </div>
          <Scrubber
            duration={duration}
            currentTime={currentTime}
            loaded={0}
            strokes={[]}
            activeId={null}
            onSelectStroke={() => {}}
            strokeTimes={[]}
            showPips={false}
            onSeek={seekTo}
            zoom={zoom}
          />           
        </div>
      )}

      {lowRes && (
        <div style={{
          fontSize: 12, color: t.warning ?? '#D97706', lineHeight: 1.5,
          background: (t.warning ?? '#D97706') + '1A',
          border: `1px solid ${(t.warning ?? '#D97706')}55`,
          borderRadius: 6, padding: '8px 12px',
        }}>
          ⚠ The selected region is ~{bboxW}×{bboxH}px on a {srcDims.w}×{srcDims.h} frame,
          below the {MIN_INPUT_PX}×{MIN_INPUT_PX}px model input minimum. You can still
          proceed, but classification quality may degrade.
        </div>
      )}

      <div style={{ display: 'flex', gap: 10 }}>
        <Btn
          variant="secondary"
          onClick={() => {
            setConfirmed(false);
            setPts(DEFAULT_CORNERS);
          }}
        >
          Reset
        </Btn>
        {!confirmed
          ? <Btn onClick={() => setConfirmed(true)}>Confirm Boundary</Btn>
          : <Btn onClick={() => onComplete(pts, srcDims)}>Next: Set Timeframe →</Btn>
        }
      </div>
    </div>
  );
}
