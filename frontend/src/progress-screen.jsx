import { useState, useEffect, useRef } from 'react';
import { useTheme, Btn, Card, Badge, SectionHeader } from './shared';

// ──── Constants ──────────────────────────────────────────────────────────────────────────────────
const PIPELINE_STAGES = [
  { label: 'Preprocessing',      desc: 'Extracting frames · normalising court perspective' },
  { label: 'Feature Extraction', desc: 'Object detection · keypoint graphs · shuttle tracking' },
  { label: 'Model Inference',    desc: 'Running selected classification models' },
  { label: 'Postprocessing',     desc: 'Aggregating results · computing evaluation metrics' },
];

// Mock log events used by the fallback timer path only.
// TODO: Remove along with mock path during clean-up once real pipeline is confirmed stable.
const LOG_EVENTS = [
  { at:  0, msg: 'Job submitted — loading video segment…' },
  { at:  4, msg: 'Applying court boundary homography transform' },
  { at:  9, msg: 'Frame extraction in progress' },
  { at: 16, msg: 'Player detection: bounding boxes extracted' },
  { at: 24, msg: 'TrackNetV3: shuttlecock trajectory computed' },
  { at: 32, msg: 'MMPose: skeleton keypoint sequences extracted' },
  { at: 52, msg: 'Model A (BST): inference started' },
  { at: 76, msg: 'Model A: inference complete' },
  { at: 82, msg: 'Writing results' },
  { at: 100, msg: '✓ Analysis complete' },
];

// TODO: add second model (BRIC)

// ──── Markup payload builder ─────────────────────────────────────────────────────────────────────
/**
 * Translates the wizard's in-memory state into the schema documented in docs/api_contract.md on 
 * feat/bric-pipeline. The backend validates this via the Pydantic Markup model in src/api/main.py.
 *
 * Markup state shape is the new multi-stroke form: `markup.annotations` is a list of 
 * {id, targetSec}.
 **/

function buildMarkupPayload(task) {
  const m = task?.markup;
  if (!m) return null;
  const video = m.video;
  const boundary = Array.isArray(m.boundary) && m.boundary.length === 4
    ? m.boundary.map(p => ({ x: p.x, y: p.y }))
    : null;
  // Intrinsic source resolution, so the backend can resolution-check the
  // normalised boundary against the model input minimum.
  const frameDims = m.frameDims && m.frameDims.w > 0 && m.frameDims.h > 0 ? m.frameDims : null;

  // Resolve the in-memory annotation list, with migration from the old
  // single-`timeframe` shape and legacy `markup.player` (1/2 → top/bottom).
  let strokes = Array.isArray(m.annotations) ? m.annotations : null;
  if (!strokes && m.timeframe) {
    const tf = m.timeframe;
    strokes = [{ targetSec: tf.targetSec }];
  }
  strokes = strokes || [];
  const startingSide = m.playerSide
    ?? (m.player === 1 ? 'top' : m.player === 2 ? 'bottom' : null);
  const flip = (s) => s === 'top' ? 'bottom' : (s === 'bottom' ? 'top' : null);

  // Drop any half-set stroke before sending — frontend guards against this at the Confirm Timeframe 
  // button, but the migration path can produce half-set entries from old persisted state.
  const annotations = strokes
    .filter(a => a && a.targetSec != null)
    .map((a, i) => ({
      target_sec: a.targetSec,
      player_side: startingSide == null ? null : (i % 2 === 0 ? startingSide : flip(startingSide)),
    }));

  // Pick the first enabled model from Configure as the explicit choice;
  // architecture defaults to 'bst'.
  const enabledModel = (task?.models || []).find(mm => task?.enabled?.[mm.id]) || null;
  return {
    architecture: enabledModel?.architecture ?? 'bst',
    model_id: enabledModel?.id ?? null,
    orientation: m.orientation || 'portrait',
    video_label: video?.filename || video?.match || null,
    boundary,
    frame_width: frameDims?.w ?? null,
    frame_height: frameDims?.h ?? null,
    annotations,
    enabled_sides: ['top', 'bottom'],
  };
}

// ──── Component ──────────────────────────────────────────────────────────────────────────────────
/**
 * Step 4 of the wizard: submits the job and polls for completion.
 * 
 * Two execution paths:
 *   Real path  — fires when task has a valid video source (upload or library).
 *                POSTs to /api/upload or /api/library_predict, polls /api/status,
 *                fetches /api/results, then calls onComplete(result).
 *   Mock path  — fires when realRun is false (task null/malformed).
 *                Drives progress via a synthetic timer. Should not fire in
 *                normal use now that DEV_FIXTURES are removed.
 *                TODO: Remove during clean-up once real pipeline is confirmed stable.
 */
export function ProgressScreen({ task, onComplete }) {
  const { t } = useTheme();

  // ──── Derived run type ─────────────────────────────────────────────────────────────────────────
  const file = task?.markup?.video?.file ?? null;
  const videoSource = task?.markup?.video?.source ?? null;
  const isUploadRun = !!file && videoSource === 'upload';
  const isLibraryRun = !file && videoSource === 'library';
  const realRun = isUploadRun || isLibraryRun;

  // ──── State ────────────────────────────────────────────────────────────────────────────────────
  const [pct,    setPct]    = useState(0);
  const [stage,  setStage]  = useState(0);
  const [log,    setLog]    = useState([]);
  const [error,  setError]  = useState(null);
  const [retryNonce, setRetryNonce] = useState(0);
  const logRef = useRef(null);

  const appendLog = (msg, at = null) => setLog(l => [...l, {
    msg,
    at: at,
    time: new Date().toLocaleTimeString('en-AU', {hour12: false}),
  }]);

  // ──── Real path ────────────────────────────────────────────────────────────────────────────────
  // Drives progress from /api/upload (or /api/library_predict) + /api/status + /api/results. 
  // The two submission shapes differ but share the same job-poll lifecycle, so most of this effect 
  // is shared.
  useEffect(() => {
    if (!realRun) return;
    let aborted = false;
    let pollId = null;
    setError(null);
    setLog([]);
    setPct(0);
    setStage(0);

    const markupPayload = buildMarkupPayload(task);

    const run = async () => {
      try {
        let upRes;
        if (isUploadRun) {
          appendLog(`Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)…`);
          // The query-param `model` is the *backend execution* model and must be in 
          // _available_models() (.pt stems + "default"). The registry-level model_id is 
          // conceptually different and rides inside the markup JSON sidecar — see Gap 1. 
          // Always use "default" here; if the frontend later needs to pick a specific
          // checkpoint, gate it on _available_models output.
          const params = new URLSearchParams({ model: 'default' });
          // Trim the upload to the bounding span across every marked stroke.
          // Legacy `markup.timeframe` (single-stroke shape) is still honoured
          // so sessions mid-flight across this change keep working.
          const strokes = Array.isArray(task?.markup?.annotations)
            ? task.markup.annotations
            : (task?.markup?.timeframe ? [task.markup.timeframe] : []);
          const starts = strokes
            .map(s => s?.startSec)
            .filter(v => v != null);
          const ends = strokes
            .map(s => s?.endSec)
            .filter(v => v != null);
          if (starts.length) params.set('start_sec', String(Math.min(...starts)));
          if (ends.length && Math.max(...ends) > (starts.length ? Math.min(...starts) : 0)) {
            params.set('end_sec', String(Math.max(...ends)));
          }
          // XHR rather than fetch - needed for upload-progress events for the
          // visible byte-flow phase (fetch does not surface them).
          upRes = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', `/api/upload?${params.toString()}`);
            xhr.upload.addEventListener('progress', (e) => {
              if (aborted || !e.lengthComputable) return;
              const frac = e.loaded / e.total;
              setPct(5 + frac * 25);  // 5% → 30% covers upload phase
            });
            xhr.addEventListener('load', () => {
              if (xhr.status >= 200 && xhr.status < 300) {
                try { resolve(JSON.parse(xhr.responseText)); }
                catch (e) { reject(new Error('Malformed upload response')); }
              } else {
                let detail = `HTTP ${xhr.status}`;
                try { detail = JSON.parse(xhr.responseText).detail || detail; } catch { /* noop */ }
                reject(new Error(detail));
              }
            });
            xhr.addEventListener('error', () => reject(new Error('Network error during upload')));
            xhr.addEventListener('abort', () => reject(new Error('Upload aborted')));
            const fd = new FormData();
            fd.append('file', file, file.name);
            if (markupPayload) {
              fd.append('markup', JSON.stringify(markupPayload));
              appendLog(`Markup sidecar: ${markupPayload.boundary ? '4 corners' : 'no boundary'}, ${markupPayload.annotations.length} annotation(s)`);
            }
            xhr.send(fd);
          });
        } else {
          // Library-match path: POST JSON to /api/library_predict. No upload bytes - jump straight 
          // to the queued/processing phase. The clip_stem is the library entry's id; the canned
          // backend stub doesn't read the video itself.
          appendLog(`Submitting library clip "${task?.markup?.video?.match || 'clip'}" for inference…`);
          setPct(20);
          const body = {
            clip_stem: task?.markup?.video?.id || task?.markup?.video?.youtubeId,
            architecture: markupPayload?.architecture || 'bst',
            model_id: markupPayload?.model_id || null,
            markup: markupPayload,
          };
          const r = await fetch('/api/library_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          });
          if (!r.ok) {
            let detail = `HTTP ${r.status}`;
            try { detail = (await r.json()).detail || detail; } catch { /* noop */ }
            throw new Error(detail);
          }
          upRes = await r.json();
          if (markupPayload) {
            appendLog(`Markup sidecar: ${markupPayload.boundary ? '4 corners' : 'no boundary'}, ${markupPayload.annotations.length} annotation(s)`);
          }
        }

        if (aborted) return;
        appendLog(`Upload accepted · job ${upRes.job_id.slice(0, 8)}`);
        setStage(1);
        setPct(30);

        const jobId = upRes.job_id;
        let lastStatus = null;

        // Poll /api/status every 250ms until terminal state.
        await new Promise((resolve, reject) => {
          pollId = setInterval(async () => {
            try {
              const r = await fetch(`/api/status/${jobId}`);
              if (!r.ok) throw new Error(`Status HTTP ${r.status}`);
              const s = await r.json();
              if (s.status !== lastStatus) {
                lastStatus = s.status;
                if (s.status === 'queued') {
                  setStage(1); setPct(p => Math.max(p, 35));
                  appendLog('Backend status: queued');
                } else if (s.status === 'processing') {
                  setStage(2); setPct(p => Math.max(p, 55));
                  appendLog('Backend status: processing');
                } else if (s.status === 'complete') {
                  clearInterval(pollId); pollId = null;
                  resolve();
                  return;
                } else if (s.status === 'failed') {
                  clearInterval(pollId); pollId = null;
                  reject(new Error('Backend reported failure'));
                  return;
                }
              }
              // Slow tick during processing so the bar feels alive without
              // racing past the backend.
              setPct(p => (p < 88 ? p + 0.6 : p));
            } catch (e) {
              if (pollId) { clearInterval(pollId); pollId = null; }
              reject(e);
            }
          }, 250);
        });
        if (aborted) return;
        appendLog('Inference complete · fetching results');
        setStage(3); setPct(95);

        const rr = await fetch(`/api/results/${jobId}`);
        if (!rr.ok) {
          let detail = `HTTP ${rr.status}`;
          try { detail = (await rr.json()).detail || detail; } catch { /* noop */ }
          throw new Error(detail);
        }
        const result = await rr.json();
        if (aborted) return;
        setPct(100);
        appendLog(`✓ Done · ${result.strokes?.length ?? 0} stroke(s) returned`, 100);
        setTimeout(() => { if (!aborted) onComplete(result); }, 800);
      } catch (e) {
        if (aborted) return;
        setError(e.message || String(e));
        appendLog(`✗ ${e.message || e}`);
      }
    };
    run();
    return () => {
      aborted = true;
      if (pollId) clearInterval(pollId);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps 
  }, [realRun, retryNonce]); // task and onComplete intentionally omitted, effect must not re-run mid-job

  // ──── Mock path ────────────────────────────────────────────────────────────────────────────────
  // Fallback timer path for when realRun is false. Should not fire in normal use now that 
  // DEV_FIXTURES are removed - kept as a safety net. 
  // TODO: Remove during clean-up once real pipeline is confirmed stable.
  useEffect(() => {
    if (realRun) return;
    let current = 0;
    const iv = setInterval(() => {
      const increment = Math.random() * 2.8 + 0.6;
      current = Math.min(current + increment, 100);
      setPct(current);

      const newEvents = LOG_EVENTS.filter(e => e.at <= current && e.at > current - increment - 0.1);
      if (newEvents.length > 0) {
        setLog(l => [...l, ...newEvents.map(e => ({ ...e, time: new Date().toLocaleTimeString('en-AU', { hour12: false }) }))]);
      }

      if (current < 25)      setStage(0);
      else if (current < 52) setStage(1);
      else if (current < 80) setStage(2);
      else                   setStage(3);

      if (current >= 100) {
        clearInterval(iv);
        setTimeout(() => onComplete(null), 1400);
      }
    }, 280);
    return () => clearInterval(iv);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [realRun]); // onComplete intentionally omitted - effect must not re-run mid-job

  // ──── Auto-scroll log ─────────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  // ──── Render ───────────────────────────────────────────────────────────────────────────────────
  const done = pct >= 100 && !error;

  return (
    <div style={{ maxWidth: 820, margin: '0 auto', padding: 32 }}>
      <SectionHeader
        title="Analysis in Progress"
        subtitle={task?.taskName}
      />
      
      {error && (
        <Card style={{ padding: 22, marginBottom: 20, borderColor: t.danger, background: t.dangerDim }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 14 }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: t.danger, marginBottom: 4 }}>
                Analysis failed
              </div>
              <div style={{ fontSize: 12, color: t.text, fontFamily: "'JetBrains Mono', monospace" }}>
                {error}
              </div>
            </div>
            <Btn onClick={() => setRetryNonce(n => n + 1)}>Try again</Btn>
          </div>
        </Card>
      )}

      {/* ── Progress bar ── */}
      <Card style={{ padding: 26, marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: t.text }}>{PIPELINE_STAGES[stage].label}</div>
            <div style={{ fontSize: 12, color: t.muted, marginTop: 2 }}>{PIPELINE_STAGES[stage].desc}</div>
          </div>
          <div style={{ fontSize: 28, fontWeight: 700, color: done ? t.success : t.blue, fontFamily: "'JetBrains Mono', monospace" }}>
            {Math.round(pct)}%
          </div>
        </div>
        <div style={{ height: 8, background: t.surface2, borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            height: '100%', borderRadius: 4,
            width: `${pct}%`,
            background: done
              ? t.success
              : `linear-gradient(90deg, ${t.blue}, ${t.blueLight})`,
            transition: 'width 0.28s ease',
          }} />
        </div>
        {done && (
          <div style={{ marginTop: 12, fontSize: 13, color: t.success, fontWeight: 600 }}>
            ✓ Analysis complete — loading results…
          </div>
        )}
      </Card>

      {/* ── Pipeline stages ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        <Card style={{ padding: 20 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 16 }}>
            Pipeline
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {PIPELINE_STAGES.map((s, i) => {
              const isDone   = i < stage || done;
              const isActive = i === stage && !done;
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                  <div style={{
                    width: 24, height: 24, borderRadius: '50%', flexShrink: 0, marginTop: 1,
                    background: isDone ? t.success : isActive ? t.blue : 'transparent',
                    border: `1.5px solid ${isDone ? t.success : isActive ? t.blue : t.border}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 10, color: isDone || isActive ? '#fff' : t.muted,
                    fontWeight: 700,
                  }}>
                    {isDone ? '✓' : isActive ? '⟳' : i + 1}
                  </div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: isActive ? 600 : 400, color: isDone ? t.muted : isActive ? t.text : t.muted }}>
                      {s.label}
                    </div>
                    <div style={{ fontSize: 11, color: t.muted, marginTop: 1 }}>{s.desc}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        {/* ── Activity log ── */}
        <Card style={{ padding: 20, display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 16 }}>
            Activity Log
          </div>
          <div
            ref={logRef}
            style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 5, maxHeight: 200 }}
          >
            {log.length === 0 && (
              <div style={{ fontSize: 12, color: t.muted }}>Waiting for pipeline…</div>
            )}
            {log.map((entry, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, fontSize: 11, alignItems: 'flex-start' }}>
                <span style={{ color: t.muted, fontFamily: "'JetBrains Mono', monospace", flexShrink: 0 }}>
                  {entry.time}
                </span>
                <span style={{ color: entry.at >= 100 ? t.success : t.text, lineHeight: 1.4 }}>
                  {entry.msg}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* ── Per-model progress bars ── */}
        {(task?.models || []).filter(m => task?.enabled?.[m.id]).map((m, i) => {
          // Model inference sits in stage 2 (52-80% of overall pipeline pct).
          // Per-model progress is synthetic — the backend status is global — so stagger only the 
          // START to keep multi-model jobs visually distinct, but have every enabled model COMPLETE
          // together at the end of the inference phase. Anchoring `complete` to the overall `done` 
          // flag guarantees no model bar is ever left blue/Running once the run has finished (and
          // stops the i=0 model lagging the rest).
          const completeAt = 78;
          const startAt    = Math.min(52 + i * 4, completeAt - 4);
          const span       = Math.max(1, completeAt - startAt);
          const modelPct   = Math.max(0, Math.min(100, ((pct - startAt) / span) * 100));
          const active     = pct > startAt;
          const complete   = done || pct >= completeAt;
          return (
            <Card key={m.id} style={{ padding: 18, gridColumn: '1 / -1' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: t.text }}>{m.name}</div>
                  <div style={{ fontSize: 11, color: t.muted }}>{m.subtitle}</div>
                </div>
                <Badge color={complete ? 'green' : active ? 'blue' : 'muted'}>
                  {complete ? 'Done' : active ? 'Running' : 'Queued'}
                </Badge>
              </div>
              <div style={{ height: 5, background: t.surface2, borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  height: '100%', borderRadius: 3,
                  width: `${complete ? 100 : modelPct}%`,
                  background: complete ? t.success : t.blue,
                  transition: 'width 0.28s ease',
                }} />
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}