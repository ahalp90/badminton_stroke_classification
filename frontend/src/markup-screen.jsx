import { useState, Fragment } from 'react';
import { useTheme, Btn, Card } from './shared';
import { CourtBoundaryStep } from './components/CourtBoundaryStep';
import { TimeframeStep } from './components/TimeframeStep';

// ─── Markup Shell ────────────────────────────────────────────────────────────────────────────────
// `orientation` is fixed to 'portrait' for v1: every official badminton
// broadcast camera is portrait. See frontend_integration_handoff.md §
// "About corners" for the contract.
const ORIENTATION = 'portrait';

const STEPS = [
  { label: 'Court Boundary', desc: 'Align perspective transform' },
  { label: 'Timeframe',      desc: 'Isolate stroke segment' },
];
/** Two-step markup wizard: court boundary alignment followed by stroke timeframe selection.
 * Builds and passes a markup payload to onNext on completion. */
export function MarkupScreen({ video, onNext, onBack }) {
  const { t } = useTheme();
  const [step, setStep] = useState(0);
  const [boundary, setBoundary] = useState(null);

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