import { useTheme } from "../shared";
import logoSrc from '../uploads/logo-1777443863198.png';

// ──── Icons ──────────────────────────────────────────────────────────────────────────────────────
/* Inline SVG sun/moon icons for the theme toggle; stroke uses currentColor
   so they inherit the button's text color. No icon-lib dependency. */
function SunIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
    </svg>
  );
}

function MoonIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

// ──── NavBar ─────────────────────────────────────────────────────────────────────────────────────
const STEPS = [
    { id: 'library',   label: 'Select Video' },
    { id: 'markup',    label: 'Markup' },
    { id: 'configure', label: 'Configure' },
    { id: 'progress',  label: 'Analysis' },
    { id: 'results',   label: 'Results' },
  ];

/**
 * Sticky top navigation bar. Renders the wizard step indicators and the
 * Model Results and theme-toggle buttons. Navigation is delegated to the
 * onNavigate handler in app.jsx where all wizard guards live.
 */
export function NavBar({ screen, onNavigate }) {
  const { t, dark, setDark } = useTheme();

  const stepIndex = STEPS.findIndex(s => s.id === screen);
  const modelResultsActive = screen === 'model-results';

  return (
    <nav style={{
      background: t.surface,
      borderBottom: `1px solid ${t.border}`,
      display: 'flex',
      alignItems: 'center',
      padding: '0 24px',
      height: 72,
      position: 'sticky',
      top: 0,
      zIndex: 100,
      gap: 0,
    }}>
      {/* ──── Logo ──── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginRight: 28, flexShrink: 0 }}>
        <img
          src={logoSrc}
          style={{ height: 48, filter: dark ? 'none' : 'invert(1) brightness(0.15)' }}
          alt="HBA"
        />
        <div style={{ width: 1, height: 24, background: t.border }} />
        <span style={{ color: t.muted, fontSize: 13, fontWeight: 500, letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>
          Stroke Classifier
        </span>
      </div>
      
      {/* ──── Wizard steps ──── */}
      <div style={{ display: 'flex', alignItems: 'center', flex: 1, gap: 2, minWidth: 0, overflowX: 'auto' }}>
        {STEPS.map((step, i) => {
          const done     = i < stepIndex;
          const active   = i === stepIndex;
          const disabled = false;
          return (
            <button
              key={step.id}
              title={step.label}
              onClick={() => !disabled && onNavigate(step.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 7,
                padding: '0 14px', height: 56,
                background: 'none', border: 'none',
                cursor: disabled ? 'default' : 'pointer',
                color: active ? t.blue : done ? t.text : disabled ? t.border : t.muted,
                fontSize: 13, fontWeight: active ? 600 : 400,
                borderBottom: active ? `2px solid ${t.blue}` : '2px solid transparent',
                transition: 'all 0.15s',
                whiteSpace: 'nowrap',
                fontFamily: "'Space Grotesk', sans-serif",
              }}
            >
              <span style={{
                width: 18, height: 18, borderRadius: '50%', flexShrink: 0,
                background: active ? t.blue : done ? t.success : 'transparent',
                border: `1.5px solid ${active ? t.blue : done ? t.success : disabled ? t.border : t.muted}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 9, fontWeight: 700,
                color: active || done ? '#fff' : disabled ? t.border : t.muted,
              }}>
                {done ? '✓' : i + 1}
              </span>
              <span className="nav-step-label">{step.label}</span>
            </button>
          );
        })}
      </div>

      {/* ──── Model Results button ──── */}
      <button
        onClick={() => onNavigate('model-results')}
        style={{
          background: modelResultsActive ? t.blueDim : 'transparent',
          border: `1px solid ${modelResultsActive ? t.blue : t.border}`,
          borderRadius: 7,
          height: 32,
          padding: '0 12px',
          boxSizing: 'border-box',
          display: 'inline-flex',
          alignItems: 'center',
          cursor: 'pointer',
          color: modelResultsActive ? t.blue : t.text,
          fontSize: 12,
          fontWeight: 600,
          fontFamily: "'Space Grotesk', sans-serif",
          marginRight: 8,
          flexShrink: 0,
        }}
      >
        Model Results
      </button>

      {/* ──── Theme toggle ──── */}
      <button
        onClick={() => setDark(d => !d)}
        title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        style={{
          background: 'transparent', border: `1px solid ${t.border}`,
          borderRadius: 7, width: 32, height: 32, padding: 0, boxSizing: 'border-box',
          cursor: 'pointer', color: t.text, lineHeight: 1,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        {dark ? <SunIcon /> : <MoonIcon />}
      </button>
    </nav>
  );
}