import { createContext, useContext, useState, useEffect, Component } from 'react';
import logoSrc from './uploads/logo-1777443863198.png';

const ThemeContext = createContext();

export const DARK = {
  bg: '#070B13',
  surface: '#0E1422',
  surface2: '#16203A',
  border: '#1E2D4A',
  blue: '#2563EB',
  blueLight: '#3B82F6',
  blueDim: 'rgba(37,99,235,0.12)',
  pine: '#D4A843',
  pineDim: 'rgba(212,168,67,0.12)',
  text: '#E4EAF6',
  muted: '#6B7FA3',
  success: '#22C55E',
  successDim: 'rgba(34,197,94,0.12)',
  danger: '#EF4444',
  dangerDim: 'rgba(239,68,68,0.12)',
  warning: '#F59E0B',
};

export const LIGHT = {
  bg: '#EEF2FA',
  surface: '#FFFFFF',
  surface2: '#E4EBF7',
  border: '#C8D4EA',
  blue: '#1D4ED8',
  blueLight: '#3B82F6',
  blueDim: 'rgba(29,78,216,0.08)',
  pine: '#A8720A',
  pineDim: 'rgba(168,114,10,0.1)',
  text: '#0A0E17',
  muted: '#5A6882',
  success: '#16A34A',
  successDim: 'rgba(22,163,74,0.1)',
  danger: '#DC2626',
  dangerDim: 'rgba(220,38,38,0.1)',
  warning: '#D97706',
};

const THEME_KEY = 'hba.theme';

function initialDark() {
  if (typeof window === 'undefined') return true;
  const stored = window.localStorage?.getItem(THEME_KEY);
  if (stored === 'dark') return true;
  if (stored === 'light') return false;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? true;
}

export function ThemeProvider({ children }) {
  const [dark, setDark] = useState(initialDark);
  const t = dark ? DARK : LIGHT;

  useEffect(() => {
    document.body.style.background = t.bg;
    document.body.style.color = t.text;
    try { window.localStorage?.setItem(THEME_KEY, dark ? 'dark' : 'light'); } catch { /* noop */ }
  }, [dark, t]);

  return (
    <ThemeContext.Provider value={{ t, dark, setDark }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() { return useContext(ThemeContext); }

export function NavBar({ screen, onNavigate }) {
  const { t, dark, setDark } = useTheme();

  const steps = [
    { id: 'library',   label: 'Select Video' },
    { id: 'markup',    label: 'Markup' },
    { id: 'configure', label: 'Configure' },
    { id: 'progress',  label: 'Analysis' },
    { id: 'results',   label: 'Results' },
  ];
  const stepIndex = steps.findIndex(s => s.id === screen);
  // const projectActive = screen === 'project'; // restore with the Project button below

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

      <div style={{ display: 'flex', alignItems: 'center', flex: 1, gap: 2 }}>
        {steps.map((step, i) => {
          const done     = i < stepIndex;
          const active   = i === stepIndex;
          const disabled = false;
          return (
            <button
              key={step.id}
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
              {step.label}
            </button>
          );
        })}
      </div>

      {/* Project showcase button hidden until placeholder content is filled in.
          The 'project' screen is unreachable while this is commented out (there
          is no URL route to it). Uncomment to restore. */}
      {/* <button
        onClick={() => onNavigate('project')}
        style={{
          background: projectActive ? t.blueDim : 'transparent',
          border: `1px solid ${projectActive ? t.blue : t.border}`,
          borderRadius: 7,
          padding: '6px 12px',
          cursor: 'pointer',
          color: projectActive ? t.blue : t.text,
          fontSize: 12,
          fontWeight: 600,
          fontFamily: "'Space Grotesk', sans-serif",
          marginRight: 8,
          flexShrink: 0,
        }}
      >
        Project
      </button> */}

      <button
        onClick={() => setDark(d => !d)}
        style={{
          background: t.surface2, border: `1px solid ${t.border}`,
          borderRadius: 7, padding: '6px 12px', cursor: 'pointer',
          color: t.muted, fontSize: 12, fontWeight: 500,
          fontFamily: "'Space Grotesk', sans-serif",
          flexShrink: 0,
        }}
      >
        {dark ? '𖤓 Light' : '☾ Dark'}
      </button>
    </nav>
  );
}

export class ScreenErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error('Screen crashed:', error, info);
  }
  componentDidUpdate(prevProps) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }
  render() {
    if (this.state.error) {
      return <ScreenErrorFallback error={this.state.error} onReset={this.props.onReset} />;
    }
    return this.props.children;
  }
}

function ScreenErrorFallback({ error, onReset }) {
  const { t } = useTheme();
  return (
    <div style={{ maxWidth: 720, margin: '48px auto', padding: 24 }}>
      <div
        style={{
          background: t.dangerDim,
          border: `1px solid ${t.danger}`,
          borderRadius: 10,
          padding: 24,
          color: t.text,
        }}
      >
        <div style={{ fontSize: 18, fontWeight: 700, color: t.danger, marginBottom: 8 }}>
          Something went wrong on this screen
        </div>
        <div style={{ fontSize: 13, color: t.muted, marginBottom: 14, lineHeight: 1.5 }}>
          The navigation bar above is still usable. You can jump to another screen, or return to the start.
        </div>
        <pre
          style={{
            background: t.surface2,
            color: t.text,
            padding: 12,
            borderRadius: 6,
            fontSize: 12,
            overflowX: 'auto',
            margin: '0 0 14px',
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          {String(error?.message ?? error)}
        </pre>
        {onReset && (
          <button
            onClick={onReset}
            style={{
              background: t.blue,
              color: '#fff',
              border: 'none',
              borderRadius: 7,
              padding: '8px 14px',
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 600,
              fontFamily: "'Space Grotesk', sans-serif",
            }}
          >
            Return to start
          </button>
        )}
      </div>
    </div>
  );
}

export function Btn({ children, variant = 'primary', onClick, disabled, style: extraStyle = {}, size = 'md' }) {
  const { t } = useTheme();
  const [hov, setHov] = useState(false);

  const pad = size === 'sm' ? '7px 14px' : '10px 20px';
  const fz  = size === 'sm' ? 12 : 14;

  const base = {
    padding: pad, borderRadius: 8, fontSize: fz, fontWeight: 600,
    cursor: disabled ? 'not-allowed' : 'pointer', border: 'none',
    transition: 'all 0.15s', opacity: disabled ? 0.4 : 1,
    fontFamily: "'Space Grotesk', sans-serif", lineHeight: 1.4,
    ...extraStyle,
  };
  const variants = {
    primary:   { background: hov && !disabled ? t.blueLight : t.blue, color: '#fff' },
    secondary: { background: hov && !disabled ? t.surface2 : 'transparent', color: t.text, border: `1px solid ${t.border}` },
    ghost:     { background: hov && !disabled ? t.blueDim : 'transparent', color: t.blue, border: `1px solid ${hov && !disabled ? t.blue : 'transparent'}` },
    danger:    { background: hov && !disabled ? '#DC2626' : t.dangerDim, color: t.danger, border: `1px solid ${t.danger}` },
  };

  return (
    <button
      style={{ ...base, ...variants[variant] }}
      onClick={disabled ? undefined : onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
    >
      {children}
    </button>
  );
}

export function Card({ children, style: extraStyle = {}, onClick }) {
  const { t } = useTheme();
  return (
    <div
      onClick={onClick}
      style={{
        background: t.surface,
        border: `1px solid ${t.border}`,
        borderRadius: 12,
        ...extraStyle,
      }}
    >
      {children}
    </div>
  );
}

export function Badge({ children, color = 'blue' }) {
  const { t } = useTheme();
  const palette = {
    blue:   { bg: t.blueDim,    text: t.blueLight },
    pine:   { bg: t.pineDim,    text: t.pine },
    green:  { bg: t.successDim, text: t.success },
    red:    { bg: t.dangerDim,  text: t.danger },
    muted:  { bg: t.surface2,   text: t.muted },
  };
  const c = palette[color] || palette.blue;
  return (
    <span style={{
      background: c.bg, color: c.text,
      padding: '2px 8px', borderRadius: 4,
      fontSize: 11, fontWeight: 600,
      fontFamily: "'JetBrains Mono', monospace",
      display: 'inline-flex', alignItems: 'center',
    }}>
      {children}
    </span>
  );
}

export function SectionHeader({ title, subtitle }) {
  const { t } = useTheme();
  return (
    <div style={{ marginBottom: 24 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, color: t.text, marginBottom: 4 }}>{title}</h1>
      {subtitle && <p style={{ fontSize: 14, color: t.muted }}>{subtitle}</p>}
    </div>
  );
}
