import { createContext, useContext, useState, useEffect, Component } from 'react';

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