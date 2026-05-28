// Human-facing labels for the dataset splits. The raw keys ('val'/'test')
// stay as-is for data lookups (e.g. `${split}_metrics`); these are display-only.
export const SPLIT_LABELS = { val: 'Validation', test: 'Test' };
export const splitLabel = (s) => SPLIT_LABELS[s] ?? s;

export const fmtTime = (s) => {
  if (!isFinite(s)) return '–:––';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = String(m).padStart(2, '0');
  const ss = String(sec).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
};