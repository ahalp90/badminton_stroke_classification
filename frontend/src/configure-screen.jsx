import { useState, useEffect, useMemo } from 'react';
import { useTheme, Btn, Card, Badge, SectionHeader } from './shared';
import { toModelCard } from './utils/adapters';
import { ModelCard } from './components/ModelCard';
import { ModelPickerModal } from './components/ModelPickerModal';

// ──── Configure Screen ───────────────────────────────────────────────────────────────────────────
/**
 * Step 3 of the wizard: select classification model/s and label the task before submission.
 * Fetches available models from the registry API.
 *
 * The registry surfaces several BST-X cells (taxonomy/split variants) plus BRIC.
 * To keep this screen calm we show just two headline cards: the default BST-X
 * cell and BRIC. The other BST-X variants are reachable behind "Change variant",
 * which swaps which BST-X cell the headline card represents. Submission is blocked
 * until at least one card is selected.
 */
export function ConfigureScreen({ markup, onSubmit, onBack }) {
  const { t } = useTheme();

  // ──── State ────────────────────────────────────────────────────────────────────────────────────
  const [models,      setModels]      = useState([]);
  const [loadError,   setLoadError]   = useState(null);
  const [activeBstId, setActiveBstId] = useState(null);   // which BST-X variant the headline card shows
  const [bstOn,       setBstOn]       = useState(true);   // headline BST-X card selected?
  const [otherOn,     setOtherOn]     = useState({});     // id -> bool for non-BST-X cards (BRIC)
  const [picking,     setPicking]     = useState(false);  // variant modal open?
  const [changeHov,   setChangeHov]   = useState(false);  // hover state for the Change-variant button
  const [taskName,    setTaskName]    = useState(
    `Analysis — ${markup?.video?.match?.split(' vs ')[0] ?? 'Video'} — ${new Date().toLocaleDateString('en-AU')}`
  );

  // ──── Load model registry ──────────────────────────────────────────────────────────────────────
  useEffect(() => {
    // `alive` flag prevents state updates if the component unmounts before the fetch resolves.
    let alive = true;
    fetch('/api/registry')
      .then(response => response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`)))
      .then(data => {
        if (!alive) return;
        const items = (data.models || []).map(toModelCard);
        setModels(items);
        // Default the headline card to the is_default BST-X cell (first BST-X as
        // a fallback). BRIC starts off, matching the old "first model on" default.
        const bst = items.filter(m => m.architecture === 'bst-x');
        const def = bst.find(m => m.isDefault) ?? bst[0];
        setActiveBstId(def?.id ?? null);
      })
      .catch(err => { if (alive) setLoadError(err.message); });
    return () => { alive = false; };
  }, []);

  // ──── Derived state ────────────────────────────────────────────────────────────────────────────
  const bstModels   = useMemo(() => models.filter(m => m.architecture === 'bst-x'), [models]);
  const otherModels = useMemo(() => models.filter(m => m.architecture !== 'bst-x'), [models]);
  const activeBst   = useMemo(() => bstModels.find(m => m.id === activeBstId) ?? null, [bstModels, activeBstId]);

  /** True when at least one card is selected - gates the submit button. */
  const anyEnabled = bstOn || otherModels.some(m => otherOn[m.id]);
  const toggleOther = id => setOtherOn(prev => ({ ...prev, [id]: !prev[id] }));

  // Final enabled map keyed by real model id, as the downstream screens expect:
  // only the active BST-X variant carries the BST selection, plus any BRIC toggles.
  const buildEnabled = () => ({
    ...otherOn,
    ...(activeBstId ? { [activeBstId]: bstOn } : {}),
  });

  // ──── Render ───────────────────────────────────────────────────────────────────────────────────
  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: 32 }}>
      <SectionHeader
        title="Configure Analysis"
        subtitle="Select models and tune parameters before submitting the classification job."
      />

      {/* ──── Model selection ──── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 24, alignItems: 'start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Classification Models
          </div>
          {loadError && (
            <div style={{ fontSize: 12, color: t.danger, padding: '8px 12px', background: t.dangerDim, borderRadius: 6 }}>
              Couldn't load model registry: {loadError}
            </div>
          )}
          {!loadError && models.length === 0 && (
            <div style={{ fontSize: 12, color: t.muted, padding: '8px 12px' }}>
              Loading models…
            </div>
          )}

          {/* Headline BST-X card + variant picker affordance. */}
          {activeBst && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <ModelCard
                model={activeBst}
                enabled={bstOn}
                disabled={false}
                onToggle={() => setBstOn(v => !v)}
              />
              {bstModels.length > 1 && (
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    onClick={() => setPicking(true)}
                    onMouseEnter={() => setChangeHov(true)}
                    onMouseLeave={() => setChangeHov(false)}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 7,
                      background: changeHov ? t.blueDim : 'none',
                      border: `1px solid ${t.blue}`,
                      color: t.blue,
                      padding: '8px 14px', borderRadius: 7,
                      fontSize: 12.5, fontWeight: 600, cursor: 'pointer',
                      fontFamily: "'Space Grotesk', sans-serif",
                      transition: 'all 0.15s',
                    }}
                  >
                    ⇄ Change variant ({bstModels.length})
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Other architectures (BRIC) as their own cards. */}
          {otherModels.map(m => (
            <ModelCard
              key={m.id}
              model={m}
              enabled={!!otherOn[m.id]}
              disabled={false}
              onToggle={() => toggleOther(m.id)}
            />
          ))}

          {models.length > 0 && (
            <div style={{ fontSize: 11, color: t.muted, lineHeight: 1.5 }}>
              Models come from <code>docs/models_registry.yaml</code>. Add a new entry there to surface it here.
            </div>
          )}
          {!anyEnabled && models.length > 0 && (
            <div style={{ fontSize: 12, color: t.danger, padding: '8px 12px', background: t.dangerDim, borderRadius: 6 }}>
              Select at least one model to continue.
            </div>
          )}
        </div>

        {/* ──── Task panel ──── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Task
          </div>
          <Card style={{ padding: 18 }}>
            <div style={{ fontSize: 12, color: t.muted, marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Task Label</div>
            <input
              value={taskName}
              onChange={e => setTaskName(e.target.value)}
              style={{
                width: '100%', background: t.surface2, border: `1px solid ${t.border}`,
                borderRadius: 6, padding: '8px 10px', color: t.text, fontSize: 13,
                fontFamily: "'Space Grotesk', sans-serif", outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </Card>

          <Card style={{ padding: 16 }}>
            <div style={{ fontSize: 12, color: t.muted, marginBottom: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Input Summary</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: t.text, marginBottom: 2 }}>{markup?.video?.match}</div>
            <div style={{ fontSize: 11, color: t.muted, marginBottom: 10 }}>{markup?.video?.tournament}</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <Badge color={markup?.video?.annotated ? 'green' : 'muted'}>
                {markup?.video?.annotated ? 'Annotated' : 'Unannotated'}
              </Badge>
              {markup?.playerSide && (
                <Badge color="blue">{`Player ${markup.playerSide}`}</Badge>
              )}
              {Array.isArray(markup?.annotations) && markup.annotations.length > 0 ? (
                <Badge color="pine">
                  {markup.annotations.length} stroke{markup.annotations.length === 1 ? '' : 's'}
                </Badge>
              ) : markup?.timeframe ? (
                <Badge color="pine">{markup.timeframe.duration}s segment</Badge>
              ) : null}
            </div>
          </Card>

          <Btn disabled={!anyEnabled} onClick={() => onSubmit({ markup, enabled: buildEnabled(), taskName, models })}>
            Submit for Analysis →
          </Btn>
          <Btn variant="secondary" onClick={onBack}>← Back</Btn>
        </div>
      </div>

      {picking && (
        <ModelPickerModal
          models={bstModels}
          activeId={activeBstId}
          onPick={id => { setActiveBstId(id); setBstOn(true); setPicking(false); }}
          onClose={() => setPicking(false)}
        />
      )}
    </div>
  );
}
