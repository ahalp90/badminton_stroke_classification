import { useState } from 'react';
import { ThemeProvider, useTheme, ScreenErrorBoundary } from './shared';
import { NavBar } from './components/NavBar';
import { LibraryScreen } from './library-screen';
import { MarkupScreen } from './markup-screen';
import { ConfigureScreen } from './configure-screen';
import { ProgressScreen } from './progress-screen';
import { ResultsScreen } from './results-screen';
import { ModelResultsScreen } from './model-results-screen';

// ──── Wizard stage order ─────────────────────────────────────────────────────────────────────────
/** Ordered list of wizard stages (excludes Model Results showcase) */
const ORDER = ['library', 'markup', 'configure', 'progress', 'results'];

// ──── Component ──────────────────────────────────────────────────────────────────────────────────
function HBAStrokeClassifier() {
  // ──── Wizard state ─────────────────────────────────────────────────────────────────────────────
  const [screen, setScreen] = useState('library');
  const [video,  setVideo]  = useState(null);
  const [markup, setMarkup] = useState(null);
  const [task,   setTask]   = useState(null);

  // ──── Actions ──────────────────────────────────────────────────────────────────────────────────
  const resetAll = () => {
    setVideo(null);
    setMarkup(null);
    setTask(null);
    setScreen('library');
  };

  /**
   * NavBar navigation handler. Normal wizard progression (onNext/onBack inside each screen)
   * calls setScreen directly and never goes through here - this is only for navbar jumps.
   * 
   * Guards (in order):
   *  1. Jumping to Model Results - always allowed.
   *  2. Returning from Model Results - allowed subject to wizard state:
   *        no video                → library only
   *        video, no markup        → library or markup
   *        video + markup, no task → library, markup, or configure
   *        video + markup + task   → any wizard screen
   *  3. Backward within wizard - always allowed.
   *  4. Forward without a video - blocked.
   *  5. Forward past markup without saved markup - blocked.
   *  6. Forward past configure without saved task - blocked.
   *  7. Otherwise - advance.
   */
  const navigate = target => {
    // Guard 1: Model Results page is outside the wizard pipeline — jump freely.
    if (target === 'model-results') {
      setScreen('model-results');
      return;
    }

    // Guard 2: Returning from Model Results - cap by wizard state.
    if (screen === 'model-results') {
      if (!video) { setScreen('library'); return; }
      if (ORDER.indexOf(target) >= ORDER.indexOf('configure') && !markup) return;
      if (ORDER.indexOf(target) >= ORDER.indexOf('progress') && !task) return;
      setScreen(target);
      return;
    }

    // Guard 3: Backward navigation within the wizard is always allowed.
    const cur = ORDER.indexOf(screen);
    const dst = ORDER.indexOf(target);
    if (dst <= cur) {
      setScreen(target);
      return;
    }

    // Guard 4: Forward navigation requires a real video.
    if (!video) return;

    // Guard 5: Reaching configure or beyond requires completed markup.
    if (dst >= ORDER.indexOf('configure') && !markup) return;

    // Guard 6: Reaching progress or beyond requires a submitted task.
    if (dst >= ORDER.indexOf('progress') && !task) return;

    // Guard 7: All guards passed - advance.
    // TODO: May need additional gating here so user cannot click through to Results screen before job is complete - TBD
    setScreen(target);
  };

  // ──── Render ─────────────────────────────────────────────────────────────────────────────
  const { t } = useTheme();

  const screens = {
    library: (
      <LibraryScreen
        onNext={v => { setVideo(v); setScreen('markup'); }}
      />
    ),
    markup: (
      <MarkupScreen
        video={video}
        onNext={m => { setMarkup(m); setScreen('configure'); }}
        onBack={() => setScreen('library')}
      />
    ),
    configure: (
      <ConfigureScreen
        markup={markup}
        onSubmit={s => { setTask(s); setScreen('progress'); }}
        onBack={() => setScreen('markup')}
      />
    ),
    progress: (
      <ProgressScreen
        task={task}
        onComplete={result => {
          if (result) setTask(prev => ({ ...prev, uploadResult: result }));
          setScreen('results');
        }}
      />
    ),
    results: (
      <ResultsScreen
        task={task}
        onNew={resetAll}
      />
    ),
    'model-results': <ModelResultsScreen />,
  };

  return (
    <div style={{ minHeight: '100vh', background: t.bg }}>
      <NavBar screen={screen} onNavigate={navigate} />
      <ScreenErrorBoundary resetKey={screen} onReset={resetAll}>
        {screens[screen]}
      </ScreenErrorBoundary>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <HBAStrokeClassifier />
    </ThemeProvider>
  );
}