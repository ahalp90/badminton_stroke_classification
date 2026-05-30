import { useState } from 'react';
import { ThemeProvider, useTheme, NavBar, ScreenErrorBoundary } from './shared';
import { LibraryScreen } from './library-screen';
import { MarkupScreen } from './markup-screen';
import { ConfigureScreen } from './configure-screen';
import { ProgressScreen } from './progress-screen';
import { ResultsScreen } from './results-screen';
import { ModelResultsScreen } from './model-results-screen';

// ──── Wizard stage order ─────────────────────────────────────────────────────────────────────────
/** Ordered list of wizard stages (excludes Model Results showcase) */
const ORDER = ['library', 'markup', 'configure', 'progress', 'results'];

// ──── Dev fixtures ───────────────────────────────────────────────────────────────────────────────

/**
 * Stub data used during development to allow downstream stage-jumping after a video is selected
 * 
 * NOTE: `video` is deliberately excluded - forward navigation without a chose video now blocks 
 * (user must pick from Library or upload a file)
 * 
 * NOTE: `markup` is deliberately excluded — the user must complete the markup steps before
 * configure is accessible.
 * 
 * TODO: Remove DEV_FIXTURES during clean-up, when live inference implementation complete
 */
const DEV_FIXTURES = {
  task: {
    taskName: 'Demo task — fixture',
    enabled: { A: true, B: false },
  },
};

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
   *        no video          → library only
   *        video, no markup  → library or markup
   *        video + markup    → library, markup, or configure (and beyond once task exists)
   *  3. Backward within wizard - always allowed.
   *  4. Forward without a video - blocked.
   *  5. Forward past markup without saved markup - blocked.
   *  6. Otherwise - advance, seeding task fixture if needed.
   *  TODO: Update Step 6 prior to deployment.
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

    // Guard 6: Advance, seeding task fixture if jumping past configure.
    const s = task ?? { ...DEV_FIXTURES.task };
    if (dst >= ORDER.indexOf('progress') && !task) setTask(s);
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
        onComplete={(result) => {
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