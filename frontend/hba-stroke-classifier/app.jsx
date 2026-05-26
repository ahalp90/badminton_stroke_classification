import { useState } from 'react';
import { ThemeProvider, useTheme, NavBar, ScreenErrorBoundary } from './shared';
import { LibraryScreen } from './library-screen';
import { MarkupScreen } from './markup-screen';
import { ConfigureScreen } from './configure-screen';
import { ProgressScreen } from './progress-screen';
import { ResultsScreen } from './results-screen';
import { ProjectScreen } from './project-screen';

// ──── Wizard stage order ─────────────────────────────────────────────────────────────────────────
/** Ordered list of wizard stages (excludes Project showcase) */
const ORDER = ['library', 'markup', 'configure', 'progress', 'results'];

// ──── Dev fixtures ───────────────────────────────────────────────────────────────────────────────

/**
 * Stub data used during development to allow downstream stage-jumping after a video is selected
 * 
 * NOTE: `video` is deliberately excluded - forward navigation without a chose video now blocks 
 * (user must pick from Library or upload a file)
 * 
 * TODO: Remove DEV_FIXTURES during clean-up, when live inference implementation complete
 */
const DEV_FIXTURES = {
  markup: {
    player: 1,
    timeframe: { duration: 30 },
  },
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

  const navigate = target => {
    // The Project showcase is outside the wizard pipeline — jump freely.
    if (target === 'project') {
      setScreen('project');
      return;
    }
    
    const cur = ORDER.indexOf(screen);
    const dst = ORDER.indexOf(target);
    if (dst <= cur) {
      setScreen(target);
      return;
    }
    // Forward navigation requires a real video. No dummy fallback - the 
    // user must explicitly select from the Match Library or upload a file 
    // first.
    if (!video) return;
    const m = markup ?? { ...DEV_FIXTURES.markup, video };
    const s = task ?? { ...DEV_FIXTURES.task, markup: m };
    if (dst >= ORDER.indexOf('configure') && !markup) setMarkup(m);
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
    project: <ProjectScreen />,
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