import { useState } from 'react';
import { ThemeProvider, useTheme, NavBar, ScreenErrorBoundary } from './shared';
import { LibraryScreen } from './library-screen';
import { MarkupScreen } from './markup-screen';
import { ConfigureScreen } from './configure-screen';
import { ProgressScreen } from './progress-screen';
import { ResultsScreen } from './results-screen';
import { ProjectScreen } from './project-screen';

const ORDER = ['library', 'markup', 'configure', 'progress', 'results'];

const DEV_FIXTURES = {
  video: {
    youtubeId: 'dQw4w9WgXcQ',
    match: 'Player One vs Player Two',
    tournament: 'Demo Tournament 2026',
    annotated: true,
    strokeTimes: [],
  },
  markup: {
    player: 1,
    timeframe: { duration: 30 },
  },
  task: {
    taskName: 'Demo task — fixture',
    enabled: { A: true, B: false },
  },
};

function HBAStrokeClassifier() {
  const [screen, setScreen] = useState('library');
  const [video,  setVideo]  = useState(null);
  const [markup, setMarkup] = useState(null);
  const [task,   setTask]   = useState(null);

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
    // Returning to the wizard from the Project page: restore wizard state.
    if (screen === 'project') {
      setScreen(target);
      return;
    }
    const cur = ORDER.indexOf(screen);
    const dst = ORDER.indexOf(target);
    if (dst <= cur) {
      setScreen(target);
      return;
    }
    const v = video ?? DEV_FIXTURES.video;
    const m = markup ?? { ...DEV_FIXTURES.markup, video: v };
    const t = task ?? { ...DEV_FIXTURES.task, markup: m };
    if (!video) setVideo(v);
    if (dst >= ORDER.indexOf('configure') && !markup) setMarkup(m);
    if (dst >= ORDER.indexOf('progress') && !task) setTask(t);
    setScreen(target);
  };

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
        onSubmit={t => { setTask(t); setScreen('progress'); }}
        onBack={() => setScreen('markup')}
      />
    ),
    progress: (
      <ProgressScreen
        task={task}
        onComplete={() => setScreen('results')}
      />
    ),
    results: (
      <ResultsScreen
        task={task}
        onNew={() => { setScreen('library'); setVideo(null); setMarkup(null); setTask(null); }}
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
