import { useState } from 'react';
import { ThemeProvider, useTheme, NavBar } from './shared';
import { LibraryScreen } from './library-screen';
import { MarkupScreen } from './markup-screen';
import { ConfigureScreen, ProgressScreen } from './configure-screen';
import { ResultsScreen } from './results-screen';

const ORDER = ['library', 'markup', 'configure', 'progress', 'results'];

function HBAStrokeClassifier() {
  const [screen, setScreen] = useState('library');
  const [video,  setVideo]  = useState(null);
  const [markup, setMarkup] = useState(null);
  const [task,   setTask]   = useState(null);

  const navigate = target => {
    const cur = ORDER.indexOf(screen);
    const dst = ORDER.indexOf(target);
    if (dst <= cur) setScreen(target);
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
  };

  return (
    <div style={{ minHeight: '100vh', background: t.bg }}>
      <NavBar screen={screen} onNavigate={navigate} />
      {screens[screen]}
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
