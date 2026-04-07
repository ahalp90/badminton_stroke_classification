import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import { Home, Analysis, Results } from './pages'

import './App.css'

function App() {

  return (
    <>
      <Header/>
      <main className="main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="analysis" element={<Analysis />} />
          <Route path="results" element={<Results />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
    </>
  )
}

export default App
