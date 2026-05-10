/*import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)*/
import { createRoot } from 'react-dom/client'
import '../hba-stroke-classifier/styles.css'
import App from '../hba-stroke-classifier/app'

createRoot(document.getElementById('root')).render(<App />)
