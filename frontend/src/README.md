# HBA Stroke Classifier

A React UI for classifying badminton strokes from match video.

The interface walks through a four-step pipeline:

1. **Select Video** — choose a match from the ShuttleSet validation set or upload your own footage
2. **Markup** — align the court boundary, select the target player, and isolate a stroke segment
3. **Configure** — choose classification models and tune inference parameters
4. **Results** — per-stroke predictions, shot distribution charts, model comparison, and class activation maps

## Models

| Model   | Architecture                                                    | Status                               |
| ------- | --------------------------------------------------------------- | ------------------------------------ |
| Model A | MMPose keypoints + TrackNetV3 shuttle → TCN → Transformer (BST) | Active                               |
| Model B | TBD                                                             | Reserved — no architecture committed |

## Integration

This is a standalone React component tree. Drop it into any React project:

```jsx
import './styles.css'
import App from './app'

function MyPage() {
  return <App />
}
```

`App` (the default export from `app.jsx`) includes its own `ThemeProvider` — no wrapping required.

### Files

```
app.jsx               Root component + screen router
shared.jsx            Theme context, shared components (Btn, Card, Badge, etc.)
library-screen.jsx    Step 1: video selection
markup-screen.jsx     Step 2: court boundary, player selection, timeframe
configure-screen.jsx  Step 3: model selection, parameters + Step 4: progress view
results-screen.jsx    Step 5: results tabs
styles.css            Global resets, scrollbar styles, range input appearance
uploads/              Static assets (logo)
```

### Logo path

The logo is imported in `shared.jsx` as `./uploads/logo-1777443863198.png`. Move the file and update the import path to match your project's asset conventions.

### Fonts

`styles.css` loads Space Grotesk and JetBrains Mono from Google Fonts. If your project already manages fonts, remove the `@import` line and ensure both families are available.

## Theme

The UI ships with dark and light themes, toggled via the navbar button. Theme tokens are exported from `shared.jsx` as `DARK` and `LIGHT` if you need to reference them elsewhere.

```js
import { DARK, LIGHT } from './shared'
```
