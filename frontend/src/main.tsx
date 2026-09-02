/**
 * Entry point.
 *
 * `StrictMode` is deliberately not used. It double-invokes effects in
 * development, which for this application means opening two Server-Sent Event
 * streams per run and creating two Leaflet layer groups per render, and the
 * resulting duplicated markers and doubled tick histories look exactly like
 * bugs in the platform rather than a development-mode artefact. The effects
 * here all clean up after themselves; the double-mount check is not worth
 * making the live demo untrustworthy.
 */

import { createRoot } from 'react-dom/client'
import App from './App.tsx'

const container = document.getElementById('root')
if (!container) throw new Error('the #root element is missing from index.html')

createRoot(container).render(<App />)
