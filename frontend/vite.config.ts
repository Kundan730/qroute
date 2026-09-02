/**
 * Vite configuration.
 *
 * The dev server proxies `/api` to the FastAPI process so the frontend is
 * same-origin in development as well as in production, which keeps CORS out of
 * the picture and lets `EventSource` connect to the run stream without any
 * special handling. `changeOrigin` is off because the target is localhost, and
 * buffering must stay disabled or Server-Sent Events arrive in one lump at the
 * end of a run instead of per iteration.
 */

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const API_TARGET = process.env.QROUTE_API ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: false,
        // Server-Sent Events must not be buffered by the proxy.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (String(proxyRes.headers['content-type'] ?? '').includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
            }
          })
        },
      },
    },
  },
  build: {
    // The three heavyweight dependencies are split out so a change to the
    // application code does not invalidate the cached vendor chunks.
    rollupOptions: {
      output: {
        manualChunks: (id: string) => {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('leaflet')) return 'leaflet'
          if (id.includes('recharts') || id.includes('d3-')) return 'charts'
          if (id.includes('/react') || id.includes('scheduler')) return 'react'
          return undefined
        },
      },
    },
  },
})
