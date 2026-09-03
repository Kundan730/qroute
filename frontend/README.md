# qroute web interface

The React front end of the qroute platform. It is served in production by the
Python backend from `dist/`, so there is nothing to deploy separately.

```bash
npm install
npm run build        # writes dist/, which the API serves at /
npm run dev          # Vite dev server, expects the API on :8000
```

The design system lives in `src/styles/global.css` and the two
information-carrying colour scales live in `src/lib/colors.ts`. Those two files
are the source of truth; components take colours from their tokens and never
declare their own.
