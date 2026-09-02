/**
 * Application shell: the top bar, the page switch, and the backend status.
 *
 * Routing is a piece of state rather than a router library. There are four
 * pages, they are always all mounted in the sense that their stores persist,
 * and the platform is presented rather than deep-linked, so a URL router would
 * add a dependency and a build step for no benefit. The one thing the shell
 * insists on is that the backend's reachability is visible at all times: a demo
 * that silently shows stale numbers when the API has died is worse than one
 * that says so.
 */

import { useEffect, useState } from 'react';
import { BenchmarkPage } from './pages/BenchmarkPage';
import { MapPage } from './pages/MapPage';
import { MethodPage } from './pages/MethodPage';
import { SolverPage } from './pages/SolverPage';
import { useAppStore } from './store/appStore';
import { useRunStore } from './store/runStore';
import './styles/global.css';

type PageId = 'map' | 'solver' | 'benchmark' | 'method';

const PAGES: { id: PageId; label: string }[] = [
  { id: 'map', label: 'Map' },
  { id: 'solver', label: 'Solver' },
  { id: 'benchmark', label: 'Benchmark' },
  { id: 'method', label: 'Method' },
];

export default function App() {
  const [page, setPage] = useState<PageId>('map');
  const backend = useAppStore((s) => s.backend);
  const health = useAppStore((s) => s.health);
  const error = useAppStore((s) => s.error);
  const bootstrap = useAppStore((s) => s.bootstrap);
  const streaming = useRunStore((s) => s.streaming);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  // Retry quietly while the backend is down, so starting the API after the page
  // is already open recovers without the operator having to reload.
  useEffect(() => {
    if (backend !== 'offline') return undefined;
    const timer = window.setInterval(() => void bootstrap(), 4000);
    return () => window.clearInterval(timer);
  }, [backend, bootstrap]);

  const statusTone =
    backend === 'online' ? 'var(--ok)' : backend === 'offline' ? 'var(--bad)' : 'var(--warn)';
  const statusLabel =
    backend === 'online'
      ? 'backend online'
      : backend === 'offline'
        ? 'backend unavailable'
        : 'connecting…';

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">qroute</span>
          <span className="brand-sub">
            Quantum-inspired route optimisation for congested road networks
          </span>
        </div>

        <nav className="nav">
          {PAGES.map((p) => (
            <button
              key={p.id}
              type="button"
              aria-current={page === p.id ? 'page' : undefined}
              onClick={() => setPage(p.id)}
            >
              {p.label}
            </button>
          ))}
        </nav>

        <div className="topbar-right">
          {streaming && (
            <span style={{ color: 'var(--accent-text)' }}>solver running</span>
          )}
          {health?.version && <span className="mono">v{health.version}</span>}
          <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: statusTone }}>
            <span className="dot" />
            {statusLabel}
          </span>
          {backend === 'offline' && (
            <button type="button" className="btn small" onClick={() => void bootstrap()}>
              Retry
            </button>
          )}
        </div>
      </header>

      {backend === 'offline' && (
        <div
          style={{
            padding: '7px 16px',
            background: '#2a1c1b',
            borderBottom: '1px solid #6b3a36',
            color: '#e0b5ae',
            fontSize: 12,
          }}
        >
          Cannot reach the API{error ? ` — ${error}` : ''}. Start it with{' '}
          <code>python -m qroute.api.app</code>; this page retries
          every few seconds.
        </div>
      )}

      <main className="page">
        {page === 'map' && <MapPage />}
        {page === 'solver' && <SolverPage />}
        {page === 'benchmark' && <BenchmarkPage />}
        {page === 'method' && <MethodPage />}
      </main>
    </div>
  );
}
