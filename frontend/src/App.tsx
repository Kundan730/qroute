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
 *
 * The status chip lives on the navy bar, which constrains what it may be
 * coloured. A crimson or teal glyph on navy does not clear the contrast a
 * meaningful graphic needs, so the three states are told apart by the *shape*
 * of the mark and by the chip's treatment: online and connecting are outlined
 * chips with pale marks, while offline inverts to a solid crimson chip with
 * white type - the one thing on the bar that cannot be mistaken for chrome.
 */

import type { CSSProperties } from 'react';
import { useEffect, useState } from 'react';
import { Mark } from './components/ui';
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

/** Shared geometry for the two small chips at the right of the top bar. */
const CHIP: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '3px 9px',
  borderRadius: 'var(--radius)',
  fontFamily: 'var(--display)',
  fontSize: 10,
  fontWeight: 600,
  letterSpacing: '0.12em',
  textTransform: 'uppercase',
  whiteSpace: 'nowrap',
};

export default function App() {
  const [page, setPage] = useState<PageId>('map');
  const backend = useAppStore((s) => s.backend);
  const health = useAppStore((s) => s.health);
  const error = useAppStore((s) => s.error);
  const bootstrap = useAppStore((s) => s.bootstrap);
  const recheck = useAppStore((s) => s.recheck);
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

  // And keep checking while it is up, so the badge tells the truth in the other
  // direction too. A run in flight already holds an open stream that would
  // report its own failure, and health is a cheap read of state already in
  // memory, so ten seconds costs nothing and never blocks a solve.
  useEffect(() => {
    if (backend !== 'online') return undefined;
    const timer = window.setInterval(() => void recheck(), 10000);
    return () => window.clearInterval(timer);
  }, [backend, recheck]);

  const offline = backend === 'offline';

  const statusChip: CSSProperties = offline
    ? { ...CHIP, background: 'var(--bad)', color: 'var(--panel)', fontWeight: 700 }
    : { ...CHIP, border: '1px solid var(--navy-400)', color: 'var(--navy-050)' };
  // Mark colours are chosen against the navy bar, not against a white panel.
  const markColor = offline
    ? 'var(--panel)'
    : backend === 'online'
      ? 'var(--navy-050)'
      : 'var(--warn)';
  const statusLabel =
    backend === 'online' ? 'backend online' : offline ? 'backend offline' : 'connecting…';

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
            <span style={{ ...CHIP, border: '1px solid var(--navy-400)', color: 'var(--accent-dim)' }}>
              <Mark shape="disc" size={6} />
              solver running
            </span>
          )}
          {health?.version && (
            <span className="mono" style={{ color: 'var(--navy-100)' }}>
              v{health.version}
            </span>
          )}
          <span style={statusChip} role="status">
            <span style={{ color: markColor, display: 'inline-flex' }}>
              <Mark
                shape={backend === 'online' ? 'disc' : offline ? 'square' : 'ring'}
                size={6}
              />
            </span>
            {statusLabel}
          </span>
          {offline && (
            <button type="button" className="btn small" onClick={() => void bootstrap()}>
              Retry
            </button>
          )}
        </div>
      </header>

      {offline && (
        <div
          role="alert"
          style={{
            display: 'flex',
            alignItems: 'baseline',
            flexWrap: 'wrap',
            gap: '4px 12px',
            padding: '8px 18px',
            background: 'var(--bad)',
            color: 'var(--panel)',
            fontSize: 12,
            lineHeight: 1.5,
          }}
        >
          <span
            style={{
              fontFamily: 'var(--display)',
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
            }}
          >
            Backend offline
          </span>
          <span>
            Cannot reach the API{error ? ` — ${error}` : ''}. Every figure on the
            other pages is either blank or stale. Start it with{' '}
            <code style={{ fontFamily: 'var(--mono)' }}>python -m qroute.api.app</code>;
            this page retries every few seconds.
          </span>
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
