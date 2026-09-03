/**
 * The live state of a running solve, drawn over the map.
 *
 * A spinner would be the obvious thing here and it would be the wrong thing.
 * The backend streams a tick per sampled iteration carrying the elapsed time,
 * the iteration count, the evaluation count and the incumbent cost, and the run
 * has a fixed time budget - so progress is genuinely determinate and the search
 * is genuinely watchable. An indeterminate spinner would throw all of that away
 * and tell the viewer only that the page has not frozen.
 *
 * What this shows instead, in the order it is read:
 *
 *   1. that a solve is running, and how far through its budget it is
 *   2. the best cost found so far, and how much better that is than the first
 *      solution the search produced - the number that answers "is it working"
 *   3. the rate: iterations, and evaluations per second
 *   4. the shape of the convergence so far, as a sparkline
 *
 * The improvement figure is the honest one to lead with, because a cost falling
 * from its own starting point is what optimisation looks like from outside. It
 * is measured against the first tick of this run rather than against a
 * best-known value, so it stays meaningful on a road network where no reference
 * cost exists.
 */

import { useEffect, useRef, useState } from 'react';
import type { RunTick } from '../../api/types';

const SPARK_W = 208;
const SPARK_H = 26;

function formatCost(value: number): string {
  if (!Number.isFinite(value)) return '—';
  if (value >= 10000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function sparkPath(ticks: RunTick[]): string {
  const points = ticks.filter((t) => Number.isFinite(t.best_cost));
  if (points.length < 2) return '';
  const costs = points.map((t) => t.best_cost);
  const hi = Math.max(...costs);
  const lo = Math.min(...costs);
  const span = hi - lo || 1;
  const step = SPARK_W / (points.length - 1);
  return points
    .map((t, i) => {
      const x = i * step;
      // Inverted: a falling cost should read as a falling line.
      const y = SPARK_H - ((hi - t.best_cost) / span) * (SPARK_H - 3) - 1.5;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

export function RunOverlay({
  streaming,
  starting,
  ticks,
  budgetSeconds,
  algorithm,
  onCancel,
}: {
  streaming: boolean;
  starting: boolean;
  ticks: RunTick[];
  budgetSeconds: number;
  algorithm: string;
  onCancel: () => void;
}) {
  const latest = ticks.length > 0 ? ticks[ticks.length - 1] : null;
  const first = ticks.length > 0 ? ticks[0] : null;

  // Flash the cost when the incumbent improves, so an improvement is felt and
  // not merely displayed. Held on a ref so a re-render for any other reason
  // does not retrigger it.
  const previousBest = useRef<number | null>(null);
  const [justImproved, setJustImproved] = useState(false);
  useEffect(() => {
    const best = latest?.best_cost;
    if (best === undefined || !Number.isFinite(best)) return undefined;
    if (previousBest.current !== null && best < previousBest.current - 1e-9) {
      setJustImproved(true);
      const timer = window.setTimeout(() => setJustImproved(false), 620);
      previousBest.current = best;
      return () => window.clearTimeout(timer);
    }
    previousBest.current = best;
    return undefined;
  }, [latest?.best_cost]);

  useEffect(() => {
    if (!streaming && !starting) previousBest.current = null;
  }, [streaming, starting]);

  if (!streaming && !starting) return null;

  const elapsed = latest?.elapsed ?? 0;
  const fraction = budgetSeconds > 0 ? Math.min(elapsed / budgetSeconds, 1) : 0;
  const improvement =
    first && latest && Number.isFinite(first.best_cost) && first.best_cost > 0
      ? ((first.best_cost - latest.best_cost) / first.best_cost) * 100
      : null;
  const rate =
    latest && latest.elapsed > 0.2 ? latest.evaluations / latest.elapsed : null;

  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        position: 'absolute',
        top: 12,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 620,
        minWidth: 302,
        background: 'rgba(255, 255, 255, 0.96)',
        border: '1px solid var(--border-strong)',
        borderRadius: 'var(--radius)',
        boxShadow: 'var(--shadow-float)',
        backdropFilter: 'blur(4px)',
        overflow: 'hidden',
      }}
    >
      {/* Determinate, because the budget is fixed and the elapsed time is
          streamed. The stripe animation only conveys liveness; the fill
          conveys the actual progress. */}
      <div style={{ position: 'relative', height: 3, background: 'var(--border)' }}>
        <div
          style={{
            position: 'absolute',
            inset: '0 auto 0 0',
            width: `${(fraction * 100).toFixed(1)}%`,
            background: 'var(--navy)',
            transition: 'width 240ms linear',
          }}
        />
        <div className="run-stripe" style={{ position: 'absolute', inset: 0 }} />
      </div>

      <div style={{ padding: '9px 12px 10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 8 }}>
          <span className="run-pulse" aria-hidden />
          <span
            style={{
              fontFamily: 'var(--display)',
              fontSize: 9.5,
              fontWeight: 700,
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              color: 'var(--navy-300)',
            }}
          >
            {starting && ticks.length === 0
              ? `${algorithm} starting`
              : `${algorithm} searching`}
          </span>
          <span
            className="mono"
            style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-dim)' }}
          >
            {elapsed.toFixed(1)} / {budgetSeconds.toFixed(0)} s
          </span>
          <button
            type="button"
            className="btn small"
            onClick={onCancel}
            style={{ marginLeft: 2 }}
          >
            Cancel
          </button>
        </div>

        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16 }}>
          <div style={{ minWidth: 96 }}>
            <div
              style={{
                fontFamily: 'var(--display)',
                fontSize: 9,
                fontWeight: 600,
                letterSpacing: '0.13em',
                textTransform: 'uppercase',
                color: 'var(--navy-300)',
                marginBottom: 2,
              }}
            >
              Best so far
            </div>
            <div
              style={{
                fontFamily: 'var(--display)',
                fontSize: 20,
                fontWeight: 600,
                letterSpacing: '-0.02em',
                lineHeight: 1.05,
                fontVariantNumeric: 'tabular-nums',
                color: justImproved ? 'var(--rose)' : 'var(--text)',
                transition: 'color 420ms ease',
              }}
            >
              {latest ? formatCost(latest.best_cost) : '—'}
            </div>
            {improvement !== null && improvement > 0.005 && (
              <div
                className="mono"
                style={{ fontSize: 10, color: 'var(--text-faint)', marginTop: 2 }}
              >
                {improvement.toFixed(1)}% below first
              </div>
            )}
          </div>

          <div style={{ flex: '1 1 auto', minWidth: 0 }}>
            <svg
              width="100%"
              height={SPARK_H}
              viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
              preserveAspectRatio="none"
              aria-hidden
              style={{ display: 'block' }}
            >
              <path
                d={sparkPath(ticks)}
                fill="none"
                stroke="var(--navy-400)"
                strokeWidth={1.4}
                vectorEffect="non-scaling-stroke"
                strokeLinejoin="round"
              />
            </svg>
            <div
              className="mono"
              style={{
                display: 'flex',
                gap: 12,
                fontSize: 10,
                color: 'var(--text-faint)',
                marginTop: 3,
              }}
            >
              <span>{latest ? latest.iteration.toLocaleString() : 0} iter</span>
              {rate !== null && <span>{Math.round(rate).toLocaleString()} eval/s</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
