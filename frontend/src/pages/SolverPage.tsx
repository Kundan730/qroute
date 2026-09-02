/**
 * The solver page: configure a run, start it, and watch it converge.
 *
 * The live panels are driven entirely by the Server-Sent Event stream, one
 * message per recorded iteration. Everything shown is a number the optimiser
 * actually reported: the incumbent cost, the population mean, the diversity of
 * the swarm, and the evaluation counter, from which the throughput figure is a
 * finite difference over the last dozen ticks rather than a smoothed estimate.
 *
 * The gap to the best-known solution is shown whenever the instance has one on
 * disk, and it is shown signed: a run that is 3 % worse than the literature
 * says so.
 */

import { useMemo, useState } from 'react';
import type { InstanceSummary, RunRequest } from '../api/types';
import { ConvergenceChart, DiversityChart } from '../components/charts';
import { Badge, Field, KeyValue, Notice, Panel, RailSection, Stat, StatGrid } from '../components/ui';
import { fmt, fmtCompact, fmtInt, fmtPercent, fmtSeconds } from '../lib/format';
import { groupInstances, useAppStore } from '../store/appStore';
import { evaluationRate, useRunStore } from '../store/runStore';

const FAMILY_LABELS: Record<string, string> = {
  cvrp: 'CVRPLIB — capacitated VRP',
  vrptw: 'Solomon — VRP with time windows',
  network: 'Generated from a road network',
};

export function SolverPage() {
  const backend = useAppStore((s) => s.backend);
  const algorithms = useAppStore((s) => s.algorithms);
  const instances = useAppStore((s) => s.instances);
  const run = useRunStore();

  const [chosenAlgorithm, setAlgorithm] = useState('qpso');
  const [chosenInstance, setInstanceName] = useState('');
  const [seed, setSeed] = useState(20260920);
  const [maxSeconds, setMaxSeconds] = useState(15);
  const [maxIterations, setMaxIterations] = useState(100000);
  // Overrides are stored with the algorithm they belong to, so a value that is
  // meaningful to one solver is never silently sent to another when the
  // selection changes. Deriving this beats resetting it from an effect.
  const [override, setOverride] = useState<{
    algorithm: string;
    values: Record<string, number | boolean | string>;
  }>({ algorithm: 'qpso', values: {} });

  const grouped = useMemo(() => groupInstances(instances), [instances]);

  // The catalogues arrive asynchronously, so the effective selection is derived
  // from what is actually available rather than corrected afterwards in an
  // effect, which would render once with a name the backend does not know.
  const algorithm =
    algorithms.some((a) => a.name === chosenAlgorithm)
      ? chosenAlgorithm
      : (algorithms[0]?.name ?? chosenAlgorithm);
  const instanceName =
    instances.some((i) => i.name === chosenInstance)
      ? chosenInstance
      : (instances.find((i) => i.name === 'A-n32-k5')?.name ?? instances[0]?.name ?? '');

  const selected: InstanceSummary | null =
    instances.find((i) => i.name === instanceName) ?? null;
  const spec = algorithms.find((a) => a.name === algorithm);
  const params = override.algorithm === algorithm ? override.values : {};

  const setParams = (values: Record<string, number | boolean | string>) =>
    setOverride({ algorithm, values });

  const ticks = run.ticks;
  const last = ticks.length > 0 ? ticks[ticks.length - 1] : null;
  const bks = run.status?.bks ?? selected?.bks ?? null;
  const bestCost = run.status?.best_cost ?? last?.best_cost ?? null;
  const gap =
    bks !== null && bestCost !== null && bks > 0 ? (100 * (bestCost - bks)) / bks : null;
  const rate = evaluationRate(ticks);

  function start() {
    if (!instanceName) return;
    const request: RunRequest = {
      algorithm,
      instance: instanceName,
      seed,
      max_seconds: maxSeconds,
      max_iterations: maxIterations,
      params: Object.keys(params).length > 0 ? params : undefined,
    };
    void run.start(request, 'solver');
  }

  if (backend !== 'online') {
    return (
      <div className="page-scroll">
        <Notice kind="error">
          <strong>Backend unavailable.</strong> Solver runs execute in the Python
          process; there is nothing meaningful to show without it. Start it with{' '}
          <code>python -m qroute.api.app</code> and reload.
        </Notice>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', height: '100%', minHeight: 0 }}>
      <aside className="rail">
        <RailSection title="Problem">
          <Field label="Instance" hint={selected ? describeInstance(selected) : undefined}>
            <select value={instanceName} onChange={(e) => setInstanceName(e.target.value)}>
              {instances.length === 0 && <option value="">no instances loaded</option>}
              {Array.from(grouped.entries()).map(([family, list]) => (
                <optgroup key={family} label={FAMILY_LABELS[family] ?? family}>
                  {list.map((i) => (
                    <option key={i.name} value={i.name}>
                      {i.name} ({i.n_customers})
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </Field>
          {selected && (
            <>
              <KeyValue label="Customers" value={fmtInt(selected.n_customers)} />
              <KeyValue label="Capacity" value={fmtInt(selected.capacity)} />
              <KeyValue
                label="Fleet"
                value={selected.n_vehicles === null ? 'unlimited' : fmtInt(selected.n_vehicles)}
              />
              <KeyValue
                label="Best known"
                value={selected.bks === null ? 'not on disk' : fmt(selected.bks, 1)}
              />
              <KeyValue label="Time windows" value={selected.has_time_windows ? 'yes' : 'no'} />
            </>
          )}
        </RailSection>

        <RailSection title="Search">
          <Field label="Algorithm" hint={spec?.description}>
            <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
              {algorithms.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Seed" hint="The same seed and budget reproduce the run exactly.">
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </Field>
          <Field label={`Time limit — ${maxSeconds} s`}>
            <input
              type="range"
              min={2}
              max={120}
              step={1}
              value={maxSeconds}
              onChange={(e) => setMaxSeconds(Number(e.target.value))}
            />
          </Field>
          <Field label="Iteration cap" hint="Whichever limit is reached first stops the run.">
            <input
              type="number"
              value={maxIterations}
              min={1}
              onChange={(e) => setMaxIterations(Math.max(1, Number(e.target.value)))}
            />
          </Field>
          <div className="btn-row">
            <button
              type="button"
              className="btn primary"
              disabled={run.starting || run.streaming || !instanceName}
              onClick={start}
            >
              {run.starting ? 'Starting…' : 'Start run'}
            </button>
            <button
              type="button"
              className="btn danger"
              disabled={!run.streaming}
              onClick={() => void run.cancel()}
            >
              Cancel
            </button>
          </div>
          {run.error && (
            <div style={{ marginTop: 8 }}>
              <Notice kind="error">{run.error}</Notice>
            </div>
          )}
        </RailSection>

        {spec?.params && spec.params.length > 0 && (
          <RailSection title="Parameters">
            {spec.params.map((p) => (
              <Field key={p.name} label={p.name.replace(/_/g, ' ')} hint={p.description}>
                {p.kind === 'bool' ? (
                  <select
                    value={String(params[p.name] ?? p.default)}
                    onChange={(e) => setParams({ ...params, [p.name]: e.target.value === 'true' })}
                  >
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                ) : p.kind === 'choice' ? (
                  <select
                    value={String(params[p.name] ?? p.default)}
                    onChange={(e) => setParams({ ...params, [p.name]: e.target.value })}
                  >
                    {(p.choices ?? []).map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    value={String(params[p.name] ?? p.default)}
                    min={p.min}
                    max={p.max}
                    step={p.step ?? (p.kind === 'int' ? 1 : 0.01)}
                    onChange={(e) => setParams({ ...params, [p.name]: Number(e.target.value) })}
                  />
                )}
              </Field>
            ))}
            <button
              type="button"
              className="btn small"
              onClick={() => setParams({})}
              disabled={Object.keys(params).length === 0}
            >
              Reset to defaults
            </button>
          </RailSection>
        )}
      </aside>

      <div className="page-scroll" style={{ flex: '1 1 auto', minWidth: 0 }}>
        <StatGrid columns={6}>
          <Stat
            label="Best cost"
            value={bestCost === null ? '—' : fmt(bestCost, 2)}
            sub={run.status ? run.status.instance : instanceName || '—'}
          />
          <Stat
            label="Gap to best known"
            value={gap === null ? '—' : fmtPercent(gap, 2)}
            tone={gap === null ? undefined : gap <= 0.001 ? 'ok' : gap < 2 ? 'warn' : 'bad'}
            sub={bks === null ? 'no reference on disk' : `reference ${fmt(bks, 1)}`}
          />
          <Stat
            label="Iterations"
            value={fmtInt(last?.iteration ?? run.status?.iterations ?? null)}
          />
          <Stat
            label="Evaluations"
            value={fmtCompact(last?.evaluations ?? run.status?.evaluations ?? 0)}
            sub={Number.isFinite(rate) ? `${fmtCompact(rate)} per second` : 'measuring…'}
          />
          <Stat
            label="Elapsed"
            value={fmtSeconds(last?.elapsed ?? run.status?.seconds ?? null)}
            sub={`budget ${maxSeconds} s`}
          />
          <Stat
            label="State"
            value={
              <span style={{ fontSize: 14 }}>
                {run.streaming ? 'running' : (run.status?.state ?? 'idle')}
              </span>
            }
            sub={
              run.status?.feasible === null || run.status?.feasible === undefined
                ? '—'
                : run.status.feasible
                  ? 'feasible'
                  : 'infeasible'
            }
          />
        </StatGrid>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12, marginTop: 12 }}>
          <Panel
            title="Convergence"
            actions={
              <Badge>
                {ticks.length > 0 ? `${fmtInt(ticks.length)} recorded iterations` : 'no data'}
              </Badge>
            }
          >
            {ticks.length > 0 ? (
              <ConvergenceChart ticks={ticks} reference={bks} height={280} />
            ) : (
              <div style={{ height: 280 }} className="empty">
                Start a run to stream the convergence history. The solid line is
                the incumbent, the dashed line the population mean, and the
                horizontal reference is the best-known cost for the instance.
              </div>
            )}
          </Panel>

          <Panel title="Solution">
            {run.status?.stats ? (
              <>
                <KeyValue label="Routes" value={fmtInt(run.status.n_routes)} />
                <KeyValue label="Distance" value={fmt(run.status.stats.distance, 2)} />
                <KeyValue label="Duration" value={fmt(run.status.stats.duration, 2)} />
                <KeyValue
                  label="Capacity violation"
                  value={fmt(run.status.stats.capacity_violation, 3)}
                />
                <KeyValue
                  label="Time-window violation"
                  value={fmt(run.status.stats.time_window_violation, 3)}
                />
                <KeyValue
                  label="Fleet violation"
                  value={fmt(run.status.stats.fleet_violation, 0)}
                />
                <KeyValue
                  label="Total violation"
                  value={fmt(run.status.stats.total_violation, 3)}
                />
                <div style={{ marginTop: 10 }}>
                  <Badge tone={run.status.feasible ? 'ok' : 'bad'}>
                    <span className="dot" />
                    {run.status.feasible ? 'Feasible' : 'Infeasible'}
                  </Badge>
                </div>
                <div
                  style={{
                    marginTop: 12,
                    fontSize: 11,
                    color: 'var(--text-faint)',
                    lineHeight: 1.5,
                  }}
                >
                  Violations are raw amounts of infeasibility in the instance's
                  own units, not penalty values, so a solution is reported as
                  feasible or not independently of how the search weighted its
                  penalties.
                </div>
              </>
            ) : (
              <div className="empty" style={{ minHeight: 200 }}>
                No completed run yet.
              </div>
            )}
          </Panel>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
          <Panel title="Population diversity">
            {ticks.length > 0 ? (
              <>
                <DiversityChart ticks={ticks} height={170} />
                <div
                  style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 6, lineHeight: 1.5 }}
                >
                  Mean pairwise distance between candidate positions in the
                  random-key space. A curve that falls to near zero well before
                  the budget expires means the population has collapsed and the
                  remaining time is being spent on local search alone.
                </div>
              </>
            ) : (
              <div className="empty" style={{ minHeight: 170 }}>
                Diversity is streamed per iteration.
              </div>
            )}
          </Panel>

          <Panel title="Routes">
            {run.status?.routes && run.status.routes.length > 0 ? (
              <div style={{ maxHeight: 220, overflow: 'auto' }}>
                <table className="grid">
                  <thead>
                    <tr>
                      <th>Vehicle</th>
                      <th>Stops</th>
                      <th style={{ textAlign: 'left' }}>Order</th>
                    </tr>
                  </thead>
                  <tbody>
                    {run.status.routes
                      .filter((r) => r.length > 0)
                      .map((route, i) => (
                        <tr key={i}>
                          <td>{i + 1}</td>
                          <td className="num">{route.length}</td>
                          <td
                            style={{
                              textAlign: 'left',
                              fontFamily: 'var(--mono)',
                              whiteSpace: 'normal',
                              wordBreak: 'break-word',
                              color: 'var(--text-dim)',
                            }}
                          >
                            0 → {route.join(' → ')} → 0
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty" style={{ minHeight: 170 }}>
                The route list appears when the run finishes.
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function describeInstance(instance: InstanceSummary): string {
  const parts = [`${instance.n_customers} customers`, `capacity ${instance.capacity}`];
  if (instance.has_time_windows) parts.push('time windows');
  if (instance.bks !== null) parts.push(`best known ${instance.bks}`);
  return parts.join(' · ');
}
