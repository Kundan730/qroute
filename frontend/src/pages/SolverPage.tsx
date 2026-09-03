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
 *
 * Two things the page is designed around. First, while a run is in flight the
 * reader has to be able to tell at a glance that it is *live* - so there is a
 * single run bar carrying the state, the elapsed time against the budget as a
 * determinate meter, and the moment of the last improvement, all of which move
 * on their own from real numbers rather than from an animation. Second, when
 * the run stops the outcome has to be unmistakable - so the bar changes state,
 * the meter settles, and a one-line verdict states the final cost, whether it
 * is feasible, and how it compares with the reference.
 */

import { useMemo, useState } from 'react';
import type { InstanceSummary, RunRequest, RunTick } from '../api/types';
import { ConvergenceChart, DiversityChart } from '../components/charts';
import {
  Badge,
  Caption,
  Empty,
  Field,
  KeyValue,
  Mark,
  Meter,
  Note,
  Notice,
  Panel,
  RailSection,
  Stat,
  StatGrid,
  Swatch,
  type Tone,
} from '../components/ui';
import { vehicleColor } from '../lib/colors';
import { fmt, fmtCompact, fmtInt, fmtPercent, fmtSeconds } from '../lib/format';
import { groupInstances, useAppStore } from '../store/appStore';
import { evaluationRate, useRunStore } from '../store/runStore';

const FAMILY_LABELS: Record<string, string> = {
  cvrp: 'CVRPLIB — capacitated VRP',
  vrptw: 'Solomon — VRP with time windows',
  network: 'Generated from a road network',
};

/**
 * How each run state is presented. The mark shape carries the state as well as
 * the colour, so the bar is readable in greyscale and on a projector.
 */
const RUN_STATES: Record<
  string,
  { label: string; tone?: Tone; mark: 'disc' | 'ring' | 'square'; solid?: boolean }
> = {
  idle: { label: 'idle', mark: 'ring' },
  queued: { label: 'queued', mark: 'ring' },
  running: { label: 'running', mark: 'disc', solid: true },
  done: { label: 'complete', tone: 'ok', mark: 'disc', solid: true },
  cancelled: { label: 'cancelled', tone: 'warn', mark: 'ring', solid: true },
  failed: { label: 'failed', tone: 'bad', mark: 'square', solid: true },
};

/** Elapsed seconds at the last tick on which the incumbent actually improved. */
function lastImprovement(ticks: RunTick[]): { elapsed: number; from: number; to: number } | null {
  for (let i = ticks.length - 1; i > 0; i -= 1) {
    const current = ticks[i].best_cost;
    const previous = ticks[i - 1].best_cost;
    if (Number.isFinite(current) && Number.isFinite(previous) && current < previous) {
      return { elapsed: ticks[i].elapsed, from: previous, to: current };
    }
  }
  return null;
}

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
  // The reference must belong to the instance the *displayed run* solved, not
  // to whatever is currently chosen in the rail. A run started from the map
  // page solves a road-network instance that has no best-known solution, and
  // falling back to the selected benchmark instance's reference would price a
  // Bengaluru drive time in seconds against a CVRPLIB cost and report a gap of
  // several hundred per cent that means nothing at all.
  const shownInstance = run.status?.instance ?? (run.streaming ? null : instanceName);
  const referenceApplies = shownInstance === null || shownInstance === instanceName;
  const bks = run.status?.bks ?? (referenceApplies ? (selected?.bks ?? null) : null);
  const bestCost = run.status?.best_cost ?? last?.best_cost ?? null;
  const gap =
    bks !== null && bestCost !== null && bks > 0 ? (100 * (bestCost - bks)) / bks : null;
  const rate = evaluationRate(ticks);

  const improvement = useMemo(() => lastImprovement(ticks), [ticks]);

  const stateKey = run.streaming ? 'running' : (run.status?.state ?? 'idle');
  const state = RUN_STATES[stateKey] ?? RUN_STATES.idle;
  const finished = !run.streaming && run.status !== null;
  const elapsed = last?.elapsed ?? run.status?.seconds ?? 0;
  // The budget the *displayed* run was given, not whatever the slider says now.
  const budget = run.request?.max_seconds ?? maxSeconds;
  const progress = budget > 0 ? elapsed / budget : 0;

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
        {/* The run bar. One line that always answers: what is being solved,
            what state is it in, and how far through its budget is it. */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            flexWrap: 'wrap',
            padding: '9px 14px',
            marginBottom: 12,
            background: 'var(--panel)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            borderLeft: `3px solid ${
              state.tone === 'ok'
                ? 'var(--ok)'
                : state.tone === 'warn'
                  ? 'var(--warn)'
                  : state.tone === 'bad'
                    ? 'var(--bad)'
                    : run.streaming
                      ? 'var(--accent)'
                      : 'var(--border-strong)'
            }`,
          }}
        >
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              padding: '3px 9px',
              borderRadius: 'var(--radius)',
              fontFamily: 'var(--display)',
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: '0.13em',
              textTransform: 'uppercase',
              whiteSpace: 'nowrap',
              border: '1px solid',
              borderColor: state.solid
                ? 'transparent'
                : 'var(--border-strong)',
              background: state.solid
                ? state.tone === 'ok'
                  ? 'var(--ok)'
                  : state.tone === 'warn'
                    ? 'var(--warn)'
                    : state.tone === 'bad'
                      ? 'var(--bad)'
                      : 'var(--accent)'
                : 'var(--bg-raised)',
              color: state.solid ? 'var(--panel)' : 'var(--text-dim)',
            }}
          >
            <Mark shape={state.mark} size={6} />
            {state.label}
          </span>

          <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>
            <span style={{ color: 'var(--text)', fontWeight: 500 }}>
              {run.status?.algorithm ?? algorithm}
            </span>{' '}
            on{' '}
            <span style={{ color: 'var(--text)', fontWeight: 500 }}>
              {run.status?.instance ?? (instanceName || '—')}
            </span>
            <span className="mono" style={{ marginLeft: 10 }}>
              seed {run.status?.seed ?? seed}
            </span>
          </span>

          <div style={{ flex: '1 1 200px', minWidth: 160 }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: 10,
                fontFamily: 'var(--mono)',
                color: 'var(--text-dim)',
                marginBottom: 3,
              }}
            >
              <span>{fmtSeconds(elapsed)} elapsed</span>
              <span>budget {budget} s</span>
            </div>
            <Meter
              value={progress}
              animated={run.streaming}
              tone={finished ? (state.tone ?? undefined) : undefined}
              title={`${fmtSeconds(elapsed)} of a ${budget} s budget`}
            />
          </div>

          {run.streaming && improvement && (
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
              last improvement at {fmtSeconds(improvement.elapsed)}
            </span>
          )}
        </div>

        {finished && (
          <div style={{ marginBottom: 12 }}>
            <Notice kind={run.status?.state === 'failed' ? 'error' : 'warn'}>
              <strong>{verdictHeadline(run.status?.state ?? 'done')}</strong>{' '}
              {run.status?.state === 'failed'
                ? (run.status?.error ?? 'The run did not produce a result.')
                : run.status?.state === 'cancelled'
                  ? `Stopped after ${fmtSeconds(run.status?.seconds ?? elapsed)} with the best solution found so far.`
                  : ''}
              {run.status?.state !== 'failed' && bestCost !== null && (
                <>
                  {' '}
                  Best cost <span className="mono">{fmt(bestCost, 2)}</span> over{' '}
                  <span className="mono">{fmtInt(run.status?.iterations ?? null)}</span> iterations
                  in <span className="mono">{fmtSeconds(run.status?.seconds ?? elapsed)}</span>,{' '}
                  {run.status?.feasible === null || run.status?.feasible === undefined
                    ? 'feasibility not reported'
                    : run.status.feasible
                      ? 'feasible'
                      : 'infeasible'}
                  {gap !== null
                    ? gap <= 0.001
                      ? '; it matched the best known solution.'
                      : `; ${fmtPercent(gap, 2)} from the best known solution.`
                    : '; no reference on disk to compare against.'}
                </>
              )}
            </Notice>
          </div>
        )}

        <StatGrid columns={5}>
          <Stat
            label={run.streaming ? 'Incumbent cost' : 'Best cost'}
            value={bestCost === null ? '—' : fmt(bestCost, 2)}
            sub={
              improvement
                ? `improved ${fmt(improvement.from - improvement.to, 2)} at ${fmtSeconds(improvement.elapsed)}`
                : (run.status?.instance ?? instanceName ?? '—')
            }
          />
          <Stat
            label="Gap to best known"
            value={gap === null ? '—' : fmtPercent(gap, 2)}
            tone={gap === null ? undefined : gap <= 0.001 ? 'ok' : gap < 2 ? 'warn' : 'bad'}
            sub={
              bks !== null
                ? `reference ${fmt(bks, 1)}`
                : shownInstance !== null && shownInstance !== instanceName
                  ? `no reference for ${shownInstance}`
                  : 'no reference on disk'
            }
          />
          <Stat
            label="Iterations"
            value={fmtInt(last?.iteration ?? run.status?.iterations ?? null)}
            sub={`${fmtInt(ticks.length)} sampled`}
          />
          <Stat
            label="Evaluations"
            value={fmtCompact(last?.evaluations ?? run.status?.evaluations ?? 0)}
            sub={
              Number.isFinite(rate)
                ? `${fmtCompact(rate)} per second`
                : run.streaming
                  ? 'measuring…'
                  : '—'
            }
          />
          <Stat
            label="Feasibility"
            value={
              <span style={{ fontSize: 15 }}>
                {run.status?.feasible === null || run.status?.feasible === undefined
                  ? '—'
                  : run.status.feasible
                    ? 'Feasible'
                    : 'Infeasible'}
              </span>
            }
            tone={
              run.status?.feasible === null || run.status?.feasible === undefined
                ? undefined
                : run.status.feasible
                  ? 'ok'
                  : 'bad'
            }
            sub={
              run.status?.stats
                ? `total violation ${fmt(run.status.stats.total_violation, 3)}`
                : run.streaming
                  ? 'reported when the run ends'
                  : '—'
            }
          />
        </StatGrid>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12, marginTop: 12 }}>
          <Panel
            title="Convergence"
            actions={
              <Badge mark={run.streaming ? 'disc' : undefined} tone={run.streaming ? 'ok' : undefined}>
                {ticks.length > 0
                  ? `${fmtInt(ticks.length)} recorded iterations`
                  : 'no data'}
              </Badge>
            }
          >
            {ticks.length > 0 ? (
              <>
                <ConvergenceChart ticks={ticks} reference={bks} height={280} />
                <Note style={{ marginTop: 6 }}>
                  The solid line is the incumbent and the dashed line the
                  population mean; the horizontal reference, when drawn, is the
                  best-known cost for the instance. Time rather than iteration
                  on the x axis, because one iteration of a swarm of thirty is
                  not comparable with one iteration of an annealer.
                </Note>
              </>
            ) : (
              <Empty style={{ height: 280 }}>
                Start a run to stream the convergence history. The solid line is
                the incumbent, the dashed line the population mean, and the
                horizontal reference is the best-known cost for the instance.
              </Empty>
            )}
          </Panel>

          <Panel title="Solution">
            {run.status?.stats ? (
              <>
                <div style={{ marginBottom: 10 }}>
                  <Badge
                    tone={run.status.feasible ? 'ok' : 'bad'}
                    mark={run.status.feasible ? 'disc' : 'square'}
                  >
                    {run.status.feasible ? 'Feasible' : 'Infeasible'}
                  </Badge>{' '}
                  <Badge>{fmtInt(run.status.n_routes)} routes</Badge>
                </div>
                <KeyValue label="Routes" value={fmtInt(run.status.n_routes)} />
                <KeyValue label="Distance" value={fmt(run.status.stats.distance, 2)} />
                <KeyValue label="Duration" value={fmt(run.status.stats.duration, 2)} />
                <Caption style={{ margin: '10px 0 4px' }}>Constraint violations</Caption>
                <KeyValue
                  label="Capacity"
                  value={violationValue(run.status.stats.capacity_violation, 3)}
                />
                <KeyValue
                  label="Time window"
                  value={violationValue(run.status.stats.time_window_violation, 3)}
                />
                <KeyValue label="Fleet" value={violationValue(run.status.stats.fleet_violation, 0)} />
                <KeyValue
                  label="Total"
                  value={violationValue(run.status.stats.total_violation, 3)}
                />
                <Note style={{ marginTop: 10 }}>
                  Violations are raw amounts of infeasibility in the instance's
                  own units, not penalty values, so a solution is reported as
                  feasible or not independently of how the search weighted its
                  penalties. A dash means the constraint is satisfied exactly.
                </Note>
              </>
            ) : (
              <Empty style={{ minHeight: 200 }}>
                {run.streaming
                  ? 'The solution is reported when the run finishes.'
                  : 'No completed run yet.'}
              </Empty>
            )}
          </Panel>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
          <Panel title="Population diversity">
            {ticks.length > 0 ? (
              <>
                <DiversityChart ticks={ticks} height={170} />
                <Note style={{ marginTop: 6 }}>
                  Mean pairwise distance between candidate positions in the
                  random-key space. A curve that falls to near zero well before
                  the budget expires means the population has collapsed and the
                  remaining time is being spent on local search alone.
                </Note>
              </>
            ) : (
              <Empty style={{ minHeight: 170 }}>
                Diversity is streamed per iteration.
              </Empty>
            )}
          </Panel>

          <Panel
            title="Routes"
            actions={
              run.status?.routes ? (
                <Badge>
                  {fmtInt(run.status.routes.filter((r) => r.length > 0).length)} in service
                </Badge>
              ) : undefined
            }
          >
            {run.status?.routes && run.status.routes.length > 0 ? (
              <div style={{ maxHeight: 220, overflow: 'auto' }}>
                <table className="grid">
                  <thead>
                    <tr>
                      <th>Vehicle</th>
                      <th style={{ textAlign: 'right' }}>Stops</th>
                      <th style={{ textAlign: 'left' }}>Order</th>
                    </tr>
                  </thead>
                  <tbody>
                    {run.status.routes
                      .filter((r) => r.length > 0)
                      .map((route, i) => (
                        <tr key={i}>
                          <td>
                            <span
                              style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
                            >
                              <Swatch color={vehicleColor(i)} />
                              {i + 1}
                            </span>
                          </td>
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
              <Empty style={{ minHeight: 170 }}>
                The route list appears when the run finishes. Vehicle colours
                match the ones the map draws.
              </Empty>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

/** A violation of exactly zero prints as a dash: satisfied, not "0.000". */
function violationValue(value: number, digits: number) {
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return <span style={{ color: 'var(--text-dim)' }}>—</span>;
  return <span style={{ color: 'var(--bad)', fontWeight: 500 }}>{fmt(value, digits)}</span>;
}

function verdictHeadline(state: string): string {
  if (state === 'failed') return 'Run failed.';
  if (state === 'cancelled') return 'Run cancelled.';
  return 'Run complete.';
}

function describeInstance(instance: InstanceSummary): string {
  const parts = [`${instance.n_customers} customers`, `capacity ${instance.capacity}`];
  if (instance.has_time_windows) parts.push('time windows');
  if (instance.bks !== null) parts.push(`best known ${instance.bks}`);
  return parts.join(' · ');
}
