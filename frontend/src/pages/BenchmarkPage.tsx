/**
 * The benchmark page: what the platform actually scored on the public
 * instances, and whether the differences between algorithms survive a
 * statistical test.
 *
 * The order of the page is deliberate. The grid of per-instance gaps comes
 * first because it is the raw evidence; the aggregate chart second because it
 * is a summary of that evidence; and the statistical verdict last, phrased so
 * that "no significant difference" is a legitimate and clearly-stated outcome
 * rather than something buried. Nothing here is computed in the browser beyond
 * arranging numbers the backend already produced.
 */

import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend as ChartLegend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ApiError, getBenchmark, getBenchmarks } from '../api/client';
import type { BenchmarkDetail, BenchmarkSummary } from '../api/types';
import { CHART_AXIS, CHART_TOOLTIP } from '../components/charts';
import { Badge, Field, KeyValue, Notice, Panel, RailSection, Stat, StatGrid } from '../components/ui';
import { algorithmColor } from '../lib/colors';
import { fmt, fmtInt, fmtP, fmtPercent, fmtSeconds } from '../lib/format';
import { useAppStore } from '../store/appStore';

export function BenchmarkPage() {
  const backend = useAppStore((s) => s.backend);
  const [list, setList] = useState<BenchmarkSummary[]>([]);
  const [name, setName] = useState('');
  const [detail, setDetail] = useState<BenchmarkDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<'gap' | 'cost'>('gap');

  useEffect(() => {
    if (backend !== 'online') return;
    getBenchmarks()
      .then((items) => {
        setList(items);
        if (items.length > 0) setName((current) => current || items[0].name);
      })
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : String(e)));
  }, [backend]);

  useEffect(() => {
    if (!name || backend !== 'online') return undefined;
    // `cancelled` guards against a slow response for a previously selected
    // result set overwriting a newer one; the loading flag is raised inside the
    // asynchronous body so the effect does not set state synchronously.
    let cancelled = false;
    void (async () => {
      setLoading(true);
      setError(null);
      try {
        const loaded = await getBenchmark(name);
        if (!cancelled) setDetail(loaded);
      } catch (e: unknown) {
        if (cancelled) return;
        setDetail(null);
        setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [name, backend]);

  const perAlgorithm = useMemo(() => {
    if (!detail) return [];
    return detail.algorithms.map((algorithm) => {
      const cells = detail.instances
        .map((instance) => detail.cells[`${instance}|${algorithm}`])
        .filter(Boolean);
      const gaps = cells.map((c) => c.gap.median).filter((v) => Number.isFinite(v));
      const meanGap = gaps.length > 0 ? gaps.reduce((a, b) => a + b, 0) / gaps.length : Number.NaN;
      const bestGap = gaps.length > 0 ? Math.min(...gaps) : Number.NaN;
      const hits = cells.reduce((a, c) => a + c.hit_bks, 0);
      const runs = cells.reduce((a, c) => a + c.runs, 0);
      const feasible = cells.reduce((a, c) => a + c.feasible_runs, 0);
      return {
        algorithm,
        meanGap,
        bestGap,
        hits,
        runs,
        feasible,
        rank: detail.omnibus?.mean_ranks[algorithm] ?? Number.NaN,
      };
    });
  }, [detail]);

  /**
   * Empirical CDF of per-run gaps, one curve per algorithm. A curve that sits
   * up and to the left is better: it reached a small gap on a larger fraction
   * of runs. This shows the whole distribution rather than a mean that a
   * couple of bad instances can dominate.
   */
  const distribution = useMemo(() => {
    if (!detail || detail.rows.length === 0) return { points: [], algorithms: [] as string[] };
    const byAlgorithm = new Map<string, number[]>();
    for (const row of detail.rows) {
      if (!Number.isFinite(row.gap)) continue;
      const list_ = byAlgorithm.get(row.algorithm);
      if (list_) list_.push(row.gap);
      else byAlgorithm.set(row.algorithm, [row.gap]);
    }
    for (const values of byAlgorithm.values()) values.sort((a, b) => a - b);
    const all = [...byAlgorithm.values()].flat();
    if (all.length === 0) return { points: [], algorithms: [] as string[] };
    const max = Math.max(...all);
    const steps = 60;
    const points: Record<string, number>[] = [];
    for (let i = 0; i <= steps; i += 1) {
      const threshold = (max * i) / steps;
      const point: Record<string, number> = { gap: Number(threshold.toFixed(4)) };
      for (const [algorithm, values] of byAlgorithm) {
        const count = values.filter((v) => v <= threshold).length;
        point[algorithm] = (100 * count) / values.length;
      }
      points.push(point);
    }
    return { points, algorithms: [...byAlgorithm.keys()] };
  }, [detail]);

  if (backend !== 'online') {
    return (
      <div className="page-scroll">
        <Notice kind="error">
          <strong>Backend unavailable.</strong> Benchmark results are read from
          the run directories on disk by the API. Start it with{' '}
          <code>python -m qroute.api.app</code> and reload.
        </Notice>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', height: '100%', minHeight: 0 }}>
      <aside className="rail">
        <RailSection title="Benchmark run">
          <Field label="Result set">
            <select value={name} onChange={(e) => setName(e.target.value)}>
              {list.length === 0 && <option value="">no results on disk</option>}
              {list.map((b) => (
                <option key={b.name} value={b.name}>
                  {b.name}
                </option>
              ))}
            </select>
          </Field>
          {detail && (
            <>
              <KeyValue label="Instances" value={fmtInt(detail.instances.length)} />
              <KeyValue label="Algorithms" value={fmtInt(detail.algorithms.length)} />
              <KeyValue label="Runs completed" value={fmtInt(detail.n_ok)} />
              <KeyValue label="Runs failed" value={fmtInt(detail.n_failed)} />
              <KeyValue label="Budget per run" value={fmtSeconds(detail.max_seconds)} />
            </>
          )}
          <Field label="Table metric">
            <select value={metric} onChange={(e) => setMetric(e.target.value as 'gap' | 'cost')}>
              <option value="gap">Median gap to best known (%)</option>
              <option value="cost">Median cost (cost units)</option>
            </select>
          </Field>
        </RailSection>

        {detail && (
          <RailSection title="Environment">
            <KeyValue label="Python" value={detail.environment.python ?? '—'} />
            <KeyValue label="Platform" value={detail.environment.platform ?? '—'} />
            <KeyValue label="CPU count" value={fmtInt(detail.environment.cpu_count ?? null)} />
            <KeyValue
              label="Commit"
              value={
                detail.environment.git_commit
                  ? `${detail.environment.git_commit.slice(0, 8)}${detail.environment.git_dirty ? '+' : ''}`
                  : '—'
              }
            />
            {Object.entries(detail.environment.packages ?? {})
              .slice(0, 6)
              .map(([pkg, version]) => (
                <KeyValue key={pkg} label={pkg} value={version} />
              ))}
            <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 8, lineHeight: 1.5 }}>
              Recorded at run time so a result can be reproduced, or its
              provenance questioned, without guessing.
            </div>
          </RailSection>
        )}
      </aside>

      <div className="page-scroll" style={{ flex: '1 1 auto', minWidth: 0 }}>
        {error && (
          <div style={{ marginBottom: 12 }}>
            <Notice kind="error">{error}</Notice>
          </div>
        )}
        {loading && !detail && <div className="empty">Loading results…</div>}
        {!loading && !detail && !error && (
          <div className="empty">
            No benchmark results are available. Run one with{' '}
            <code style={{ marginLeft: 4 }}>python -m qroute.cli bench</code>.
          </div>
        )}

        {detail && (
          <>
            <StatGrid columns={4}>
              <Stat
                label="Best mean gap"
                value={fmtPercent(Math.min(...perAlgorithm.map((a) => a.meanGap)), 2)}
                sub={
                  perAlgorithm.reduce(
                    (best, a) => (a.meanGap < best.meanGap ? a : best),
                    perAlgorithm[0] ?? { algorithm: '—', meanGap: Number.NaN },
                  ).algorithm
                }
              />
              <Stat
                label="Optima matched"
                value={fmtInt(perAlgorithm.reduce((a, b) => a + b.hits, 0))}
                sub={`of ${fmtInt(perAlgorithm.reduce((a, b) => a + b.runs, 0))} runs`}
              />
              <Stat
                label="Feasible runs"
                value={fmtInt(perAlgorithm.reduce((a, b) => a + b.feasible, 0))}
                sub={`of ${fmtInt(perAlgorithm.reduce((a, b) => a + b.runs, 0))}`}
              />
              <Stat
                label="Omnibus p-value"
                value={detail.omnibus ? fmtP(detail.omnibus.p_value) : '—'}
                sub={detail.omnibus ? `Friedman, ${detail.omnibus.n_instances} instances` : 'not run'}
              />
            </StatGrid>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
              <Panel title="Mean gap by algorithm">
                <div style={{ width: '100%', height: 220 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={perAlgorithm}
                      margin={{ top: 8, right: 16, bottom: 24, left: 8 }}
                    >
                      <CartesianGrid stroke="#232b35" strokeDasharray="2 4" vertical={false} />
                      <XAxis
                        dataKey="algorithm"
                        stroke={CHART_AXIS.stroke}
                        tick={CHART_AXIS.tick}
                        label={{
                          value: 'algorithm',
                          position: 'insideBottom',
                          offset: -14,
                          style: CHART_AXIS.label,
                        }}
                      />
                      <YAxis
                        stroke={CHART_AXIS.stroke}
                        tick={CHART_AXIS.tick}
                        width={58}
                        tickFormatter={(v: number) => fmt(v, 1)}
                        label={{
                          value: 'mean gap (%)',
                          angle: -90,
                          position: 'insideLeft',
                          offset: 6,
                          style: { ...CHART_AXIS.label, textAnchor: 'middle' },
                        }}
                      />
                      <Tooltip
                        contentStyle={CHART_TOOLTIP}
                        cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                        formatter={(value: unknown) => [`${fmt(Number(value), 3)} %`, 'mean gap']}
                      />
                      <Bar dataKey="meanGap" isAnimationActive={false} radius={[2, 2, 0, 0]}>
                        {perAlgorithm.map((a) => (
                          <Cell key={a.algorithm} fill={algorithmColor(a.algorithm)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-faint)', lineHeight: 1.5 }}>
                  Gap is <code>100 (cost − best known) / best known</code>,
                  averaged over instances after taking the median across seeds.
                  Lower is better; zero means the best-known solution was
                  matched.
                </div>
              </Panel>

              <Panel title="Gap distribution across all runs">
                {distribution.points.length > 0 ? (
                  <>
                    <div style={{ width: '100%', height: 220 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={distribution.points}
                          margin={{ top: 8, right: 16, bottom: 24, left: 8 }}
                        >
                          <CartesianGrid stroke="#232b35" strokeDasharray="2 4" />
                          <XAxis
                            dataKey="gap"
                            type="number"
                            stroke={CHART_AXIS.stroke}
                            tick={CHART_AXIS.tick}
                            tickFormatter={(v: number) => fmt(v, 1)}
                            label={{
                              value: 'gap to best known (%)',
                              position: 'insideBottom',
                              offset: -14,
                              style: CHART_AXIS.label,
                            }}
                          />
                          <YAxis
                            stroke={CHART_AXIS.stroke}
                            tick={CHART_AXIS.tick}
                            width={58}
                            domain={[0, 100]}
                            tickFormatter={(v: number) => `${fmt(v, 0)}`}
                            label={{
                              value: 'runs within gap (%)',
                              angle: -90,
                              position: 'insideLeft',
                              offset: 6,
                              style: { ...CHART_AXIS.label, textAnchor: 'middle' },
                            }}
                          />
                          <Tooltip
                            contentStyle={CHART_TOOLTIP}
                            labelFormatter={(v) => `gap ≤ ${fmt(Number(v), 2)} %`}
                            formatter={(value: unknown, key: unknown) => [
                              `${fmt(Number(value), 1)} % of runs`,
                              String(key),
                            ]}
                          />
                          <ChartLegend wrapperStyle={{ fontSize: 11, color: '#94a1b2' }} />
                          {distribution.algorithms.map((a) => (
                            <Line
                              key={a}
                              type="stepAfter"
                              dataKey={a}
                              stroke={algorithmColor(a)}
                              strokeWidth={1.8}
                              dot={false}
                              isAnimationActive={false}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-faint)', lineHeight: 1.5 }}>
                      Empirical cumulative distribution over every seeded run. A
                      curve further up and to the left reached a small gap on a
                      larger fraction of runs.
                    </div>
                  </>
                ) : (
                  <div className="empty" style={{ minHeight: 220 }}>
                    Per-run rows were not included in this result set.
                  </div>
                )}
              </Panel>
            </div>

            <div style={{ marginTop: 12 }}>
              <Panel
                title={`Per-instance results — ${metric === 'gap' ? 'median gap (%)' : 'median cost'}`}
                actions={<Badge>{fmtInt(detail.instances.length)} instances</Badge>}
                flush
              >
                <div style={{ maxHeight: 420, overflow: 'auto' }}>
                  <table className="grid">
                    <thead>
                      <tr>
                        <th>Instance</th>
                        <th>Best known</th>
                        {detail.algorithms.map((a) => (
                          <th key={a} style={{ color: algorithmColor(a) }}>
                            {a}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {detail.instances.map((instance) => {
                        const cells = detail.algorithms.map(
                          (a) => detail.cells[`${instance}|${a}`],
                        );
                        const values = cells.map((c) =>
                          c ? (metric === 'gap' ? c.gap.median : c.cost.median) : Number.NaN,
                        );
                        const best = Math.min(...values.filter(Number.isFinite));
                        const bks = detail.rows.find((r) => r.instance === instance)?.bks ?? null;
                        return (
                          <tr key={instance}>
                            <td>{instance}</td>
                            <td className="num" style={{ color: 'var(--text-faint)' }}>
                              {bks === null ? '—' : fmt(bks, 1)}
                            </td>
                            {values.map((value, i) => (
                              <td
                                key={detail.algorithms[i]}
                                className="num"
                                style={{
                                  color:
                                    Number.isFinite(value) && value === best
                                      ? 'var(--ok)'
                                      : 'var(--text)',
                                  fontWeight:
                                    Number.isFinite(value) && value === best ? 600 : 400,
                                }}
                              >
                                {Number.isFinite(value)
                                  ? fmt(value, metric === 'gap' ? 2 : 1)
                                  : '—'}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </div>

            <div style={{ marginTop: 12 }}>
              <Panel title="Statistical verdict">
                {detail.omnibus ? (
                  <>
                    <p style={{ color: 'var(--text-dim)', maxWidth: 900 }}>
                      Friedman test over {fmtInt(detail.omnibus.n_instances)} instances
                      and {fmtInt(detail.omnibus.algorithms.length)} algorithms:{' '}
                      <span className="mono">χ² = {fmt(detail.omnibus.statistic, 3)}</span>,{' '}
                      <span className="mono">p = {fmtP(detail.omnibus.p_value)}</span>.{' '}
                      {detail.omnibus.p_value <= 0.05
                        ? 'The algorithms are not all equivalent; the pairwise comparisons below are against the best-ranked control, with Holm correction for multiple testing.'
                        : 'The null hypothesis that all algorithms perform equally is not rejected at the 5 % level. On this instance set, the differences between them are within run-to-run noise.'}
                    </p>

                    <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 16 }}>
                      <div>
                        <h3 style={{ marginBottom: 6, color: 'var(--text-dim)' }}>Mean rank</h3>
                        <table className="grid">
                          <thead>
                            <tr>
                              <th>Algorithm</th>
                              <th>Rank</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(detail.omnibus.mean_ranks)
                              .sort((a, b) => a[1] - b[1])
                              .map(([a, rank]) => (
                                <tr key={a}>
                                  <td style={{ color: algorithmColor(a) }}>{a}</td>
                                  <td className="num">{fmt(rank, 2)}</td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                        <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 6 }}>
                          1 is best. Ranks are assigned within each instance and
                          averaged.
                        </div>
                      </div>

                      <div>
                        <h3 style={{ marginBottom: 6, color: 'var(--text-dim)' }}>
                          Pairwise, against {detail.omnibus.control ?? 'control'}
                        </h3>
                        <table className="grid">
                          <thead>
                            <tr>
                              <th>Comparison</th>
                              <th>n</th>
                              <th>Median A</th>
                              <th>Median B</th>
                              <th>p</th>
                              <th>p (Holm)</th>
                              <th>Effect</th>
                              <th style={{ textAlign: 'left' }}>Verdict</th>
                            </tr>
                          </thead>
                          <tbody>
                            {detail.omnibus.post_hoc.map((test) => {
                              const p = test.p_adjusted ?? test.p_value;
                              const significant = Number.isFinite(p) && p <= 0.05;
                              return (
                                <tr key={`${test.a}-${test.b}`}>
                                  <td>
                                    {test.a} vs {test.b}
                                  </td>
                                  <td className="num">{fmtInt(test.n)}</td>
                                  <td className="num">{fmt(test.median_a, 3)}</td>
                                  <td className="num">{fmt(test.median_b, 3)}</td>
                                  <td className="num">{fmtP(test.p_value)}</td>
                                  <td className="num">{fmtP(test.p_adjusted)}</td>
                                  <td className="num">{fmt(Math.abs(test.effect_size), 2)}</td>
                                  <td style={{ textAlign: 'left' }}>
                                    {significant ? (
                                      <Badge tone="ok">
                                        {test.winner ??
                                          (test.median_a < test.median_b ? test.a : test.b)}{' '}
                                        better
                                      </Badge>
                                    ) : (
                                      <Badge>no significant difference</Badge>
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                        <div
                          style={{
                            fontSize: 11,
                            color: 'var(--text-faint)',
                            marginTop: 6,
                            lineHeight: 1.5,
                          }}
                        >
                          Paired Wilcoxon signed-rank tests on the per-instance
                          median gap, with Holm correction across the family.
                          Effect size is the matched-pairs rank-biserial
                          correlation.
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="empty" style={{ minHeight: 120 }}>
                    No omnibus test was computed for this result set. The
                    Friedman test needs at least three algorithms measured on the
                    same instances.
                  </div>
                )}
              </Panel>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
