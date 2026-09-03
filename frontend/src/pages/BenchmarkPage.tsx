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
 *
 * Three presentation rules are enforced throughout, because a results table is
 * the easiest place in a project like this to flatter yourself by accident.
 *
 *   - A number that was measured and a number that could not be computed must
 *     never look alike. Every absent value prints as an em dash, and a cell
 *     whose solver returned no solution at all is additionally hatched and
 *     annotated, so a blank is never mistaken for a zero.
 *   - The best value in a row is emphasised by weight and ground together, so
 *     it survives greyscale and a projector, and a value no other solver
 *     matched additionally carries a rule - which is the interesting case,
 *     because on the easy instances most of the field ties at zero.
 *   - A verdict is stated in words. "Significant", "not after Holm" and "no
 *     difference" are three different marks with three different shapes, and
 *     the corrected p-value - the one that actually decides - is the emphasised
 *     column, not the raw one.
 */

import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ApiError, getBenchmark, getBenchmarks } from '../api/client';
import type { BenchmarkDetail, BenchmarkSummary } from '../api/types';
import {
  Badge,
  Caption,
  Empty,
  Field,
  KeyValue,
  Meter,
  Note,
  Notice,
  Panel,
  RailSection,
  Stat,
  StatGrid,
  Swatch,
} from '../components/ui';
import { algorithmColor } from '../lib/colors';
import { fmt, fmtInt, fmtP, fmtPercent, fmtSeconds } from '../lib/format';
import { useAppStore } from '../store/appStore';

/** The threshold every significance statement on this page is made against. */
const ALPHA = 0.05;

/**
 * Height of one summary row in the per-instance grid, in pixels.
 *
 * The grid scrolls inside a bounded box so its sticky header stays put, which
 * means the bottom edge falls wherever the box ends - through the middle of a
 * row, if nothing stops it. The two summary rows are therefore pinned to the
 * bottom of that box, and the upper one needs to know how tall the lower one
 * is. The number mirrors `table.grid td` in global.css: 6px padding top and
 * bottom around an 18px line box, plus the 1px rule above the block.
 */
const SUMMARY_ROW_H = 31;

/** Pins a summary row to the bottom of the scrolling grid. */
function summaryRow(offset: number): CSSProperties {
  return { position: 'sticky', bottom: offset, zIndex: 2 };
}

function plural(count: number, noun: string): string {
  return `${fmtInt(count)} ${noun}${count === 1 ? '' : 's'}`;
}

/**
 * Chart colours, read from the design tokens at run time.
 *
 * Recharts wants literal colour strings for SVG attributes, and the palette
 * lives in `styles/global.css`, so the tokens are resolved from the document
 * rather than duplicated here. That way a change to a token moves the charts
 * with the rest of the interface and there is no second copy to drift.
 */
function readChartTheme() {
  const style = getComputedStyle(document.documentElement);
  const token = (name: string) => style.getPropertyValue(name).trim();
  return {
    axis: token('--border-strong'),
    grid: token('--border'),
    tick: token('--text-dim'),
    label: token('--navy-300'),
    outline: token('--navy-300'),
    panel: token('--panel'),
    text: token('--text'),
    borderStrong: token('--border-strong'),
  };
}

export function BenchmarkPage() {
  const backend = useAppStore((s) => s.backend);
  const [list, setList] = useState<BenchmarkSummary[]>([]);
  const [name, setName] = useState('');
  const [detail, setDetail] = useState<BenchmarkDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<'gap' | 'cost'>('gap');

  const theme = useMemo(readChartTheme, []);

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
        /** Instances this algorithm was measured on at all. */
        measured: cells.length,
        rank: detail.omnibus?.mean_ranks[algorithm] ?? Number.NaN,
      };
    });
  }, [detail]);

  /**
   * Per-run provenance the grid needs but the aggregated cells cannot carry:
   * the best-known cost for an instance, and how many seeds of a given
   * instance-by-algorithm pair returned no solution at all. A pair with no
   * cell *and* failed seeds is a solver that could not finish the instance,
   * which is a materially different statement from a pair that was never run.
   */
  const provenance = useMemo(() => {
    const bks = new Map<string, number>();
    const failures = new Map<string, { failed: number; total: number }>();
    for (const row of detail?.rows ?? []) {
      if (!bks.has(row.instance) && row.bks !== null && Number.isFinite(row.bks)) {
        bks.set(row.instance, row.bks);
      }
      const key = `${row.instance}|${row.algorithm}`;
      const entry = failures.get(key) ?? { failed: 0, total: 0 };
      entry.total += 1;
      if (row.status !== 'ok') entry.failed += 1;
      failures.set(key, entry);
    }
    const unsolved = [...failures.entries()].filter(([, v]) => v.failed > 0);
    const failedRuns = [...failures.values()].reduce((a, v) => a + v.failed, 0);
    return { bks, failures, unsolved, failedRuns };
  }, [detail]);

  /** The rows of the per-instance grid, with the best value already located. */
  const grid = useMemo(() => {
    if (!detail) return [];
    return detail.instances.map((instance) => {
      const values = detail.algorithms.map((algorithm) => {
        const cell = detail.cells[`${instance}|${algorithm}`];
        if (!cell) return Number.NaN;
        return metric === 'gap' ? cell.gap.median : cell.cost.median;
      });
      const finite = values.filter((v) => Number.isFinite(v));
      const best = finite.length > 0 ? Math.min(...finite) : Number.NaN;
      return {
        instance,
        values,
        best,
        /** How many algorithms hold the best value; more than one is a tie. */
        bestCount: finite.filter((v) => v === best).length,
        bks: provenance.bks.get(instance) ?? null,
      };
    });
  }, [detail, metric, provenance]);

  /** Column footers: the mean down each column, and how often it won a row. */
  const columnSummary = useMemo(() => {
    if (!detail) return [];
    return detail.algorithms.map((algorithm, i) => {
      const column = grid.map((r) => r.values[i]).filter((v) => Number.isFinite(v));
      const wins = grid.filter(
        (r) => Number.isFinite(r.values[i]) && r.values[i] === r.best,
      ).length;
      return {
        algorithm,
        mean: column.length > 0 ? column.reduce((a, b) => a + b, 0) / column.length : Number.NaN,
        wins,
        measured: column.length,
      };
    });
  }, [detail, grid]);

  /**
   * The headline "best mean gap" is always a *gap*, whatever the grid is
   * currently showing. Deriving it from the column summary instead would
   * silently print a mean cost with a per-cent sign the moment the reader
   * switched the table metric.
   */
  const bestGap = useMemo(() => {
    const measured = perAlgorithm.filter((a) => Number.isFinite(a.meanGap));
    if (measured.length === 0) return { value: Number.NaN, algorithm: '—' };
    const winner = measured.reduce((best, a) => (a.meanGap < best.meanGap ? a : best));
    return { value: winner.meanGap, algorithm: winner.algorithm };
  }, [perAlgorithm]);

  /**
   * Instances on which every algorithm produced a result. The Friedman test is
   * a complete-block design, so this is the number it can actually use, and
   * stating it prevents the reader from wondering why the test reports fewer
   * instances than the grid shows.
   */
  const completeInstances = useMemo(() => {
    if (!detail) return 0;
    return detail.instances.filter((instance) =>
      detail.algorithms.every((a) => Boolean(detail.cells[`${instance}|${a}`])),
    ).length;
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

  const tooltipStyle = {
    background: theme.panel,
    border: `1px solid ${theme.borderStrong}`,
    borderRadius: 'var(--radius)',
    fontSize: 12,
    color: theme.text,
    padding: '6px 9px',
    boxShadow: 'var(--shadow-float)',
  };
  // Recharts paints a tooltip row in the colour of the series it belongs to.
  // That is fine for the swatch, which is a graphic, but not for the words: the
  // palette contains series pale enough that the name would drop to about
  // 2.3:1 on a white panel, which is unreadable. The mark keeps the hue and the
  // text is set in ink. The chart key is built in the DOM for the same reason,
  // plus one Recharts cannot solve: see the note above it.
  const tooltipItemStyle = { color: theme.text };
  const axisTick = { fill: theme.tick, fontSize: 11 };
  const axisLabel = { fill: theme.label, fontSize: 11 };

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

  const totalRuns = perAlgorithm.reduce((a, b) => a + b.runs, 0);
  const totalHits = perAlgorithm.reduce((a, b) => a + b.hits, 0);
  const totalFeasible = perAlgorithm.reduce((a, b) => a + b.feasible, 0);
  const survivingComparisons =
    detail?.omnibus?.post_hoc.filter((t) => (t.p_adjusted ?? t.p_value) <= ALPHA).length ?? 0;

  return (
    <div style={{ display: 'flex', height: '100%', minHeight: 0 }}>
      <aside className="rail">
        <RailSection title="Benchmark run">
          <Field label="Result set">
            <select value={name} onChange={(e) => setName(e.target.value)}>
              {list.length === 0 && <option value="">no results on disk</option>}
              {list.map((b) => (
                <option key={b.name} value={b.name}>
                  {b.name} — {b.n_instances}×{b.n_algorithms}, {b.n_runs} runs
                </option>
              ))}
            </select>
          </Field>
          {detail && (
            <>
              <KeyValue label="Instances" value={fmtInt(detail.instances.length)} />
              <KeyValue label="Algorithms" value={fmtInt(detail.algorithms.length)} />
              <KeyValue label="Runs completed" value={fmtInt(detail.n_ok)} />
              <KeyValue
                label="Runs without solution"
                value={fmtInt(provenance.failedRuns)}
                title="Seeds on which a solver returned no complete solution inside its budget."
              />
              <KeyValue
                label="Seeds per cell"
                value={fmtInt(
                  Math.max(0, ...Object.values(detail.cells).map((c) => c.runs)),
                )}
              />
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
            {/*
              The platform string is a single unbroken machine identifier
              (`macOS-26.2-arm64-arm-64bit-Mach-O`) and is far wider than the
              rail. It has to fold rather than ellipse: this panel exists so a
              result can be reproduced, and half a platform string reproduces
              nothing.
            */}
            <KeyValue
              label="Platform"
              value={detail.environment.platform ?? '—'}
              title={detail.environment.platform ?? undefined}
              wrap
            />
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
            <Note style={{ marginTop: 8 }}>
              Recorded at run time so a result can be reproduced, or its
              provenance questioned, without guessing.
            </Note>
          </RailSection>
        )}
      </aside>

      <div className="page-scroll" style={{ flex: '1 1 auto', minWidth: 0 }}>
        {error && (
          <div style={{ marginBottom: 12 }}>
            <Notice kind="error">{error}</Notice>
          </div>
        )}
        {loading && !detail && <Empty>Loading results…</Empty>}
        {!loading && !detail && !error && (
          <Empty>
            No benchmark results are available. Run one with{' '}
            <code style={{ marginLeft: 4 }}>python -m qroute.cli bench</code>.
          </Empty>
        )}

        {detail && (
          <>
            <div
              style={{
                display: 'flex',
                alignItems: 'baseline',
                gap: 12,
                flexWrap: 'wrap',
                marginBottom: 12,
              }}
            >
              <h1 style={{ fontSize: 18 }}>{detail.name}</h1>
              <span
                className="mono"
                style={{ fontSize: 11, color: 'var(--text-dim)' }}
              >
                {fmtInt(totalRuns)} runs · {plural(detail.instances.length, 'instance')} ·{' '}
                {plural(detail.algorithms.length, 'solver')} ·{' '}
                {fmtSeconds(detail.max_seconds)} per run
              </span>
              {detail.environment.git_commit && (
                <Badge title="Commit the results were produced at">
                  {detail.environment.git_commit.slice(0, 8)}
                  {detail.environment.git_dirty ? ' + uncommitted' : ''}
                </Badge>
              )}
            </div>

            <StatGrid columns={5}>
              <Stat
                label="Best mean gap"
                value={fmtPercent(bestGap.value, 2)}
                sub={bestGap.algorithm}
              />
              <Stat
                label="Optima matched"
                value={fmtInt(totalHits)}
                sub={`of ${fmtInt(totalRuns)} runs`}
              />
              <Stat
                label="Feasible runs"
                value={fmtInt(totalFeasible)}
                sub={`of ${fmtInt(totalRuns)} scored`}
              />
              <Stat
                label="No solution"
                value={fmtInt(provenance.failedRuns)}
                tone={provenance.failedRuns > 0 ? 'warn' : undefined}
                sub={
                  provenance.unsolved.length > 0
                    ? `${provenance.unsolved.length} solver–instance pair${provenance.unsolved.length === 1 ? '' : 's'}`
                    : 'every seed returned a plan'
                }
              />
              <Stat
                label="Omnibus p-value"
                value={detail.omnibus ? fmtP(detail.omnibus.p_value) : '—'}
                tone={
                  detail.omnibus
                    ? detail.omnibus.p_value <= ALPHA
                      ? 'ok'
                      : undefined
                    : undefined
                }
                sub={
                  detail.omnibus
                    ? `Friedman, ${fmtInt(detail.omnibus.n_instances)} complete blocks`
                    : 'not run'
                }
              />
            </StatGrid>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
              <Panel title="Mean gap by algorithm">
                <div style={{ width: '100%', height: 220 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={perAlgorithm}
                      margin={{ top: 8, right: 16, bottom: 30, left: 8 }}
                    >
                      <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" vertical={false} />
                      {/*
                        `interval={0}` forces a label on every bar. Left to its
                        own devices Recharts drops the ticks it cannot fit
                        horizontally, and in a two-up panel that meant nine bars
                        carrying five names - the reader could not tell ga from
                        pso from qiea from random. Rotating is the cost of
                        labelling all nine at this width.
                      */}
                      <XAxis
                        dataKey="algorithm"
                        stroke={theme.axis}
                        tick={axisTick}
                        interval={0}
                        angle={-38}
                        textAnchor="end"
                        height={48}
                        tickMargin={2}
                        label={{
                          value: 'algorithm',
                          position: 'insideBottom',
                          offset: -20,
                          style: axisLabel,
                        }}
                      />
                      <YAxis
                        stroke={theme.axis}
                        tick={axisTick}
                        width={58}
                        tickFormatter={(v: number) => fmt(v, 1)}
                        label={{
                          value: 'mean gap (%)',
                          angle: -90,
                          position: 'insideLeft',
                          offset: 6,
                          style: { ...axisLabel, textAnchor: 'middle' },
                        }}
                      />
                      <Tooltip
                        contentStyle={tooltipStyle}
                        itemStyle={tooltipItemStyle}
                        cursor={{ fill: theme.grid, fillOpacity: 0.5 }}
                        formatter={(value: unknown) => [`${fmt(Number(value), 3)} %`, 'mean gap']}
                      />
                      <Bar
                        dataKey="meanGap"
                        isAnimationActive={false}
                        radius={[2, 2, 0, 0]}
                        stroke={theme.outline}
                        strokeWidth={1}
                      >
                        {perAlgorithm.map((a) => (
                          <Cell key={a.algorithm} fill={algorithmColor(a.algorithm)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <Note>
                  Gap is <code>100 (cost − best known) / best known</code>,
                  averaged over instances after taking the median across seeds.
                  Lower is better; zero means the best-known solution was
                  matched. Every bar is outlined, so a pale series is still a
                  visible mark.
                </Note>
              </Panel>

              <Panel title="Gap distribution across all runs">
                {distribution.points.length > 0 ? (
                  <>
                    {/*
                      The key is drawn in the DOM above the plot rather than by
                      Recharts. Recharts reserves a fixed band for its legend,
                      and with nine series that band overflows into the plot
                      below roughly 1050px - the names print over the curves and
                      the y-axis ticks. A flex row wraps into as many lines as it
                      needs and pushes the chart down instead of covering it, and
                      it reuses the same Swatch the per-instance table uses, so
                      one key serves both.
                    */}
                    <div
                      style={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        alignItems: 'center',
                        gap: '3px 12px',
                        marginBottom: 8,
                      }}
                    >
                      {distribution.algorithms.map((a) => (
                        <span
                          key={a}
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 5,
                            fontSize: 11,
                            fontFamily: 'var(--mono)',
                            color: 'var(--text)',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          <Swatch color={algorithmColor(a)} />
                          {a}
                        </span>
                      ))}
                    </div>
                    <div style={{ width: '100%', height: 220 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={distribution.points}
                          margin={{ top: 4, right: 16, bottom: 24, left: 8 }}
                        >
                          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" />
                          <XAxis
                            dataKey="gap"
                            type="number"
                            stroke={theme.axis}
                            tick={axisTick}
                            tickFormatter={(v: number) => fmt(v, 1)}
                            label={{
                              value: 'gap to best known (%)',
                              position: 'insideBottom',
                              offset: -14,
                              style: axisLabel,
                            }}
                          />
                          <YAxis
                            stroke={theme.axis}
                            tick={axisTick}
                            width={58}
                            domain={[0, 100]}
                            tickFormatter={(v: number) => `${fmt(v, 0)}`}
                            label={{
                              value: 'runs within gap (%)',
                              angle: -90,
                              position: 'insideLeft',
                              offset: 6,
                              style: { ...axisLabel, textAnchor: 'middle' },
                            }}
                          />
                          <Tooltip
                            contentStyle={tooltipStyle}
                            itemStyle={tooltipItemStyle}
                            labelFormatter={(v) => `gap ≤ ${fmt(Number(v), 2)} %`}
                            formatter={(value: unknown, key: unknown) => [
                              `${fmt(Number(value), 1)} % of runs`,
                              String(key),
                            ]}
                          />
                          {distribution.algorithms.map((a) => (
                            <Line
                              key={a}
                              type="stepAfter"
                              dataKey={a}
                              stroke={algorithmColor(a)}
                              strokeWidth={2}
                              dot={false}
                              isAnimationActive={false}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <Note>
                      Empirical cumulative distribution over every seeded run. A
                      curve further up and to the left reached a small gap on a
                      larger fraction of runs.
                    </Note>
                  </>
                ) : (
                  <Empty style={{ minHeight: 220 }}>
                    Per-run rows were not included in this result set.
                  </Empty>
                )}
              </Panel>
            </div>

            <div style={{ marginTop: 12 }}>
              <Panel
                title={`Per-instance results — ${metric === 'gap' ? 'median gap (%)' : 'median cost'}`}
                actions={
                  <>
                    <Badge>{fmtInt(detail.instances.length)} instances</Badge>{' '}
                    <Badge>median of {fmtInt(Math.max(0, ...Object.values(detail.cells).map((c) => c.runs)))} seeds</Badge>
                  </>
                }
                flush
              >
                {/*
                  `table.grid th` in global.css is `position: sticky; top: 0`, which only
                  does anything if this wrapper is the nearest *vertical* scroll container.
                  `overflow-x: auto` alone forces `overflow-y` to compute to `auto` but the
                  wrapper still sizes to its content, so it never scrolls vertically and the
                  header used to scroll away with the page - leaving nine numeric columns
                  with no way to tell which solver each one was. Bounding the height makes
                  the wrapper the scroll container the sticky rule was written for.
                */}
                <div style={{ overflow: 'auto', maxHeight: 'calc(100vh - 300px)' }}>
                  <table className="grid">
                    <thead>
                      <tr>
                        <th style={{ minWidth: 108 }}>Instance</th>
                        <th style={{ textAlign: 'right' }}>Best known</th>
                        {detail.algorithms.map((a) => (
                          <th key={a} style={{ textAlign: 'right' }}>
                            <span
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: 5,
                                justifyContent: 'flex-end',
                              }}
                            >
                              <Swatch color={algorithmColor(a)} />
                              {a}
                            </span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {grid.map((row) => (
                        <tr key={row.instance}>
                          <td>{row.instance}</td>
                          <td className="num" style={{ color: 'var(--text-dim)' }}>
                            {row.bks === null ? '—' : fmt(row.bks, 1)}
                          </td>
                          {row.values.map((value, i) => {
                            const algorithm = detail.algorithms[i];
                            const failure = provenance.failures.get(`${row.instance}|${algorithm}`);
                            const unsolved = !Number.isFinite(value) && (failure?.failed ?? 0) > 0;
                            const isBest = Number.isFinite(value) && value === row.best;
                            // The rule is reserved for a value no other solver
                            // matched. On the easy instances most of the field
                            // ties at zero, and ruling nine cells at once would
                            // say nothing while looking like column banding.
                            const isOutright = isBest && row.bestCount === 1;
                            return (
                              <td
                                key={algorithm}
                                className="num"
                                title={
                                  unsolved
                                    ? `${algorithm} returned no complete solution on ${failure?.failed} of ${failure?.total} seeds of ${row.instance}`
                                    : isOutright
                                      ? `${algorithm} was strictly better than every other solver on ${row.instance}`
                                      : Number.isFinite(value)
                                        ? `${algorithm} on ${row.instance}`
                                        : `${algorithm} was not run on ${row.instance}`
                                }
                                style={{
                                  fontWeight: isBest ? 600 : 400,
                                  color: isBest ? 'var(--accent-text)' : 'var(--text)',
                                  background: isBest
                                    ? 'var(--bg)'
                                    : unsolved
                                      ? 'repeating-linear-gradient(45deg, var(--panel-alt), var(--panel-alt) 3px, var(--bg) 3px, var(--bg) 6px)'
                                      : undefined,
                                  boxShadow: isOutright ? 'inset 2px 0 0 var(--accent)' : undefined,
                                }}
                              >
                                {Number.isFinite(value)
                                  ? fmt(value, metric === 'gap' ? 2 : 1)
                                  : '—'}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                    <tfoot>
                      {metric === 'gap' && (
                        <tr>
                          <td
                            style={{
                              ...summaryRow(SUMMARY_ROW_H),
                              fontWeight: 600,
                              background: 'var(--panel-alt)',
                              borderTop: '1px solid var(--border-strong)',
                            }}
                          >
                            Mean gap (%)
                          </td>
                          <td
                            className="num"
                            style={{
                              ...summaryRow(SUMMARY_ROW_H),
                              background: 'var(--panel-alt)',
                              borderTop: '1px solid var(--border-strong)',
                            }}
                          >
                            —
                          </td>
                          {columnSummary.map((c) => (
                            <td
                              key={c.algorithm}
                              className="num"
                              style={{
                                ...summaryRow(SUMMARY_ROW_H),
                                background: 'var(--panel-alt)',
                                borderTop: '1px solid var(--border-strong)',
                                fontWeight: c.mean === bestGap.value ? 600 : 400,
                                color:
                                  c.mean === bestGap.value ? 'var(--accent-text)' : 'var(--text)',
                              }}
                            >
                              {fmt(c.mean, 2)}
                            </td>
                          ))}
                        </tr>
                      )}
                      <tr>
                        <td
                          style={{ ...summaryRow(0), fontWeight: 600, background: 'var(--panel-alt)' }}
                        >
                          Best in row
                        </td>
                        <td className="num" style={{ ...summaryRow(0), background: 'var(--panel-alt)' }}>
                          —
                        </td>
                        {columnSummary.map((c) => (
                          <td
                            key={c.algorithm}
                            className="num"
                            style={{
                              ...summaryRow(0),
                              background: 'var(--panel-alt)',
                              color: 'var(--text-dim)',
                            }}
                            title={`${c.algorithm} held the best value on ${c.wins} of ${c.measured} instances it was measured on`}
                          >
                            {fmtInt(c.wins)}/{fmtInt(c.measured)}
                          </td>
                        ))}
                      </tr>
                    </tfoot>
                  </table>
                </div>
                <div
                  style={{
                    padding: '10px 14px',
                    borderTop: '1px solid var(--border)',
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '6px 22px',
                    alignItems: 'center',
                  }}
                >
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 7,
                      fontSize: 11,
                      color: 'var(--text-dim)',
                    }}
                  >
                    <span
                      style={{
                        width: 22,
                        height: 14,
                        background: 'var(--bg)',
                        boxShadow: 'inset 2px 0 0 var(--accent)',
                        border: '1px solid var(--border)',
                      }}
                    />
                    best in the row; a rule marks a value no other solver matched
                  </span>
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 7,
                      fontSize: 11,
                      color: 'var(--text-dim)',
                    }}
                  >
                    <span
                      style={{
                        width: 22,
                        height: 14,
                        border: '1px solid var(--border)',
                        background:
                          'repeating-linear-gradient(45deg, var(--panel-alt), var(--panel-alt) 3px, var(--bg) 3px, var(--bg) 6px)',
                      }}
                    />
                    no complete solution inside the budget
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                    <span className="mono">—</span> no measured value
                  </span>
                </div>
                {provenance.unsolved.length > 0 && (
                  <Note style={{ padding: '0 14px 12px' }}>
                    {provenance.unsolved
                      .map(([key, v]) => {
                        const [instance, algorithm] = key.split('|');
                        return `${algorithm} returned no complete solution on ${instance} (${v.failed} of ${v.total} seeds)`;
                      })
                      .join('; ')}
                    . Those seeds are excluded from every mean on this page
                    rather than scored as zero.
                  </Note>
                )}
              </Panel>
            </div>

            <div style={{ marginTop: 12 }}>
              <Panel
                title="Statistical verdict"
                actions={
                  detail.omnibus ? (
                    <Badge
                      tone={detail.omnibus.p_value <= ALPHA ? 'ok' : undefined}
                      mark={detail.omnibus.p_value <= ALPHA ? 'disc' : 'square'}
                    >
                      {detail.omnibus.p_value <= ALPHA
                        ? 'algorithms differ'
                        : 'no detectable difference'}
                    </Badge>
                  ) : undefined
                }
              >
                {detail.omnibus ? (
                  <>
                    <p style={{ color: 'var(--text-dim)', maxWidth: 980 }}>
                      Friedman test over {fmtInt(detail.omnibus.n_instances)} instances
                      and {fmtInt(detail.omnibus.algorithms.length)} algorithms:{' '}
                      <span className="mono">χ² = {fmt(detail.omnibus.statistic, 3)}</span>,{' '}
                      <span className="mono">p = {fmtP(detail.omnibus.p_value)}</span>.{' '}
                      {detail.omnibus.p_value <= ALPHA
                        ? 'The algorithms are not all equivalent; the pairwise comparisons below are against the best-ranked control, with Holm correction for multiple testing.'
                        : 'The null hypothesis that all algorithms perform equally is not rejected at the 5 % level. On this instance set, the differences between them are within run-to-run noise.'}
                      {completeInstances === detail.omnibus.n_instances &&
                        completeInstances !== detail.instances.length && (
                          <>
                            {' '}The test is a complete-block design, so it uses only the{' '}
                            {fmtInt(completeInstances)} of {fmtInt(detail.instances.length)}{' '}
                            instances on which every algorithm returned a solution.
                          </>
                        )}
                    </p>

                    <div style={{ display: 'grid', gridTemplateColumns: '288px 1fr', gap: 18 }}>
                      <div>
                        <Caption style={{ marginBottom: 7 }}>Mean rank</Caption>
                        <table className="grid">
                          <thead>
                            <tr>
                              <th>Algorithm</th>
                              <th style={{ textAlign: 'right' }}>Rank</th>
                              <th style={{ width: 78 }}>of {detail.omnibus.algorithms.length}</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(detail.omnibus.mean_ranks)
                              .sort((a, b) => a[1] - b[1])
                              .map(([a, rank], index) => (
                                <tr key={a}>
                                  <td>
                                    <span
                                      style={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        gap: 6,
                                      }}
                                    >
                                      <Swatch color={algorithmColor(a)} />
                                      <span style={{ fontWeight: index === 0 ? 600 : 500 }}>
                                        {a}
                                      </span>
                                      {a === detail.omnibus?.control && (
                                        <span
                                          style={{
                                            fontSize: 9,
                                            fontFamily: 'var(--display)',
                                            letterSpacing: '0.1em',
                                            textTransform: 'uppercase',
                                            color: 'var(--navy-300)',
                                          }}
                                        >
                                          control
                                        </span>
                                      )}
                                    </span>
                                  </td>
                                  <td
                                    className="num"
                                    style={{ fontWeight: index === 0 ? 600 : 400 }}
                                  >
                                    {fmt(rank, 2)}
                                  </td>
                                  <td>
                                    <Meter
                                      value={rank / detail.omnibus!.algorithms.length}
                                      title={`mean rank ${fmt(rank, 2)} of ${detail.omnibus!.algorithms.length}`}
                                    />
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                        <Note style={{ marginTop: 7 }}>
                          1 is best. Ranks are assigned within each instance and
                          averaged, so a shorter bar is a better algorithm.
                        </Note>
                      </div>

                      <div>
                        <Caption style={{ marginBottom: 7 }}>
                          Pairwise, against {detail.omnibus.control ?? 'control'}
                        </Caption>
                        <table className="grid">
                          <thead>
                            <tr>
                              <th>Comparison</th>
                              <th style={{ textAlign: 'right' }}>n</th>
                              <th style={{ textAlign: 'right' }}>Median A</th>
                              <th style={{ textAlign: 'right' }}>Median B</th>
                              <th style={{ textAlign: 'right' }}>p raw</th>
                              <th style={{ textAlign: 'right' }}>p Holm</th>
                              <th style={{ textAlign: 'right' }}>|r|</th>
                              <th style={{ textAlign: 'left' }}>Verdict at α = {ALPHA}</th>
                            </tr>
                          </thead>
                          <tbody>
                            {detail.omnibus.post_hoc.map((test) => {
                              const adjusted = test.p_adjusted ?? test.p_value;
                              const significant = Number.isFinite(adjusted) && adjusted <= ALPHA;
                              const nominal =
                                !significant &&
                                Number.isFinite(test.p_value) &&
                                test.p_value <= ALPHA;
                              const winner =
                                test.winner ?? (test.median_a < test.median_b ? test.a : test.b);
                              return (
                                <tr key={`${test.a}-${test.b}`}>
                                  <td>
                                    {test.a} vs {test.b}
                                  </td>
                                  <td className="num">{fmtInt(test.n)}</td>
                                  <td className="num">{fmt(test.median_a, 3)}</td>
                                  <td className="num">{fmt(test.median_b, 3)}</td>
                                  <td className="num" style={{ color: 'var(--text-dim)' }}>
                                    {fmtP(test.p_value)}
                                  </td>
                                  <td
                                    className="num"
                                    style={{
                                      fontWeight: significant ? 600 : 400,
                                      color: significant ? 'var(--accent-text)' : 'var(--text)',
                                      background: significant ? 'var(--bg)' : undefined,
                                    }}
                                  >
                                    {fmtP(test.p_adjusted)}
                                  </td>
                                  <td className="num">{fmt(Math.abs(test.effect_size), 2)}</td>
                                  <td style={{ textAlign: 'left' }}>
                                    {significant ? (
                                      <Badge tone="ok" mark="disc">
                                        {winner} better
                                      </Badge>
                                    ) : nominal ? (
                                      <Badge
                                        tone="warn"
                                        mark="ring"
                                        title={`Raw p = ${fmtP(test.p_value)} would pass, but the Holm correction across ${detail.omnibus?.post_hoc.length} comparisons puts it at ${fmtP(test.p_adjusted)}.`}
                                      >
                                        not after Holm
                                      </Badge>
                                    ) : (
                                      <Badge mark="square">no difference</Badge>
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                        <Note style={{ marginTop: 7 }}>
                          Paired Wilcoxon signed-rank tests on the per-instance
                          median gap, with Holm correction across the family of{' '}
                          {fmtInt(detail.omnibus.post_hoc.length)} comparisons.{' '}
                          <strong style={{ color: 'var(--text)' }}>
                            {fmtInt(survivingComparisons)} of{' '}
                            {fmtInt(detail.omnibus.post_hoc.length)}
                          </strong>{' '}
                          survive the correction at α = {ALPHA}; the corrected column
                          is the one that decides, and it is the emphasised one.
                          Effect size is the matched-pairs rank-biserial
                          correlation, reported as a magnitude.
                        </Note>
                      </div>
                    </div>
                  </>
                ) : (
                  <Empty style={{ minHeight: 120 }}>
                    No omnibus test was computed for this result set. The
                    Friedman test needs at least three algorithms measured on the
                    same instances.
                  </Empty>
                )}
              </Panel>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
