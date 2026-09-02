/**
 * Recharts wrappers with a shared dark theme.
 *
 * Every chart in the platform is built from these so that grid lines, axis
 * colours, tick density and tooltip styling are decided once. Axis labels
 * always carry units: an unlabelled convergence curve is the single easiest way
 * to make a result look better than it is, and this project is trying to do the
 * opposite.
 */

import type { ReactNode } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { RunTick } from '../api/types';
import { fmt, fmtCompact } from '../lib/format';

const AXIS = {
  stroke: '#4a5462',
  tick: { fill: '#8b97a8', fontSize: 11 },
  label: { fill: '#6b7889', fontSize: 11 },
};

const TOOLTIP_STYLE = {
  background: '#171d25',
  border: '1px solid #35404e',
  borderRadius: 4,
  fontSize: 12,
  color: '#dce3ec',
  padding: '6px 9px',
};

export function ChartFrame({ height, children }: { height: number; children: ReactNode }) {
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        {children as React.ReactElement}
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Best and mean cost against elapsed wall-clock seconds.
 *
 * Time rather than iteration on the x axis, because one iteration of a swarm of
 * 30 is not comparable with one iteration of an annealer, and the honest
 * comparison between algorithms is what they achieved for the same budget.
 * `reference`, when given, is the best-known cost from the literature.
 */
export function ConvergenceChart({
  ticks,
  reference,
  height = 260,
}: {
  ticks: RunTick[];
  reference: number | null;
  height?: number;
}) {
  const data = ticks.map((t) => ({
    elapsed: Number(t.elapsed.toFixed(3)),
    best: Number.isFinite(t.best_cost) ? t.best_cost : null,
    mean: Number.isFinite(t.mean_cost) ? t.mean_cost : null,
  }));

  return (
    <ChartFrame height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 22, left: 8 }}>
        <CartesianGrid stroke="#232b35" strokeDasharray="2 4" />
        <XAxis
          dataKey="elapsed"
          type="number"
          domain={['dataMin', 'dataMax']}
          stroke={AXIS.stroke}
          tick={AXIS.tick}
          tickFormatter={(v: number) => `${fmt(v, 1)}`}
          label={{ value: 'elapsed (s)', position: 'insideBottom', offset: -12, style: AXIS.label }}
        />
        <YAxis
          stroke={AXIS.stroke}
          tick={AXIS.tick}
          width={62}
          domain={['auto', 'auto']}
          tickFormatter={(v: number) => fmtCompact(v)}
          label={{
            value: 'objective (cost units)',
            angle: -90,
            position: 'insideLeft',
            offset: 6,
            style: { ...AXIS.label, textAnchor: 'middle' },
          }}
        />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          labelFormatter={(v) => `t = ${fmt(Number(v), 2)} s`}
          formatter={(value: unknown, name: unknown) => [fmt(Number(value), 2), String(name)]}
        />
        {reference !== null && Number.isFinite(reference) && (
          <ReferenceLine
            y={reference}
            stroke="#7fc7a2"
            strokeDasharray="5 4"
            label={{
              value: `best known ${fmt(reference, 1)}`,
              position: 'insideTopRight',
              fill: '#7fc7a2',
              fontSize: 11,
            }}
          />
        )}
        <Line
          type="monotone"
          dataKey="best"
          name="best"
          stroke="#5b93e6"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="mean"
          name="population mean"
          stroke="#7c8797"
          strokeWidth={1.2}
          strokeDasharray="3 3"
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
      </LineChart>
    </ChartFrame>
  );
}

/**
 * Swarm diversity against elapsed time.
 *
 * Diversity is the mean pairwise distance between particle positions in the
 * random-key space, as recorded by `IterationRecord.diversity`. Watching it
 * collapse is what tells you whether the search still has anywhere to go.
 */
export function DiversityChart({ ticks, height = 150 }: { ticks: RunTick[]; height?: number }) {
  const data = ticks.map((t) => ({
    elapsed: Number(t.elapsed.toFixed(3)),
    diversity: Number.isFinite(t.diversity) ? t.diversity : null,
  }));
  return (
    <ChartFrame height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 22, left: 8 }}>
        <CartesianGrid stroke="#232b35" strokeDasharray="2 4" />
        <XAxis
          dataKey="elapsed"
          type="number"
          domain={['dataMin', 'dataMax']}
          stroke={AXIS.stroke}
          tick={AXIS.tick}
          tickFormatter={(v: number) => fmt(v, 1)}
          label={{ value: 'elapsed (s)', position: 'insideBottom', offset: -12, style: AXIS.label }}
        />
        <YAxis
          stroke={AXIS.stroke}
          tick={AXIS.tick}
          width={62}
          tickFormatter={(v: number) => fmt(v, 2)}
          label={{
            value: 'mean pairwise distance',
            angle: -90,
            position: 'insideLeft',
            offset: 6,
            style: { ...AXIS.label, textAnchor: 'middle' },
          }}
        />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          labelFormatter={(v) => `t = ${fmt(Number(v), 2)} s`}
          formatter={(value: unknown) => [fmt(Number(value), 4), 'diversity']}
        />
        <Line
          type="monotone"
          dataKey="diversity"
          stroke="#c9a63f"
          strokeWidth={1.6}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
      </LineChart>
    </ChartFrame>
  );
}

export { AXIS as CHART_AXIS, TOOLTIP_STYLE as CHART_TOOLTIP };
