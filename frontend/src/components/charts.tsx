/**
 * Recharts wrappers bound to the light design system.
 *
 * Every chart in the platform is built from these so that grid lines, axis
 * colours, tick density and tooltip styling are decided once. Axis labels
 * always carry units: an unlabelled convergence curve is the single easiest way
 * to make a result look better than it is, and this project is trying to do the
 * opposite.
 *
 * Recharts needs literal colour strings rather than `var(--token)`, because the
 * values reach SVG attributes and a canvas-side measurement pass. So the tokens
 * are read back off `:root` at use time through `token()` below, and nothing in
 * this file carries a hex of its own. The read is memoised only once it returns
 * a non-empty string, which means a call that happens before the stylesheet has
 * been applied falls back rather than poisoning the cache.
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
import type { DotItemDotProps } from 'recharts';
import type { RunTick } from '../api/types';
import { fmt, fmtInt } from '../lib/format';

/* -------------------------------------------------------------------------- */
/* design tokens                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Last-resort values, used only when this module is evaluated with no document
 * (tests, SSR) or before `global.css` has been applied - which in this app is
 * unreachable, since the stylesheet is a render-blocking link and the module is
 * deferred. They mirror `:root` in `global.css` and must be re-synced with it;
 * nothing here is a colour decision of its own.
 */
const FALLBACK: Record<string, string> = {
  '--panel': '#ffffff',
  '--border': '#dce0e8',
  '--border-strong': '#c2c8d4',
  '--text': '#1c273b',
  '--text-dim': '#5a6376',
  '--navy-300': '#686f7c',
  '--accent': '#1c273b',
  '--violet': '#5b4fcf',
  '--rose': '#a8436a',
  '--radius': '3px',
  '--shadow-float': '0 1px 2px rgba(28, 39, 59, 0.08), 0 8px 24px rgba(28, 39, 59, 0.1)',
  '--font': "'DM Sans', ui-sans-serif, system-ui, sans-serif",
  '--display': "'Urbanist', 'DM Sans', ui-sans-serif, system-ui, sans-serif",
  '--mono': "'DM Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
};

const TOKEN_CACHE = new Map<string, string>();

/** Read a custom property off `:root`, memoised once it actually resolves. */
export function token(name: string): string {
  const cached = TOKEN_CACHE.get(name);
  if (cached) return cached;
  let read = '';
  if (typeof document !== 'undefined' && document.documentElement) {
    read = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  if (read) {
    TOKEN_CACHE.set(name, read);
    return read;
  }
  return FALLBACK[name] ?? FALLBACK['--text'];
}

/**
 * A token colour at a given alpha, for the faint fills under a series.
 *
 * Both hex lengths are accepted because the CSS minifier shortens what it can
 * on the way into `dist` - `--panel: #ffffff` reads back as `#fff` in the built
 * app - and a token that silently failed to parse here would come back fully
 * opaque rather than as a halo.
 */
function alpha(hex: string, a: number): string {
  const s = hex.trim();
  const long = /^#?([0-9a-f]{6})$/i.exec(s);
  const short = /^#?([0-9a-f]{3})$/i.exec(s);
  const six = long ? long[1] : short ? short[1].replace(/./g, (c) => c + c) : null;
  if (six === null) return s;
  const n = parseInt(six, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/**
 * Axis chrome. Hairlines in `--border`, marks in `--text-faint`, and the tick
 * text one step darker in `--text-dim`: `--text-faint` measures 3.25:1 on the
 * white panel, which is fine for a mark but short of the 4.5:1 a label owes the
 * reader. Getters rather than a frozen literal, so the values survive a module
 * evaluated ahead of first paint.
 */
const AXIS = {
  get stroke(): string {
    return token('--border');
  },
  get tick(): { fill: string; fontSize: number; fontFamily: string } {
    return { fill: token('--text-dim'), fontSize: 10, fontFamily: token('--mono') };
  },
  get label(): { fill: string; fontSize: number; fontFamily: string; letterSpacing: string } {
    return {
      fill: token('--text-dim'),
      fontSize: 10,
      fontFamily: token('--display'),
      letterSpacing: '0.06em',
    };
  },
};

const TOOLTIP_STYLE = {
  get background(): string {
    return token('--panel');
  },
  get border(): string {
    return `1px solid ${token('--border-strong')}`;
  },
  get borderRadius(): string {
    return token('--radius');
  },
  get boxShadow(): string {
    return token('--shadow-float');
  },
  get color(): string {
    return token('--text');
  },
  get fontFamily(): string {
    return token('--font');
  },
  fontSize: 12,
  padding: '7px 10px',
};

/** The `t = …` line above the values: a label, so it takes the label voice. */
function tooltipLabelStyle() {
  return {
    color: token('--text-dim'),
    fontFamily: token('--display'),
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: '0.08em',
    marginBottom: 3,
  };
}

/**
 * Values are digits, so they are mono and tabular. No `color`: recharts then
 * paints each value in its own series hue, which keys the tooltip to the lines
 * without needing a swatch. Every series colour used here clears 4.5:1 on the
 * white panel: accent/navy 12.6:1, navy-300 5.1:1, violet 6.1:1, rose 5.7:1.
 */
function tooltipItemStyle() {
  return {
    fontFamily: token('--mono'),
    fontSize: 11,
    fontVariantNumeric: 'tabular-nums' as const,
    padding: 0,
  };
}

/**
 * Grid and cursor dash pattern. Shared rather than repeated at each call site,
 * so the hairlines behind every chart in the platform stay one decision.
 */
const GRID_DASH = '2 4';

/**
 * Tick text for a cost axis, at a precision the axis can actually carry.
 *
 * `fmtCompact` drops to whole thousands above 10k, which is right for a count
 * of road segments and wrong for an objective: a run on the road network whose
 * costs live between 10,040 and 10,900 rendered five ticks reading
 * `11k 11k 10k 10k 10k`, an axis that says nothing at all. So the precision is
 * taken from the span the chart actually covers - one tick step has to be
 * visible in the printed digits - and the `k` scale is kept only while it can
 * be afforded, otherwise the figure is printed in full.
 */
function costTick(values: (number | null)[]): (v: number) => string {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (typeof v !== 'number' || !Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const full = (v: number) => fmtInt(v);
  if (!Number.isFinite(min)) return full;
  // Recharts draws about five ticks, so one step is a fifth of the span. A
  // series that never moved has no span and nothing to abbreviate towards.
  const step = (max - min) / 5;
  if (!(step > 0)) return full;
  const magnitude = Math.max(Math.abs(min), Math.abs(max));
  const scale = magnitude >= 1e6 ? 1e6 : magnitude >= 1e3 ? 1e3 : 1;
  if (scale === 1) return full;
  // Decimals enough that one step still shows in the printed digits. Past one
  // (two on the millions scale) the abbreviation has stopped paying for itself
  // and the figure is better read in full.
  const digits = Math.ceil(-Math.log10(step / scale));
  if (digits > (scale === 1e6 ? 2 : 1)) return full;
  const suffix = scale === 1e6 ? 'M' : 'k';
  return (v: number) => `${fmt(v / scale, Math.max(0, digits))}${suffix}`;
}

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
 * Marks the final sample of a series and nothing else.
 *
 * A convergence curve is read for where it ended up, so the last point gets a
 * filled dot inside a halo of the same hue; every other index renders an empty
 * group, which keeps the line clean.
 */
function finalDot(color: string, lastIndex: number) {
  return function FinalDot(props: DotItemDotProps) {
    const { cx, cy, index } = props;
    const key = `pt-${index}`;
    if (index !== lastIndex || typeof cx !== 'number' || typeof cy !== 'number') {
      return <g key={key} />;
    }
    return (
      <g key={key}>
        <circle cx={cx} cy={cy} r={5} fill={alpha(color, 0.16)} />
        <circle cx={cx} cy={cy} r={2.6} fill={color} stroke={token('--panel')} strokeWidth={1} />
      </g>
    );
  };
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

  // The searched series takes the accent, the population mean sits back in
  // grey, and the literature's best-known cost is an annotation rather than a
  // series, so it takes --rose: one of the two hues the palette reserves for
  // the map and the charts, and the one thing here that must stay legible
  // whatever the accent happens to be.
  const bestColor = token('--accent');
  const meanColor = token('--navy-300');
  const refColor = token('--rose');

  // The reference, when there is one, is part of what the axis has to resolve:
  // a best-known cost far below the search's range would otherwise widen the
  // domain without widening the tick precision.
  const yValues: (number | null)[] = data.flatMap((d) => [d.best, d.mean]);
  if (reference !== null && Number.isFinite(reference)) yValues.push(reference);
  const yTick = costTick(yValues);

  return (
    <ChartFrame height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 22, left: 8 }}>
        <CartesianGrid stroke={token('--border')} strokeDasharray={GRID_DASH} vertical={false} />
        <XAxis
          dataKey="elapsed"
          type="number"
          domain={['dataMin', 'dataMax']}
          stroke={AXIS.stroke}
          tick={AXIS.tick}
          tickLine={{ stroke: token('--border') }}
          minTickGap={24}
          tickFormatter={(v: number) => `${fmt(v, 1)}`}
          label={{ value: 'elapsed (s)', position: 'insideBottom', offset: -12, style: AXIS.label }}
        />
        <YAxis
          stroke={AXIS.stroke}
          tick={AXIS.tick}
          tickLine={{ stroke: token('--border') }}
          width={62}
          domain={['auto', 'auto']}
          tickFormatter={yTick}
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
          labelStyle={tooltipLabelStyle()}
          itemStyle={tooltipItemStyle()}
          cursor={{ stroke: token('--border-strong'), strokeDasharray: GRID_DASH }}
          labelFormatter={(v) => `t = ${fmt(Number(v), 2)} s`}
          formatter={(value: unknown, name: unknown) => [fmt(Number(value), 2), String(name)]}
        />
        {reference !== null && Number.isFinite(reference) && (
          <ReferenceLine
            y={reference}
            stroke={refColor}
            strokeDasharray="5 4"
            strokeWidth={1}
            label={{
              value: `best known ${fmt(reference, 1)}`,
              position: 'insideTopRight',
              fill: refColor,
              fontSize: 10,
              fontFamily: token('--mono'),
            }}
          />
        )}
        <Line
          type="monotone"
          dataKey="best"
          name="best"
          stroke={bestColor}
          strokeWidth={1.8}
          dot={finalDot(bestColor, data.length - 1)}
          activeDot={{ r: 3, fill: bestColor, stroke: token('--panel'), strokeWidth: 1 }}
          isAnimationActive={false}
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="mean"
          name="population mean"
          stroke={meanColor}
          strokeWidth={1.1}
          strokeDasharray="3 3"
          dot={false}
          activeDot={{ r: 2.6, fill: meanColor, stroke: token('--panel'), strokeWidth: 1 }}
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
  const lineColor = token('--violet');

  return (
    <ChartFrame height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 22, left: 8 }}>
        <CartesianGrid stroke={token('--border')} strokeDasharray={GRID_DASH} vertical={false} />
        <XAxis
          dataKey="elapsed"
          type="number"
          domain={['dataMin', 'dataMax']}
          stroke={AXIS.stroke}
          tick={AXIS.tick}
          tickLine={{ stroke: token('--border') }}
          minTickGap={24}
          tickFormatter={(v: number) => fmt(v, 1)}
          label={{ value: 'elapsed (s)', position: 'insideBottom', offset: -12, style: AXIS.label }}
        />
        <YAxis
          stroke={AXIS.stroke}
          tick={AXIS.tick}
          tickLine={{ stroke: token('--border') }}
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
          labelStyle={tooltipLabelStyle()}
          itemStyle={tooltipItemStyle()}
          cursor={{ stroke: token('--border-strong'), strokeDasharray: GRID_DASH }}
          labelFormatter={(v) => `t = ${fmt(Number(v), 2)} s`}
          formatter={(value: unknown) => [fmt(Number(value), 4), 'diversity']}
        />
        <Line
          type="monotone"
          dataKey="diversity"
          name="diversity"
          stroke={lineColor}
          strokeWidth={1.5}
          dot={finalDot(lineColor, data.length - 1)}
          activeDot={{ r: 2.8, fill: lineColor, stroke: token('--panel'), strokeWidth: 1 }}
          isAnimationActive={false}
          connectNulls
        />
      </LineChart>
    </ChartFrame>
  );
}

export { AXIS as CHART_AXIS, GRID_DASH as CHART_GRID_DASH, TOOLTIP_STYLE as CHART_TOOLTIP };
