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
import { fmt, fmtCompact } from '../lib/format';

/* -------------------------------------------------------------------------- */
/* design tokens                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Last-resort values, used only when this module is evaluated with no document
 * (tests, SSR) or before `global.css` has been applied. They mirror `:root`.
 */
const FALLBACK: Record<string, string> = {
  '--panel': '#ffffff',
  '--border': '#dce0e8',
  '--border-strong': '#c2c8d4',
  '--text': '#1c273b',
  '--text-dim': '#5a6376',
  '--text-faint': '#868da0',
  '--navy': '#1c273b',
  '--navy-300': '#686f7c',
  '--accent': '#1c273b',
  '--violet': '#5b4fcf',
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

/** `#rrggbb` at a given alpha, for the faint fills under a series. */
function alpha(hex: string, a: number): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
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
 * white panel (accent 5.7:1, navy-300 5.1:1, violet 6.1:1).
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
 * Legend voice, kept here so every chart in the platform agrees. Nothing in
 * this file renders a `<Legend>` - both charts name their series in the tooltip
 * instead - but a chart elsewhere that needs one should take these values.
 */
const LEGEND = {
  get color(): string {
    return token('--text-dim');
  },
  get fontFamily(): string {
    return token('--font');
  },
  fontSize: 11,
};

const GRID_DASH = '2 4';

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

  const bestColor = token('--accent');
  const meanColor = token('--navy-300');
  const refColor = token('--navy');

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
          tickFormatter={(v: number) => `${fmt(v, 1)}`}
          label={{ value: 'elapsed (s)', position: 'insideBottom', offset: -12, style: AXIS.label }}
        />
        <YAxis
          stroke={AXIS.stroke}
          tick={AXIS.tick}
          tickLine={{ stroke: token('--border') }}
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

export { AXIS as CHART_AXIS, TOOLTIP_STYLE as CHART_TOOLTIP, LEGEND as CHART_LEGEND };
