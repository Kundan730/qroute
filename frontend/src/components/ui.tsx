/**
 * The small set of presentational primitives the pages are assembled from.
 *
 * They exist so that a panel, a labelled field, a statistic and a status badge
 * look identical on all four pages without every page importing the same class
 * names by hand. Nothing here holds state or talks to the network.
 *
 * Colour. Almost nothing here names a colour: the class names are styled by
 * `styles/global.css`, which owns the palette. The exceptions are the few
 * places where a value has to be computed at render time - a tone on a
 * statistic, the fill of a meter - and those read a design token through
 * `var(...)` rather than carrying a literal.
 *
 * State is encoded in form as well as in hue. A badge can carry a mark whose
 * *shape* says what it means - a filled disc for an affirmative result, a
 * hollow ring for a borderline one, a filled square for a negative one - so
 * that the interface survives a projector, a greyscale print of the report, and
 * a reader who cannot separate red from green.
 */

import type { CSSProperties, ReactNode } from 'react';

export type Tone = 'ok' | 'warn' | 'bad';

/** The shape of a status mark. Chosen so the three read apart in greyscale. */
export type MarkShape = 'disc' | 'ring' | 'square';

const TONE_COLOR: Record<Tone, string> = {
  ok: 'var(--ok)',
  warn: 'var(--warn)',
  bad: 'var(--bad)',
};

export function toneColor(tone: Tone | undefined): string | undefined {
  return tone ? TONE_COLOR[tone] : undefined;
}

export function Panel({
  title,
  actions,
  children,
  flush = false,
  style,
  className = '',
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  style?: CSSProperties;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`} style={style}>
      {title !== undefined && (
        <header className="panel-head">
          <h2>{title}</h2>
          {actions && <div className="spacer" />}
          {actions}
        </header>
      )}
      <div className={`panel-body${flush ? ' flush' : ''}`}>{children}</div>
    </section>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
      {hint !== undefined && (
        // `.field-hint` is painted --text-faint by global.css, which is 3.3:1 on
        // a white panel. Hints are prose a reader is expected to read, so the
        // primitive darkens them to --text-dim (5.9:1) rather than leaving a
        // shared class failing contrast on every page that uses a Field.
        <span className="field-hint" style={{ color: 'var(--text-dim)' }}>
          {hint}
        </span>
      )}
    </div>
  );
}

export function Stat({
  label,
  value,
  unit,
  sub,
  tone,
  title,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  sub?: ReactNode;
  tone?: Tone;
  title?: string;
}) {
  const color = toneColor(tone);
  return (
    <div className="stat" title={title}>
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={color ? { color } : undefined}>
        {value}
        {unit && <span className="unit">{unit}</span>}
      </div>
      {sub !== undefined && (
        // Same reason as the field hint: --text-faint on --panel is 3.3:1, and
        // the sub-line carries the denominator a stat is meaningless without.
        <div className="stat-sub" style={{ color: 'var(--text-dim)' }}>
          {sub}
        </div>
      )}
    </div>
  );
}

export function StatGrid({ columns, children }: { columns: number; children: ReactNode }) {
  return (
    <div className="stats" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
      {children}
    </div>
  );
}

/**
 * A small status mark drawn in `currentColor`, so it takes the colour of
 * whatever it sits inside. The shape carries the meaning on its own: hue is a
 * second, redundant channel rather than the only one.
 */
export function Mark({
  shape = 'disc',
  size = 7,
  style,
}: {
  shape?: MarkShape;
  size?: number;
  style?: CSSProperties;
}) {
  return (
    <span
      aria-hidden="true"
      style={{
        display: 'inline-block',
        flex: '0 0 auto',
        width: size,
        height: size,
        borderRadius: shape === 'square' ? 1 : '50%',
        background: shape === 'ring' ? 'transparent' : 'currentColor',
        border: shape === 'ring' ? '1.5px solid currentColor' : undefined,
        ...style,
      }}
    />
  );
}

export function Badge({
  tone,
  mark,
  title,
  children,
}: {
  tone?: Tone;
  mark?: MarkShape;
  title?: string;
  children: ReactNode;
}) {
  return (
    <span className={`badge${tone ? ` ${tone}` : ''}`} title={title}>
      {mark && <Mark shape={mark} size={6} />}
      {children}
    </span>
  );
}

/**
 * A colour key for a series, outlined so that the shape is visible whatever
 * the fill does - a pale series colour on a white panel would otherwise fall
 * below the contrast a meaningful graphic has to reach.
 */
export function Swatch({ color, size = 8 }: { color: string; size?: number }) {
  return (
    <span
      aria-hidden="true"
      style={{
        display: 'inline-block',
        flex: '0 0 auto',
        width: size,
        height: size,
        background: color,
        border: '1px solid var(--navy-300)',
        borderRadius: 1,
      }}
    />
  );
}

/**
 * A determinate bar. Used for elapsed time against a budget and for a mean
 * rank against the number of algorithms - never as decoration, always as a
 * second reading of a number printed next to it.
 */
export function Meter({
  value,
  tone,
  height = 4,
  title,
  animated = false,
}: {
  /** Fraction in `[0, 1]`; values outside are clamped. */
  value: number;
  tone?: Tone;
  height?: number;
  title?: string;
  animated?: boolean;
}) {
  const fraction = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
  return (
    <div
      title={title}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(fraction * 100)}
      style={{
        height,
        background: 'var(--border)',
        borderRadius: 1,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          width: `${fraction * 100}%`,
          height: '100%',
          background: toneColor(tone) ?? 'var(--accent)',
          transition: animated ? 'width 220ms linear' : undefined,
        }}
      />
    </div>
  );
}

export function KeyValue({
  label,
  value,
  title,
  wrap = false,
}: {
  label: string;
  value: ReactNode;
  title?: string;
  /**
   * Let a long value fold onto a second line instead of running off the rail.
   *
   * `.kv` in global.css is a baseline flex row whose value is `nowrap` and
   * whose label ellipses. That is right for the numbers this row usually
   * carries, but a value with no spaces - a platform string, a commit, a file
   * name - first squeezes the label to nothing (so the reader cannot tell what
   * they are looking at) and then overflows the rail's padding, because a
   * `nowrap` span cannot shrink below its text. Opting into `wrap` pins the
   * label at its natural width and lets the value break, which keeps the row
   * inside the rail and the whole value on screen. Off by default, so every
   * existing caller keeps the single-line row it was written for.
   */
  wrap?: boolean;
}) {
  return (
    <div className="kv" title={title}>
      <span style={wrap ? { flex: '0 0 auto', overflow: 'visible' } : undefined}>{label}</span>
      <span
        style={
          wrap
            ? { whiteSpace: 'normal', overflowWrap: 'anywhere', textAlign: 'right', minWidth: 0 }
            : undefined
        }
      >
        {value}
      </span>
    </div>
  );
}

export function Notice({
  kind = 'warn',
  children,
}: {
  kind?: 'warn' | 'error';
  children: ReactNode;
}) {
  return <div className={`notice${kind === 'error' ? ' error' : ''}`}>{children}</div>;
}

export function Empty({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  // An empty state is the only thing on screen when it shows, so it has to be
  // readable: --text-dim rather than the --text-faint the shared class uses.
  // `style` exists so a caller can reserve the height of the chart or table the
  // panel would otherwise be showing, without reaching for the bare class name
  // and losing the colour correction with it.
  return (
    <div className="empty" style={{ color: 'var(--text-dim)', ...style }}>
      {children}
    </div>
  );
}

/**
 * The explanatory line that sits under a chart or a table. Deliberately set in
 * `--text-dim` rather than `--text-faint`: it is prose a judge is expected to
 * read, and it has to clear 4.5:1 on the panel behind it.
 */
export function Note({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div style={{ fontSize: 11, lineHeight: 1.55, color: 'var(--text-dim)', ...style }}>
      {children}
    </div>
  );
}

/** The small uppercase label that titles a group inside a panel. */
export function Caption({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div
      style={{
        fontFamily: 'var(--display)',
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: '0.13em',
        textTransform: 'uppercase',
        color: 'var(--navy-300)',
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function RailSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rail-section">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

export function CheckLine({
  checked,
  onChange,
  children,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  children: ReactNode;
}) {
  return (
    <label className="checkline">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {children}
    </label>
  );
}
