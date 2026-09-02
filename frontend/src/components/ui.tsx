/**
 * The small set of presentational primitives the pages are assembled from.
 *
 * They exist so that a panel, a labelled field, a statistic and a status badge
 * look identical on all four pages without every page importing the same class
 * names by hand. Nothing here holds state or talks to the network.
 */

import type { CSSProperties, ReactNode } from 'react';

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
      {hint !== undefined && <span className="field-hint">{hint}</span>}
    </div>
  );
}

export function Stat({
  label,
  value,
  unit,
  sub,
  tone,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  sub?: ReactNode;
  tone?: 'ok' | 'warn' | 'bad';
}) {
  const color = tone === 'ok' ? 'var(--ok)' : tone === 'warn' ? 'var(--warn)' : tone === 'bad' ? 'var(--bad)' : undefined;
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={color ? { color } : undefined}>
        {value}
        {unit && <span className="unit">{unit}</span>}
      </div>
      {sub !== undefined && <div className="stat-sub">{sub}</div>}
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

export function Badge({
  tone,
  children,
}: {
  tone?: 'ok' | 'warn' | 'bad';
  children: ReactNode;
}) {
  return <span className={`badge${tone ? ` ${tone}` : ''}`}>{children}</span>;
}

export function KeyValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="kv">
      <span>{label}</span>
      <span>{value}</span>
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

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
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
