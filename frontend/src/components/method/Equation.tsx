/**
 * Typeset display equations without a maths engine.
 *
 * The four update rules this project needs to show are short, so a full LaTeX
 * renderer would add a few hundred kilobytes to the bundle to typeset a dozen
 * lines. Instead, variables are set in an italic serif with real subscripts,
 * operators get proper spacing, and each equation carries a plain-language
 * gloss underneath, which is what makes it readable to a panel rather than only
 * to someone who already knows the algorithm.
 */

import type { ReactNode } from 'react';

export function Eq({ children, note }: { children: ReactNode; note?: ReactNode }) {
  return (
    <div style={{ margin: '10px 0 14px' }}>
      <div
        style={{
          fontFamily: 'Georgia, "Times New Roman", serif',
          fontSize: 16,
          lineHeight: 1.9,
          color: 'var(--text)',
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderLeft: '3px solid var(--accent-dim)',
          borderRadius: 'var(--radius)',
          padding: '10px 14px',
          overflowX: 'auto',
        }}
      >
        {children}
      </div>
      {note && (
        <div style={{ fontSize: 11.5, color: 'var(--text-faint)', marginTop: 5, lineHeight: 1.55 }}>
          {note}
        </div>
      )}
    </div>
  );
}

/** An italic variable name, optionally with a subscript. */
export function V({ children, sub }: { children: ReactNode; sub?: ReactNode }) {
  return (
    <span style={{ fontStyle: 'italic' }}>
      {children}
      {sub !== undefined && (
        <sub style={{ fontStyle: 'normal', fontSize: '0.72em', letterSpacing: '0.02em' }}>
          {sub}
        </sub>
      )}
    </span>
  );
}

/** An operator or relation, with the spacing maths typesetting would give it. */
export function Op({ children }: { children: ReactNode }) {
  return <span style={{ margin: '0 0.35em', fontStyle: 'normal' }}>{children}</span>;
}

/** Non-italic text inside an equation, such as a function name or a condition. */
export function T({ children }: { children: ReactNode }) {
  return <span style={{ fontStyle: 'normal', fontFamily: 'var(--font)', fontSize: '0.88em' }}>{children}</span>;
}
