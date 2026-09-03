/**
 * Typeset display equations without a maths engine.
 *
 * The four update rules this project needs to show are short, so a full LaTeX
 * renderer would add a few hundred kilobytes to the bundle to typeset a dozen
 * lines. Instead the equation is set in the product's mono face, on generous
 * leading, between two hairline rules - the drafting convention for a displayed
 * result. There is no tinted box: on a white panel a block of monospaced glyphs
 * with air around it already separates itself from the prose, and a coloured
 * card would spend the accent on something that is not an action.
 *
 * Every symbol is then defined in a description list rather than in a paragraph,
 * so a reader checking the implementation can find "what is beta" by scanning a
 * column instead of reading a sentence.
 */

import type { ReactNode } from 'react';

export interface SymbolDef {
  /** The symbol as it appears in the equation above. */
  sym: ReactNode;
  /** What it denotes, in words a reader can check against the source. */
  desc: ReactNode;
}

export function Eq({
  children,
  defs,
  note,
}: {
  children: ReactNode;
  defs?: SymbolDef[];
  note?: ReactNode;
}) {
  return (
    <figure style={{ margin: '14px 0 18px' }}>
      <div
        style={{
          fontFamily: 'var(--mono)',
          fontSize: 14,
          lineHeight: 2.15,
          letterSpacing: '-0.005em',
          fontVariantNumeric: 'tabular-nums',
          color: 'var(--text)',
          borderTop: '1px solid var(--border-strong)',
          borderBottom: '1px solid var(--border-strong)',
          padding: '13px 2px 14px',
          overflowX: 'auto',
          whiteSpace: 'nowrap',
        }}
      >
        {children}
      </div>

      {defs && defs.length > 0 && (
        <dl
          style={{
            display: 'grid',
            gridTemplateColumns: 'max-content minmax(0, 1fr)',
            columnGap: 14,
            rowGap: 5,
            margin: '10px 0 0',
            fontSize: 12,
            lineHeight: 1.55,
          }}
        >
          {defs.map((d, i) => (
            <div key={i} style={{ display: 'contents' }}>
              <dt
                style={{
                  fontFamily: 'var(--mono)',
                  fontSize: 12,
                  color: 'var(--text)',
                  textAlign: 'right',
                  whiteSpace: 'nowrap',
                }}
              >
                {d.sym}
              </dt>
              <dd style={{ margin: 0, color: 'var(--text-dim)' }}>{d.desc}</dd>
            </div>
          ))}
        </dl>
      )}

      {note && (
        <figcaption
          style={{
            fontSize: 12,
            color: 'var(--text-dim)',
            marginTop: defs && defs.length > 0 ? 9 : 8,
            lineHeight: 1.6,
            borderTop: defs && defs.length > 0 ? '1px solid var(--border)' : undefined,
            paddingTop: defs && defs.length > 0 ? 9 : 0,
          }}
        >
          {note}
        </figcaption>
      )}
    </figure>
  );
}

/** An italic variable name, optionally with a subscript. */
export function V({ children, sub }: { children: ReactNode; sub?: ReactNode }) {
  return (
    <span style={{ fontFamily: 'var(--mono)', fontStyle: 'italic' }}>
      {children}
      {sub !== undefined && (
        <sub
          style={{
            fontStyle: 'normal',
            fontSize: '0.72em',
            letterSpacing: '0.02em',
            color: 'var(--text-dim)',
          }}
        >
          {sub}
        </sub>
      )}
    </span>
  );
}

/**
 * An operator or relation, with the spacing maths typesetting would give it.
 * Set in the UI face: DM Mono has no arrows, sigma or set relations, so leaving
 * these to the mono fallback would mix two unrelated monospaced fonts on one
 * line.
 */
export function Op({ children }: { children: ReactNode }) {
  return (
    <span style={{ margin: '0 0.3em', fontStyle: 'normal', fontFamily: 'var(--font)' }}>
      {children}
    </span>
  );
}

/** Non-italic text inside an equation, such as a function name or a condition. */
export function T({ children }: { children: ReactNode }) {
  return (
    <span style={{ fontStyle: 'normal', fontFamily: 'var(--font)', fontSize: '0.94em' }}>
      {children}
    </span>
  );
}
