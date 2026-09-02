/**
 * Runtime coercion helpers for JSON that arrives as `unknown`.
 *
 * The backend is written by a separate process and a payload can legitimately
 * be thinner than the declared contract, particularly while both halves are
 * still being finished. Rather than casting responses with `as` and letting a
 * missing field surface as a crash inside a chart, every response is walked
 * through these helpers, which return a well-typed value and a caller-supplied
 * fallback when the field is absent or of the wrong shape.
 *
 * They are deliberately small and total: no helper here ever throws.
 */

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function rec(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

export function num(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

/** Like `num` but preserves a genuine null, which several fields use as "unknown". */
export function numOrNull(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string' && value !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function str(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

export function bool(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  return fallback;
}

export function boolOrNull(value: unknown): boolean | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  return null;
}

export function arr(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

export function numArray(value: unknown): number[] {
  return arr(value).map((v) => num(v));
}

export function strArray(value: unknown): string[] {
  return arr(value).filter((v): v is string => typeof v === 'string');
}

/** `[[a, b], ...]` pairs, dropping anything that is not a two-number tuple. */
export function pairArray(value: unknown): [number, number][] {
  const out: [number, number][] = [];
  for (const item of arr(value)) {
    if (Array.isArray(item) && item.length >= 2) {
      out.push([num(item[0]), num(item[1])]);
    }
  }
  return out;
}

/** A list of index lists, used for route encodings. */
export function intMatrix(value: unknown): number[][] {
  return arr(value).map((row) => numArray(row).map((v) => Math.round(v)));
}

/**
 * Read the first key that is present, so a field renamed between the contract
 * and the implementation (`size` versus `n_customers`, say) still resolves.
 */
export function pick(source: Record<string, unknown>, ...keys: string[]): unknown {
  for (const key of keys) {
    if (source[key] !== undefined && source[key] !== null) return source[key];
  }
  return undefined;
}

/**
 * Unwrap the common envelope shapes: a bare array, `{items: [...]}`, or
 * `{<name>: [...]}` for a caller-supplied key.
 */
export function listOf(value: unknown, ...keys: string[]): unknown[] {
  if (Array.isArray(value)) return value;
  const record = rec(value);
  for (const key of ['items', 'results', 'data', ...keys]) {
    if (Array.isArray(record[key])) return record[key] as unknown[];
  }
  return [];
}
