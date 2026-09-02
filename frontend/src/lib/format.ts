/**
 * Number and unit formatting.
 *
 * Every figure the interface prints carries a unit, and the units come from the
 * backend's own conventions: distances in metres, travel times in seconds,
 * congestion as the fractional delay `(t - t0) / t0`, and benchmark costs in
 * the instance's native cost units. Formatting is centralised so the same
 * quantity never appears with two different precisions on two pages.
 */

const NBSP = ' '; // thin space, so "12.4 km" does not wrap

export function fmt(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toLocaleString('en-GB', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return Math.round(value).toLocaleString('en-GB');
}

/** Metres, shown as kilometres above one kilometre. */
export function fmtDistance(metres: number | null | undefined): string {
  if (metres === null || metres === undefined || !Number.isFinite(metres)) return '—';
  if (metres >= 1000) return `${fmt(metres / 1000, 2)}${NBSP}km`;
  return `${fmtInt(metres)}${NBSP}m`;
}

/** Seconds, shown as h:mm:ss above an hour and m:ss above a minute. */
export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—';
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  if (m > 0) return `${m}:${String(s).padStart(2, '0')}`;
  return `${s}${NBSP}s`;
}

/** Wall-clock seconds of a solver run, where sub-second precision matters. */
export function fmtSeconds(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—';
  if (seconds < 10) return `${fmt(seconds, 2)}${NBSP}s`;
  if (seconds < 120) return `${fmt(seconds, 1)}${NBSP}s`;
  return fmtDuration(seconds);
}

export function fmtPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${fmt(value, digits)}${NBSP}%`;
}

/** A simulator clock in minutes since midnight, printed as a 24-hour time. */
export function fmtClock(minuteOfWeek: number): string {
  const minuteOfDay = ((minuteOfWeek % 1440) + 1440) % 1440;
  const h = Math.floor(minuteOfDay / 60);
  const m = Math.floor(minuteOfDay % 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

const DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export function dayName(dayOfWeek: number): string {
  return DAY_NAMES[((dayOfWeek % 7) + 7) % 7] ?? 'Monday';
}

/** Compact counts for axis ticks: 1.2k, 34k, 1.1M. */
export function fmtCompact(value: number): string {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1e6) return `${fmt(value / 1e6, 1)}M`;
  if (abs >= 1e3) return `${fmt(value / 1e3, abs >= 1e4 ? 0 : 1)}k`;
  return fmtInt(value);
}

/** p-values, where the exact magnitude below 0.001 is not worth the digits. */
export function fmtP(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return '—';
  if (p < 0.001) return '< 0.001';
  return p.toPrecision(3);
}
