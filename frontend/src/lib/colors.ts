/**
 * The two colour scales the platform uses to carry information.
 *
 * Everything else in the interface is neutral, so these are the only places a
 * colour means something. The congestion ramp reproduces the band edges in
 * `qroute.traffic.bpr.CONGESTION_BANDS` exactly, so the map legend and the
 * backend's own band histogram always agree; the vehicle palette is a
 * categorical set chosen to stay distinguishable on a dark map and under the
 * washed-out colour of a projector.
 */

import type { CongestionBand } from '../api/types';

export interface Band {
  name: CongestionBand;
  /** Upper bound of the fractional delay `(t - t0) / t0`, exclusive. */
  max: number;
  color: string;
  label: string;
}

/** Band edges copied from `qroute.traffic.bpr.CONGESTION_BANDS`. */
export const CONGESTION_BANDS: Band[] = [
  { name: 'free', max: 0.1, color: '#3f8f6b', label: 'Free flow' },
  { name: 'light', max: 0.35, color: '#86a94e', label: 'Light' },
  { name: 'moderate', max: 0.75, color: '#c9a63f', label: 'Moderate' },
  { name: 'heavy', max: 1.5, color: '#cd7c42', label: 'Heavy' },
  { name: 'severe', max: Number.POSITIVE_INFINITY, color: '#c0524f', label: 'Severe' },
];

/** Descriptions of each band in the units a reader can check: delay percentage. */
export const BAND_RANGES: Record<CongestionBand, string> = {
  free: '0 – 10 % delay',
  light: '10 – 35 % delay',
  moderate: '35 – 75 % delay',
  heavy: '75 – 150 % delay',
  severe: '> 150 % delay',
};

export function congestionBand(level: number): Band {
  for (const band of CONGESTION_BANDS) {
    if (level < band.max) return band;
  }
  return CONGESTION_BANDS[CONGESTION_BANDS.length - 1];
}

export function congestionColor(level: number): string {
  return congestionBand(level).color;
}

/**
 * Ten hues for vehicle routes. Ordered so that any prefix of the list stays
 * mutually distinguishable, which matters because most instances use fewer
 * than ten vehicles and the first colours are the ones usually seen.
 */
export const VEHICLE_COLORS: string[] = [
  '#5b93e6',
  '#e8894a',
  '#57b487',
  '#c96ec4',
  '#d8b93f',
  '#5fbccf',
  '#dd6f74',
  '#8a8fe0',
  '#8fae4c',
  '#c98a5f',
];

export function vehicleColor(index: number): string {
  return VEHICLE_COLORS[((index % VEHICLE_COLORS.length) + VEHICLE_COLORS.length) % VEHICLE_COLORS.length];
}

/** Fixed colours per algorithm so a series keeps its hue across every chart. */
const ALGORITHM_COLORS: Record<string, string> = {
  qpso: '#5b93e6',
  pso: '#57b487',
  ga: '#e8894a',
  sa: '#c96ec4',
  aco: '#d8b93f',
  random: '#7c8797',
  ortools: '#5fbccf',
  pyvrp: '#8a8fe0',
  cpsat: '#dd6f74',
};

export function algorithmColor(name: string): string {
  const key = name.trim().toLowerCase();
  if (ALGORITHM_COLORS[key]) return ALGORITHM_COLORS[key];
  // Deterministic fallback for a name the palette does not know, so repeated
  // renders of the same benchmark never shuffle their colours.
  let hash = 0;
  for (let i = 0; i < key.length; i += 1) hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  return VEHICLE_COLORS[hash % VEHICLE_COLORS.length];
}

/** Road-class line widths, so arterials read above residential streets. */
export function edgeWeight(highway: string, congested: boolean): number {
  const base =
    highway.includes('motorway') || highway.includes('trunk')
      ? 3.0
      : highway.includes('primary')
        ? 2.4
        : highway.includes('secondary')
          ? 1.9
          : highway.includes('tertiary')
            ? 1.5
            : 1.0;
  return congested ? base + 0.5 : base;
}
