/**
 * The two colour scales the platform uses to carry information.
 *
 * Everything else in the interface is neutral, so these are the only places a
 * colour means something. The congestion ramp reproduces the band edges in
 * `qroute.traffic.bpr.CONGESTION_BANDS` exactly, so the map legend and the
 * backend's own band histogram always agree.
 *
 * The ramp is a temperature rather than a traffic light. Green, gold and blue
 * are all held out of the product's palette, so it runs from grey through slate
 * and violet into rose and crimson. Saturation and perceived heat both rise
 * together, which means the ordering survives greyscale printing, and a
 * free-flowing road ends up the quietest thing on the map rather than a bright
 * line competing for attention. It also stays readable for the roughly one man
 * in twelve who cannot separate red from green, which the conventional scheme
 * does not.
 *
 * The vehicle palette is categorical rather than ordered, and every hue is dark
 * enough to hold its own over a pale basemap or over satellite imagery.
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
  { name: 'free', max: 0.1, color: '#a6adb9', label: 'Free flow' },
  { name: 'light', max: 0.35, color: '#7d8698', label: 'Light' },
  { name: 'moderate', max: 0.75, color: '#8e6aa8', label: 'Moderate' },
  { name: 'heavy', max: 1.5, color: '#a8497e', label: 'Heavy' },
  { name: 'severe', max: Number.POSITIVE_INFINITY, color: '#b5364a', label: 'Severe' },
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
  '#1c273b',
  '#a8436a',
  '#4f5b6e',
  '#8e4a2f',
  '#6f4a9c',
  '#b5364a',
  '#2f5d63',
  '#7a3f5c',
  '#3f4a5c',
  '#93553a',
];

export function vehicleColor(index: number): string {
  return VEHICLE_COLORS[((index % VEHICLE_COLORS.length) + VEHICLE_COLORS.length) % VEHICLE_COLORS.length];
}

/** Fixed colours per algorithm so a series keeps its hue across every chart. */
const ALGORITHM_COLORS: Record<string, string> = {
  // The two proposed engines take the brand navy and the rose, so they are the
  // two series a reader picks out first on every chart; the classical field is
  // held in slates and browns behind them.
  qpso: '#1c273b',
  qiea: '#a8436a',
  qrk: '#7a3f5c',
  pso: '#4f5b6e',
  ga: '#8e4a2f',
  sa: '#6f4a9c',
  aco: '#2f5d63',
  random: '#a6adb9',
  ortools: '#93553a',
  pyvrp: '#b5364a',
  cpsat: '#3f4a5c',
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
