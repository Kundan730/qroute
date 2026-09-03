/**
 * The two colour scales the platform uses to carry information.
 *
 * Everything else in the interface is neutral, so these are the only places a
 * colour means something. The congestion ramp reproduces the band edges in
 * `qroute.traffic.bpr.CONGESTION_BANDS` exactly, so the map legend and the
 * backend's own band histogram always agree.
 *
 * The ramp is the conventional traffic scale: green, through olive and amber,
 * into orange and red. An earlier version avoided green and gold on palette
 * grounds and ran grey to crimson instead; it satisfied every measurable
 * constraint and still failed the one that matters, because a reader could not
 * tell at a glance which end meant trouble.
 *
 * Two things are held on top of the convention, since the naive green-yellow-red
 * fails both. Every band clears 3:1 against the pale basemap, where a bright
 * yellow sits at about 2.3:1. And luminance falls monotonically from free flow
 * to severe, so severity reads as darker as well as redder - which preserves the
 * order in greyscale and for red-green colour blindness, the case the
 * conventional scale is usually criticised for.
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
  { name: 'free', max: 0.1, color: '#0F9D4F', label: 'Free flow' },
  { name: 'light', max: 0.35, color: '#6E8A12', label: 'Light' },
  { name: 'moderate', max: 0.75, color: '#9C6708', label: 'Moderate' },
  { name: 'heavy', max: 1.5, color: '#A94714', label: 'Heavy' },
  { name: 'severe', max: Number.POSITIVE_INFINITY, color: '#A81E19', label: 'Severe' },
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
  random: '#8e939d',
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

/**
 * Road-class line widths, so arterials read above residential streets, with
 * congested links drawn heavier still.
 *
 * Weight does a second job here beyond hierarchy. Colour alone is a weak signal
 * on a line one or two pixels wide over a basemap, and it is no signal at all to
 * a reader who cannot separate the two ends of the scale. Making a failing
 * corridor thicker as well as redder means the thing that needs attention is
 * legible from across a room, on a projector, and to someone who sees the
 * colours differently.
 */
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
  return congested ? base * 1.6 + 0.6 : base;
}
