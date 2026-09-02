/**
 * Turning API payloads into the polylines Leaflet draws, and walking a vehicle
 * along one of them.
 *
 * Two conversions matter here. GeoJSON stores positions as `[longitude,
 * latitude]` while Leaflet takes `[latitude, longitude]`, and getting that
 * backwards puts Bengaluru in the Indian Ocean, so the swap happens in exactly
 * one place. And a route drawn as a sequence of stops is not the route a
 * vehicle drives: when the backend supplies road-following geometry it is used,
 * and the straight-line fallback is marked as such in the interface rather than
 * being passed off as a real path.
 */

import type { RouteCollection } from '../../api/types';

export type LatLng = [number, number];

export interface RouteLine {
  vehicle: number;
  points: LatLng[];
  /** Cumulative length in metres at each point, for the animation. */
  cumulative: number[];
  totalMetres: number;
  /** False when the line is a straight-line fallback rather than a road path. */
  onRoad: boolean;
  /** Backend-measured drive time for this leg, when it supplied one. */
  seconds: number | null;
  /** Customer indices served, in order, for the route table. */
  stops: number[];
}

const EARTH_RADIUS_M = 6371008.8;

/** Equirectangular approximation; accurate to well under a metre at city scale. */
export function metresBetween(a: LatLng, b: LatLng): number {
  const latRad = ((a[0] + b[0]) / 2) * (Math.PI / 180);
  const dLat = (b[0] - a[0]) * (Math.PI / 180);
  const dLon = (b[1] - a[1]) * (Math.PI / 180) * Math.cos(latRad);
  return Math.sqrt(dLat * dLat + dLon * dLon) * EARTH_RADIUS_M;
}

function withCumulative(points: LatLng[]): { cumulative: number[]; total: number } {
  const cumulative = new Array<number>(points.length);
  let total = 0;
  cumulative[0] = 0;
  for (let i = 1; i < points.length; i += 1) {
    total += metresBetween(points[i - 1], points[i]);
    cumulative[i] = total;
  }
  return { cumulative, total };
}

/**
 * Build one line per vehicle.
 *
 * `geojson`, when present, is the road-following geometry the backend traced
 * through the network and is preferred. Otherwise stop coordinates are joined
 * directly, which is honest as a schematic but is not a drivable path.
 */
export function buildRouteLines(
  routes: number[][] | null,
  coords: LatLng[] | null,
  geojson: RouteCollection | null,
): RouteLine[] {
  const stopsByVehicle = (routes ?? []).filter((r) => r.length > 0);

  if (geojson && geojson.features.length > 0) {
    return geojson.features
      .map((feature) => {
        const points: LatLng[] = feature.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
        const { cumulative, total } = withCumulative(points);
        return {
          vehicle: feature.properties.vehicle,
          points,
          cumulative,
          totalMetres: feature.properties.length_m ?? total,
          onRoad: true,
          seconds: feature.properties.travel_time_s ?? null,
          stops: stopsByVehicle[feature.properties.vehicle] ?? [],
        };
      })
      .filter((line) => line.points.length >= 2);
  }

  if (!coords || coords.length === 0) return [];

  return stopsByVehicle
    .map((stops, vehicle) => {
      const points: LatLng[] = [coords[0], ...stops.map((c) => coords[c]).filter(Boolean), coords[0]];
      const { cumulative, total } = withCumulative(points);
      return {
        vehicle,
        points,
        cumulative,
        totalMetres: total,
        onRoad: false,
        seconds: null,
        stops,
      };
    })
    .filter((line) => line.points.length >= 2);
}

/**
 * Position at a given fraction of the way along a line, by arc length.
 *
 * Interpolating by vertex index instead would make a vehicle crawl through
 * dense junction geometry and then leap along a long straight, which reads as a
 * bug even though the route is right.
 */
export function pointAt(line: RouteLine, fraction: number): LatLng {
  const points = line.points;
  if (points.length === 0) return [0, 0];
  if (points.length === 1) return points[0];
  const target = Math.min(Math.max(fraction, 0), 1) * line.cumulative[points.length - 1];

  // Binary search for the segment containing the target arc length.
  let lo = 0;
  let hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (line.cumulative[mid] <= target) lo = mid;
    else hi = mid;
  }
  const span = line.cumulative[hi] - line.cumulative[lo];
  const t = span > 0 ? (target - line.cumulative[lo]) / span : 0;
  return [
    points[lo][0] + (points[hi][0] - points[lo][0]) * t,
    points[lo][1] + (points[hi][1] - points[lo][1]) * t,
  ];
}

/** Bounding box of every supplied point, as Leaflet's `[[s, w], [n, e]]`. */
export function boundsOf(points: LatLng[]): [LatLng, LatLng] | null {
  if (points.length === 0) return null;
  let minLat = points[0][0];
  let maxLat = points[0][0];
  let minLon = points[0][1];
  let maxLon = points[0][1];
  for (const [lat, lon] of points) {
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (lon < minLon) minLon = lon;
    if (lon > maxLon) maxLon = lon;
  }
  return [
    [minLat, minLon],
    [maxLat, maxLon],
  ];
}
