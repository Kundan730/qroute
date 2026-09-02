/**
 * Imperative Leaflet layers driven by React state.
 *
 * These are written against the Leaflet API directly rather than as
 * `react-leaflet` elements because of scale: an arterial view of the Bengaluru
 * extract is several thousand polylines, and mounting one React component per
 * road segment costs far more than creating the same number of Leaflet paths on
 * a single shared canvas. Each component below therefore owns one
 * `L.LayerGroup`, rebuilds it when its inputs change, and removes it on
 * unmount.
 */

import L from 'leaflet';
import { useEffect, useMemo, useRef } from 'react';
import { useMap } from 'react-leaflet';
import type { EdgeCollection, EdgeProperties, ExactRoute, InstanceDetail } from '../../api/types';
import { congestionColor, edgeWeight, vehicleColor } from '../../lib/colors';
import { fmt, fmtDistance, fmtDuration } from '../../lib/format';
import type { LatLng, RouteLine } from './geometry';
import { pointAt } from './geometry';

/** One canvas renderer shared by every path layer, so the map has one surface. */
function useCanvasRenderer(): L.Canvas {
  return useMemo(() => L.canvas({ padding: 0.3 }), []);
}

// ------------------------------------------------------------------- roads

export function EdgeLayer({
  edges,
  visible,
  selectedEdge,
  onSelect,
}: {
  edges: EdgeCollection | null;
  visible: boolean;
  selectedEdge: number | null;
  onSelect: (edge: EdgeProperties) => void;
}) {
  const map = useMap();
  const renderer = useCanvasRenderer();
  // Held in a ref so that changing the click handler does not force several
  // thousand polylines to be rebuilt; written in an effect, because mutating a
  // ref during render is not safe under concurrent rendering.
  const selectRef = useRef(onSelect);
  useEffect(() => {
    selectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!edges || !visible) return undefined;
    const group = L.layerGroup([], { pane: 'overlayPane' });
    for (const feature of edges.features) {
      const points: LatLng[] = feature.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
      const props = feature.properties;
      const line = L.polyline(points, {
        renderer,
        color: congestionColor(props.congestion),
        weight: edgeWeight(props.highway, props.congestion >= 0.75),
        opacity: 0.82,
        interactive: true,
        bubblingMouseEvents: false,
      });
      line.on('click', () => selectRef.current(props));
      line.bindTooltip(
        `${props.highway} · ${fmtDistance(props.length_m)} · ${fmt(props.congestion * 100, 0)} % delay`,
        { sticky: true, direction: 'top', opacity: 0.95 },
      );
      group.addLayer(line);
    }
    group.addTo(map);
    return () => {
      group.remove();
    };
  }, [edges, visible, map, renderer]);

  // The selection highlight is a separate, thin layer so picking a different
  // edge does not rebuild the thousands of polylines underneath it.
  useEffect(() => {
    if (!edges || selectedEdge === null) return undefined;
    const feature = edges.features.find((f) => f.properties.edge === selectedEdge);
    if (!feature) return undefined;
    const points: LatLng[] = feature.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
    const halo = L.polyline(points, {
      color: '#f2f6fb',
      weight: edgeWeight(feature.properties.highway, false) + 4,
      opacity: 0.75,
      interactive: false,
    }).addTo(map);
    return () => {
      halo.remove();
    };
  }, [edges, selectedEdge, map]);

  return null;
}

// ------------------------------------------------------------------- stops

export function StopsLayer({ instance }: { instance: InstanceDetail | null }) {
  const map = useMap();
  const renderer = useCanvasRenderer();

  useEffect(() => {
    if (!instance || !instance.geographic || instance.coords.length === 0) return undefined;
    const group = L.layerGroup();
    instance.coords.forEach((coord, index) => {
      const isDepot = index === 0;
      const demand = instance.demand[index] ?? 0;
      const marker = L.circleMarker(coord, {
        renderer: isDepot ? undefined : renderer,
        radius: isDepot ? 7 : 4,
        color: isDepot ? '#f2f6fb' : '#0e1116',
        weight: isDepot ? 2 : 1,
        fillColor: isDepot ? '#e8894a' : '#93a3b6',
        fillOpacity: 1,
      });
      const window_ = instance.time_windows?.[index];
      marker.bindPopup(
        isDepot
          ? `<strong>Depot</strong><br/>node ${instance.node_ids?.[0] ?? '—'}`
          : [
              `<strong>Customer ${index}</strong>`,
              `demand ${fmt(demand, 0)} units`,
              window_ ? `window ${fmtDuration(window_[0])} – ${fmtDuration(window_[1])}` : null,
              instance.node_ids ? `OSM node ${instance.node_ids[index]}` : null,
            ]
              .filter(Boolean)
              .join('<br/>'),
      );
      group.addLayer(marker);
    });
    group.addTo(map);
    return () => {
      group.remove();
    };
  }, [instance, map, renderer]);

  return null;
}

// ------------------------------------------------------------------ routes

export function RouteLayer({ lines, dim }: { lines: RouteLine[]; dim: boolean }) {
  const map = useMap();

  useEffect(() => {
    if (lines.length === 0) return undefined;
    const group = L.layerGroup();
    for (const line of lines) {
      const color = vehicleColor(line.vehicle);
      // A dark casing under each route keeps it readable where it crosses a
      // congested arterial of a similar hue.
      group.addLayer(
        L.polyline(line.points, {
          color: '#0b0e12',
          weight: 6.5,
          opacity: dim ? 0.35 : 0.65,
          interactive: false,
        }),
      );
      const path = L.polyline(line.points, {
        color,
        weight: 3,
        opacity: dim ? 0.5 : 0.95,
        dashArray: line.onRoad ? undefined : '6 5',
      });
      path.bindTooltip(
        `Vehicle ${line.vehicle + 1} · ${line.stops.length} stops · ${fmtDistance(line.totalMetres)}` +
          (line.onRoad ? '' : ' (straight-line approximation)'),
        { sticky: true, direction: 'top' },
      );
      group.addLayer(path);
    }
    group.addTo(map);
    return () => {
      group.remove();
    };
  }, [lines, dim, map]);

  return null;
}

// ---------------------------------------------------------------- vehicles

/**
 * Walk one marker per vehicle along its route.
 *
 * Speed is expressed in metres of route per second of wall clock and scaled so
 * that the longest route takes `LAP_SECONDS`, which keeps every vehicle on
 * screen for the same time regardless of how unbalanced the solution is. This
 * is a visualisation of the route order, not a simulation of the schedule, and
 * the interface says so.
 */
const LAP_SECONDS = 16;

export function VehicleLayer({ lines, running }: { lines: RouteLine[]; running: boolean }) {
  const map = useMap();

  useEffect(() => {
    if (!running || lines.length === 0) return undefined;
    const markers = lines.map((line) =>
      L.circleMarker(pointAt(line, 0), {
        radius: 5.5,
        color: '#0b0e12',
        weight: 2,
        fillColor: vehicleColor(line.vehicle),
        fillOpacity: 1,
        interactive: false,
      }).addTo(map),
    );

    const longest = Math.max(...lines.map((l) => l.totalMetres), 1);
    const start = performance.now();
    let frame = 0;

    const step = (now: number) => {
      const elapsed = (now - start) / 1000;
      lines.forEach((line, i) => {
        const lap = LAP_SECONDS * (line.totalMetres / longest || 1);
        const fraction = lap > 0 ? (elapsed % lap) / lap : 0;
        markers[i].setLatLng(pointAt(line, fraction));
      });
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);

    return () => {
      cancelAnimationFrame(frame);
      for (const marker of markers) marker.remove();
    };
  }, [lines, running, map]);

  return null;
}

// ------------------------------------------------------------- view control

/** Fit the map to a bounding box whenever the box identity changes. */
export function FitBounds({ bounds }: { bounds: [LatLng, LatLng] | null }) {
  const map = useMap();
  useEffect(() => {
    if (!bounds) return;
    map.fitBounds(bounds, { padding: [40, 40], animate: true });
  }, [bounds, map]);
  return null;
}

// ------------------------------------------------------------- exact path

/**
 * The exact shortest path between two stops, drawn as a bright thin line over
 * everything else.
 *
 * It is deliberately visually distinct from the vehicle routes: this is the
 * output of A* on the road graph, an exactly optimal answer to a polynomial
 * subproblem, and it should not be mistaken for the metaheuristic's output.
 */
export function ExactRouteLayer({ route }: { route: ExactRoute | null }) {
  const map = useMap();

  useEffect(() => {
    if (!route || route.points.length < 2) return undefined;
    const group = L.layerGroup();
    group.addLayer(
      L.polyline(route.points, { color: '#0b0e12', weight: 7, opacity: 0.7, interactive: false }),
    );
    const path = L.polyline(route.points, {
      color: '#f2f6fb',
      weight: 2.5,
      opacity: 0.95,
      dashArray: '9 5',
    });
    path.bindTooltip(
      `Exact shortest path · ${fmtDistance(route.distance_m)} · ${fmtDuration(route.travel_time_s)} ` +
        `(${fmt(route.delay_ratio, 2)} x free flow) · ${route.nodes_expanded} nodes expanded`,
      { sticky: true, direction: 'top' },
    );
    group.addLayer(path);
    group.addTo(map);
    return () => {
      group.remove();
    };
  }, [route, map]);

  return null;
}
