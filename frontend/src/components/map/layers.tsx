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

// ------------------------------------------------------------------ colour

/**
 * Read one design token as a literal string.
 *
 * Leaflet writes colours into canvas fill/stroke state and into SVG
 * presentation attributes, neither of which resolves `var(--token)`, so the
 * map is the one place in the interface that needs the computed value rather
 * than the reference. Resolving it here — rather than copying a hex into this
 * file — keeps `global.css` the single source of truth. The palette is fixed
 * (the product is light-only, with no theme switch), so a resolved value is
 * cached for the life of the page; an empty result, which is what a call
 * before the stylesheet has applied returns, is deliberately not cached.
 */
const TOKENS = new Map<string, string>();

function token(name: string): string {
  const cached = TOKENS.get(name);
  if (cached) return cached;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  if (value) TOKENS.set(name, value);
  return value;
}

/**
 * The casing colour used under everything the map draws over the basemap.
 *
 * Three base layers have to be survived at once — a pale vector map, aerial
 * imagery, and a dark vector map — and no single ink is legible on all three.
 * A white casing is, because it is lighter than the darkest layer and the
 * marks it carries are all darker than the lightest one. It is the standard
 * cartographic halo and it is why a route on satellite imagery still reads.
 */
function casing(): string {
  return token('--panel');
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
        // Half a pixel wider and close to opaque, because these lines have to
        // hold up over aerial imagery, where a thin translucent stroke on a
        // photograph of a city simply disappears. There are several thousand
        // of them, so they cannot each afford a casing the way a route can.
        weight: edgeWeight(props.highway, props.congestion >= 0.75) + 0.6,
        opacity: 0.92,
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
    const width = edgeWeight(feature.properties.highway, false) + 0.6;
    // Three strokes, because one cannot be seen on all three base layers. The
    // white outer casing is what makes the highlight visible over satellite
    // imagery and the dark map; the accent ring is what makes it visible over
    // the pale one; and the segment is then redrawn in its own congestion
    // colour on top, so selecting an edge never hides the value that the edge
    // was selected to inspect.
    const halo = L.layerGroup([
      L.polyline(points, {
        color: casing(),
        weight: width + 8,
        opacity: 0.9,
        interactive: false,
      }),
      L.polyline(points, {
        color: token('--accent'),
        weight: width + 4,
        opacity: 0.95,
        interactive: false,
      }),
      L.polyline(points, {
        color: congestionColor(feature.properties.congestion),
        weight: width,
        opacity: 1,
        interactive: false,
      }),
    ]).addTo(map);
    return () => {
      halo.remove();
    };
  }, [edges, selectedEdge, map]);

  return null;
}

// ------------------------------------------------------------------- stops

/**
 * The depot, drawn as a diamond rather than as a larger dot.
 *
 * Shape is what makes it findable. In a field of a hundred customers a reader
 * picks out the one mark that is not a circle immediately, without having to
 * compare two sizes or two hues against each other, and that holds at any
 * zoom and for any reader. Colour then reinforces it: the accent is the one
 * saturated hue in the chrome, and the depot is the one place on the map that
 * earns it.
 *
 * It is a div icon rather than a Leaflet vector so the casing can be written
 * in tokens directly: a white border to lift it off aerial imagery and the
 * dark basemap, and a navy hairline outside that border, because on the pale
 * basemap a white casing on a near-white background would otherwise be no
 * casing at all. There is exactly one of these per instance, so the cost of a
 * DOM node is irrelevant here in a way it would not be for the customers.
 */
const DEPOT_ICON = L.divIcon({
  className: '',
  iconSize: [20, 20],
  iconAnchor: [10, 10],
  popupAnchor: [0, -11],
  html:
    '<span style="display:block;width:12px;height:12px;margin:2px;' +
    'transform:rotate(45deg);background:var(--accent);' +
    'border:2px solid var(--panel);box-shadow:0 0 0 1px var(--navy);"></span>',
});

export function StopsLayer({ instance }: { instance: InstanceDetail | null }) {
  const map = useMap();
  const renderer = useCanvasRenderer();

  useEffect(() => {
    if (!instance || !instance.geographic || instance.coords.length === 0) return undefined;
    const group = L.layerGroup();
    instance.coords.forEach((coord, index) => {
      const isDepot = index === 0;
      const demand = instance.demand[index] ?? 0;
      // Customers are navy dots in a white casing. Navy is dark enough to read
      // on the pale basemap and on satellite imagery; the casing is what keeps
      // them from dissolving into the dark one, and what separates two stops
      // that sit on the same junction.
      const marker = isDepot
        ? L.marker(coord, { icon: DEPOT_ICON, keyboard: false })
        : L.circleMarker(coord, {
            renderer,
            radius: 4.5,
            color: casing(),
            weight: 1.5,
            fillColor: token('--navy'),
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
      // A white casing under each route does two jobs at once: it separates
      // the route from a congested arterial of a similar hue underneath it,
      // and it is what stops the route from being swallowed by aerial imagery
      // or by the dark basemap. Every vehicle hue is dark, so the casing never
      // competes with the line it carries.
      group.addLayer(
        L.polyline(line.points, {
          color: casing(),
          weight: 7,
          opacity: dim ? 0.4 : 0.9,
          interactive: false,
        }),
      );
      const path = L.polyline(line.points, {
        color,
        weight: 3.4,
        opacity: dim ? 0.55 : 1,
        dashArray: line.onRoad ? undefined : '6 5',
      });
      path.bindTooltip(
        `Vehicle ${line.vehicle + 1} · ${line.stops.length} stops · ${fmtDistance(line.totalMetres)}` +
          (line.seconds !== null ? ` · ${fmtDuration(line.seconds)}` : '') +
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
        radius: 6,
        color: casing(),
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

/**
 * Fit the map to a bounding box whenever the box identity changes.
 *
 * The size guard matters more than it looks. Leaflet computes a fitting zoom
 * from its cached container size, and when that size is still zero — which it
 * is on the first effect pass, before the flex layout has settled — `fitBounds`
 * silently returns `maxZoom` and the map opens at street level over a corner of
 * the city. So the fit is skipped until the container genuinely has a size, and
 * a resize observer retries once it does.
 */
export function FitBounds({ bounds }: { bounds: [LatLng, LatLng] | null }) {
  const map = useMap();

  useEffect(() => {
    if (!bounds) return undefined;
    let fitted = false;

    const fit = () => {
      map.invalidateSize({ animate: false, pan: false });
      const size = map.getSize();
      if (size.x < 2 || size.y < 2) return;
      // Not animated: an interrupted zoom animation strands the map at an
      // intermediate zoom that looks like a broken fit.
      map.fitBounds(bounds, { padding: [40, 40], animate: false });
      fitted = true;
    };

    fit();
    if (fitted) return undefined;
    const observer = new ResizeObserver(() => {
      if (!fitted) fit();
    });
    observer.observe(map.getContainer());
    return () => observer.disconnect();
  }, [bounds, map]);

  return null;
}

/**
 * Keep Leaflet's idea of the container size in step with the real one.
 *
 * Without this the map is drawn for the wrong viewport after the window is
 * resized or a projector changes the resolution mid-demonstration: tiles do not
 * fill the pane and clicks land on the wrong road.
 */
export function ResizeWatcher() {
  const map = useMap();
  useEffect(() => {
    const observer = new ResizeObserver(() =>
      map.invalidateSize({ animate: false, pan: false }),
    );
    observer.observe(map.getContainer());
    return () => observer.disconnect();
  }, [map]);
  return null;
}

// ------------------------------------------------------------- exact path

/**
 * The exact shortest path between two stops, drawn over everything else as a
 * navy ribbon with a white dashed core.
 *
 * It is deliberately visually distinct from the vehicle routes: this is the
 * output of A* on the road graph, an exactly optimal answer to a polynomial
 * subproblem, and it should not be mistaken for the metaheuristic's output.
 * The routes are light-cased and dark-cored; this is the inverse, so the two
 * are told apart by their construction and not only by their colour. The
 * inversion is also what carries it across all three base layers: the navy
 * ribbon reads on the pale map and on imagery, and where the ribbon itself
 * goes quiet — on the dark basemap — the white dashes take over.
 */
export function ExactRouteLayer({ route }: { route: ExactRoute | null }) {
  const map = useMap();

  useEffect(() => {
    if (!route || route.points.length < 2) return undefined;
    const group = L.layerGroup();
    group.addLayer(
      L.polyline(route.points, {
        color: token('--navy'),
        weight: 8,
        opacity: 0.85,
        interactive: false,
      }),
    );
    const path = L.polyline(route.points, {
      color: casing(),
      weight: 3,
      opacity: 1,
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
