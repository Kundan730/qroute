/**
 * Base layers for the map, and why these three.
 *
 * All of them are key-free, and that is a hard requirement rather than a
 * convenience: the demonstration is given on venue wifi in front of a panel, and
 * a basemap that needs an account is a basemap that can fail in the one minute
 * it matters. If tiles do not arrive at all the road network still draws,
 * because the roads are the data and the basemap is only context.
 *
 * All three come from Esri's free tile services. CARTO's vector basemaps were
 * used first and had to be dropped: they fetch perfectly well from a terminal
 * but, given a browser Referer, return a tile stamped API KEY REQUIRED straight
 * across the map. That is the precise failure this comment exists to prevent,
 * and it is only visible if someone actually looks at the running application.
 *
 * Each layer is paired with a body class so the stylesheet can hold it back by
 * a different amount. A pale vector basemap needs almost no restraint; satellite
 * imagery is busy and needs a good deal, or the coloured routes drawn over it
 * stop reading.
 */

export type BasemapId = 'light' | 'satellite' | 'dark';

export interface Basemap {
  id: BasemapId;
  label: string;
  /** Short reason this layer exists, shown in the map's own provenance note. */
  purpose: string;
  url: string;
  attribution: string;
  maxZoom: number;
  /** Class applied to the map wrapper, read by `.basemap-*` rules in the CSS. */
  bodyClass: string;
  subdomains?: string;
  /**
   * Deepest zoom the provider actually publishes. Leaflet upscales beyond it
   * rather than requesting tiles that come back blank, which matters for the
   * grey canvas layers: they stop at 16 while the map goes to 19.
   */
  maxNativeZoom?: number;
}

export const BASEMAPS: Basemap[] = [
  {
    id: 'light',
    label: 'Light',
    purpose:
      'A pale grey canvas. Street geometry stays legible while the congestion ' +
      'colours and the vehicle routes carry all the saturation on screen. This is ' +
      'the default because it is the one that reads on a projector and in print.',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attribution:
      'Tiles &copy; <a href="https://www.esri.com/">Esri</a> &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
    maxZoom: 19,
    maxNativeZoom: 16,
    bodyClass: 'basemap-light',
  },
  {
    id: 'satellite',
    label: 'Satellite',
    purpose:
      'Aerial imagery, for showing that the depot and the stops sit on real ' +
      'buildings and real junctions rather than on abstract coordinates. Harder ' +
      'to read routes over, so it is a layer to switch to and back from.',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution:
      'Imagery &copy; <a href="https://www.esri.com/">Esri</a>, Maxar, Earthstar Geographics',
    maxZoom: 19,
    bodyClass: 'basemap-satellite',
  },
  {
    id: 'dark',
    label: 'Dark',
    purpose:
      'A dark grey canvas. Highest contrast for the route colours, and the right ' +
      'choice in a dim room, though it fights the rest of the interface.',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attribution:
      'Tiles &copy; <a href="https://www.esri.com/">Esri</a> &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
    maxZoom: 19,
    maxNativeZoom: 16,
    bodyClass: 'basemap-dark',
  },
];

export const DEFAULT_BASEMAP: BasemapId = 'light';

export function getBasemap(id: BasemapId): Basemap {
  return BASEMAPS.find((b) => b.id === id) ?? BASEMAPS[0];
}
