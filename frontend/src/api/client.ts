/**
 * The one module that talks to the network.
 *
 * Every HTTP call the application makes lives here, is typed against
 * `./types.ts`, and normalises its response through `./coerce.ts` so that no
 * component ever handles a raw `unknown`. Keeping the boundary in a single file
 * means the "backend unavailable" behaviour is implemented once: any call that
 * fails throws an `ApiError`, callers record it in the store, and the shell
 * shows a single honest banner instead of each panel inventing its own empty
 * state.
 *
 * The base URL defaults to `/api`, which the Vite dev server proxies to the
 * FastAPI process on port 8000, and which in a production build is served by
 * the same origin as the static bundle.
 */

import {
  arr,
  bool,
  boolOrNull,
  intMatrix,
  listOf,
  num,
  numArray,
  numOrNull,
  pairArray,
  pick,
  rec,
  str,
  strArray,
} from './coerce';
import type {
  AlgorithmInfo,
  BenchmarkCell,
  BenchmarkDetail,
  BenchmarkRow,
  BenchmarkSummary,
  CongestionBand,
  EdgeCollection,
  EdgeFeature,
  EventRequest,
  ExactRoute,
  FriedmanResult,
  Health,
  InstanceDetail,
  InstanceFamily,
  InstanceRequest,
  InstanceSummary,
  NetworkSummary,
  PairwiseResult,
  ParamSpec,
  RouteCollection,
  RunHandle,
  RunRequest,
  RunState,
  RunStatus,
  RunTick,
  SolutionStats,
  TrafficEventDto,
  TrafficState,
} from './types';

const envBase: unknown = import.meta.env.VITE_API_BASE;
export const API_BASE: string = typeof envBase === 'string' && envBase ? envBase : '/api';

/** A failed HTTP call, carrying the status code when there was a response. */
export class ApiError extends Error {
  readonly status: number;
  readonly path: string;

  constructor(message: string, path: string, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.path = path;
    this.status = status;
  }
}

async function request(path: string, init?: RequestInit): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : 'network error';
    throw new ApiError(`cannot reach the backend (${detail})`, path);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = rec(await response.json());
      detail = str(pick(body, 'detail', 'message', 'error'), detail);
    } catch {
      /* a non-JSON error body is not worth reporting verbatim */
    }
    // 502/503/504 do not come from the API at all: they are the dev proxy (or a
    // reverse proxy) saying it could not reach it. The bare status text for
    // that case is "Bad Gateway", which sends the reader looking for a bug in
    // the request rather than at the process that is not running.
    if (response.status >= 502 && response.status <= 504) {
      throw new ApiError(
        `cannot reach the backend (HTTP ${response.status} for ${path}); ` +
          'start it with `python -m qroute.api.app`',
        path,
        response.status,
      );
    }
    throw new ApiError(detail || `HTTP ${response.status}`, path, response.status);
  }
  if (response.status === 204) return {};
  try {
    return await response.json();
  } catch {
    throw new ApiError('response was not valid JSON', path, response.status);
  }
}

async function postJson(path: string, body: unknown): Promise<unknown> {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// ------------------------------------------------------------- normalisation

function toParamSpec(value: unknown): ParamSpec | null {
  const r = rec(value);
  const name = str(pick(r, 'name', 'key'));
  if (!name) return null;
  const rawKind = str(pick(r, 'kind', 'type'), 'float');
  const kind: ParamSpec['kind'] =
    rawKind === 'int' || rawKind === 'bool' || rawKind === 'choice' ? rawKind : 'float';
  const rawDefault = pick(r, 'default', 'value');
  const fallback: number | boolean | string = kind === 'bool' ? false : kind === 'choice' ? '' : 0;
  const def =
    typeof rawDefault === 'number' || typeof rawDefault === 'boolean' || typeof rawDefault === 'string'
      ? rawDefault
      : fallback;
  return {
    name,
    kind,
    default: def,
    min: numOrNull(r.min) ?? undefined,
    max: numOrNull(r.max) ?? undefined,
    step: numOrNull(r.step) ?? undefined,
    choices: Array.isArray(r.choices) ? strArray(r.choices) : undefined,
    description: str(r.description) || undefined,
  };
}

function toAlgorithm(value: unknown): AlgorithmInfo {
  const r = rec(value);
  const specs = arr(pick(r, 'params', 'parameters'))
    .map(toParamSpec)
    .filter((p): p is ParamSpec => p !== null);
  return {
    name: str(pick(r, 'name', 'id'), 'unknown'),
    description: str(pick(r, 'description', 'summary')),
    params: specs.length > 0 ? specs : undefined,
  };
}

function toFamily(value: unknown, hasWindows: boolean): InstanceFamily {
  const raw = str(value).toLowerCase();
  if (raw === 'cvrp' || raw === 'vrptw' || raw === 'network') return raw;
  return hasWindows ? 'vrptw' : 'cvrp';
}

function toInstanceSummary(value: unknown): InstanceSummary {
  const r = rec(value);
  const hasWindows = bool(pick(r, 'has_time_windows', 'time_windows'), false);
  return {
    name: str(pick(r, 'name', 'instance'), 'unknown'),
    family: toFamily(pick(r, 'family', 'kind', 'set'), hasWindows),
    n_customers: num(pick(r, 'n_customers', 'customers', 'size')),
    capacity: num(r.capacity),
    n_vehicles: numOrNull(pick(r, 'n_vehicles', 'vehicles')),
    bks: numOrNull(pick(r, 'bks', 'best_known', 'best_known_cost')),
    has_time_windows: hasWindows,
  };
}

function toInstanceDetail(value: unknown): InstanceDetail {
  const r = rec(value);
  const coords = pairArray(pick(r, 'coords', 'coordinates'));
  const windows = pairArray(r.time_windows);
  const geoRaw = pick(r, 'geographic', 'is_geographic');
  // A road instance carries OSM node ids; failing that, latitude/longitude are
  // recognisable by their range, which is how a plain CVRPLIB grid is excluded.
  const looksGeographic =
    coords.length > 0 &&
    coords.every(([a, b]) => a >= -90 && a <= 90 && b >= -180 && b <= 180) &&
    coords.some(([a, b]) => !Number.isInteger(a) || !Number.isInteger(b));
  return {
    ...toInstanceSummary(r),
    coords,
    demand: numArray(r.demand),
    time_windows: windows.length > 0 ? windows : undefined,
    service_time: Array.isArray(r.service_time) ? numArray(r.service_time) : undefined,
    node_ids: Array.isArray(r.node_ids) ? numArray(r.node_ids) : undefined,
    geographic:
      typeof geoRaw === 'boolean' ? geoRaw : Array.isArray(r.node_ids) ? true : looksGeographic,
    meta: rec(r.meta),
  };
}

function toNetwork(value: unknown): NetworkSummary {
  const r = rec(value);
  const centre = pairArray([pick(r, 'center', 'centre')])[0] ?? [0, 0];
  const box = numArray(r.bbox);
  return {
    id: str(pick(r, 'id', 'name', 'key'), 'unknown'),
    name: str(pick(r, 'name', 'title', 'id'), 'unknown'),
    n_nodes: num(pick(r, 'n_nodes', 'nodes')),
    n_edges: num(pick(r, 'n_edges', 'edges')),
    center: centre,
    bbox:
      box.length === 4 ? [box[0], box[1], box[2], box[3]] : [centre[0], centre[1], centre[0], centre[1]],
  };
}

function toEdgeCollection(value: unknown): EdgeCollection {
  const r = rec(value);
  const features: EdgeFeature[] = [];
  for (const raw of arr(r.features)) {
    const f = rec(raw);
    const geom = rec(f.geometry);
    const line = pairArray(geom.coordinates);
    if (line.length < 2) continue;
    const p = rec(f.properties);
    features.push({
      type: 'Feature',
      geometry: { type: 'LineString', coordinates: line },
      properties: {
        edge: Math.round(num(pick(p, 'edge', 'index', 'id'), -1)),
        u: Math.round(num(p.u)),
        v: Math.round(num(p.v)),
        highway: str(pick(p, 'highway', 'road_class'), 'unclassified'),
        length_m: num(pick(p, 'length_m', 'length')),
        free_flow_s: num(pick(p, 'free_flow_s', 'free_flow_time')),
        travel_time_s: num(pick(p, 'travel_time_s', 'travel_time')),
        congestion: num(pick(p, 'congestion', 'congestion_level')),
        speed_kph: num(pick(p, 'speed_kph', 'speed')),
      },
    });
  }
  return { type: 'FeatureCollection', features, properties: rec(r.properties) };
}

function toRouteCollection(value: unknown): RouteCollection | null {
  const r = rec(value);
  if (!Array.isArray(r.features)) return null;
  const features: RouteCollection['features'] = [];
  for (const raw of arr(r.features)) {
    const f = rec(raw);
    const line = pairArray(rec(f.geometry).coordinates);
    if (line.length < 2) continue;
    const p = rec(f.properties);
    features.push({
      type: 'Feature',
      geometry: { type: 'LineString', coordinates: line },
      properties: {
        vehicle: Math.round(num(p.vehicle)),
        // qroute.graph.network.route_geojson names these `distance_m` and
        // `duration_s`; the other spellings are accepted so a rename on the
        // backend degrades to the computed polyline length instead of silently
        // showing nothing.
        length_m: numOrNull(pick(p, 'distance_m', 'length_m', 'length')) ?? undefined,
        travel_time_s:
          numOrNull(pick(p, 'duration_s', 'travel_time_s', 'travel_time')) ?? undefined,
      },
    });
  }
  return features.length > 0 ? { type: 'FeatureCollection', features } : null;
}

const BAND_KEYS: CongestionBand[] = ['free', 'light', 'moderate', 'heavy', 'severe'];

function toEvent(value: unknown): TrafficEventDto {
  const r = rec(value);
  const kindRaw = str(r.kind, 'lane_blockage');
  const kind = kindRaw === 'closure' || kindRaw === 'slowdown' ? kindRaw : 'lane_blockage';
  const edges = numArray(r.edges).map((e) => Math.round(e));
  return {
    event_id: Math.round(num(pick(r, 'event_id', 'id'))),
    kind,
    edges,
    n_edges: Math.round(num(r.n_edges, edges.length)),
    start_minute: num(r.start_minute),
    duration_minutes: num(r.duration_minutes, 60),
    end_minute: num(r.end_minute, num(r.start_minute) + num(r.duration_minutes, 60)),
    severity: num(r.severity, 1),
    lanes: Math.round(num(r.lanes, 2)),
    blockage: typeof r.blockage === 'string' ? r.blockage : null,
    capacity_multiplier: num(r.capacity_multiplier, 1),
    time_multiplier: num(r.time_multiplier, 1),
    hcm_tabulated: bool(r.hcm_tabulated),
    description: str(r.description),
  };
}

function toTrafficState(value: unknown): TrafficState {
  const r = rec(value);
  const c = rec(r.congestion);
  const bandsRaw = rec(c.bands);
  const bands = {} as Record<CongestionBand, number>;
  for (const key of BAND_KEYS) bands[key] = Math.round(num(bandsRaw[key]));
  const tt = rec(r.travel_time_seconds);
  const worst = arr(r.worst_edges).map((w) => {
    const e = rec(w);
    return {
      index: Math.round(num(pick(e, 'index', 'edge'))),
      road_class: str(pick(e, 'road_class', 'highway'), 'unclassified'),
      congestion_level: num(pick(e, 'congestion_level', 'congestion')),
      travel_time_s: num(e.travel_time_s),
      free_flow_s: num(e.free_flow_s),
    };
  });
  return {
    time_minutes: num(pick(r, 'time_minutes', 'minute')),
    hour_of_day: num(r.hour_of_day),
    day_of_week: Math.round(num(r.day_of_week)),
    weekend: bool(r.weekend),
    vdf: str(r.vdf, 'bpr'),
    seed: numOrNull(r.seed),
    profile: str(r.profile, 'default'),
    n_edges: Math.round(num(r.n_edges)),
    n_closed: Math.round(num(r.n_closed)),
    reference_saturation: num(r.reference_saturation),
    congestion: {
      mean_level_length_weighted: num(c.mean_level_length_weighted),
      vkt_weighted_ratio: num(c.vkt_weighted_ratio, 1),
      mean_level: num(c.mean_level),
      median_level: num(c.median_level),
      p95_level: num(c.p95_level),
      max_level: num(c.max_level),
      bands,
    },
    travel_time_seconds: {
      total_free_flow: num(tt.total_free_flow),
      total_current: num(tt.total_current),
      network_ratio: num(tt.network_ratio, 1),
    },
    events: arr(r.events).map(toEvent),
    n_active_events: Math.round(num(r.n_active_events)),
    worst_edges: worst.length > 0 ? worst : undefined,
  };
}

export function toRunTick(value: unknown): RunTick {
  const r = rec(value);
  const routes = pick(r, 'routes', 'best_routes');
  return {
    iteration: Math.round(num(pick(r, 'iteration', 'i'))),
    best_cost: num(pick(r, 'best_cost', 'c'), Number.NaN),
    mean_cost: num(pick(r, 'mean_cost', 'm'), Number.NaN),
    diversity: num(pick(r, 'diversity', 'd'), Number.NaN),
    elapsed: num(pick(r, 'elapsed', 't')),
    evaluations: Math.round(num(pick(r, 'evaluations', 'e'))),
    feasible: boolOrNull(r.feasible) ?? undefined,
    routes: Array.isArray(routes) ? intMatrix(routes) : undefined,
  };
}

function toStats(value: unknown): SolutionStats | null {
  if (!value) return null;
  const r = rec(value);
  return {
    distance: num(r.distance),
    duration: num(r.duration),
    capacity_violation: num(r.capacity_violation),
    time_window_violation: num(r.time_window_violation),
    duration_violation: num(r.duration_violation),
    fleet_violation: num(r.fleet_violation),
    edge_load_violation: num(r.edge_load_violation),
    total_violation: num(r.total_violation),
    feasible: num(r.feasible),
  };
}

const RUN_STATES: RunState[] = ['queued', 'running', 'done', 'cancelled', 'failed'];

/** Exposed so the run stream can parse the terminal event without a refetch. */
export function parseRunStatus(value: unknown): RunStatus {
  const r = rec(value);
  const rawState = str(pick(r, 'state', 'status'), 'queued');
  const state = (RUN_STATES as string[]).includes(rawState) ? (rawState as RunState) : 'queued';
  const routes = pick(r, 'routes', 'best_routes');
  const coords = pick(r, 'coords', 'coordinates');
  return {
    run_id: str(pick(r, 'run_id', 'id')),
    state,
    algorithm: str(r.algorithm),
    instance: str(r.instance),
    seed: num(r.seed),
    bks: numOrNull(pick(r, 'bks', 'best_known')),
    best_cost: numOrNull(pick(r, 'best_cost', 'cost')),
    n_routes: numOrNull(r.n_routes),
    feasible: boolOrNull(r.feasible),
    routes: Array.isArray(routes) ? intMatrix(routes) : null,
    stats: toStats(r.stats),
    iterations: Math.round(num(r.iterations)),
    evaluations: Math.round(num(r.evaluations)),
    seconds: num(r.seconds),
    params: rec(r.params),
    geojson: toRouteCollection(pick(r, 'geojson', 'routes_geojson')),
    coords: Array.isArray(coords) ? pairArray(coords) : null,
    error: typeof r.error === 'string' ? r.error : null,
    history: arr(r.history).map(toRunTick),
    baseline_cost: numOrNull(r.baseline_cost),
    parent_run_id: typeof r.parent_run_id === 'string' ? r.parent_run_id : null,
    warm_started: r.warm_started === true,
  };
}

function toStatSummary(value: unknown) {
  const r = rec(value);
  return {
    n: Math.round(num(r.n)),
    best: num(r.best, Number.NaN),
    mean: num(r.mean, Number.NaN),
    median: num(r.median, Number.NaN),
    std: num(r.std),
    iqr: num(r.iqr),
    worst: num(r.worst, Number.NaN),
  };
}

function toCell(value: unknown): BenchmarkCell {
  const r = rec(value);
  return {
    instance: str(r.instance),
    algorithm: str(r.algorithm),
    cost: toStatSummary(r.cost),
    gap: toStatSummary(r.gap),
    feasible_runs: Math.round(num(r.feasible_runs)),
    runs: Math.round(num(r.runs)),
    mean_seconds: num(r.mean_seconds),
    mean_iterations: num(r.mean_iterations),
    hit_bks: Math.round(num(r.hit_bks)),
    median_time_to_1pct: numOrNull(r.median_time_to_1pct),
  };
}

function toRow(value: unknown): BenchmarkRow {
  const r = rec(value);
  return {
    instance: str(r.instance),
    algorithm: str(r.algorithm),
    seed: num(r.seed),
    cost: num(r.cost, Number.NaN),
    gap: num(r.gap, Number.NaN),
    bks: numOrNull(r.bks),
    n_routes: Math.round(num(r.n_routes)),
    feasible: bool(r.feasible),
    iterations: Math.round(num(r.iterations)),
    evaluations: Math.round(num(r.evaluations)),
    seconds: num(r.seconds),
    status: str(r.status, 'ok'),
  };
}

function toPairwise(value: unknown): PairwiseResult {
  const r = rec(value);
  return {
    a: str(r.a),
    b: str(r.b),
    n: Math.round(num(r.n)),
    median_a: num(r.median_a, Number.NaN),
    median_b: num(r.median_b, Number.NaN),
    statistic: num(r.statistic, Number.NaN),
    p_value: num(r.p_value, Number.NaN),
    p_adjusted: numOrNull(r.p_adjusted),
    effect_size: num(r.effect_size),
    winner: typeof r.winner === 'string' ? r.winner : null,
  };
}

function toOmnibus(value: unknown): FriedmanResult | null {
  if (!value) return null;
  const r = rec(value);
  const ranksRaw = rec(r.mean_ranks);
  const mean_ranks: Record<string, number> = {};
  for (const [key, v] of Object.entries(ranksRaw)) mean_ranks[key] = num(v, Number.NaN);
  return {
    algorithms: strArray(r.algorithms),
    mean_ranks,
    statistic: num(r.statistic, Number.NaN),
    p_value: num(r.p_value, Number.NaN),
    n_instances: Math.round(num(r.n_instances)),
    post_hoc: arr(r.post_hoc).map(toPairwise),
    control: typeof r.control === 'string' ? r.control : null,
  };
}

// ---------------------------------------------------------------- public API

export async function getHealth(): Promise<Health> {
  const r = rec(await request('/health'));
  return {
    status: str(pick(r, 'status', 'state'), 'ok'),
    version: str(r.version) || undefined,
    networks: numOrNull(r.networks) ?? undefined,
    instances: numOrNull(r.instances) ?? undefined,
    algorithms: numOrNull(r.algorithms) ?? undefined,
  };
}

export async function getAlgorithms(): Promise<AlgorithmInfo[]> {
  return listOf(await request('/algorithms'), 'algorithms').map(toAlgorithm);
}

export async function getInstances(): Promise<InstanceSummary[]> {
  const payload = await request('/instances');
  // The loader groups instances by family; accept both the grouped shape and a
  // flat list so the UI does not depend on which the backend chose.
  if (isGrouped(payload)) {
    const out: InstanceSummary[] = [];
    for (const [family, items] of Object.entries(rec(payload))) {
      for (const item of arr(items)) {
        const parsed = toInstanceSummary(typeof item === 'string' ? { name: item } : item);
        out.push({ ...parsed, family: toFamily(family, parsed.has_time_windows) });
      }
    }
    return out;
  }
  return listOf(payload, 'instances').map(toInstanceSummary);
}

function isGrouped(payload: unknown): boolean {
  if (Array.isArray(payload)) return false;
  const r = rec(payload);
  const keys = Object.keys(r);
  return keys.length > 0 && keys.every((k) => Array.isArray(r[k])) && !('items' in r);
}

export async function getInstance(name: string): Promise<InstanceDetail> {
  return toInstanceDetail(await request(`/instances/${encodeURIComponent(name)}`));
}

export async function getNetworks(): Promise<NetworkSummary[]> {
  return listOf(await request('/networks'), 'networks').map(toNetwork);
}

export async function getNetworkEdges(
  id: string,
  options: { minImportance?: number; maxEdges?: number } = {},
): Promise<EdgeCollection> {
  const query = new URLSearchParams();
  if (options.minImportance !== undefined) query.set('min_importance', String(options.minImportance));
  if (options.maxEdges !== undefined) query.set('max_edges', String(options.maxEdges));
  const suffix = query.toString() ? `?${query}` : '';
  return toEdgeCollection(await request(`/networks/${encodeURIComponent(id)}/edges${suffix}`));
}

export async function createNetworkInstance(
  id: string,
  body: InstanceRequest,
): Promise<InstanceDetail> {
  return toInstanceDetail(await postJson(`/networks/${encodeURIComponent(id)}/instance`, body));
}

export async function getTrafficState(id: string, topK = 8): Promise<TrafficState> {
  return toTrafficState(await request(`/traffic/${encodeURIComponent(id)}/state?top_k=${topK}`));
}

export async function setTrafficTime(id: string, minute: number): Promise<TrafficState> {
  return toTrafficState(
    await postJson(`/traffic/${encodeURIComponent(id)}/time`, { minute: Math.round(minute) }),
  );
}

export async function addTrafficEvent(id: string, body: EventRequest): Promise<TrafficState> {
  const payload = await postJson(`/traffic/${encodeURIComponent(id)}/events`, body);
  // The endpoint may answer with the created event or with the whole state;
  // asking for the state afterwards keeps the caller's contract simple.
  const r = rec(payload);
  if ('congestion' in r) return toTrafficState(payload);
  return getTrafficState(id);
}

export async function getExactRoute(params: {
  network: string;
  fromNode: number;
  toNode: number;
  departMinute?: number;
  weight?: 'travel_time' | 'length';
}): Promise<ExactRoute> {
  const query = new URLSearchParams({
    network: params.network,
    from_node: String(params.fromNode),
    to_node: String(params.toNode),
    weight: params.weight ?? 'travel_time',
  });
  if (params.departMinute !== undefined) query.set('depart_minute', String(params.departMinute));
  const feature = rec(await request(`/route/exact?${query}`));
  const p = rec(feature.properties);
  return {
    points: pairArray(rec(feature.geometry).coordinates).map(([lon, lat]) => [lat, lon]),
    distance_m: num(pick(p, 'length_m', 'distance_m', 'distance')),
    travel_time_s: num(pick(p, 'duration_s', 'travel_time_s', 'travel_time')),
    free_flow_s: num(p.free_flow_s),
    delay_ratio: num(p.delay_ratio, 1),
    nodes_expanded: Math.round(num(p.nodes_expanded)),
    search_seconds: num(p.search_seconds),
    from_node: Math.round(num(p.from_node)),
    to_node: Math.round(num(p.to_node)),
  };
}

export async function createRun(body: RunRequest): Promise<RunHandle> {
  const r = rec(await postJson('/runs', body));
  const id = str(pick(r, 'run_id', 'id'));
  if (!id) throw new ApiError('the backend did not return a run id', '/runs');
  return { run_id: id };
}

export async function getRun(id: string): Promise<RunStatus> {
  return parseRunStatus(await request(`/runs/${encodeURIComponent(id)}`));
}

export async function cancelRun(id: string): Promise<void> {
  await postJson(`/runs/${encodeURIComponent(id)}/cancel`, {});
}

export interface ReoptimizeRequest {
  algorithm?: string;
  seed?: number;
  max_seconds?: number;
  max_iterations?: number;
  /** Rebuild the travel-time matrices from the traffic state as it now stands. */
  refresh_traffic?: boolean;
  /** Seed the new search with the parent run's incumbent, where supported. */
  warm_start?: boolean;
  params?: Record<string, number | boolean | string>;
}

export async function reoptimizeRun(id: string, body: ReoptimizeRequest = {}): Promise<RunHandle> {
  const r = rec(await postJson(`/runs/${encodeURIComponent(id)}/reoptimize`, body));
  const next = str(pick(r, 'run_id', 'id'), id);
  return { run_id: next };
}

/** URL of the Server-Sent Events stream for a run, consumed by `useRunStream`. */
export function runStreamUrl(id: string): string {
  return `${API_BASE}/runs/${encodeURIComponent(id)}/stream`;
}

export async function getBenchmarks(): Promise<BenchmarkSummary[]> {
  return listOf(await request('/benchmarks'), 'benchmarks').map((value) => {
    const r = rec(value);
    return {
      name: str(pick(r, 'name', 'id'), 'unknown'),
      n_instances: Math.round(num(pick(r, 'n_instances', 'instances'))),
      n_algorithms: Math.round(num(pick(r, 'n_algorithms', 'algorithms'))),
      n_runs: Math.round(num(pick(r, 'n_runs', 'runs', 'n_tasks'))),
      max_seconds: num(r.max_seconds),
      timestamp: numOrNull(pick(r, 'timestamp', 'timestamp_unix')) ?? undefined,
    };
  });
}

export async function getBenchmark(name: string): Promise<BenchmarkDetail> {
  const r = rec(await request(`/benchmarks/${encodeURIComponent(name)}`));
  const summary = rec(pick(r, 'summary', 'result'));
  const cellsRaw = rec(pick(r, 'cells', 'grid') ?? summary.cells);
  const cells: Record<string, BenchmarkCell> = {};
  for (const [key, value] of Object.entries(cellsRaw)) cells[key] = toCell(value);
  const environment = rec(pick(r, 'environment', 'meta'));
  const algorithms = strArray(pick(r, 'algorithms') ?? summary.algorithms);
  const instances = strArray(pick(r, 'instances') ?? summary.instances);
  return {
    name: str(pick(r, 'name'), name),
    algorithms:
      algorithms.length > 0
        ? algorithms
        : uniqueSorted(Object.values(cells).map((c) => c.algorithm)),
    instances:
      instances.length > 0 ? instances : uniqueSorted(Object.values(cells).map((c) => c.instance)),
    cells,
    rows: arr(pick(r, 'rows', 'runs')).map(toRow),
    omnibus: toOmnibus(pick(r, 'omnibus', 'friedman') ?? summary.omnibus),
    n_ok: Math.round(num(pick(r, 'n_ok') ?? summary.n_ok)),
    n_failed: Math.round(num(pick(r, 'n_failed') ?? summary.n_failed)),
    max_seconds: num(pick(r, 'max_seconds') ?? rec(r.config).max_seconds),
    environment: {
      python: str(environment.python) || undefined,
      platform: str(environment.platform) || undefined,
      cpu_count: numOrNull(environment.cpu_count) ?? undefined,
      packages: Object.fromEntries(
        Object.entries(rec(environment.packages)).map(([k, v]) => [k, str(v)]),
      ),
      git_commit: str(environment.git_commit) || undefined,
      git_dirty: boolOrNull(environment.git_dirty) ?? undefined,
      timestamp_unix: numOrNull(environment.timestamp_unix) ?? undefined,
    },
  };
}

function uniqueSorted(values: string[]): string[] {
  return Array.from(new Set(values)).sort();
}
