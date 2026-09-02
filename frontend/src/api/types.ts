/**
 * Wire types for the qroute HTTP API.
 *
 * These declarations are the single description of what the backend at
 * `qroute/api/` is expected to send and receive. Every network call in the
 * application goes through `src/api/client.ts` and is typed against this file,
 * so a change in the backend contract shows up as a compile error in one place
 * rather than as an undefined property somewhere deep in a chart.
 *
 * Fields the backend may reasonably omit are declared optional rather than
 * assumed present, because the frontend is developed alongside the backend and
 * must render honestly when a payload is thinner than expected.
 */

// --------------------------------------------------------------- capabilities

/** Response of `GET /api/health`. */
export interface Health {
  status: string;
  version?: string;
  networks?: number;
  instances?: number;
  algorithms?: number;
}

/** One entry of `GET /api/algorithms`. */
export interface AlgorithmInfo {
  name: string;
  description: string;
  /** Tunable parameters with their defaults, when the backend exposes them. */
  params?: ParamSpec[];
}

/** Declaration of a single algorithm parameter, used to build the solver form. */
export interface ParamSpec {
  name: string;
  kind: 'int' | 'float' | 'bool' | 'choice';
  default: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
  choices?: string[];
  description?: string;
}

// ------------------------------------------------------------------ instances

export type InstanceFamily = 'cvrp' | 'vrptw' | 'network';

/** One entry of `GET /api/instances`. */
export interface InstanceSummary {
  name: string;
  family: InstanceFamily;
  /** Number of customers, excluding the depot. */
  n_customers: number;
  capacity: number;
  /** Fleet size; null means unlimited (the CVRPLIB convention). */
  n_vehicles: number | null;
  /** Best-known solution cost from the literature, when one is on disk. */
  bks: number | null;
  has_time_windows: boolean;
}

/** Response of `GET /api/instances/{name}`, and of the generate-from-map call. */
export interface InstanceDetail extends InstanceSummary {
  /** `[latitude, longitude]` per stop for road instances, `[x, y]` otherwise. */
  coords: [number, number][];
  demand: number[];
  /** `[earliest, latest]` service start per stop, when the instance has windows. */
  time_windows?: [number, number][];
  service_time?: number[];
  /** OSM node id per stop, present only for instances generated from a network. */
  node_ids?: number[];
  /** Whether `coords` are geographic and can be drawn on the Leaflet map. */
  geographic: boolean;
  meta: Record<string, unknown>;
}

// ------------------------------------------------------------------- networks

/** One entry of `GET /api/networks`. */
export interface NetworkSummary {
  id: string;
  name: string;
  n_nodes: number;
  n_edges: number;
  /** `[latitude, longitude]` of the extract centre, for the initial map view. */
  center: [number, number];
  /** `[minLat, minLon, maxLat, maxLon]`. */
  bbox: [number, number, number, number];
}

/** Properties carried by every feature of `GET /api/networks/{id}/edges`. */
export interface EdgeProperties {
  /** Index into the simulator's edge array; the handle used to inject events. */
  edge: number;
  u: number;
  v: number;
  highway: string;
  length_m: number;
  free_flow_s: number;
  travel_time_s: number;
  /** Fractional delay `(t - t0) / t0`; 0 is free flow, 1 is twice as long. */
  congestion: number;
  speed_kph: number;
}

export interface EdgeFeature {
  type: 'Feature';
  geometry: { type: 'LineString'; coordinates: [number, number][] };
  properties: EdgeProperties;
}

export interface EdgeCollection {
  type: 'FeatureCollection';
  features: EdgeFeature[];
  properties?: { network?: string; n_features?: number; min_importance?: number };
}

/** Body of `POST /api/networks/{id}/instance`. */
export interface InstanceRequest {
  n_customers: number;
  seed: number;
  /** Simulator clock in minutes since midnight Monday, so matrices match the map. */
  minute?: number;
  capacity?: number;
  n_vehicles?: number;
}

// -------------------------------------------------------------------- traffic

export type CongestionBand = 'free' | 'light' | 'moderate' | 'heavy' | 'severe';

export interface CongestionSummary {
  mean_level_length_weighted: number;
  vkt_weighted_ratio: number;
  mean_level: number;
  median_level: number;
  p95_level: number;
  max_level: number;
  bands: Record<CongestionBand, number>;
}

export interface TravelTimeSummary {
  total_free_flow: number;
  total_current: number;
  /** Network-wide `sum(t) / sum(t0)`; the headline congestion figure. */
  network_ratio: number;
}

export type EventKind = 'lane_blockage' | 'closure' | 'slowdown';

/** A scheduled disruption, as serialised by `TrafficEvent.as_dict`. */
export interface TrafficEventDto {
  event_id: number;
  kind: EventKind;
  edges: number[];
  n_edges: number;
  start_minute: number;
  duration_minutes: number;
  end_minute: number;
  severity: number;
  lanes: number;
  blockage: string | null;
  capacity_multiplier: number;
  time_multiplier: number;
  /** True when the capacity figure is a Highway Capacity Manual table value. */
  hcm_tabulated: boolean;
  description: string;
}

/** Response of `GET /api/traffic/{id}/state`. */
export interface TrafficState {
  time_minutes: number;
  hour_of_day: number;
  day_of_week: number;
  weekend: boolean;
  vdf: string;
  seed: number | null;
  profile: string;
  n_edges: number;
  n_closed: number;
  reference_saturation: number;
  congestion: CongestionSummary;
  travel_time_seconds: TravelTimeSummary;
  events: TrafficEventDto[];
  n_active_events: number;
  worst_edges?: WorstEdge[];
}

export interface WorstEdge {
  index: number;
  road_class: string;
  congestion_level: number;
  travel_time_s: number;
  free_flow_s: number;
}

/** Body of `POST /api/traffic/{id}/events`. */
export interface EventRequest {
  kind: EventKind;
  edges: number[];
  start_minute?: number;
  duration_minutes?: number;
  severity?: number;
  lanes?: number;
  blockage?: string;
  speed_multiplier?: number;
  description?: string;
}

// ------------------------------------------------------------------ solutions

export interface SolutionStats {
  distance: number;
  duration: number;
  capacity_violation: number;
  time_window_violation: number;
  duration_violation: number;
  fleet_violation: number;
  edge_load_violation: number;
  total_violation: number;
  feasible: number;
}

/** A route is the ordered list of customer indices between two depot visits. */
export type RouteIndices = number[];

/** GeoJSON of the road polyline actually followed by each vehicle. */
export interface RouteFeature {
  type: 'Feature';
  geometry: { type: 'LineString'; coordinates: [number, number][] };
  properties: { vehicle: number; length_m?: number; travel_time_s?: number };
}

export interface RouteCollection {
  type: 'FeatureCollection';
  features: RouteFeature[];
}

// ----------------------------------------------------------------------- runs

export type RunState = 'queued' | 'running' | 'done' | 'cancelled' | 'failed';

/** Body of `POST /api/runs`. */
export interface RunRequest {
  algorithm: string;
  /** Name of a benchmark instance, or of an instance generated from a network. */
  instance: string;
  seed: number;
  max_seconds: number;
  max_iterations: number;
  params?: Record<string, number | boolean | string>;
}

/** Response of `POST /api/runs`. */
export interface RunHandle {
  run_id: string;
}

/** One Server-Sent Event from `GET /api/runs/{id}/stream`. */
export interface RunTick {
  iteration: number;
  best_cost: number;
  mean_cost: number;
  diversity: number;
  /** Seconds since the run started. */
  elapsed: number;
  evaluations: number;
  feasible?: boolean;
  /** Present on ticks where the incumbent improved, so the map can follow. */
  routes?: RouteIndices[];
}

/** Response of `GET /api/runs/{id}`. */
export interface RunStatus {
  run_id: string;
  state: RunState;
  algorithm: string;
  instance: string;
  seed: number;
  /** Best-known cost for the instance, when one exists; drawn as a target line. */
  bks: number | null;
  best_cost: number | null;
  n_routes: number | null;
  feasible: boolean | null;
  routes: RouteIndices[] | null;
  stats: SolutionStats | null;
  iterations: number;
  evaluations: number;
  seconds: number;
  params: Record<string, unknown>;
  /** Road-following geometry, present only for instances built on a network. */
  geojson: RouteCollection | null;
  /** Stops in map order, so the map can draw a run without refetching. */
  coords: [number, number][] | null;
  error: string | null;
  /** The complete history, so a page reload after a run still shows the curve. */
  history: RunTick[];
}

// ----------------------------------------------------------------- benchmarks

/** One entry of `GET /api/benchmarks`. */
export interface BenchmarkSummary {
  name: string;
  n_instances: number;
  n_algorithms: number;
  n_runs: number;
  max_seconds: number;
  timestamp?: number;
}

export interface StatSummary {
  n: number;
  best: number;
  mean: number;
  median: number;
  std: number;
  iqr: number;
  worst: number;
}

/** One cell of the instance-by-algorithm result grid. */
export interface BenchmarkCell {
  instance: string;
  algorithm: string;
  cost: StatSummary;
  gap: StatSummary;
  feasible_runs: number;
  runs: number;
  mean_seconds: number;
  mean_iterations: number;
  hit_bks: number;
  median_time_to_1pct: number | null;
}

/** A single seeded run, as stored in `rows.jsonl`. */
export interface BenchmarkRow {
  instance: string;
  algorithm: string;
  seed: number;
  cost: number;
  gap: number;
  bks: number | null;
  n_routes: number;
  feasible: boolean;
  iterations: number;
  evaluations: number;
  seconds: number;
  status: string;
}

export interface PairwiseResult {
  a: string;
  b: string;
  n: number;
  median_a: number;
  median_b: number;
  statistic: number;
  p_value: number;
  p_adjusted: number | null;
  effect_size: number;
  winner: string | null;
}

export interface FriedmanResult {
  algorithms: string[];
  mean_ranks: Record<string, number>;
  statistic: number;
  p_value: number;
  n_instances: number;
  post_hoc: PairwiseResult[];
  control: string | null;
}

export interface BenchmarkEnvironment {
  python?: string;
  platform?: string;
  cpu_count?: number;
  packages?: Record<string, string>;
  git_commit?: string;
  git_dirty?: boolean;
  timestamp_unix?: number;
}

/** Response of `GET /api/benchmarks/{name}`. */
export interface BenchmarkDetail {
  name: string;
  algorithms: string[];
  instances: string[];
  cells: Record<string, BenchmarkCell>;
  rows: BenchmarkRow[];
  omnibus: FriedmanResult | null;
  n_ok: number;
  n_failed: number;
  max_seconds: number;
  environment: BenchmarkEnvironment;
}

// -------------------------------------------------------------- exact routing

/**
 * Response of `GET /api/route/exact`: a GeoJSON `Feature` tracing the exact
 * shortest path, with the search statistics in its properties. The endpoint
 * runs A* with a great-circle lower bound, which is exact rather than
 * heuristic, and reports how many nodes it expanded.
 */
export interface ExactRoute {
  /** `[latitude, longitude]` along the road polyline, already swapped. */
  points: [number, number][];
  distance_m: number;
  travel_time_s: number;
  free_flow_s: number;
  /** Travel time divided by free-flow time along the same path. */
  delay_ratio: number;
  nodes_expanded: number;
  search_seconds: number;
  from_node: number;
  to_node: number;
}
