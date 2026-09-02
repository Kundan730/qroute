/**
 * Everything the map page needs: which network is loaded, the road geometry,
 * the simulated traffic state, and the delivery instance built on top of them.
 *
 * The three are deliberately kept separate because they change on different
 * clocks. Road geometry is fetched once per network and per level of detail;
 * the traffic state changes whenever the time-of-day slider moves or an
 * incident is injected; the instance changes only when the operator asks for a
 * new one. Conflating them would refetch 4000 polylines every time the slider
 * moved a minute.
 */

import { create } from 'zustand';
import {
  addTrafficEvent,
  ApiError,
  createNetworkInstance,
  getExactRoute,
  getNetworkEdges,
  getTrafficState,
  setTrafficTime,
} from '../api/client';
import type {
  EdgeCollection,
  EdgeProperties,
  EventKind,
  ExactRoute,
  InstanceDetail,
  TrafficState,
} from '../api/types';

/** Level-of-detail presets; the numbers are `min_importance` in the backend. */
export const DETAIL_LEVELS = [
  { value: 4, label: 'Arterials', hint: 'motorway, trunk and primary' },
  { value: 3, label: 'Main roads', hint: 'secondary and above' },
  { value: 2, label: 'Through roads', hint: 'tertiary and above' },
  { value: 0, label: 'All roads', hint: 'every mapped segment' },
] as const;

export interface IncidentDraft {
  kind: EventKind;
  durationMinutes: number;
  severity: number;
  lanes: number;
  speedMultiplier: number;
}

export const DEFAULT_INCIDENT: IncidentDraft = {
  kind: 'lane_blockage',
  durationMinutes: 45,
  severity: 1,
  lanes: 2,
  speedMultiplier: 0.5,
};

export interface MapState {
  networkId: string | null;
  detail: number;
  edges: EdgeCollection | null;
  edgesLoading: boolean;
  traffic: TrafficState | null;
  trafficLoading: boolean;
  /** Simulator clock in minutes since midnight of the simulated day. */
  minute: number;
  instance: InstanceDetail | null;
  instanceLoading: boolean;
  instanceSize: number;
  instanceSeed: number;
  selectedEdge: EdgeProperties | null;
  incident: IncidentDraft;
  animate: boolean;
  showEdges: boolean;
  /** The exact A* shortest path between two stops, when one has been asked for. */
  exactRoute: ExactRoute | null;
  exactFrom: number;
  exactTo: number;
  exactLoading: boolean;
  error: string | null;

  selectNetwork: (id: string) => Promise<void>;
  setDetail: (detail: number) => Promise<void>;
  setMinute: (minute: number) => Promise<void>;
  refreshTraffic: () => Promise<void>;
  selectEdge: (edge: EdgeProperties | null) => void;
  setIncident: (patch: Partial<IncidentDraft>) => void;
  injectIncident: () => Promise<void>;
  setInstanceSize: (size: number) => void;
  setInstanceSeed: (seed: number) => void;
  generateInstance: () => Promise<InstanceDetail | null>;
  setAnimate: (on: boolean) => void;
  setShowEdges: (on: boolean) => void;
  setExactPair: (from: number, to: number) => void;
  computeExactRoute: () => Promise<void>;
  clearExactRoute: () => void;
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return 'unknown error';
}

/** Edge polylines are capped so panning stays smooth on a laptop GPU. */
const MAX_EDGES = 5000;

export const useMapStore = create<MapState>()((set, get) => ({
  networkId: null,
  detail: 3,
  edges: null,
  edgesLoading: false,
  traffic: null,
  trafficLoading: false,
  minute: 9 * 60,
  instance: null,
  instanceLoading: false,
  instanceSize: 24,
  instanceSeed: 20260920,
  selectedEdge: null,
  incident: DEFAULT_INCIDENT,
  animate: true,
  showEdges: true,
  exactRoute: null,
  exactFrom: 0,
  exactTo: 1,
  exactLoading: false,
  error: null,

  selectNetwork: async (id) => {
    if (get().networkId === id) return;
    set({
      networkId: id,
      edges: null,
      traffic: null,
      instance: null,
      selectedEdge: null,
      error: null,
    });
    await Promise.all([loadEdges(set, get), get().refreshTraffic()]);
  },

  setDetail: async (detail) => {
    set({ detail });
    await loadEdges(set, get);
  },

  setMinute: async (minute) => {
    const { networkId } = get();
    set({ minute });
    if (!networkId) return;
    set({ trafficLoading: true });
    try {
      const traffic = await setTrafficTime(networkId, minute);
      set({ traffic, trafficLoading: false, error: null });
      // Travel times changed, so the drawn congestion is now stale.
      await loadEdges(set, get);
    } catch (error) {
      set({ trafficLoading: false, error: describe(error) });
    }
  },

  refreshTraffic: async () => {
    const { networkId } = get();
    if (!networkId) return;
    set({ trafficLoading: true });
    try {
      const traffic = await getTrafficState(networkId, 8);
      set({ traffic, trafficLoading: false, minute: traffic.time_minutes, error: null });
    } catch (error) {
      set({ trafficLoading: false, error: describe(error) });
    }
  },

  selectEdge: (edge) => set({ selectedEdge: edge }),

  setIncident: (patch) => set((s) => ({ incident: { ...s.incident, ...patch } })),

  injectIncident: async () => {
    const { networkId, selectedEdge, incident, minute } = get();
    if (!networkId || !selectedEdge) return;
    set({ trafficLoading: true });
    try {
      const traffic = await addTrafficEvent(networkId, {
        kind: incident.kind,
        edges: [selectedEdge.edge],
        start_minute: minute,
        duration_minutes: incident.durationMinutes,
        severity: incident.severity,
        lanes: incident.lanes,
        speed_multiplier: incident.speedMultiplier,
        description: `${incident.kind} on ${selectedEdge.highway} (edge ${selectedEdge.edge})`,
      });
      set({ traffic, trafficLoading: false, error: null });
      await loadEdges(set, get);
    } catch (error) {
      set({ trafficLoading: false, error: describe(error) });
    }
  },

  setInstanceSize: (instanceSize) => set({ instanceSize }),
  setInstanceSeed: (instanceSeed) => set({ instanceSeed }),

  generateInstance: async () => {
    const { networkId, instanceSize, instanceSeed, minute } = get();
    if (!networkId) return null;
    set({ instanceLoading: true, error: null });
    try {
      const instance = await createNetworkInstance(networkId, {
        n_customers: instanceSize,
        seed: instanceSeed,
        minute,
      });
      set({ instance, instanceLoading: false, exactRoute: null, exactFrom: 0, exactTo: 1 });
      return instance;
    } catch (error) {
      set({ instanceLoading: false, error: describe(error) });
      return null;
    }
  },

  setAnimate: (animate) => set({ animate }),
  setShowEdges: (showEdges) => set({ showEdges }),

  setExactPair: (exactFrom, exactTo) => set({ exactFrom, exactTo, exactRoute: null }),

  computeExactRoute: async () => {
    const { networkId, instance, exactFrom, exactTo, minute } = get();
    const nodes = instance?.node_ids;
    if (!networkId || !nodes || exactFrom === exactTo) return;
    const from = nodes[exactFrom];
    const to = nodes[exactTo];
    if (from === undefined || to === undefined) return;
    set({ exactLoading: true });
    try {
      const exactRoute = await getExactRoute({
        network: networkId,
        fromNode: from,
        toNode: to,
        departMinute: minute,
      });
      set({ exactRoute, exactLoading: false, error: null });
    } catch (error) {
      set({ exactLoading: false, exactRoute: null, error: describe(error) });
    }
  },

  clearExactRoute: () => set({ exactRoute: null }),
}));

type Setter = (partial: Partial<MapState>) => void;
type Getter = () => MapState;

async function loadEdges(set: Setter, get: Getter): Promise<void> {
  const { networkId, detail } = get();
  if (!networkId) return;
  set({ edgesLoading: true });
  try {
    // The edge weights the endpoint reports are those of the simulator's
    // current clock, so moving the clock is followed by a refetch rather than
    // by passing a time here.
    const edges = await getNetworkEdges(networkId, {
      minImportance: detail,
      maxEdges: MAX_EDGES,
    });
    // A slower request for an earlier network must not overwrite a newer one.
    if (get().networkId !== networkId) return;
    set({ edges, edgesLoading: false, error: null });
  } catch (error) {
    set({ edgesLoading: false, error: describe(error) });
  }
}
