/**
 * Catalogue state: what the backend offers, and whether it is reachable at all.
 *
 * The three catalogues (algorithms, benchmark instances, road networks) are
 * fetched once at start-up and shared by every page, so switching tabs does not
 * refetch them. `backend` is the single source of truth for the "backend
 * unavailable" banner: nothing in the interface pretends to have data when this
 * store says the API could not be reached.
 */

import { create } from 'zustand';
import { ApiError, getAlgorithms, getHealth, getInstances, getNetworks } from '../api/client';
import type { AlgorithmInfo, Health, InstanceSummary, NetworkSummary } from '../api/types';

export type BackendState = 'unknown' | 'checking' | 'online' | 'offline';

export interface AppState {
  backend: BackendState;
  health: Health | null;
  error: string | null;
  algorithms: AlgorithmInfo[];
  instances: InstanceSummary[];
  networks: NetworkSummary[];
  /** Incremented on every successful bootstrap, so pages can refetch on reconnect. */
  generation: number;
  bootstrap: () => Promise<void>;
  /**
   * Re-check reachability without refetching the catalogues.
   *
   * The shell polls this while the backend is up. Without it the status badge
   * only ever moved from offline to online, so an API process that died
   * mid-demonstration left a green "backend online" on screen while every
   * action failed with nothing but an HTTP status text to explain it.
   */
  recheck: () => Promise<void>;
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return 'unknown error';
}

export const useAppStore = create<AppState>()((set, get) => ({
  backend: 'unknown',
  health: null,
  error: null,
  algorithms: [],
  instances: [],
  networks: [],
  generation: 0,

  bootstrap: async () => {
    if (get().backend === 'checking') return;
    set({ backend: 'checking', error: null });
    try {
      const health = await getHealth();
      // The catalogues are independent; one failing should not blank the others,
      // so each is settled separately and its failure recorded as an empty list.
      const [algorithms, instances, networks] = await Promise.all([
        getAlgorithms().catch(() => [] as AlgorithmInfo[]),
        getInstances().catch(() => [] as InstanceSummary[]),
        getNetworks().catch(() => [] as NetworkSummary[]),
      ]);
      set((s) => ({
        backend: 'online',
        health,
        error: null,
        algorithms,
        instances,
        networks,
        generation: s.generation + 1,
      }));
    } catch (error) {
      set({ backend: 'offline', health: null, error: describe(error) });
    }
  },

  recheck: async () => {
    // Only meaningful once a bootstrap has succeeded; while offline or in
    // flight the bootstrap retry owns the transition and this would race it.
    if (get().backend !== 'online') return;
    try {
      const health = await getHealth();
      set({ health, error: null });
    } catch (error) {
      // The catalogues are deliberately left in place: they are still an
      // accurate description of the backend that was there, and blanking them
      // would empty every picker on the way back up. The badge carries the bad
      // news, and bootstrap refills everything when the API returns.
      set({ backend: 'offline', health: null, error: describe(error) });
    }
  },
}));

/** Instances grouped by family, for the two-level picker on the solver page. */
export function groupInstances(instances: InstanceSummary[]): Map<string, InstanceSummary[]> {
  const groups = new Map<string, InstanceSummary[]>();
  for (const instance of instances) {
    const list = groups.get(instance.family);
    if (list) list.push(instance);
    else groups.set(instance.family, [instance]);
  }
  for (const list of groups.values()) list.sort((a, b) => a.name.localeCompare(b.name));
  return groups;
}
