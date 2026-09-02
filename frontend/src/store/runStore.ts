/**
 * The live solver run: request, Server-Sent Event stream, and final result.
 *
 * There is exactly one active run at a time, which is what the interface
 * assumes: a run started from the map is the same object the solver page draws
 * a convergence curve for, so a judge can inject an incident on the map, switch
 * tab, and watch the same search converge.
 *
 * The stream the backend emits is named-event SSE: `start` once, `tick` per
 * sampled iteration, and exactly one of `done`, `cancelled` or `error` at the
 * end, the terminal event carrying the complete run record. That means the
 * common case needs no follow-up request at all. The one subtlety is that a
 * server-sent event named `error` arrives on the same listener as a transport
 * failure, so the two are told apart by whether the event carries data.
 *
 * The `EventSource` itself lives outside the store. Putting a live connection
 * into React state means every tick re-renders anything that touches it, and it
 * makes the connection lifecycle implicit; keeping it in a module variable
 * keeps it explicit and single.
 */

import { create } from 'zustand';
import {
  ApiError,
  cancelRun,
  createRun,
  getRun,
  parseRunStatus,
  reoptimizeRun,
  runStreamUrl,
  toRunTick,
} from '../api/client';
import type { ReoptimizeRequest } from '../api/client';
import type { RunRequest, RunStatus, RunTick } from '../api/types';

export interface RunSlice {
  runId: string | null;
  request: RunRequest | null;
  status: RunStatus | null;
  ticks: RunTick[];
  /** The best routes seen so far, updated from ticks that carry them. */
  routes: number[][] | null;
  streaming: boolean;
  starting: boolean;
  error: string | null;
  /** Where the run was launched from, so the map knows whether to follow it. */
  origin: 'solver' | 'map' | null;

  start: (request: RunRequest, origin: 'solver' | 'map') => Promise<string | null>;
  cancel: () => Promise<void>;
  /** Re-solve under the traffic state as it now stands, warm-started. */
  reoptimize: (options?: ReoptimizeRequest) => Promise<string | null>;
  clear: () => void;
}

let source: EventSource | null = null;
let pollTimer: number | null = null;

function closeStream(): void {
  if (source) {
    source.close();
    source = null;
  }
  if (pollTimer !== null) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return 'unknown error';
}

function parse(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}

export const useRunStore = create<RunSlice>()((set, get) => {
  /** Fetch the run record; used only when the terminal event did not arrive. */
  async function finalise(runId: string): Promise<void> {
    try {
      const status = await getRun(runId);
      if (get().runId !== runId) return;
      set((s) => ({
        status,
        streaming: false,
        // Keep the streamed ticks when the server did not persist a history,
        // so the convergence chart never goes blank at the end of a run.
        ticks: status.history.length > 0 ? status.history : s.ticks,
        routes: status.routes ?? s.routes,
      }));
    } catch (error) {
      if (get().runId !== runId) return;
      set({ streaming: false, error: describe(error) });
    }
  }

  function applyTerminal(runId: string, payload: unknown): void {
    if (get().runId !== runId) return;
    const status = parseRunStatus(payload);
    set((s) => ({
      status,
      streaming: false,
      ticks: status.history.length > 0 ? status.history : s.ticks,
      routes: status.routes ?? s.routes,
      error: status.error ?? s.error,
    }));
  }

  /**
   * Open the SSE stream. Falls back to polling `GET /api/runs/{id}` when the
   * stream cannot be established, which is the difference between a degraded
   * live view and no live view at all.
   */
  function openStream(runId: string): void {
    closeStream();
    let connected = false;
    const es = new EventSource(runStreamUrl(runId));
    source = es;

    es.addEventListener('start', () => {
      connected = true;
    });

    es.addEventListener('tick', (event) => {
      if (get().runId !== runId) return;
      connected = true;
      const payload = parse((event as MessageEvent<string>).data);
      if (payload === null) return;
      const tick = toRunTick(payload);
      set((s) => ({
        ticks: [...s.ticks, tick],
        routes: tick.routes && tick.routes.length > 0 ? tick.routes : s.routes,
      }));
    });

    for (const name of ['done', 'cancelled'] as const) {
      es.addEventListener(name, (event) => {
        const payload = parse((event as MessageEvent<string>).data);
        closeStream();
        if (payload !== null) applyTerminal(runId, payload);
        else void finalise(runId);
      });
    }

    // A named `error` event from the server and a transport failure both land
    // here; only the former carries a data payload.
    es.addEventListener('error', (event) => {
      const message = event as MessageEvent<string | undefined>;
      if (typeof message.data === 'string') {
        const payload = parse(message.data);
        closeStream();
        if (payload !== null) applyTerminal(runId, payload);
        else void finalise(runId);
        return;
      }
      if (es.readyState !== EventSource.CLOSED) return; // still reconnecting
      closeStream();
      if (connected) void finalise(runId);
      else startPolling(runId);
    });
  }

  function startPolling(runId: string): void {
    const tick = async () => {
      if (get().runId !== runId) return;
      try {
        const status = await getRun(runId);
        if (get().runId !== runId) return;
        const done = status.state !== 'running' && status.state !== 'queued';
        set((s) => ({
          status,
          ticks: status.history.length > 0 ? status.history : s.ticks,
          routes: status.routes ?? s.routes,
          streaming: !done,
        }));
        if (!done) pollTimer = window.setTimeout(() => void tick(), 500);
      } catch (error) {
        if (get().runId !== runId) return;
        set({ streaming: false, error: describe(error) });
      }
    };
    void tick();
  }

  return {
    runId: null,
    request: null,
    status: null,
    ticks: [],
    routes: null,
    streaming: false,
    starting: false,
    error: null,
    origin: null,

    start: async (request, origin) => {
      closeStream();
      set({
        starting: true,
        error: null,
        ticks: [],
        routes: null,
        status: null,
        runId: null,
        request,
        origin,
      });
      try {
        const handle = await createRun(request);
        set({ runId: handle.run_id, starting: false, streaming: true });
        openStream(handle.run_id);
        return handle.run_id;
      } catch (error) {
        set({ starting: false, streaming: false, error: describe(error) });
        return null;
      }
    },

    cancel: async () => {
      const { runId } = get();
      if (!runId) return;
      try {
        await cancelRun(runId);
      } catch (error) {
        set({ error: describe(error) });
      }
      // The stream emits `cancelled` of its own accord; nothing more to do.
    },

    reoptimize: async (options = {}) => {
      const { runId, request, origin } = get();
      if (!runId) {
        if (request) return get().start(request, origin ?? 'map');
        return null;
      }
      closeStream();
      set({ starting: true, error: null, ticks: [], routes: null, status: null });
      try {
        const handle = await reoptimizeRun(runId, {
          max_seconds: request?.max_seconds ?? 10,
          max_iterations: request?.max_iterations ?? 1000000,
          refresh_traffic: true,
          warm_start: true,
          ...options,
        });
        set({ runId: handle.run_id, starting: false, streaming: true });
        openStream(handle.run_id);
        return handle.run_id;
      } catch (error) {
        set({ starting: false, streaming: false, error: describe(error) });
        return null;
      }
    },

    clear: () => {
      closeStream();
      set({
        runId: null,
        request: null,
        status: null,
        ticks: [],
        routes: null,
        streaming: false,
        starting: false,
        error: null,
        origin: null,
      });
    },
  };
});

/** Evaluations per second over the most recent window of the stream. */
export function evaluationRate(ticks: RunTick[], window = 12): number {
  if (ticks.length < 2) return Number.NaN;
  const head = ticks[Math.max(0, ticks.length - window)];
  const tail = ticks[ticks.length - 1];
  const dt = tail.elapsed - head.elapsed;
  if (!(dt > 0)) return Number.NaN;
  return (tail.evaluations - head.evaluations) / dt;
}
