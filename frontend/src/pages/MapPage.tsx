/**
 * The map page: a real road network, live simulated traffic, and the routes a
 * solver produced on top of both.
 *
 * The intended demonstration is a loop. Pick a city extract, generate a
 * delivery instance on it, optimise, watch the vehicles walk their routes, drag
 * the time-of-day slider until the evening peak recolours the arterials, click
 * a road to block a lane, re-optimise, and see the routes move away from the
 * incident. Every step of that loop is a real backend call; nothing on this
 * page is pre-baked.
 */

import { useEffect, useMemo, useState } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { EventKind } from '../api/types';
import { Legend } from '../components/map/Legend';
import { boundsOf, buildRouteLines } from '../components/map/geometry';
import type { LatLng } from '../components/map/geometry';
import {
  EdgeLayer,
  FitBounds,
  RouteLayer,
  StopsLayer,
  VehicleLayer,
} from '../components/map/layers';
import { Badge, CheckLine, Field, KeyValue, Notice, RailSection, Stat, StatGrid } from '../components/ui';
import { congestionBand } from '../lib/colors';
import { dayName, fmt, fmtClock, fmtDistance, fmtInt, fmtSeconds } from '../lib/format';
import { useAppStore } from '../store/appStore';
import { DETAIL_LEVELS, useMapStore } from '../store/mapStore';
import { useRunStore } from '../store/runStore';

const INCIDENT_KINDS: { value: EventKind; label: string; note: string }[] = [
  {
    value: 'lane_blockage',
    label: 'Lane blockage',
    note: 'Residual capacity from the Highway Capacity Manual incident table.',
  },
  { value: 'closure', label: 'Full closure', note: 'Capacity falls to zero; the link is removed.' },
  { value: 'slowdown', label: 'Slowdown', note: 'Free speed scaled; capacity unchanged.' },
];

export function MapPage() {
  const backend = useAppStore((s) => s.backend);
  const networks = useAppStore((s) => s.networks);
  const algorithms = useAppStore((s) => s.algorithms);

  const map = useMapStore();
  const run = useRunStore();

  const [algorithm, setAlgorithm] = useState('qpso');
  const [seconds, setSeconds] = useState(10);

  useEffect(() => {
    if (algorithms.length > 0 && !algorithms.some((a) => a.name === algorithm)) {
      setAlgorithm(algorithms[0].name);
    }
  }, [algorithms, algorithm]);

  // Load the first network as soon as the catalogue arrives, so the page is
  // never an empty grey rectangle waiting for a click.
  useEffect(() => {
    if (!map.networkId && networks.length > 0) void map.selectNetwork(networks[0].id);
  }, [networks, map]);

  const network = networks.find((n) => n.id === map.networkId) ?? null;

  const routes = run.routes ?? run.status?.routes ?? null;
  const coords: LatLng[] | null = map.instance?.geographic
    ? map.instance.coords
    : (run.status?.coords ?? null);
  const lines = useMemo(
    () => buildRouteLines(routes, coords, run.status?.geojson ?? null),
    [routes, coords, run.status],
  );

  const stopBounds = useMemo(() => (coords ? boundsOf(coords) : null), [coords]);
  const networkBounds = useMemo<[LatLng, LatLng] | null>(
    () =>
      network
        ? [
            [network.bbox[0], network.bbox[1]],
            [network.bbox[2], network.bbox[3]],
          ]
        : null,
    [network],
  );

  const center: LatLng = network?.center ?? [12.9345, 77.6183];
  const traffic = map.traffic;
  const ratio = traffic?.travel_time_seconds.network_ratio ?? 1;
  const straightLine = lines.length > 0 && lines.every((l) => !l.onRoad);

  async function optimise() {
    const instance = map.instance ?? (await map.generateInstance());
    if (!instance) return;
    await run.start(
      {
        algorithm,
        instance: instance.name,
        seed: map.instanceSeed,
        max_seconds: seconds,
        max_iterations: 100000,
      },
      'map',
    );
  }

  if (backend !== 'online') {
    return (
      <div className="page-scroll">
        <Notice kind="error">
          <strong>Backend unavailable.</strong> The map draws a live road network,
          a live traffic simulation and live solver output; none of that can be
          shown without the API. Start it with{' '}
          <code>uvicorn qroute.api.app:app --port 8000</code> and reload.
        </Notice>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', height: '100%', minHeight: 0 }}>
      {/* ------------------------------------------------------- left rail */}
      <aside className="rail">
        <RailSection title="Network">
          <Field label="City extract">
            <select
              value={map.networkId ?? ''}
              onChange={(e) => void map.selectNetwork(e.target.value)}
            >
              {networks.length === 0 && <option value="">no networks on disk</option>}
              {networks.map((n) => (
                <option key={n.id} value={n.id}>
                  {n.name}
                </option>
              ))}
            </select>
          </Field>
          {network && (
            <>
              <KeyValue label="Intersections" value={fmtInt(network.n_nodes)} />
              <KeyValue label="Road segments" value={fmtInt(network.n_edges)} />
            </>
          )}
          <Field
            label="Detail"
            hint={DETAIL_LEVELS.find((d) => d.value === map.detail)?.hint ?? ''}
          >
            <select
              value={map.detail}
              onChange={(e) => void map.setDetail(Number(e.target.value))}
            >
              {DETAIL_LEVELS.map((d) => (
                <option key={d.value} value={d.value}>
                  {d.label}
                </option>
              ))}
            </select>
          </Field>
          <div className="kv">
            <span>Drawn</span>
            <span>
              {map.edgesLoading ? 'loading…' : fmtInt(map.edges?.features.length ?? 0)}
            </span>
          </div>
          <CheckLine checked={map.showEdges} onChange={map.setShowEdges}>
            Show road segments
          </CheckLine>
        </RailSection>

        {/* ----------------------------------------------------- time slider */}
        <RailSection title="Time of day">
          <div
            style={{
              display: 'flex',
              alignItems: 'baseline',
              justifyContent: 'space-between',
              marginBottom: 6,
            }}
          >
            <span style={{ fontFamily: 'var(--mono)', fontSize: 22 }}>
              {fmtClock(map.minute)}
            </span>
            <span style={{ fontSize: 11, color: 'var(--text-faint)' }}>
              {traffic ? dayName(traffic.day_of_week) : 'Monday'}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={1439}
            step={5}
            value={map.minute % 1440}
            onChange={(e) => useMapStore.setState({ minute: Number(e.target.value) })}
            onMouseUp={(e) => void map.setMinute(Number(e.currentTarget.value))}
            onTouchEnd={(e) => void map.setMinute(Number(e.currentTarget.value))}
            onKeyUp={(e) => void map.setMinute(Number(e.currentTarget.value))}
          />
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 10,
              color: 'var(--text-faint)',
            }}
          >
            <span>00:00</span>
            <span>08:00</span>
            <span>12:00</span>
            <span>18:00</span>
            <span>24:00</span>
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
            {[
              ['07:30', 450],
              ['12:00', 720],
              ['18:30', 1110],
              ['23:00', 1380],
            ].map(([label, minute]) => (
              <button
                key={label}
                type="button"
                className="btn small"
                onClick={() => void map.setMinute(Number(minute))}
              >
                {label}
              </button>
            ))}
          </div>
          {traffic && (
            <div style={{ marginTop: 10 }}>
              <KeyValue
                label="Network travel time"
                value={`${fmt(ratio, 2)} x free flow`}
              />
              <KeyValue
                label="Mean delay"
                value={`${fmt(traffic.congestion.mean_level_length_weighted * 100, 1)} %`}
              />
              <KeyValue label="Closed links" value={fmtInt(traffic.n_closed)} />
              <KeyValue label="Active incidents" value={fmtInt(traffic.n_active_events)} />
            </div>
          )}
        </RailSection>

        {/* -------------------------------------------------------- incident */}
        <RailSection title="Inject an incident">
          {map.selectedEdge ? (
            <>
              <div style={{ fontSize: 12, marginBottom: 8, color: 'var(--text-dim)' }}>
                Selected <strong style={{ color: 'var(--text)' }}>{map.selectedEdge.highway}</strong>{' '}
                segment, {fmtDistance(map.selectedEdge.length_m)}, currently{' '}
                {congestionBand(map.selectedEdge.congestion).label.toLowerCase()} (
                {fmt(map.selectedEdge.congestion * 100, 0)} % delay).
              </div>
              <Field
                label="Kind"
                hint={INCIDENT_KINDS.find((k) => k.value === map.incident.kind)?.note}
              >
                <select
                  value={map.incident.kind}
                  onChange={(e) => map.setIncident({ kind: e.target.value as EventKind })}
                >
                  {INCIDENT_KINDS.map((k) => (
                    <option key={k.value} value={k.value}>
                      {k.label}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={`Duration — ${map.incident.durationMinutes} min`}>
                <input
                  type="range"
                  min={5}
                  max={180}
                  step={5}
                  value={map.incident.durationMinutes}
                  onChange={(e) => map.setIncident({ durationMinutes: Number(e.target.value) })}
                />
              </Field>
              {map.incident.kind === 'lane_blockage' && (
                <Field label="Lanes on the link">
                  <select
                    value={map.incident.lanes}
                    onChange={(e) => map.setIncident({ lanes: Number(e.target.value) })}
                  >
                    {[2, 3, 4].map((n) => (
                      <option key={n} value={n}>
                        {n} lanes
                      </option>
                    ))}
                  </select>
                </Field>
              )}
              {map.incident.kind === 'slowdown' && (
                <Field
                  label={`Speed multiplier — ${fmt(map.incident.speedMultiplier, 2)}`}
                  hint="0.5 halves free speed, doubling free-flow travel time."
                >
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={map.incident.speedMultiplier}
                    onChange={(e) => map.setIncident({ speedMultiplier: Number(e.target.value) })}
                  />
                </Field>
              )}
              <div className="btn-row">
                <button
                  type="button"
                  className="btn primary"
                  disabled={map.trafficLoading}
                  onClick={() => void map.injectIncident()}
                >
                  Inject
                </button>
                <button type="button" className="btn" onClick={() => map.selectEdge(null)}>
                  Clear selection
                </button>
              </div>
            </>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--text-faint)' }}>
              Click a road segment on the map to select it, then choose a
              disruption to apply from the simulated present onwards.
            </div>
          )}
        </RailSection>
      </aside>

      {/* ------------------------------------------------------------ map */}
      <div style={{ position: 'relative', flex: '1 1 auto', minWidth: 0 }}>
        <MapContainer
          center={center}
          zoom={14}
          preferCanvas
          style={{ height: '100%', width: '100%' }}
          zoomControl
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
            attribution='Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, tiles &copy; <a href="https://carto.com/attributions">CARTO</a>'
            maxZoom={19}
          />
          <EdgeLayer
            edges={map.edges}
            visible={map.showEdges}
            selectedEdge={map.selectedEdge?.edge ?? null}
            onSelect={map.selectEdge}
          />
          <RouteLayer lines={lines} dim={run.streaming} />
          <StopsLayer instance={map.instance} />
          <VehicleLayer lines={lines} running={map.animate && !run.streaming} />
          <FitBounds bounds={stopBounds ?? networkBounds} />
        </MapContainer>
        <Legend
          congestion={traffic?.congestion ?? null}
          vehicles={lines.length}
          straightLine={straightLine}
        />
        {map.error && (
          <div style={{ position: 'absolute', top: 12, left: 12, right: 12, zIndex: 500 }}>
            <Notice kind="error">{map.error}</Notice>
          </div>
        )}
      </div>

      {/* ------------------------------------------------------ right rail */}
      <aside className="rail right">
        <RailSection title="Delivery instance">
          <Field label={`Customers — ${map.instanceSize}`}>
            <input
              type="range"
              min={8}
              max={80}
              step={1}
              value={map.instanceSize}
              onChange={(e) => map.setInstanceSize(Number(e.target.value))}
            />
          </Field>
          <Field label="Seed" hint="Stop selection and demands are reproducible from this seed.">
            <input
              type="number"
              value={map.instanceSeed}
              onChange={(e) => map.setInstanceSeed(Number(e.target.value))}
            />
          </Field>
          <div className="btn-row">
            <button
              type="button"
              className="btn"
              disabled={map.instanceLoading || !map.networkId}
              onClick={() => void map.generateInstance()}
            >
              {map.instanceLoading ? 'Building matrices…' : 'Generate instance'}
            </button>
          </div>
          {map.instance && (
            <div style={{ marginTop: 10 }}>
              <KeyValue label="Name" value={map.instance.name} />
              <KeyValue label="Customers" value={fmtInt(map.instance.n_customers)} />
              <KeyValue label="Capacity" value={fmtInt(map.instance.capacity)} />
              <KeyValue
                label="Total demand"
                value={fmtInt(map.instance.demand.reduce((a, b) => a + b, 0))}
              />
            </div>
          )}
        </RailSection>

        <RailSection title="Optimise">
          <Field label="Algorithm">
            <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
              {algorithms.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label={`Time budget — ${seconds} s`}>
            <input
              type="range"
              min={2}
              max={60}
              step={1}
              value={seconds}
              onChange={(e) => setSeconds(Number(e.target.value))}
            />
          </Field>
          <div className="btn-row">
            <button
              type="button"
              className="btn primary"
              disabled={run.streaming || run.starting || !map.networkId}
              onClick={() => void optimise()}
            >
              {run.starting ? 'Starting…' : run.streaming ? 'Running…' : 'Optimise'}
            </button>
            {run.streaming ? (
              <button type="button" className="btn danger" onClick={() => void run.cancel()}>
                Cancel
              </button>
            ) : (
              <button
                type="button"
                className="btn"
                disabled={!run.runId}
                onClick={() => void run.reoptimize()}
              >
                Re-optimise
              </button>
            )}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 8, lineHeight: 1.45 }}>
            Re-optimise rebuilds the travel-time matrices under the traffic state
            as it now stands, incidents included, and searches again from
            scratch.
          </div>
          <CheckLine checked={map.animate} onChange={map.setAnimate}>
            Animate vehicles along routes
          </CheckLine>
          {run.error && (
            <div style={{ marginTop: 8 }}>
              <Notice kind="error">{run.error}</Notice>
            </div>
          )}
        </RailSection>

        {run.status && (
          <RailSection title="Current solution">
            <StatGrid columns={2}>
              <Stat
                label="Objective"
                value={fmt(run.status.best_cost, 1)}
                sub={run.status.instance}
              />
              <Stat label="Vehicles" value={fmtInt(run.status.n_routes)} />
              <Stat
                label="Distance"
                value={fmtDistance(run.status.stats?.distance ?? null)}
              />
              <Stat
                label="Drive time"
                value={fmtSeconds(run.status.stats?.duration ?? null)}
              />
            </StatGrid>
            <div style={{ marginTop: 8 }}>
              <Badge tone={run.status.feasible ? 'ok' : 'bad'}>
                <span className="dot" />
                {run.status.feasible ? 'Feasible' : 'Infeasible'}
              </Badge>{' '}
              <Badge>{fmtInt(run.status.iterations)} iterations</Badge>{' '}
              <Badge>{fmtSeconds(run.status.seconds)}</Badge>
            </div>
          </RailSection>
        )}

        {traffic && traffic.events.length > 0 && (
          <RailSection title="Incidents">
            {traffic.events.map((event) => (
              <div
                key={event.event_id}
                style={{
                  borderLeft: '2px solid var(--bad)',
                  paddingLeft: 8,
                  marginBottom: 8,
                  fontSize: 11.5,
                  color: 'var(--text-dim)',
                }}
              >
                <div style={{ color: 'var(--text)' }}>{event.kind.replace('_', ' ')}</div>
                <div>
                  {fmtClock(event.start_minute)} – {fmtClock(event.end_minute)} ·{' '}
                  {fmtInt(event.n_edges)} link{event.n_edges === 1 ? '' : 's'}
                </div>
                <div>
                  capacity x{fmt(event.capacity_multiplier, 2)}, time x
                  {fmt(event.time_multiplier, 2)}
                  {event.hcm_tabulated ? ' (HCM table)' : ''}
                </div>
              </div>
            ))}
          </RailSection>
        )}
      </aside>
    </div>
  );
}
