/**
 * The map legend.
 *
 * It states the units explicitly, because "congestion" as a colour is
 * meaningless without saying what quantity is being coloured. The quantity is
 * the fractional delay `(t - t0) / t0` computed by
 * `qroute.traffic.bpr.congestion_level`, and the band edges are the same ones
 * the backend uses to build its own histogram, so the counts shown beside each
 * swatch are the backend's numbers, not a recount done in the browser.
 */

import type { CongestionSummary } from '../../api/types';
import { BAND_RANGES, CONGESTION_BANDS, vehicleColor } from '../../lib/colors';
import { fmtInt } from '../../lib/format';

export function Legend({
  congestion,
  vehicles,
  straightLine,
}: {
  congestion: CongestionSummary | null;
  vehicles: number;
  straightLine: boolean;
}) {
  return (
    <div
      style={{
        position: 'absolute',
        left: 12,
        bottom: 22,
        zIndex: 500,
        background: 'rgba(20, 25, 32, 0.93)',
        border: '1px solid var(--border-strong)',
        borderRadius: 'var(--radius-lg)',
        padding: '9px 11px',
        minWidth: 236,
        maxWidth: 292,
        backdropFilter: 'blur(3px)',
      }}
    >
      <div
        style={{
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          color: 'var(--text-faint)',
          marginBottom: 6,
        }}
      >
        Congestion — delay over free flow
      </div>
      {CONGESTION_BANDS.map((band) => (
        <div
          key={band.name}
          style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '1px 0' }}
        >
          <span
            style={{
              width: 16,
              height: 3,
              background: band.color,
              borderRadius: 2,
              flex: '0 0 auto',
            }}
          />
          <span style={{ fontSize: 11, color: 'var(--text)', width: 62 }}>{band.label}</span>
          <span style={{ fontSize: 11, color: 'var(--text-faint)', flex: '1 1 auto' }}>
            {BAND_RANGES[band.name]}
          </span>
          {congestion && (
            <span
              style={{
                fontSize: 11,
                fontFamily: 'var(--mono)',
                color: 'var(--text-dim)',
                minWidth: 42,
                textAlign: 'right',
              }}
            >
              {fmtInt(congestion.bands[band.name])}
            </span>
          )}
        </div>
      ))}

      {vehicles > 0 && (
        <>
          <div
            style={{
              fontSize: 10,
              textTransform: 'uppercase',
              letterSpacing: '0.07em',
              color: 'var(--text-faint)',
              margin: '9px 0 5px',
              paddingTop: 8,
              borderTop: '1px solid var(--border)',
            }}
          >
            Vehicles
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {Array.from({ length: vehicles }, (_, i) => (
              <span
                key={i}
                style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11 }}
              >
                <span
                  style={{
                    width: 12,
                    height: 3,
                    background: vehicleColor(i),
                    borderRadius: 2,
                  }}
                />
                {i + 1}
              </span>
            ))}
          </div>
          {straightLine && (
            <div style={{ fontSize: 10.5, color: 'var(--warn)', marginTop: 6, lineHeight: 1.4 }}>
              Routes drawn as straight lines between stops: the backend did not
              return road geometry for this run.
            </div>
          )}
        </>
      )}

      <div
        style={{
          fontSize: 10.5,
          color: 'var(--text-faint)',
          marginTop: 8,
          paddingTop: 7,
          borderTop: '1px solid var(--border)',
          lineHeight: 1.45,
        }}
      >
        Counts are road segments per band across the whole network, from the
        backend's own histogram; the map draws the most important few thousand.
        Line width follows the OSM highway class.
      </div>
    </div>
  );
}
