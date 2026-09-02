/**
 * A side-by-side simulation of one classical PSO particle and one
 * quantum-behaved particle in a single dimension.
 *
 * This is not a decorative animation. Both particles are stepped with the
 * actual update rules used by `qroute/algorithms/pso.py` and
 * `qroute/algorithms/qpso.py`, from a seeded generator, and the histograms
 * underneath accumulate where each particle has actually been. The point the
 * picture makes is a specific and checkable one: the classical particle's
 * reachable set is bounded by its velocity clamp, so its histogram stays inside
 * an envelope around the attractor, whereas the quantum particle's sampling
 * distribution has unbounded support, so its histogram has thin but non-zero
 * tails across the whole domain.
 *
 * Everything is drawn on one canvas at device pixel ratio, which keeps the axis
 * labels crisp on a projector.
 */

import { useEffect, useRef, useState } from 'react';

/** Small deterministic generator, so the illustration is the same every load. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const BINS = 96;
const DOMAIN: [number, number] = [-1, 1];
/** Where the swarm's best and the particle's own best sit, in domain units. */
const GBEST = 0.18;
const PBEST = -0.32;

interface Particle {
  x: number;
  v: number;
  history: Float64Array;
  hits: number;
}

function newParticle(x: number): Particle {
  return { x, v: 0, history: new Float64Array(BINS), hits: 0 };
}

function record(p: Particle, x: number): void {
  const t = (x - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]);
  const bin = Math.floor(t * BINS);
  if (bin >= 0 && bin < BINS) {
    p.history[bin] += 1;
    p.hits += 1;
  }
}

export function SamplingDemo() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [beta, setBeta] = useState(0.75);
  const [running, setRunning] = useState(true);
  const [resetKey, setResetKey] = useState(0);
  // The animation loop reads beta every step; keeping it in a ref means moving
  // the slider does not tear down and restart the simulation, which would throw
  // away the histogram the picture is about.
  const betaRef = useRef(beta);
  useEffect(() => {
    betaRef.current = beta;
  }, [beta]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;

    const rng = mulberry32(20260920 + resetKey);
    const classical = newParticle(0.6);
    const quantum = newParticle(0.6);
    // The mean best position; with one particle it is simply its own best.
    const mbest = PBEST;

    let frame = 0;
    let steps = 0;

    /** Classical PSO with constriction, exactly as in `pso.py`. */
    const stepClassical = (): void => {
      const chi = 0.7298;
      const c = 2.05;
      const r1 = rng();
      const r2 = rng();
      classical.v =
        chi * (classical.v + c * r1 * (PBEST - classical.x) + c * r2 * (GBEST - classical.x));
      // Velocity clamp at a quarter of the domain width, the usual convention;
      // it is exactly this clamp that bounds the reachable set.
      const vmax = 0.25 * (DOMAIN[1] - DOMAIN[0]);
      classical.v = Math.max(-vmax, Math.min(vmax, classical.v));
      classical.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], classical.x + classical.v));
      record(classical, classical.x);
    };

    /** QPSO: sample from the delta-well density by inverting its CDF. */
    const stepQuantum = (): void => {
      const phi = rng();
      const p = phi * PBEST + (1 - phi) * GBEST;
      const u = Math.max(rng(), 1e-12);
      const width = betaRef.current * Math.abs(mbest - quantum.x);
      const jump = width * Math.log(1 / u);
      quantum.x = rng() < 0.5 ? p + jump : p - jump;
      // Positions outside the domain are real events, not errors; they are
      // reflected back so the histogram stays comparable, and reflection is
      // what the decoder's key clipping does in the actual implementation.
      if (quantum.x < DOMAIN[0]) quantum.x = DOMAIN[0] + (DOMAIN[0] - quantum.x) * 0.5;
      if (quantum.x > DOMAIN[1]) quantum.x = DOMAIN[1] - (quantum.x - DOMAIN[1]) * 0.5;
      quantum.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], quantum.x));
      record(quantum, quantum.x);
    };

    const draw = (): void => {
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const padL = 12;
      const padR = 12;
      const plotW = width - padL - padR;
      const toX = (v: number) => padL + ((v - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * plotW;

      const panels = [
        { p: classical, label: 'Classical PSO particle', color: '#57b487', y: 14 },
        { p: quantum, label: 'Quantum-behaved particle', color: '#5b93e6', y: height / 2 + 6 },
      ];
      const panelH = height / 2 - 30;

      for (const panel of panels) {
        const baseline = panel.y + panelH;

        ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
        ctx.fillStyle = '#94a1b2';
        ctx.textAlign = 'left';
        ctx.fillText(panel.label, padL, panel.y - 2);

        // Attractor references.
        for (const [value, label, colour] of [
          [PBEST, 'personal best', '#6b7889'],
          [GBEST, 'swarm best', '#c9a63f'],
        ] as [number, string, string][]) {
          ctx.strokeStyle = colour;
          ctx.setLineDash([3, 3]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(toX(value), panel.y + 2);
          ctx.lineTo(toX(value), baseline);
          ctx.stroke();
          ctx.setLineDash([]);
          if (panel === panels[0]) {
            ctx.fillStyle = colour;
            ctx.textAlign = 'center';
            ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
            ctx.fillText(label, toX(value), panel.y + 11);
          }
        }

        // Histogram of visited positions, normalised to the tallest bin so the
        // shape is comparable between the two panels.
        const max = Math.max(1, ...panel.p.history);
        const binW = plotW / BINS;
        for (let i = 0; i < BINS; i += 1) {
          const h = (panel.p.history[i] / max) * (panelH - 16);
          if (h <= 0) continue;
          ctx.fillStyle = panel.color;
          ctx.globalAlpha = 0.42;
          ctx.fillRect(padL + i * binW, baseline - h, Math.max(binW - 0.6, 0.8), h);
          ctx.globalAlpha = 1;
        }

        // Axis and the particle's current position.
        ctx.strokeStyle = '#35404e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, baseline + 0.5);
        ctx.lineTo(width - padR, baseline + 0.5);
        ctx.stroke();

        ctx.fillStyle = panel.color;
        ctx.beginPath();
        ctx.arc(toX(panel.p.x), baseline - 5, 4, 0, Math.PI * 2);
        ctx.fill();

        // How much of the domain the particle has ever reached.
        const visited = panel.p.history.reduce((a, b) => a + (b > 0 ? 1 : 0), 0);
        ctx.fillStyle = '#6b7889';
        ctx.font = '10px ui-monospace, SFMono-Regular, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(
          `${((100 * visited) / BINS).toFixed(0)} % of the domain reached in ${steps} steps`,
          width - padR,
          panel.y - 2,
        );
      }
    };

    const loop = () => {
      for (let i = 0; i < 3; i += 1) {
        stepClassical();
        stepQuantum();
        steps += 1;
      }
      draw();
      frame = requestAnimationFrame(loop);
    };

    if (running) {
      frame = requestAnimationFrame(loop);
    } else {
      draw();
    }

    return () => cancelAnimationFrame(frame);
  }, [running, resetKey]);

  return (
    <div>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: 288,
          display: 'block',
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
        }}
      />
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginTop: 10 }}>
        <button type="button" className="btn small" onClick={() => setRunning((r) => !r)}>
          {running ? 'Pause' : 'Resume'}
        </button>
        <button
          type="button"
          className="btn small"
          onClick={() => setResetKey((k) => k + 1)}
        >
          Restart
        </button>
        <label
          style={{ display: 'flex', alignItems: 'center', gap: 8, flex: '1 1 auto', fontSize: 12 }}
        >
          <span style={{ color: 'var(--text-dim)', whiteSpace: 'nowrap' }}>
            contraction–expansion β = {beta.toFixed(2)}
          </span>
          <input
            type="range"
            min={0.2}
            max={1.6}
            step={0.05}
            value={beta}
            onChange={(e) => setBeta(Number(e.target.value))}
          />
        </label>
      </div>
      <p style={{ fontSize: 11.5, color: 'var(--text-faint)', marginTop: 8, lineHeight: 1.55 }}>
        Both particles are stepped from the same seeded generator with the same
        personal and swarm bests. Raising β widens the sampling distribution and
        the quantum particle explores further; Sun et al.'s stability analysis
        puts the convergence threshold near β = 1.78, and above roughly that
        value the particle stops settling at all.
      </p>
    </div>
  );
}
