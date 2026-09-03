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
 * Drawing rules, so the figure survives the two places it will be seen:
 *
 *  - Every colour is read from the design tokens at draw time rather than
 *    written as a literal, with a fallback so a first paint before the
 *    stylesheet settles still draws something sane.
 *  - The contrast the figure is about is carried by colour AND by shape. The
 *    classical particle is neutral navy and the quantum one is the accent; the
 *    mean-best reference is a hollow square on a dashed rule and the swarm best
 *    is a filled triangle on a solid one, so a greyscale print of this panel
 *    still says which line is which.
 *  - The backing store is sized by devicePixelRatio, so the axis labels stay
 *    crisp on a retina display and on a projector.
 *  - If the reader has asked for reduced motion, the simulation is run to its
 *    settled state and drawn once instead of animating.
 */

import { useEffect, useRef, useState } from 'react';

import { V } from './Equation';

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
/** Steps drawn immediately when motion is suppressed. */
const STATIC_STEPS = 1500;

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

/**
 * The design tokens this figure draws with, resolved from the document element.
 * Canvas cannot take a `var()`, so the values are read once per mount; the
 * fallbacks are the same values global.css defines, for the case where the
 * first paint happens before the stylesheet has been applied.
 */
interface Tokens {
  classical: string;
  quantum: string;
  meanBest: string;
  swarmBest: string;
  axis: string;
  grid: string;
  label: string;
  dim: string;
  surface: string;
  panel: string;
  display: string;
  mono: string;
}

function readTokens(): Tokens {
  const fallback: Tokens = {
    classical: '#686f7c',
    quantum: '#1c273b',
    meanBest: '#424b5c',
    swarmBest: '#5b4fcf',
    axis: '#c2c8d4',
    grid: '#dce0e8',
    label: '#686f7c',
    dim: '#5a6376',
    surface: '#f6f8fb',
    panel: '#ffffff',
    display: "'Urbanist', sans-serif",
    mono: "'DM Mono', monospace",
  };
  if (typeof window === 'undefined') return fallback;
  const cs = getComputedStyle(document.documentElement);
  const get = (name: string, fb: string): string => cs.getPropertyValue(name).trim() || fb;
  return {
    classical: get('--navy-300', fallback.classical),
    quantum: get('--accent', fallback.quantum),
    meanBest: get('--navy-400', fallback.meanBest),
    swarmBest: get('--violet', fallback.swarmBest),
    axis: get('--border-strong', fallback.axis),
    grid: get('--border', fallback.grid),
    label: get('--navy-300', fallback.label),
    dim: get('--text-dim', fallback.dim),
    surface: get('--panel-alt', fallback.surface),
    panel: get('--panel', fallback.panel),
    display: get('--display', fallback.display),
    mono: get('--mono', fallback.mono),
  };
}

/** `letterSpacing` is well supported but not in every DOM typing; keep it soft. */
function setTracking(ctx: CanvasRenderingContext2D, value: string): void {
  (ctx as CanvasRenderingContext2D & { letterSpacing?: string }).letterSpacing = value;
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

export function SamplingDemo() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [beta, setBeta] = useState(0.75);
  const [reduced] = useState(prefersReducedMotion);
  const [running, setRunning] = useState(!reduced);
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

    const tk = readTokens();
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

    /** A hollow square: the mean-best reference. */
    const square = (x: number, y: number, s: number, colour: string): void => {
      ctx.strokeStyle = colour;
      ctx.fillStyle = tk.panel;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.rect(x - s / 2, y - s / 2, s, s);
      ctx.fill();
      ctx.stroke();
    };

    /** A filled triangle: the swarm-best reference. */
    const triangle = (x: number, y: number, s: number, colour: string): void => {
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.moveTo(x, y + s * 0.55);
      ctx.lineTo(x - s * 0.55, y - s * 0.45);
      ctx.lineTo(x + s * 0.55, y - s * 0.45);
      ctx.closePath();
      ctx.fill();
    };

    const draw = (): void => {
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (width <= 0 || height <= 0) return;
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = tk.surface;
      ctx.fillRect(0, 0, width, height);

      const padL = 16;
      const padR = 16;
      const plotW = width - padL - padR;
      const toX = (v: number) => padL + ((v - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * plotW;

      const blockH = height / 2;
      const panels = [
        {
          p: classical,
          label: 'Classical PSO particle',
          short: 'Classical',
          note: 'velocity-clamped',
          colour: tk.classical,
          top: 4,
        },
        {
          p: quantum,
          label: 'Quantum-behaved particle',
          short: 'Quantum-behaved',
          note: 'unbounded support',
          colour: tk.quantum,
          top: blockH + 2,
        },
      ];

      // What the heading line can hold is decided once, for both panels
      // together, so the two rows of a comparison never disagree about their
      // format. It is measured against the wider of the two headings and
      // against a full-width "100 %", so the line does not reflow when one
      // particle crosses 99 %. A step counter grows without bound, so it is the
      // first thing dropped; then the headings fall back to the one word that
      // carries the contrast. Nothing is ever allowed to run off the plot or
      // into its neighbour.
      const HEAD_GAP = 14;
      ctx.font = `700 9px ${tk.display}`;
      setTracking(ctx, '0.13em');
      const longHeadW = Math.max(...panels.map((p) => ctx.measureText(p.label.toUpperCase()).width));
      const shortHeadW = Math.max(...panels.map((p) => ctx.measureText(p.short.toUpperCase()).width));
      setTracking(ctx, '0em');
      ctx.font = `9px ${tk.mono}`;
      const line = width - padL - padR;
      const fullStatW = ctx.measureText(`100 % visited · ${steps} steps`).width;
      const pctStatW = ctx.measureText('100 % visited').width;
      let statMode: 'full' | 'pct' | 'none' = 'none';
      let useShort = true;
      if (longHeadW + HEAD_GAP + fullStatW <= line) {
        statMode = 'full';
        useShort = false;
      } else if (longHeadW + HEAD_GAP + pctStatW <= line) {
        statMode = 'pct';
        useShort = false;
      } else if (shortHeadW + HEAD_GAP + pctStatW <= line) {
        statMode = 'pct';
      } else if (longHeadW <= line) {
        useShort = false;
      }

      for (const panel of panels) {
        const labelBase = panel.top + 12;
        const plotTop = panel.top + 24;
        const baseline = panel.top + blockH - 30;
        const isTop = panel === panels[0];

        // --- panel heading -------------------------------------------------
        ctx.textBaseline = 'alphabetic';
        ctx.font = `700 9px ${tk.display}`;
        setTracking(ctx, '0.13em');
        ctx.fillStyle = tk.label;
        ctx.textAlign = 'left';
        const heading = (useShort ? panel.short : panel.label).toUpperCase();
        ctx.fillText(heading, padL, labelBase);
        const labelW = ctx.measureText(heading).width;
        setTracking(ctx, '0em');

        // How much of the domain the particle has ever reached, on the right of
        // the same line. The qualifier between them is dropped rather than
        // allowed to collide when the column is narrow.
        const visited = panel.p.history.reduce((a, b) => a + (b > 0 ? 1 : 0), 0);
        const pct = `${((100 * visited) / BINS).toFixed(0)} % visited`;
        const stat = statMode === 'full' ? `${pct} · ${steps} steps` : statMode === 'pct' ? pct : '';
        ctx.font = `9px ${tk.mono}`;
        ctx.fillStyle = tk.dim;
        const statW = stat ? ctx.measureText(stat).width : 0;
        const noteW = ctx.measureText(panel.note).width;
        if (padL + labelW + 10 + noteW + HEAD_GAP + statW < width - padR) {
          ctx.textAlign = 'left';
          ctx.fillText(panel.note, padL + labelW + 10, labelBase);
        }
        if (stat) {
          ctx.textAlign = 'right';
          ctx.fillText(stat, width - padR, labelBase);
        }

        // --- attractor references, distinguished by shape ------------------
        const refs: [number, string, string, boolean][] = [
          [PBEST, 'mean best', tk.meanBest, true],
          [GBEST, 'swarm best', tk.swarmBest, false],
        ];
        for (const [value, text, colour, dashed] of refs) {
          const x = toX(value);
          ctx.strokeStyle = colour;
          ctx.lineWidth = 1;
          ctx.setLineDash(dashed ? [3, 3] : []);
          ctx.beginPath();
          ctx.moveTo(Math.round(x) + 0.5, plotTop + 6);
          ctx.lineTo(Math.round(x) + 0.5, baseline);
          ctx.stroke();
          ctx.setLineDash([]);
          if (dashed) square(x, plotTop, 7, colour);
          else triangle(x, plotTop, 9, colour);
          if (isTop) {
            ctx.font = `600 9px ${tk.display}`;
            setTracking(ctx, '0.1em');
            ctx.fillStyle = colour;
            const caption = text.toUpperCase();
            // The left-hand label is set back from its marker, unless that
            // would push it off the plot, in which case it starts at the edge.
            if (dashed && x - 8 - ctx.measureText(caption).width < padL) {
              ctx.textAlign = 'left';
              ctx.fillText(caption, padL, plotTop + 3);
            } else {
              ctx.textAlign = dashed ? 'right' : 'left';
              ctx.fillText(caption, x + (dashed ? -8 : 9), plotTop + 3);
            }
            setTracking(ctx, '0em');
          }
        }

        // --- histogram of visited positions --------------------------------
        // Normalised to the tallest bin so the shape is comparable between the
        // two panels; the interesting difference is width, not height.
        const max = Math.max(1, ...panel.p.history);
        const binW = plotW / BINS;
        const span = baseline - plotTop - 12;
        ctx.fillStyle = panel.colour;
        for (let i = 0; i < BINS; i += 1) {
          const h = (panel.p.history[i] / max) * span;
          if (h <= 0) continue;
          // A bin that has been touched at all must be visible: the tails are
          // the whole argument, and a half-pixel bar would erase them.
          ctx.fillRect(padL + i * binW, baseline - Math.max(h, 1.5), Math.max(binW - 0.7, 0.9), Math.max(h, 1.5));
        }

        // --- axis, ticks and the particle ----------------------------------
        ctx.strokeStyle = tk.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, baseline + 0.5);
        ctx.lineTo(width - padR, baseline + 0.5);
        ctx.stroke();

        ctx.font = `9px ${tk.mono}`;
        ctx.fillStyle = tk.dim;
        // Half-unit ticks are dropped rather than allowed to collide: an axis
        // reading "-1.00.5 0 0.51.0" is worse than one reading -1, 0, 1.
        const tickLabelW = ctx.measureText('−0.5').width;
        const ticks = plotW / 4 >= tickLabelW + 8 ? [-1, -0.5, 0, 0.5, 1] : [-1, 0, 1];
        for (const t of ticks) {
          const x = Math.round(toX(t)) + 0.5;
          ctx.strokeStyle = tk.axis;
          ctx.beginPath();
          ctx.moveTo(x, baseline + 1);
          ctx.lineTo(x, baseline + 4);
          ctx.stroke();
          ctx.textAlign = t === DOMAIN[0] ? 'left' : t === DOMAIN[1] ? 'right' : 'center';
          // A true minus, so the ticks match the sign used in the equations.
          ctx.fillText(t === 0 ? '0' : t.toFixed(1).replace('-', '−'), x, baseline + 14);
        }

        // The particle itself, ringed in the panel colour so it stays legible
        // where it sits on top of its own histogram.
        const px = toX(panel.p.x);
        ctx.beginPath();
        ctx.arc(px, baseline - 6, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = panel.colour;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = tk.panel;
        ctx.stroke();
      }

      // The rule between the two panels, so they read as a comparison.
      ctx.strokeStyle = tk.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, Math.round(blockH) + 0.5);
      ctx.lineTo(width - padR, Math.round(blockH) + 0.5);
      ctx.stroke();
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

    if (reduced) {
      // Motion is suppressed: run the simulation to its settled state and draw
      // the result once, so the reader still sees the thing being argued.
      for (let i = 0; i < STATIC_STEPS; i += 1) {
        stepClassical();
        stepQuantum();
        steps += 1;
      }
      draw();
    } else if (running) {
      frame = requestAnimationFrame(loop);
    } else {
      draw();
    }

    // Keep the backing store matched to the element when the column resizes.
    const observer =
      typeof ResizeObserver === 'function' ? new ResizeObserver(() => draw()) : null;
    observer?.observe(canvas);

    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
    };
  }, [running, resetKey, reduced]);

  return (
    <div>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: 300,
          display: 'block',
          background: 'var(--panel-alt)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
        }}
      />
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 10 }}>
        {!reduced && (
          <button type="button" className="btn small" onClick={() => setRunning((r) => !r)}>
            {running ? 'Pause' : 'Resume'}
          </button>
        )}
        <button type="button" className="btn small" onClick={() => setResetKey((k) => k + 1)}>
          Restart
        </button>
        <label
          style={{ display: 'flex', alignItems: 'center', gap: 8, flex: '1 1 auto', fontSize: 12 }}
        >
          <span
            style={{
              color: 'var(--text-dim)',
              whiteSpace: 'nowrap',
              fontFamily: 'var(--mono)',
              fontVariantNumeric: 'tabular-nums',
              fontSize: 11.5,
            }}
          >
            β = {beta.toFixed(2)}
          </span>
          <input
            type="range"
            min={0.2}
            max={1.6}
            step={0.05}
            value={beta}
            onChange={(e) => setBeta(Number(e.target.value))}
            aria-label="contraction-expansion coefficient beta"
          />
        </label>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 9, lineHeight: 1.6 }}>
        Both particles are stepped from the same seeded generator with the same
        personal and swarm bests. The hollow square on the dashed rule is the
        particle's personal best, which with a swarm of one is also the mean
        best <V>m</V>; the filled triangle on the solid rule is the swarm best{' '}
        <V>g</V>. Raising β widens the sampling distribution and the quantum
        particle explores further; Sun et al.'s stability analysis puts the
        convergence threshold near β = 1.78, and above roughly that value the
        particle stops settling at all.
        {reduced && (
          <>
            {' '}
            Motion is switched off because this system asks for reduced motion,
            so the figure shows the state after {STATIC_STEPS} steps.
          </>
        )}
      </p>
    </div>
  );
}
