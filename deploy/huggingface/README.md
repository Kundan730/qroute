---
title: qroute
emoji: 🚦
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Quantum-inspired vehicle routing on real road networks
---

# qroute

Quantum-inspired route optimisation over real Indian road networks, with a live
traffic model and a reproducible benchmark against classical metaheuristics and
exact solvers.

Built for Smart India Hackathon 2026, problem statement 26137.

## What you can do here

* **Map** — pick a city extract, move the time-of-day slider and watch the road
  network recolour as congestion builds, generate a delivery instance on real
  intersections, and solve it. Block a road and re-optimise.
* **Solver** — run any of eleven solvers on a published benchmark instance and
  watch the search converge live.
* **Benchmark** — 1,520 runs over 17 instances and 9 solvers, with the Friedman
  test and Holm-corrected pairwise comparisons.
* **Method** — the update rules, written out, with what is and is not claimed.

## Honest positioning

The optimiser is a *classical* algorithm whose update rule is derived from the
quantum mechanics of a particle in a delta potential well. It runs on ordinary
hardware. No quantum speedup is claimed and no quantum hardware is involved.
Single-pair shortest paths are solved exactly by Dijkstra and A\*; a
metaheuristic is not used where an exact polynomial algorithm exists.

Measured results, including the ones that do not flatter the method, are in
`docs/findings.md` in the source repository.

## Note on this Space

The road networks are about 40 MB and are rebuilt on first use rather than
shipped in the image, so the first city you select takes a few seconds longer
than the rest.

**No API key is needed for anything.** The base maps come from Esri's free tile
services and the benchmark instances ship with the image. Traffic is simulated
from a calibrated time-of-day profile, and the interface says so rather than
presenting it as observed data. A live traffic feed is supported and optional:
setting `TOMTOM_API_KEY` as a Space secret switches it on, and without it the
platform falls back to simulation and labels itself accordingly.
