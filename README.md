# qroute — Quantum-Inspired Intelligent Traffic Route Optimisation

Reference implementation for **Smart India Hackathon 2026, Problem Statement 26137**
(Egreen Quanta, *Transportation & Logistics*).

`qroute` solves large vehicle routing problems on real road networks using
**Quantum Particle Swarm Optimisation** and a second quantum-inspired engine based
on **quantum rotation gates**, and it measures the result honestly against
classical metaheuristics, a state-of-the-art solver and exact methods.

## What it is

A transportation network is modelled as a weighted directed graph whose edge
weights change with traffic. On top of that graph the platform solves the
capacitated vehicle routing problem, with optional time windows, and answers
single-pair shortest-path queries exactly.

The optimiser is a memetic quantum-behaved particle swarm: particles are
continuous random keys, sorted into a customer ordering, cut into vehicle routes
by an optimal split, and refined by granular local search.

## Honest positioning

* QPSO is a **classical** algorithm whose update rule is derived from the
  quantum mechanics of a particle in a delta potential well. It runs on ordinary
  hardware. No quantum speedup is claimed and no quantum hardware is used.
* Single-pair shortest paths are solved **exactly** by Dijkstra and A\*. A
  metaheuristic is not used where an exact polynomial algorithm exists.
* Every benchmark number is a gap against a published best-known solution, taken
  from the instance's own reference file, over multiple seeds with statistical
  tests.

## Status

Verified so far:

* 138 CVRPLIB and Solomon reference solutions re-evaluate to their published cost
  exactly, using each family's own distance convention.
* The route split is provably optimal: 300 random instances agree with brute-force
  enumeration to machine precision.
* Local search is correct on asymmetric (one-way street) cost matrices.

## Layout

| Path | Contents |
| --- | --- |
| `qroute/problems` | instance model, objective and constraints, benchmark loaders |
| `qroute/algorithms` | QPSO, quantum rotation-gate engine, classical baselines, compiled kernels |
| `qroute/graph` | road network handling and travel-time matrices |
| `qroute/traffic` | congestion model and the dynamic weight-update engine |
| `qroute/exact` | exact methods used as ground truth |
| `qroute/benchmark` | reproducible experiments, statistics, reporting |
| `qroute/api` | FastAPI service |
| `qroute/cli` | command line interface |
| `frontend/` | React map and dashboard |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Data

Benchmark instances (CVRPLIB sets A, B, P, X and the Solomon VRPTW set) and three
Indian city road graphs are held under `data/`.

## Licence

MIT. Benchmark instances belong to their original authors. Road network data is
© OpenStreetMap contributors, ODbL.
