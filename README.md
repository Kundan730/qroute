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

## Results

1,520 benchmark runs over 17 instances, 9 solvers and 10 seeds, at an equal
twenty-second budget with every run pinned to one thread. No run returned an
infeasible solution. Friedman over 16 instances gives p = 1.25e-09.

| Solver | Mean gap to best known | Reached best known |
| --- | ---: | ---: |
| PyVRP, hybrid genetic search | 0.24% | 120/170 |
| **Quantum rotation gate (ours)** | 0.65% | 104/170 |
| Genetic algorithm | 0.66% | 105/170 |
| Ant colony optimisation | 0.77% | 100/170 |
| **Quantum particle swarm (ours)** | 0.85% | 100/170 |
| Simulated annealing | 0.93% | 81/170 |
| Classical particle swarm | 0.99% | 80/170 |
| Random multi-start | 1.21% | 77/170 |
| OR-Tools guided local search | 2.23% | 60/160 |

The comparison the problem statement asks for is the quantum-behaved swarm
against the classical one it derives from, under an identical decoder, local
search and budget. It comes out positive: 0.85% against 0.99%, p = 1.4e-05 over
170 paired runs. It also beats simulated annealing and multi-start local search.

It does **not** beat the genetic algorithm or ant colony optimisation, and it is
well behind PyVRP, a specialised state-of-the-art solver. Those results are on
every table rather than omitted.

Also established:

* 138 CVRPLIB and Solomon reference solutions re-evaluate to their published cost
  exactly, using each family's own distance convention.
* The route split is provably optimal: 300 random instances agree with brute-force
  enumeration to machine precision.
* Local search is correct on asymmetric (one-way street) cost matrices.
* The contraction coefficient standard in the QPSO literature is wrong for a
  random-key encoding by roughly a factor of twenty; correcting it is significant
  at p = 1.3e-04, reproduced twice.

The full record, including the negative results and a confound that invalidated
an earlier version of them, is in [docs/findings.md](docs/findings.md).

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
| `docs/` | formulation, algorithms, benchmark protocol, architecture, findings |

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
