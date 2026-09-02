# Platform Architecture

How the pieces fit together, and why the boundaries are where they are.

```
                         React front end (map, solver, benchmark, method)
                                        |
                                  HTTP + Server-Sent Events
                                        |
                         FastAPI service  (qroute/api)
                          |            |             |
              ------------+            |             +-------------
              |                        |                          |
      road network layer        optimisation engines        benchmark layer
      (qroute/graph)            (qroute/algorithms)         (qroute/benchmark)
              |                        |                          |
        traffic model            problem model               exact methods
       (qroute/traffic)        (qroute/problems)      (qroute/exact, qroute/baselines)
```

## The layers

**Problem model** (`qroute/problems`) owns the mathematical formulation: a single
`Instance` type that covers the capacitated problem, the time-window variant and
the time-dependent road-network case, because they differ only in which optional
arrays are present. It also owns the benchmark loaders and, importantly, the
*reference* implementation of the objective. Every reported cost is recomputed by
that reference implementation, never taken from a solver's own accounting, so a
bug in a compiled kernel cannot inflate a result.

**Road network layer** (`qroute/graph`) turns an OpenStreetMap extract into
something a solver can use: a compressed sparse adjacency for fast shortest
paths, imputed free-flow speeds by road class, extraction of the largest strongly
connected component, and exact Dijkstra and A\*. Its single most important
operation is `update_weights`, which rewrites edge travel times in place in time
proportional to the number of edges, with no graph rebuild. That is what makes
the dynamic weight update fast enough to be interactive.

**Traffic model** (`qroute/traffic`) supplies those weights: a volume-delay
function, time-of-day profiles, an incident queue, and a simulator that owns the
clock. Live traffic enters through a `TrafficSource` interface, so the simulated
and live cases are interchangeable and the interface can state which is in use.

**Optimisation engines** (`qroute/algorithms`) are the search rules, sharing one
decoder and one local search so that comparisons between them are meaningful.

**Exact methods and reference solvers** (`qroute/exact`, `qroute/baselines`)
provide ground truth. They are deliberately separate from the engines: they are
not competitors in the same class, they are the yardstick.

**Benchmark layer** (`qroute/benchmark`) runs experiments reproducibly and
analyses them statistically. It has no knowledge of any particular algorithm; it
dispatches by name and treats every solver as a function from an instance to a
result.

**Service and interface** (`qroute/api`, `frontend`) expose all of it. The API
runs solvers in worker processes and streams progress, so a long optimisation
never blocks the server and can be cancelled.

## Boundaries worth explaining

**Why the solvers do not know about roads.** The optimisation engines see only a
cost matrix. The road network and the traffic model are what produce that matrix.
This keeps the engines testable against standard benchmark instances, and it
means the same engine works unchanged for any graph-shaped routing problem.

**Why the compiled kernels are separate from the model.** The inner loops are
compiled and operate on flat arrays, which makes them fast and unreadable. The
readable reference implementation lives in the problem model and the tests check
one against the other, so the code that is easy to verify and the code that is
fast are both present and are kept in agreement.

**Why runs happen in separate processes.** Partly so the API stays responsive and
runs can be cancelled, and partly for measurement: a benchmark is only fair if
every solver gets the same CPU, which means one thread each, which means process
isolation.

**Why the reference implementation always has the last word.** A solver reports
whatever it computed. Before that number is shown to anyone, the routes are
re-evaluated by the model and validated for completeness. It costs microseconds
and it makes an entire category of error impossible to hide.
