# Benchmark Protocol

This document fixes how the platform is measured, before any result is quoted. It
exists because a metaheuristic can be made to look good in a dozen ways that are
not lies but are not evidence either, and the only defence is to decide the
protocol in advance and publish it.

---

## 1. What is measured against what

Every result is reported as a **gap** to a reference value:

```
gap % = 100 · (cost found − reference cost) / reference cost
```

The reference is one of two things, and the report always says which:

* A **proven optimum**, for instances the CP-SAT model closes within its budget.
  On these, "near-optimal" is a checkable statement.
* A **published best-known solution**, read from the instance's own reference
  file. Every one of the 138 reference solutions shipped with the platform has
  been re-evaluated by the platform's own objective function and reproduces its
  published cost exactly. This is not a formality: the two instance families use
  different and incompatible distance conventions, and getting either wrong
  shifts every gap by around half a percent.

| Family | Distance convention | Verified against |
| --- | --- | --- |
| CVRPLIB (sets A, B, P, X) | Euclidean, rounded to nearest integer | A-n32-k5, A-n80-k10, B-n31-k5, P-n16-k8, X-n101-k25, X-n502-k39, X-n1001-k43 |
| Solomon VRPTW | Euclidean, truncated to one decimal | C101, C201, R101, R112, R201, RC101 |

A solver that used unrounded distances would report costs about 0.5% higher than
the published optima and would appear worse than it is; one that compared
unrounded costs against rounded optima would appear better than it is.

---

## 2. Instance selection

Instances are chosen in tiers so that a claim can be matched to the strongest
available evidence.

| Tier | Instances | What it establishes |
| --- | --- | --- |
| 0 | P-n16-k8, P-n23-k8, A-n32-k5, A-n33-k5, A-n37-k5, A-n39-k5 | small enough for an exact solver, so the gap is to a proven optimum |
| 1 | A-n45-k7, A-n54-k7, A-n65-k9, A-n80-k10, B-n50-k7, B-n78-k10, P-n76-k4, P-n101-k4 | medium, proven-optimal published values |
| 2 | X-n101-k25, X-n153-k22, X-n200-k36 | scale, where the gap grows and the differences between algorithms appear |
| 3 | Solomon C101, R101, RC101, C201, R201 | time windows |
| 4 | Koramangala road network with generated demand | the real network under real traffic |

Small instances are kept in the set even though most algorithms solve them
optimally, because an algorithm that *fails* on them is disqualified regardless
of how it performs elsewhere.

---

## 3. Run protocol

* **Ten independent seeds** per algorithm and instance, derived from one master
  seed so that run *k* is identical regardless of execution order or worker count.
* **Equal wall-clock budget** for every algorithm on a given instance. Comparing
  by iteration count would be meaningless, since one iteration of a swarm of
  thirty is not comparable to one iteration of simulated annealing.
* **One thread per run.** Every solver process pins its thread-count environment
  variables to one. Without this, a solver whose library happens to parallelise
  would silently receive several times the CPU of its competitors and the
  wall-clock comparison would be void.
* **Compilation excluded.** The compiled kernels are warmed once before timing
  starts, so a run measures search, not the just-in-time compiler.
* **Deterministic solvers are labelled.** OR-Tools' search takes no seed. Running
  it ten times would produce ten identical rows and would overstate how much
  evidence the table contains, so it is marked as deterministic in the report.

---

## 4. Metrics

Per instance and algorithm:

* best, mean, median, standard deviation and worst cost across seeds
* the same as gaps to the reference
* number of runs that reached the reference value
* number of runs that ended feasible, reported separately from cost, because a
  cheap infeasible solution is not a solution
* median time and median iteration count to reach within 1% and 2% of the
  reference; a run that never got there is reported as "not reached" and is never
  silently dropped from the median
* evaluations per second, which separates "searches well" from "searches fast"

---

## 5. Statistical treatment

One seed proves nothing, and ten seeds prove something only if analysed properly.

* Two algorithms are compared with the **Wilcoxon signed-rank test** on runs
  paired by seed. It is non-parametric, which matters because gap distributions
  are bounded below and skewed, so a t-test's assumptions do not hold.
* All algorithms together are compared with the **Friedman test** on per-instance
  ranks, followed by **Holm's step-down correction** on the comparisons against
  the control. Correction is not optional: with eight algorithms there are
  twenty-eight pairwise comparisons, and at a 5% threshold more than one is
  expected to look significant purely by chance.
* Every significant result is reported with its **effect size** alongside the
  p-value. A p-value says a difference is unlikely to be noise; it does not say
  the difference is large enough to care about.
* Where the tests do not separate two algorithms, the report says they are not
  separated. That is a finding, not a gap in the evidence.

---

## 6. Reproducibility

Each run directory contains:

* the resolved configuration, including every parameter actually used
* the environment: Python version, platform, processor, core count, and the
  versions of numpy, scipy, numba, OR-Tools, networkx, osmnx, vrplib and PyVRP
* the git commit and whether the working tree was dirty
* one JSON line per run with its seed, cost, gap, feasibility, timings and
  convergence history

Regenerating a table is one command against the same configuration file.

---

## 7. Known limitations

Stated here rather than discovered by a reader:

* Ten seeds is enough to detect a moderate difference and not enough to detect a
  small one. Where the tests are inconclusive, that is the honest reading.
* A twenty-second budget is short. Published results for the strongest solvers
  use minutes per instance, so the absolute gaps here are larger than the
  literature's and are not directly comparable to it. What is comparable is the
  ranking of algorithms measured under identical conditions.
* The road-network instances use generated demand, because no public delivery
  demand dataset exists for these areas. The road network, its geometry and its
  travel times are real; the customers are not, and the report says so.
* Congestion is simulated from a calibrated profile rather than measured live,
  unless a traffic API key is supplied. The interface labels which source is in
  use, and never presents simulated data as observed.
