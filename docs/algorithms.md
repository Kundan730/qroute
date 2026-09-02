# The Optimisation Engines

This document describes every search algorithm in the platform, what it actually
does, and what can honestly be claimed for it. It is deliverable 3 of the problem
statement, and it is deliberately written so that a reader can check the claims
against the code.

---

## 1. The shared pipeline

Every metaheuristic in `qroute` solves the same problem through the same three
stages, and only the first stage differs between them.

```
  search rule  ->  customer ordering  ->  optimal split  ->  local search
  (QPSO, PSO,      (a permutation of     (routes cut at    (2-opt, Or-opt,
   GA, SA, ACO)     1..n)                 the best places)  relocate, swap, 2-opt*)
```

This is a deliberate experimental design, not a convenience. If each algorithm had
its own decoder and its own improvement operators, a difference in results would
tell us nothing about the search rules, because it could equally come from a
better decoder. Sharing stages two and three means a difference in outcome is
attributable to stage one.

It also sets up the most important comparison in the project. `RandomRestart`
runs stages two and three on *random* orderings, with no search rule at all.
Any algorithm that cannot beat it is contributing nothing.

### 1.1 The encoding

A particle, chromosome or state is a vector of `n` real numbers, one per
customer, called random keys. Sorting the keys ascending gives a customer
ordering. Two properties make this the right representation here:

* Any real vector decodes to a valid ordering, so no operator can produce an
  invalid solution and no repair is needed at this level.
* Small changes to a key usually leave the ordering unchanged, and occasionally
  move one customer, so the search space is locally smooth in a way a raw
  permutation space is not.

### 1.2 The split

Given an ordering, the platform computes the *cheapest possible* way to cut it
into vehicle routes, by solving a shortest-path problem on a directed acyclic
graph whose arcs represent candidate routes. This is Prins' split procedure.

The consequence is worth stating plainly: **the search never has to guess where
routes begin and end.** It proposes an ordering, and the route boundaries are
placed optimally for that ordering, in `O(nL)` time where `L` is how many
customers fit in one vehicle. The implementation is verified against brute-force
enumeration on 300 random instances.

### 1.3 The local search

Decoded routes are improved by first-improvement descent over five neighbourhoods:
2-opt and Or-opt within a route, and relocate, swap and 2-opt\* between routes.
Two implementation choices matter:

* **Granularity.** Only moves involving a customer's nearest neighbours are
  considered, which reduces the neighbourhood from `O(n^2)` to `O(nK)`.
* **Don't-look bits.** A customer is examined only while a move involving it
  might still pay off. Without this the scan restarts after every improvement,
  which is what makes naive implementations unusable past a few hundred stops.

Improvements are written back into the keys, so the search inherits what local
search discovered instead of rediscovering it.

---

## 2. Quantum Particle Swarm Optimisation

### 2.1 Where the update rule comes from

In classical particle swarm optimisation a particle has a position and a
velocity. Because the velocity is bounded, so is the region the particle can
reach next, and the reachable region shrinks as the swarm converges. A converged
classical swarm therefore cannot escape a deep local optimum.

QPSO, introduced by Sun, Feng and Xu in 2004, removes the velocity. A particle is
modelled as a quantum particle bound in a delta potential well centred on a point
called the local attractor. In quantum mechanics such a particle has no
trajectory, only a probability density of being found somewhere. For the delta
well that density is a double exponential, and sampling from it by inverting the
cumulative distribution gives

```
x(t+1) = p  ±  (L/2) · ln(1/u),        u ~ U(0,1)
```

The support of that distribution is the entire real line. That is the precise,
checkable sense in which QPSO searches more globally than PSO: at every single
iteration, every particle has non-zero probability of appearing anywhere in the
search space. It costs one logarithm per dimension, and it removes the velocity
array entirely.

### 2.2 The complete update

For each particle `i` and each dimension `d`:

```
  phi   ~ U(0,1)
  p     = phi · pbest[i][d]  +  (1 - phi) · gbest[d]            local attractor
  mbest = (1/M) · sum over particles of pbest                   mean best position
  u     ~ U(0,1)
  x[i][d] = p  ±  beta · |mbest[d] - x[i][d]| · ln(1/u)         sign by fair coin
```

The characteristic length is taken from the swarm's **mean best position**. This
is the part that makes the algorithm self-regulating: while the personal bests
are spread out, the sampling width is large and the search explores; as they
cluster, the width shrinks on its own and the search intensifies. There is no
separate exploration schedule to tune.

### 2.3 The one parameter that matters

`beta`, the contraction-expansion coefficient, is the only critical parameter.
Sun et al.'s stability analysis shows the swarm converges when `beta` stays below
roughly 1.78, and the standard schedule decreases it linearly from 1.0 to 0.5
over the run. The implementation uses that schedule and asserts the bound.

One implementation detail is easy to get wrong and was caught here by testing:
the schedule must be indexed on whichever budget actually binds. Benchmark runs
are given a wall-clock budget with the iteration cap left effectively infinite.
A schedule that divided the current iteration by that cap would compute a
progress fraction of nearly zero for the whole run, hold `beta` at its starting
value, and never contract at all. The exploration-to-exploitation transition
would silently never happen.

### 2.4 What the parameter study found

A sweep over swarm size, contraction schedule, mutation and local-search policy
on four instances with three seeds and an equal budget produced mean gaps
between about 1.6% and 2.1%, against a run-to-run standard deviation near 1.0.
**No setting was significantly better than another.**

This is reported rather than buried because it is informative. Once an optimal
split and a strong local search are present, they do most of the work, and the
swarm rule mainly decides which orderings get refined. The defaults are the best
observed mean, not a claim of tuned superiority.

---

## 3. The quantum rotation-gate engine

The deliverable table asks for "quantum rotation / update rules", which refers to
a different family from QPSO: quantum-inspired evolutionary algorithms, due to
Han and Kim (2002). The platform implements this as a second, switchable engine.

A register of `m` qubits is held as amplitude pairs `(alpha, beta)` with
`alpha^2 + beta^2 = 1`, initialised to equal superposition. Observing the register
collapses it to a bit string, bit `i` being one with probability `beta_i^2`. The
register is then updated by applying a rotation to each qubit,

```
  [alpha']   [ cos t   -sin t ] [alpha]
  [beta' ] = [ sin t    cos t ] [beta ]
```

with the angle chosen from a lookup table that rotates each qubit toward the bit
value held by the better of the current and best-known solutions.

This is a classical simulation of qubit-like state. The register is a product
state and represents no entanglement, so it is a probabilistic model with a
convenient trigonometric update rule, not a quantum computation. The
documentation says so, and the published comparisons are not flattering:
Ma and Cheah's 2024 study on TSPLIB found conventional genetic algorithms beat
quantum-inspired ones and ran considerably faster. The engine is included because
it is what the deliverable asks for and because it is measured honestly alongside
everything else, not because it is expected to win.

---

## 4. Classical baselines

| Algorithm | Search rule | Why it is here |
| --- | --- | --- |
| PSO | velocity with inertia and constriction, `w = 0.729`, `c1 = c2 = 1.49445` | the direct ancestor of QPSO; the single most important comparison |
| GA | order crossover, swap and inversion mutation, tournament selection | population search without swarm dynamics |
| SA | relocate, swap, 2-opt and Or-opt moves, geometric cooling with a calibrated initial temperature | single-trajectory search; no population at all |
| ACO | pheromone on arcs, pseudo-random-proportional rule, min-max bounds | a different kind of memory: on arcs rather than on positions |
| Random restart | none | the control; anything that cannot beat this is contributing nothing |

All five use the same decoder, the same local search and the same stopping rule.

---

## 5. Reference solvers

These exist so that "near-optimal" means something measurable.

* **OR-Tools** routing with guided local search, a strong and widely used
  classical solver.
* **PyVRP**, a hybrid genetic search implementation that is close to the state of
  the art for this problem. The platform does not claim to beat it, and reports
  the gap to it.
* **CP-SAT**, which proves optimality on instances small enough to close, so on
  those the reported gap is a gap to a *proven* optimum rather than to a
  best-known value.
* **Held-Karp** dynamic programming for very small travelling-salesman cases, and
  valid lower bounds for bracketing large instances.

---

## 6. What is and is not claimed

Claimed, and measured:

* The shortest-path layer is exact.
* The route split is optimal for the ordering it is given, verified against brute
  force.
* Reported solutions are feasible, or the violation is stated explicitly.
* Every gap is measured against a published best-known solution or a proven
  optimum, over multiple seeds, with statistical tests.

Not claimed:

* No quantum speedup. The algorithms run on ordinary hardware and their
  asymptotic complexity is the same as their classical counterparts.
* No quantum hardware is used, and no result depends on any.
* No claim that a metaheuristic beats an exact method where one is applicable.
* No claim that QPSO is universally better than PSO. Where the measurements do
  not separate them, the report says so.
