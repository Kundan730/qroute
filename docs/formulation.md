# Mathematical Formulation

This document is deliverable 2 of the problem statement: the complete optimisation
model of the traffic routing problem, its decision variables, its objective and its
constraints. Section 6 maps every constraint onto the code that enforces it, so the
model and the implementation can be checked against each other.

---

## 1. The network

The transportation network is a weighted directed graph

$$G = (V, A)$$

where $V$ is the set of road intersections and $A \subseteq V \times V$ the set of
road segments. A directed graph is required rather than an undirected one because
real road networks contain one-way streets: in the Koramangala network used for the
demonstration, travel time from $u$ to $v$ frequently differs from $v$ to $u$, and
for many pairs one direction does not exist at all.

Each arc $(u,v) \in A$ carries

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $\ell_{uv}$ | length of the road segment | metres |
| $t^0_{uv}$ | free-flow travel time | seconds |
| $c_{uv}$ | capacity | vehicles per hour |
| $t_{uv}(\tau)$ | travel time when entering at time $\tau$ | seconds |

The dependence of $t_{uv}$ on $\tau$ is what makes this a *traffic* routing problem
rather than a static one. Section 4 defines it.

### 1.1 Reduction to the customer graph

The routing problem is not posed over all of $V$. A delivery instance names a depot
$0$ and customers $1,\dots,n$, all of which are nodes of $G$. What the routing model
needs is the cost of travelling between those $n+1$ locations, which is the cost of
the *shortest path* between them in $G$:

$$d_{ij} = \min_{P \in \mathcal{P}(i,j)} \sum_{(u,v) \in P} \ell_{uv},
\qquad
t_{ij} = \min_{P \in \mathcal{P}(i,j)} \sum_{(u,v) \in P} t_{uv}$$

These are computed exactly, by Dijkstra's algorithm run from each of the $n+1$
locations. This is worth being explicit about, because it is the honest division of
labour in the whole platform: **the shortest-path subproblem is solved exactly, in
polynomial time, and no metaheuristic is used for it.** The hard part is not finding
a path between two points; it is deciding which customers each vehicle serves and in
what order, and that is where the search algorithms operate.

The result is a complete directed graph on $\{0,1,\dots,n\}$ with arc costs $d_{ij}$
and $t_{ij}$, which is in general **asymmetric**, since it inherits the one-way
structure of the road network.

---

## 2. Decision variables

$$
x_{ij} \in \{0,1\}, \quad (i,j) \in A' \qquad
\text{arc } i \to j \text{ is traversed by some vehicle}
$$

$$
f_{ij} \ge 0, \quad (i,j) \in A' \qquad
\text{load carried on arc } i \to j
$$

$$
T_i \ge 0, \quad i \in \{0,\dots,n\} \qquad
\text{time at which service starts at } i
$$

where $A'$ is the arc set of the complete customer graph.

This is the **two-index** formulation. A three-index formulation with $x_{ijk}$ per
vehicle $k$ is more natural to read but has $K$ times as many variables and, for a
homogeneous fleet, an enormous symmetry group: any permutation of vehicle labels
gives an equivalent solution, which cripples branch-and-bound. The two-index model
with flow variables avoids that entirely.

---

## 3. Objective

The problem statement asks to minimise travel time, distance and congestion
together, so the objective is a weighted sum:

$$
\min \; Z = \sum_{(i,j) \in A'} \Big( w_T \, t_{ij} + w_D \, d_{ij} + w_C \, \gamma_{ij} \Big) x_{ij}
\;+\; w_K \sum_{j} x_{0j}
$$

with weights $w_T, w_D, w_C, w_K \ge 0$ exposed in the user interface.

### 3.1 What "congestion" means, precisely

The word names two different quantities in this system, and both are useful, so
they are defined here and each is used consistently in its own place.

**Delay fraction**, the quantity the objective uses:

$$\lambda_e = 1 - \frac{t^0_e}{t_e} \in [0, 1)$$

This is the fraction of the time spent on arc $e$ that is *lost* to congestion.
A link running at free flow scores 0; one taking twice as long as free flow
scores 0.5. It is bounded, which matters because it is aggregated along a path:
the congestion of the shortest path from $i$ to $j$ is the travel-time-weighted
mean of $\lambda_e$ over the road segments that path uses,

$$\gamma_{ij} = \frac{\sum_{e \in P_{ij}} \lambda_e \, t_e}{\sum_{e \in P_{ij}} t_e} \in [0, 1]$$

so $\gamma_{ij}$ is the fraction of that journey's duration spent in traffic. The
congestion term of the objective, $w_C \sum \gamma_{ij} t_{ij} x_{ij}$, is then
literally the number of vehicle-seconds the plan spends sitting in congestion,
which is what makes it addable to the time term.

**Delay ratio**, the quantity the interface colours by:

$$\delta_e = \frac{t_e - t^0_e}{t^0_e} \in [0, \infty)$$

This is delay expressed as a multiple of free-flow time, and it is unbounded: a
blocked arterial can reach 30 or more. It is the right choice for a colour scale
because it separates a badly congested road from a merely slow one, which the
bounded measure compresses (both sit near 1). The map's bands, free flow, light,
moderate, heavy and severe, are cut on $\delta_e$ at 0.10, 0.35, 0.75 and 1.50.

The two are monotonically related, $\lambda = \delta / (1 + \delta)$, so neither
is more correct; they answer different questions. The objective needs an additive,
bounded quantity and uses $\lambda$. The display needs a discriminating one and
uses $\delta$. Code that reads `edge_congestion` on the road network is getting
$\lambda$; code that calls `congestion_level` in the traffic model is getting
$\delta$.

The last term prices the fleet. With $w_K > 0$ the model trades distance against the
number of vehicles used, which is the trade-off a logistics operator actually faces.

On the published benchmark instances the weights are set to $w_D = 1$ and all others
zero, because that is the objective under which every best-known solution in the
literature was computed. Comparing against those values under any other objective
would be meaningless.

---

## 4. Constraints

### 4.1 Visit each customer exactly once

$$\sum_{j : (i,j) \in A'} x_{ij} = 1 \qquad \forall i \in \{1,\dots,n\}$$
$$\sum_{i : (i,j) \in A'} x_{ij} = 1 \qquad \forall j \in \{1,\dots,n\}$$

Every customer has exactly one successor and one predecessor.

### 4.2 Fleet size and depot balance

$$\sum_{j} x_{0j} \le K, \qquad \sum_{j} x_{0j} = \sum_{i} x_{i0}$$

At most $K$ vehicles leave the depot, and as many return as left.

### 4.3 Capacity, and elimination of subtours

Capacity is enforced through a single-commodity flow. Let $q_i$ be the demand of
customer $i$ and $Q$ the vehicle capacity:

$$\sum_{j} f_{ji} - \sum_{j} f_{ij} = q_i \qquad \forall i \in \{1,\dots,n\}$$

$$q_j \, x_{ij} \;\le\; f_{ij} \;\le\; (Q - q_i)\, x_{ij} \qquad \forall (i,j) \in A'$$

The first line says each customer consumes exactly its demand from the load passing
through it. The second links flow to arc usage and bounds it by capacity.

These constraints do double duty. They enforce capacity, and they also eliminate
subtours: a cycle disconnected from the depot would have no source for its flow, so
the flow-conservation equations would be infeasible. This is the Gavish–Graves
single-commodity flow formulation. The alternative Miller–Tucker–Zemlin constraints
are smaller but have a substantially weaker linear relaxation, which matters a great
deal to an exact solver. Both are implemented in `qroute/exact/milp.py` so the
difference can be demonstrated rather than merely asserted.

### 4.4 Time windows

Each customer $i$ has a service window $[a_i, b_i]$ and a service duration $s_i$:

$$T_i + s_i + t_{ij} - M_{ij}(1 - x_{ij}) \;\le\; T_j
\qquad \forall (i,j) \in A', \; j \neq 0$$

$$a_i \;\le\; T_i \;\le\; b_i \qquad \forall i \in \{1,\dots,n\}$$

The first is the standard big-$M$ time propagation: if arc $(i,j)$ is used, service
at $j$ cannot start before service at $i$ finished plus the travel time. A vehicle
arriving before $a_i$ waits, which is permitted; arriving after $b_i$ is not.
Choosing $M_{ij} = \max(0,\; b_i + s_i + t_{ij} - a_j)$ gives the tightest valid
value.

### 4.5 Route duration

$$\sum_{(i,j) \in R} \big( t_{ij} + s_j \big) \;\le\; D \qquad \text{for every route } R$$

A driver shift limit, optional.

### 4.6 Edge flow capacity (congestion-aware extension)

The constraints above route each vehicle as if the others did not exist. When many
vehicles are dispatched at once they interact, because they compete for the same
road capacity. Let $\mathcal{E}$ be the set of *road* arcs, and let $\rho_e$ count
the vehicles of this fleet whose chosen path uses road arc $e$:

$$\rho_e = \sum_{(i,j)} \mathbb{1}[\, e \in P_{ij} \,] \; x_{ij}, \qquad
\rho_e \;\le\; \kappa_e$$

This is the flow constraint in its congestion-aware form, and it is what stops a
solver from routing an entire fleet down one attractive arterial. It is treated as
an extension rather than part of the core model, and it is stated here as a bound
that is checked and penalised, because the fully coupled version, where each
vehicle's travel time depends on the others' choices, is a constrained system
optimum problem and is considerably harder than the VRP itself.

---

## 5. Dynamic weights: how traffic enters the model

Travel time on a road arc is a function of how loaded it is. The platform uses the
Bureau of Public Roads volume-delay function:

$$t_e(\tau) = t^0_e \left( 1 + \alpha \left( \frac{v_e(\tau)}{c_e} \right)^{\beta} \right), \qquad \alpha = 0.15, \; \beta = 4$$

with the volume $v_e(\tau)$ produced by a time-of-day profile, a road-class
sensitivity and, when an incident is active, a reduced capacity $c_e$.

Because $\beta = 4$, travel time is nearly flat while a road is below capacity and
rises very steeply past it. This is the behaviour that makes congestion interesting
to optimise around: the difference between a road at 90% and 110% of capacity is far
larger than the difference between 40% and 60%.

**Recomputation, not rebuilding.** When traffic changes, the arc costs $t_{ij}$ of
the customer graph change, but the road graph itself does not. The dynamic weight
update therefore writes new values into the existing sparse adjacency array and
recomputes only the affected shortest paths. The optimisation is then warm started
from the previous solution rather than restarted.

**A note on time dependence and correctness.** If travel time varies with departure
time, a naive implementation multiplies the free-flow time by the factor for the
departure period. That is wrong, and importantly so: it allows a vehicle leaving
later to arrive earlier, which violates the first-in-first-out property, and once
FIFO fails, Dijkstra's algorithm is no longer guaranteed to return a shortest path.
The platform instead advances a vehicle along an arc at the speed prevailing in each
period it passes through, which preserves FIFO and keeps the exact shortest-path
claim honest.

---

## 6. Where each constraint lives in the code

| Constraint | Enforced by | File |
| --- | --- | --- |
| Visit once (4.1) | the permutation encoding; every solution is a permutation of customers | `algorithms/decoder.py`, checked in `core/types.py::Solution.validate` |
| Fleet size (4.2) | fleet-limited split, then a penalty if it must be exceeded | `algorithms/kernels.py::_split_fleet` |
| Capacity (4.3) | the split refuses to extend a route past capacity; residual excess is penalised | `algorithms/kernels.py`, `algorithms/penalty.py` |
| Time windows (4.4) | arrival times propagated during evaluation; lateness penalised | `problems/instance.py::evaluate` |
| Route duration (4.5) | accumulated per route and penalised | `problems/instance.py::evaluate` |
| Edge flow (4.6) | counted over chosen paths and reported as a violation | `problems/instance.py` (`edge_load_violation`) |
| Objective weights (3) | folded into a single arc-cost matrix once per traffic update | `problems/instance.py::_build_cost_matrix` |

---

## 7. Complexity, stated honestly

The vehicle routing problem is NP-hard: it contains the travelling salesman problem
as the special case $K=1$, $Q = \sum_i q_i$. No algorithm in this platform changes
that, and none is claimed to.

What the platform does claim, and measures:

* The **shortest-path** layer is solved exactly in $O((|V| + |A|)\log|V|)$ per source.
* Given a fixed customer ordering, the partition of that ordering into routes is
  solved **to optimality** in $O(nL)$ time, where $L$ is the number of customers that
  fit in one vehicle. So the metaheuristic searches only over orderings; it never has
  to guess where routes should be cut.
* The search itself is a heuristic. Its output is compared against proven optima
  where an exact method can close the instance, and against published best-known
  solutions otherwise, with the gap reported for every run.
