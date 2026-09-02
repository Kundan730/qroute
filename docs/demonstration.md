# The Demonstration

Deliverable 5 asks for a demonstration on "at least one realistic urban network
showing near-optimal routes under varying traffic conditions". This document is
the script for that demonstration and the record of what it actually produced.

All figures below were measured on the development machine, an Apple M4 with ten
cores and 16 GB of memory, running single-threaded unless stated.

---

## 1. The network

`data/osm/bengaluru_koramangala.graphml`, a four-kilometre radius of Koramangala,
Bengaluru, taken from OpenStreetMap.

| Property | Value |
| --- | --- |
| Nodes (road intersections) | 13,343 after taking the largest strongly connected component |
| Directed edges (road segments) | 34,266 |
| Total road length | 2,050 km |
| Load time from disk | about 10 seconds |

Two further networks are bundled for scalability: Connaught Place in New Delhi
(7,445 nodes) and Anna Nagar in Chennai (13,132 nodes).

Free-flow speeds are imputed by road class, because Indian OpenStreetMap data
carries almost no speed limit tags and the library defaults would otherwise put
residential streets at 10 km/h. The imputation table is a module-level constant so
it can be audited, and the slide states it is an assumption.

---

## 2. Traffic varies through the day

The volume-delay function is the Bureau of Public Roads form with the standard
coefficients, driven by an hour-of-day profile with per-road-class sensitivity.
Measured peak-to-free-flow travel time ratios at 09:00:

| Road class | Segments | Peak / free-flow |
| --- | --- | --- |
| Trunk | 159 | 1.65 |
| Primary | 575 | 1.55 |
| Secondary | 1,176 | 1.47 |
| Tertiary | 3,388 | 1.27 |
| Residential | 27,705 | 1.06 |
| Whole network | 34,266 | 1.10 |

Arterials slow by roughly half at peak while residential streets barely change,
which is the right shape: congestion concentrates on the roads that carry the
traffic. The network-wide figure is low only because four fifths of the segments
are residential. This is worth saying out loud during the demonstration, because
a single network-wide number would otherwise look like the model was doing
nothing.

---

## 3. The routing instance

Forty delivery stops and a depot are placed on real intersections, with demand
generated reproducibly from a seed. The travel-time matrix between them is
computed by exact Dijkstra over the road graph.

| Property | Value |
| --- | --- |
| Customers | 40 |
| Vehicle capacity | 103 |
| Total demand | 511, so at least five vehicles are needed |
| Matrix build | about 2 seconds for 41 by 41 over 13k nodes |
| Travel times | 121 s to 1,768 s between stops |

The customers are generated, and the demonstration says so. The road network, its
geometry and its travel times are real.

---

## 4. Optimise under morning traffic

Running QPSO for ten seconds at 09:00:

| Result | Value |
| --- | --- |
| Total vehicle time | 1,986 vehicle-minutes |
| Routes | 5, the minimum possible for this demand |
| Route loads | 103, 103, 103, 100, 102 against a capacity of 103 |
| Feasible | yes |

The load profile is worth pausing on. Five vehicles of capacity 103 carry 515
units and the customers demand 511, so there are four units of slack in the whole
plan. Finding a five-vehicle solution at all requires very tight packing.

This is also where the platform's honesty machinery shows itself. The search
initially returned a plan that was two units over capacity on one vehicle,
because the search is deliberately allowed to cross infeasible ground where the
good moves are. A repair pass escalates the constraint penalties and re-optimises
until the plan is dispatchable, which cost 5% more vehicle time. The platform
reports the feasible, more expensive plan, not the cheaper impossible one.

---

## 5. Disrupt it

One lane is blocked on fifteen road segments that the current routes actually
use, for ninety minutes. Travel time on those segments rises by a factor of 49 on
average, because the volume-delay function grows with the fourth power of
saturation and those links were already near capacity.

| Step | Result |
| --- | --- |
| Travel-time matrix rebuilt | 0.48 seconds |
| Cost of keeping the existing plan | 2,105 vehicle-minutes, 6.0% worse |
| Re-optimised cold, 2 seconds | 2,074 vehicle-minutes, recovers 1.4% |
| Re-optimised **warm**, 2 seconds | 2,030 vehicle-minutes, recovers 3.6% |

The warm start is the point. Seeding the swarm from the previous solution and
letting it adapt recovers more than twice as much as starting from scratch in the
same two seconds. That is the argument for keeping an optimiser resident and
warm in a live traffic system rather than re-solving from nothing on every event.

---

## 6. What a viewer should take away

1. The network is real, the shortest paths are exact, and the traffic model is a
   standard one with published coefficients.
2. The optimiser produces a feasible plan, and when it cannot, it says so rather
   than quietly reporting an impossible one.
3. When the road network changes, the plan can be repaired in about two seconds,
   and warm starting is measurably better than restarting.
4. Every claim on screen is a number the platform computed and can recompute.

## 7. Reproducing it

```bash
qroute osm demo --network bengaluru_koramangala --hour 9 --customers 40
```
