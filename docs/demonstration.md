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

> **These figures are being re-measured and are not currently quoted.** An audit
> found that the numbers previously printed here came from an ad-hoc script
> rather than from the `qroute osm demo` command section 7 tells you to run, and
> did not reproduce from it. Two further statements in this section were wrong on
> their face and have been removed: the disruption was described as a blocked
> lane when the code injects a speed reduction, and the resulting slowdown was
> given as a factor of 49, which a speed multiplier of 0.1 cannot produce because
> it caps the increase at ten-fold. The correct description of the mechanism is
> below; the table returns when it has been regenerated from the documented
> command with every flag pinned.

What the demonstration actually does: it selects the road segments the current
plan uses most heavily, applies a **slowdown event** that cuts their speed to a
tenth for ninety minutes, and applies a milder reduction to adjacent links.
This is a speed reduction, not a capacity reduction, so the volume-delay
function is not what produces the effect and the travel time on an affected
segment rises by at most a factor of ten.

The platform then rebuilds the travel-time matrices under the new weights,
prices the existing plan against them, and re-optimises both from scratch and
from a warm start seeded with the previous solution. The command prints all
three costs and states plainly which start won, including when the warm start
loses.

## 6. What a viewer should take away

1. The network is real, the shortest paths are exact, and the traffic model is a
   standard one with published coefficients.
2. The optimiser produces a feasible plan, and when it cannot, it says so rather
   than quietly reporting an impossible one.
3. When the road network changes, the plan can be re-optimised in seconds, and
   the command reports honestly whether warm starting beat a cold restart on
   that run rather than assuming it did.
4. Every claim on screen is a number the platform computed and can recompute.

## 7. Reproducing it

```bash
qroute osm demo --network bengaluru_koramangala --hour 9 --customers 40
```
