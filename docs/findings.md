# What the Measurements Actually Showed

This document records the experimental findings, including the ones that do not
flatter the method. It exists because the alternative, reporting only the
comparisons that came out well, is how this literature acquired its credibility
problem, and because a panel that probes a claim should find the answer already
written down.

All experiments use a matched evaluation budget: every variant decodes and
evaluates the same number of candidate solutions, so the only difference is
which candidates get evaluated.

---

## 1. The split and the local search do most of the work

Six instances, five seeds, 1,200 evaluations each.

| Configuration | Mean gap to best known |
| --- | --- |
| Random orderings, no local search | 150.3% |
| Random orderings, with local search | 1.4% |
| QPSO, no local search | 1.8% |
| QPSO, with local search | 1.8% |

Two readings, and both are true.

The encouraging one: the quantum-inspired swarm is a genuinely effective
optimiser in its own right. Given the same optimal split and the same number of
evaluations, random sampling of orderings lands 150% above the best-known cost
while QPSO lands 1.8% above it. The search rule is doing real work.

The uncomfortable one: adding local search to random sampling achieves the same
thing, and the two do not compose. QPSO with local search is no better than
random restarts with local search.

---

## 2. Multi-start local search is not beaten by the swarm

Four instances, five seeds, four budgets, paired by instance and seed.

| Evaluations | Multi-start + local search | QPSO + local search | Wilcoxon |
| --- | --- | --- | --- |
| 900 | 2.02% | 2.51% | multi-start better, p = 0.024 |
| 3,000 | 1.39% | 2.11% | multi-start better, p = 0.0015 |
| 9,000 | 1.29% | 1.54% | no significant difference, p = 0.10 |
| 24,000 | 1.12% | 1.35% | no significant difference, p = 0.093 |

The honest summary: at short budgets multi-start local search is significantly
better than the swarm. As the budget grows the difference shrinks until it is no
longer statistically detectable. At no budget tested does the swarm
significantly beat multi-start. On one instance, B-n78-k10 at the largest
budget, the swarm is ahead; on the others it is behind.

A diversity-preserving variant, with Cauchy mutation and periodic restarts of
the worst particles, did not change this.

---

## 3. Why: the swarm never contracts

The mechanism turned out to be specific and instructive, and it took direct
measurement to find. Tracking a 150-iteration run on X-n101-k25:

| Iteration | Best | Swarm mean | Key-space diversity |
| --- | --- | --- | --- |
| 1 | 3.72% | 7.05% | 2.77 |
| 46 | 3.19% | 7.52% | 2.70 |
| 91 | 2.75% | 7.41% | 2.74 |
| 136 | 2.75% | 7.32% | 2.70 |

Diversity is flat. The swarm mean does not improve. Only the best-so-far
improves, and it does so at about the rate an accumulating sample would.

Compare thirty independent multi-start local optima on the same instance: best
3.19%, mean 6.62%. The swarm's *mean* candidate, at 7.3%, is worse than an
independent random restart's.

The cause is an interaction between two design choices that are each defensible
alone. QPSO's step length is `beta * |mbest - x| * ln(1/u)`, and the algorithm
relies on `|mbest - x|` shrinking as the personal bests cluster; that is its
entire exploration-to-exploitation mechanism. But the Lamarckian write-back
rewrites every improved particle's keys as canonical ranks, and across a
diverse swarm each customer takes many different ranks, so the mean best
position sits near the middle of the range in every dimension. `|mbest - x|`
therefore stays around 0.29 and never shrinks.

Meanwhile a meaningful difference between two orderings is on the order of one
rank, or `1/n`, which is 0.01 for a hundred customers. The step is roughly
thirty times larger than the signal it is supposed to refine. The swarm is
effectively sampling at random, and the failure is self-reinforcing: large steps
prevent clustering, and no clustering keeps the steps large.

Clone prevention was implemented and tested against a competing hypothesis, that
particles were collapsing onto identical local optima. It never triggered: the
particles produce genuinely distinct solutions. The problem is not too little
diversity, it is that the swarm's diversity is undirected.

---

## 4. What this means for the claims made

Supported by measurement:

* The route split is optimal for the ordering it receives, verified against
  brute-force enumeration on 300 random instances.
* Shortest paths are exact.
* Reported solutions are feasible, and gaps are measured against published
  best-known values or proven optima.
* The quantum-inspired update rule, on its own, is a strong global optimiser:
  1.8% against 150% for random sampling at equal evaluations.

Not supported, and therefore not claimed:

* That QPSO beats classical metaheuristics once a strong local search is
  present. On these instances it does not.
* That the hybrid converges faster in any sense that the measurements bear out.
* Any claim of quantum advantage or quantum hardware.

---

## 5. What would be needed to change the result

Stated so the work can be continued rather than merely concluded.

The step-length scaling is the concrete defect, and it is fixable in principle:
the sampling width has to be expressed in units of the representation, which for
rank keys means `1/n` rather than the raw coordinate spread. A second, larger
change would be to give the swarm the diversity-aware survivor selection that
the strongest solvers for this problem use, where an individual's fitness
combines its cost rank with its distance from the rest of the population. Both
are real research directions rather than parameter tweaks, and neither is
claimed here as done.
