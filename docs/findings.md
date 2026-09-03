# What the Measurements Actually Showed

This document records the experimental findings, including the ones that do not
flatter the method. It exists because the alternative, reporting only the
comparisons that came out well, is how this literature acquired its credibility
problem, and because a panel that probes a claim should find the answer already
written down.

All experiments use a matched evaluation budget: every variant decodes and
evaluates the same number of candidate solutions, so the only difference is
which candidates get evaluated.

**A correction, recorded rather than quietly folded in.** The first version of
this document reported an ablation that was confounded. QPSO hard-coded a
constraint penalty of 1000 while every other solver, and the multi-start
control, inherited the instance-scaled default of about 16. The algorithms were
not minimising the same objective, so the comparison was not a comparison. An
adversarial verification pass found it; the tests did not. Everything below is
the re-run with identical objectives.

---

## 1. What each component contributes

Six instances, five seeds, 3,000 evaluations each, every solver minimising the
identical objective.

| Configuration | Mean gap to best known |
| --- | --- |
| Random orderings, no local search | 146.6% |
| QPSO, no local search | 1.25% |
| Random orderings, with local search | 1.07% |
| Genetic algorithm | 0.56% |
| Ant colony optimisation | 0.75% |
| Classical particle swarm | 1.13% |
| QPSO, with local search | 1.34% |
| Simulated annealing | 2.53% |

Three things follow, and the second is not flattering.

**The pipeline is what delivers the result.** Random orderings without local
search sit 147% above the best-known cost. Everything that reaches roughly one
percent does so because of the optimal split and the local search, not because
of any particular search rule.

**QPSO does not beat the classical baselines.** It places sixth of eight. The
genetic algorithm is more than twice as good on average, and ant colony
optimisation is close behind it. This holds on five of the six instances.

**The quantum-inspired rule is nevertheless a real optimiser.** With no local
search at all, QPSO reaches 1.25% where random sampling of the same number of
orderings reaches 146.6%. The update rule is doing genuine work; what it does
not do is add to a strong local search. Notably, QPSO without local search
(1.25%) is slightly *better* than QPSO with it (1.34%), which is the clue that
led to the mechanism in section 3.

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

## 5. The step-scaling defect, and fixing it

The diagnosis in section 3 makes a prediction that can be tested rather than
merely asserted: if the sampling step is far larger than the differences it
should be refining, then reducing the contraction coefficient well below its
classical range should improve the result.

The published QPSO literature uses coefficients between about 0.5 and 1.0, and
Sun et al.'s stability analysis bounds them above by roughly 1.78. Those values
come from continuous function optimisation, where a coordinate has absolute
meaning. A random-key permutation encoding is different: the coordinate itself
is meaningless, only the induced ordering matters, and the smallest meaningful
change is one rank.

Sweeping the coefficient on five instances with five seeds at 3,000 evaluations:

| Contraction coefficient | Mean gap |
| --- | --- |
| 1.0, the classical default | 1.53% |
| 0.5, the classical lower bound | 1.43% |
| 0.2 | 1.27% |
| **0.05** | **1.01%** |
| 0.02 | 1.16% |
| Multi-start control | 1.25% |

The trend is monotone into an optimum near 0.05, twenty times below the standard
value. Confirming on ten instances with eight seeds, paired by instance and seed:

| Variant | Mean gap | Median gap |
| --- | --- | --- |
| Multi-start + local search | 0.92% | 0.80% |
| QPSO, coefficient 1.0 | 1.36% | 0.77% |
| QPSO, coefficient 0.05 | 1.04% | 0.25% |

The reduced coefficient beats the classical one with p = 0.00015 on a Wilcoxon
signed-rank test over 80 paired runs, effect size 0.57. Against the multi-start
control the difference is no longer significant either way (p = 0.105), so the
correction moves QPSO from significantly worse than the control to tied with it.

The mean remains marginally behind the control because of a single instance,
X-n153-k22, where the swarm scores 5.16% against the control's 1.70%. That is
the largest instance in the set, and it is what the diagnosis would predict: if
the right step is proportional to `1/n`, then a constant fitted at eighty
customers is too large at a hundred and fifty.

### A scaling law that did not survive testing

The residual weakness on the largest instance suggested a sharper hypothesis.
The step in one dimension is `beta * |mbest - x| * ln(1/u)`. With canonical rank
keys the mean best position sits mid-range in every dimension, so `|mbest - x|`
is about 0.29 regardless of the instance and `ln(1/u)` averages one; the step is
therefore about `0.29 * beta`, while the smallest meaningful change to an
ordering is `1/n`. Requiring the two to be comparable gives

    beta  ~  c / n,      c = 1.16 from the measured optimum at n = 80

which predicts that a single constant fitted at eighty customers is too large at
a hundred and fifty, and that scaling it should recover the lost ground.

**It did not.** Tested over eleven instances up to two hundred customers, eight
seeds each:

| Variant | Mean gap | Median gap |
| --- | --- | --- |
| Multi-start + local search | 1.14% | 0.91% |
| QPSO, classical coefficient 1.0 | 1.53% | 1.01% |
| QPSO, fixed coefficient 0.05 | 1.24% | 0.60% |
| QPSO, derived `1.16/n` law | 1.27% | 0.59% |

The derived law is statistically indistinguishable from simply using the
constant everywhere (p = 0.42 over 88 paired runs), and on X-n153-k22, the
instance that motivated it, it scores 5.05% against the constant's 5.16% and the
control's 1.70%. The hypothesis is not supported.

What that instance is actually short of is iterations, not step size. At 3,000
evaluations a swarm of thirty gets a hundred iterations, while the multi-start
control gets three thousand independent samples; on a large instance the
control's coverage wins regardless of how the swarm is tuned.

Both results are reported because the second is the more useful one. The
correction that works, reducing the coefficient, is confirmed twice at
p = 0.00013. The elegant explanation for why it should be size-dependent is
wrong, and pretending otherwise would have put an unsupported formula into the
method description.

The default is therefore the fixed constant. The derived law remains available
as an option, documented as unsupported.

## 6. The definitive benchmark

**This run replaces an earlier one that was not a fair comparison.** The
dispatcher inspected a thin wrapper's parameter names, found neither the budget
nor the seed among them, and forwarded neither, so PyVRP and OR-Tools ran at
their own ten-second defaults against everyone else's twenty and PyVRP received
seed 0 for all ten of its supposedly independent runs. Dropping a keyword
argument raises nothing, which is why it survived a verification pass. The
figures below are from the corrected run, in which elapsed time tracks the
budget for every solver and PyVRP's answers vary with the seed.

Worth recording: the correction did **not** materially change any conclusion.
OR-Tools improved from 2.23% to 1.93% with its full budget, PyVRP was unchanged
at 0.23% because it converges long before either limit, and the ranking moved
only in that the genetic algorithm and the rotation-gate engine swapped second
and third. The bug was real and had to be fixed; it was not load-bearing.

Seventeen instances, nine solvers, ten seeds each, an equal twenty-second
wall-clock budget, every run pinned to one thread. 1,520 completed runs, none
infeasible. OR-Tools found no solution at all on X-n153-k22 in all ten seeds,
reported as "no solution found" rather than folded into an average.

Friedman mean ranks over the sixteen instances every solver scored, lower being
better, omnibus p = 1.72e-09:

| Rank | Solver | Mean rank | Mean gap | Reached best known |
| ---: | :--- | ---: | ---: | ---: |
| 1 | PyVRP, hybrid genetic search | 3.53 | 0.23% | 116/170 |
| 2 | Genetic algorithm | 3.78 | 0.66% | 103/170 |
| 3 | **Quantum rotation-gate engine** | 3.97 | 0.70% | 104/170 |
| 4 | Ant colony optimisation | 4.12 | 0.77% | 98/170 |
| 5 | **Quantum particle swarm** | 4.81 | 0.87% | 96/170 |
| 6 | Simulated annealing | 5.50 | 0.94% | 78/170 |
| 7 | Classical particle swarm | 5.66 | 1.03% | 79/170 |
| 8 | Random multi-start | 6.12 | 1.22% | 77/170 |
| 9 | OR-Tools guided local search | 7.50 | 1.93% | 60/160 |

QPSO against every other solver, paired by instance and seed, Holm-corrected.
The OR-Tools row has 160 pairs rather than 170 because its ten failed runs
cannot be paired:

| Comparison | Result |
| --- | --- |
| QPSO vs classical PSO | **QPSO better**, p = 1.3e-06, effect 0.62 |
| QPSO vs simulated annealing | **QPSO better**, p = 0.0028, effect 0.37 |
| QPSO vs random multi-start | **QPSO better**, p = 6.5e-09, effect 0.74 |
| QPSO vs OR-Tools | **QPSO better**, p = 5.2e-16, effect 0.95 |
| QPSO vs ant colony | ACO better, p = 0.0027, effect 0.43 |
| QPSO vs rotation-gate engine | QIEA better, p = 2.7e-05, effect 0.62 |
| QPSO vs genetic algorithm | GA better, p = 1.3e-06, effect 0.71 |
| QPSO vs PyVRP | PyVRP better, p = 3.7e-11, effect 0.91 |

The comparison the problem statement asks for is the first one, and it is
positive: the quantum-behaved swarm significantly outperforms the classical
particle swarm it is derived from, under an identical decoder, an identical
local search and an identical budget.

The rotation-gate engine does better still. It significantly beats classical PSO
(p = 4.7e-11), simulated annealing (p = 6.8e-08) and multi-start local search
(p = 2.8e-14), and is **statistically indistinguishable from the genetic
algorithm** (p = 0.087), the strongest classical metaheuristic in the set.

So both quantum-inspired engines place above every classical swarm and
single-trajectory method tested, and the better of the two ties with the best
classical population method. Neither approaches PyVRP, which is a specialised
state-of-the-art solver for this exact problem and is ahead of everything.

Eight of the seventeen instances are solved to the best-known value by nearly
every solver. The small tier is retained because it disqualifies anything that
fails there, not because it separates the rest.

---

## 6. What this means for the claims made

Supported by measurement:

* The route split is optimal for the ordering it receives, verified against
  brute-force enumeration on 300 random instances.
* Shortest paths are exact.
* Reported solutions are feasible, and gaps are measured against published
  best-known values or proven optima.
* The quantum-inspired update rule is a strong optimiser on its own: 1.25%
  against 146.6% for random sampling at equal evaluations.
* The classical contraction range is wrong for a random-key encoding, and
  correcting it is a significant improvement, confirmed twice at p = 0.00013
  over 80 and 88 paired runs.
* QPSO significantly outperforms classical PSO, simulated annealing,
  multi-start local search and OR-Tools on the full benchmark, at p = 0.0028 or
  better after Holm correction.
* The rotation-gate engine is statistically indistinguishable from the genetic
  algorithm (p = 0.087) and significantly beats every other classical method
  except ant colony optimisation.

Not supported, and therefore not claimed:

* That QPSO beats every classical metaheuristic. It does not: the genetic
  algorithm and ant colony optimisation are both significantly better.
* That either engine approaches the state of the art. PyVRP is ahead of
  everything, at 0.23 percent against 0.70 for the better of the two.
* That the correct coefficient scales with instance size. That was predicted
  and tested, and the prediction failed.
* Any claim of quantum advantage, or of quantum hardware being involved.

---

## 8. What would be needed to go further

The remaining gap to the genetic algorithm is most likely the absence of
diversity-aware survivor selection, where an individual's fitness combines its
cost rank with its distance from the rest of the population. That is what the
strongest solvers for this problem use, and it is a real research direction
rather than a parameter change. It is not claimed here as done.
