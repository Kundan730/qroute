# Benchmark report

Run name: `report-demo`  
Budget: 8 s wall clock per run, 5 seeds from master seed 20260920  
Environment: Python 3.13.7 on macOS-26.2-arm64-arm-64bit-Mach-O, 10 CPUs, numpy 2.5.2  
Git commit: `c0373bb1bb67` (working tree dirty)  

75 completed runs over 3 instances and 5 algorithms; no runs failed.

All gaps are percentages above the best known cost, so lower is better. A value that could not be computed is printed as an em dash.

## Main results: gap to best known, mean (best) over seeds

| Instance | Size | Best known | ga | pso | qpso | qpso-nols | random | Best on row |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| A-n32-k5 | 32 | 784 | **0.00 (0.00)** | **0.00 (0.00)** | **0.00 (0.00)** | **0.00 (0.00)** | **0.00 (0.00)** | tie (all equal) |
| A-n45-k7 | 45 | 1146 | **0.19 (0.00)** | 0.59 (0.17) | 0.37 (0.00) | 0.52 (0.00) | 0.86 (0.00) | ga |
| A-n80-k10 | 80 | 1763 | **1.57 (1.30)** | 2.47 (1.87) | 3.34 (2.78) | 3.55 (3.06) | 3.44 (2.95) | ga |

Cells are the mean percentage gap above the best known cost over all seeds, with the single best seed in brackets. Lower is better; the best mean on each row is emphasised and named in the last column, and every algorithm tied with it at two decimal places is marked too. An em dash means no run of that algorithm produced a comparable result.

## Summary by instance-size tier

| Tier | Algorithm | Instances | Runs | Mean gap % | Runs at best known | Mean seconds |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| small (< 50 nodes) | ga | 2 | 10 | 0.10 | 6/10 | 8.113 |
| small (< 50 nodes) | pso | 2 | 10 | 0.30 | 5/10 | 8.120 |
| small (< 50 nodes) | qpso | 2 | 10 | 0.18 | 6/10 | 8.077 |
| small (< 50 nodes) | qpso-nols | 2 | 10 | 0.26 | 6/10 | 8.118 |
| small (< 50 nodes) | random | 2 | 10 | 0.43 | 6/10 | 8.060 |
| medium (50-99 nodes) | ga | 1 | 5 | 1.57 | 0/5 | 8.220 |
| medium (50-99 nodes) | pso | 1 | 5 | 2.47 | 0/5 | 8.333 |
| medium (50-99 nodes) | qpso | 1 | 5 | 3.34 | 0/5 | 8.325 |
| medium (50-99 nodes) | qpso-nols | 1 | 5 | 3.55 | 0/5 | 8.332 |
| medium (50-99 nodes) | random | 1 | 5 | 3.44 | 0/5 | 8.154 |

Tier boundaries are on node count: small (< 50 nodes), medium (50-99 nodes), large (>= 100 nodes). "Runs at best known" counts runs whose final cost equalled the best known cost, over the runs that have a best known cost to compare against.

## Statistical comparison: Friedman ranks and Holm-corrected pairwise tests

| Algorithm | Mean rank | Holm-corrected p vs control | Effect size | Finding |
| :--- | ---: | ---: | ---: | :--- |
| ga | 1.67 | — | — | control algorithm (lowest mean rank) |
| qpso | 2.67 | 1 | 1.00 | no significant difference from ga at alpha = 0.05 (n = 3 instances) |
| pso | 3.00 | 1 | 1.00 | no significant difference from ga at alpha = 0.05 (n = 3 instances) |
| qpso-nols | 3.33 | 1 | 1.00 | no significant difference from ga at alpha = 0.05 (n = 3 instances) |
| random | 4.33 | 1 | 1.00 | no significant difference from ga at alpha = 0.05 (n = 3 instances) |

Friedman omnibus test over 3 instances and 5 algorithms: chi-square = 6.800, p = 0.147, so at alpha = 0.05 no overall difference between the algorithms was detected. Ranks are over the per-instance median gap, 1 = best.
Pairwise tests are paired Wilcoxon signed-rank tests against the control (ga), with Holm's step-down correction over the 4 comparisons. Instances used: A-n32-k5, A-n45-k7, A-n80-k10.

## Convergence: median effort to reach a target quality

| Instance | Algorithm | Iterations to 1% | Seconds to 1% | Runs reaching 1% | Iterations to 2% | Seconds to 2% | Runs reaching 2% |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-n32-k5 | ga | 1 | 0.302 | 5/5 | 1 | 0.302 | 5/5 |
| A-n32-k5 | pso | 1 | 0.968 | 5/5 | 1 | 0.968 | 5/5 |
| A-n32-k5 | qpso | 1 | 1.075 | 5/5 | 1 | 1.060 | 5/5 |
| A-n32-k5 | qpso-nols | 1 | 1.690 | 5/5 | 1 | 1.572 | 5/5 |
| A-n32-k5 | random | 1 | 0.045 | 5/5 | 1 | 0.045 | 5/5 |
| A-n45-k7 | ga | 5 | 1.989 | 5/5 | 1 | 0.997 | 5/5 |
| A-n45-k7 | pso | 3 | 1.022 | 4/5 | 1 | 0.580 | 5/5 |
| A-n45-k7 | qpso | 8 | 1.856 | 5/5 | 2 | 0.638 | 5/5 |
| A-n45-k7 | qpso-nols | 8 | 4.192 | 4/5 | 3 | 1.758 | 5/5 |
| A-n45-k7 | random | 22 | 3.872 | 4/5 | 12 | 1.774 | 4/5 |
| A-n80-k10 | ga | not reached | not reached | 0/5 | 10 | 3.763 | 4/5 |
| A-n80-k10 | pso | not reached | not reached | 0/5 | 5 | 2.844 | 1/5 |
| A-n80-k10 | qpso | not reached | not reached | 0/5 | not reached | not reached | 0/5 |
| A-n80-k10 | qpso-nols | not reached | not reached | 0/5 | not reached | not reached | 0/5 |
| A-n80-k10 | random | not reached | not reached | 0/5 | not reached | not reached | 0/5 |

Medians are over the runs that reached the target; the last column of each block gives how many of the measurable seeds did. "not reached" means no seed reached that target within the time budget, which is a result; an em dash with a k/0 count means the target could not be evaluated for those runs at all.

## Ablation: contribution of each component

| Variant | What changed | Instances | Runs | Mean gap % | Change vs full method | Runs at best known | Mean seconds |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| qpso | full method: quantum-behaved swarm with local search | 3 | 15 | 1.23 | — | 6/15 | 8.160 |
| qpso-nols | local search removed | 3 | 15 | 1.36 | +0.13 | 6/15 | 8.189 |
| random | control: random restart with the same local search | 3 | 15 | 1.43 | +0.20 | 6/15 | 8.092 |

All arms are compared on the 3 instance(s) that every arm solved, with the same seeds and the same time budget. "Change vs full method" is the difference in mean gap against qpso; a positive number means removing that component made the result worse.

## Per-instance detail

| Instance | Size | Best known | Algorithm | Runs | Feasible runs | Best cost | Mean cost | Median cost | Std dev | Worst cost | Best gap % | Mean gap % |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-n32-k5 | 32 | 784 | ga | 5 | 5 | 784.00 | 784.00 | 784.00 | 0.00 | 784.00 | 0.00 | 0.00 |
| A-n32-k5 | 32 | 784 | pso | 5 | 5 | 784.00 | 784.00 | 784.00 | 0.00 | 784.00 | 0.00 | 0.00 |
| A-n32-k5 | 32 | 784 | qpso | 5 | 5 | 784.00 | 784.00 | 784.00 | 0.00 | 784.00 | 0.00 | 0.00 |
| A-n32-k5 | 32 | 784 | qpso-nols | 5 | 5 | 784.00 | 784.00 | 784.00 | 0.00 | 784.00 | 0.00 | 0.00 |
| A-n32-k5 | 32 | 784 | random | 5 | 5 | 784.00 | 784.00 | 784.00 | 0.00 | 784.00 | 0.00 | 0.00 |
| A-n45-k7 | 45 | 1146 | ga | 5 | 5 | 1146.00 | 1148.20 | 1147.00 | 2.77 | 1153.00 | 0.00 | 0.19 |
| A-n45-k7 | 45 | 1146 | pso | 5 | 5 | 1148.00 | 1152.80 | 1152.00 | 4.97 | 1161.00 | 0.17 | 0.59 |
| A-n45-k7 | 45 | 1146 | qpso | 5 | 5 | 1146.00 | 1150.20 | 1150.00 | 4.32 | 1157.00 | 0.00 | 0.37 |
| A-n45-k7 | 45 | 1146 | qpso-nols | 5 | 5 | 1146.00 | 1152.00 | 1151.00 | 5.61 | 1160.00 | 0.00 | 0.52 |
| A-n45-k7 | 45 | 1146 | random | 5 | 5 | 1146.00 | 1155.80 | 1153.00 | 10.13 | 1173.00 | 0.00 | 0.86 |
| A-n80-k10 | 80 | 1763 | ga | 5 | 5 | 1786.00 | 1790.60 | 1788.00 | 5.41 | 1799.00 | 1.30 | 1.57 |
| A-n80-k10 | 80 | 1763 | pso | 5 | 5 | 1796.00 | 1806.60 | 1809.00 | 7.96 | 1817.00 | 1.87 | 2.47 |
| A-n80-k10 | 80 | 1763 | qpso | 5 | 5 | 1812.00 | 1821.80 | 1819.00 | 10.64 | 1840.00 | 2.78 | 3.34 |
| A-n80-k10 | 80 | 1763 | qpso-nols | 5 | 5 | 1817.00 | 1825.60 | 1820.00 | 13.76 | 1850.00 | 3.06 | 3.55 |
| A-n80-k10 | 80 | 1763 | random | 5 | 5 | 1815.00 | 1823.60 | 1822.00 | 7.02 | 1834.00 | 2.95 | 3.44 |

Costs are the final cost of each run, re-scored by the reference evaluator. The standard deviation is the sample standard deviation over seeds; it is 0.00 for a single run and an em dash when no run produced a cost.
