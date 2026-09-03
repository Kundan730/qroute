# Benchmark report

Run name: `quick`  
Budget: 3.0 s wall clock per run, 2 seeds from master seed 20260920  
Environment: Python 3.13.7 on macOS-26.2-arm64-arm-64bit-Mach-O, 10 CPUs, numpy 2.5.2  
Git commit: `623524abcdfd` (working tree dirty)  

40 completed runs over 4 instances and 5 algorithms; no runs failed.

All gaps are percentages above the best known cost, so lower is better. A value that could not be computed is printed as an em dash.

## Main results: gap to best known, mean (best) over seeds

| Instance | Size | Best known | ga | pso | qpso | random | sa | Best on row |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| A-n60-k9 | 60 | 1354 | **0.33 (0.30)** | 0.85 (0.66) | 1.74 (0.89) | 0.96 (0.89) | 0.96 (0.74) | ga |
| B-n64-k9 | 64 | 861 | 10.16 (3.02) | 4.59 (3.25) | 5.34 (5.23) | 6.91 (6.27) | **4.53 (3.02)** | sa |
| P-n76-k5 | 76 | 627 | **0.72 (0.48)** | 3.03 (2.55) | 2.95 (2.87) | 1.28 (0.96) | 2.23 (2.07) | ga |
| A-n80-k10 | 80 | 1763 | **2.13 (1.42)** | 3.72 (3.12) | 3.32 (2.89) | 3.74 (3.40) | 4.59 (4.37) | ga |

Cells are the mean percentage gap above the best known cost over all seeds, with the single best seed in brackets. Lower is better; the best mean on each row is emphasised and named in the last column, and every algorithm tied with it at two decimal places is marked too. An em dash means no run of that algorithm produced a comparable result.

## Summary by instance-size tier

| Tier | Algorithm | Instances | Runs | Mean gap % | Runs at best known | Mean seconds |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| medium (50-99 nodes) | ga | 4 | 8 | 3.33 | 0/8 | 3.173 |
| medium (50-99 nodes) | pso | 4 | 8 | 3.05 | 0/8 | 3.143 |
| medium (50-99 nodes) | qpso | 4 | 8 | 3.34 | 0/8 | 3.237 |
| medium (50-99 nodes) | random | 4 | 8 | 3.22 | 0/8 | 3.065 |
| medium (50-99 nodes) | sa | 4 | 8 | 3.08 | 0/8 | 3.020 |

Tier boundaries are on node count: small (< 50 nodes), medium (50-99 nodes), large (>= 100 nodes). "Runs at best known" counts runs whose final cost equalled the best known cost, over the runs that have a best known cost to compare against.

## Statistical comparison: Friedman ranks and Holm-corrected pairwise tests

| Algorithm | Mean rank | Holm-corrected p vs control | Effect size | Finding |
| :--- | ---: | ---: | ---: | :--- |
| ga | 2.00 | — | — | control algorithm (lowest mean rank) |
| pso | 3.00 | 1 | 0.20 | no significant difference from ga at alpha = 0.05 (n = 4 instances) |
| random | 3.25 | 1 | 0.20 | no significant difference from ga at alpha = 0.05 (n = 4 instances) |
| sa | 3.25 | 1 | 0.20 | no significant difference from ga at alpha = 0.05 (n = 4 instances) |
| qpso | 3.50 | 1 | 0.20 | no significant difference from ga at alpha = 0.05 (n = 4 instances) |

Friedman omnibus test over 4 instances and 5 algorithms: chi-square = 2.200, p = 0.699, so at alpha = 0.05 no overall difference between the algorithms was detected. Ranks are over the per-instance median gap, 1 = best.
Pairwise tests are paired Wilcoxon signed-rank tests against the control (ga), with Holm's step-down correction over the 4 comparisons. Instances used: A-n60-k9, B-n64-k9, P-n76-k5, A-n80-k10.

## Convergence: median effort to reach a target quality

| Instance | Algorithm | Iterations to 1% | Seconds to 1% | Runs reaching 1% | Iterations to 2% | Seconds to 2% | Runs reaching 2% |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-n60-k9 | ga | 4 | 1.535 | 2/2 | 1 | 0.896 | 2/2 |
| A-n60-k9 | pso | 5 | 2.694 | 1/2 | 1 | 1.297 | 2/2 |
| A-n60-k9 | qpso | 1 | 1.373 | 1/2 | 1 | 1.373 | 1/2 |
| A-n60-k9 | random | 8 | 0.996 | 1/2 | 4 | 0.419 | 2/2 |
| A-n60-k9 | sa | 15 | 0.483 | 1/2 | 36 | 1.018 | 2/2 |
| B-n64-k9 | ga | 3 | 1.319 | 1/2 | 2 | 1.044 | 2/2 |
| B-n64-k9 | pso | 3 | 1.325 | 1/2 | 2 | 1.182 | 2/2 |
| B-n64-k9 | qpso | not reached | not reached | 0/2 | not reached | not reached | 0/2 |
| B-n64-k9 | random | not reached | not reached | 0/2 | 12 | 1.482 | 2/2 |
| B-n64-k9 | sa | not reached | not reached | 0/2 | 51 | 1.415 | 2/2 |
| P-n76-k5 | ga | 4 | 1.647 | 2/2 | 2 | 1.267 | 2/2 |
| P-n76-k5 | pso | not reached | not reached | 0/2 | not reached | not reached | 0/2 |
| P-n76-k5 | qpso | not reached | not reached | 0/2 | not reached | not reached | 0/2 |
| P-n76-k5 | random | 6 | 0.888 | 1/2 | 6 | 0.892 | 2/2 |
| P-n76-k5 | sa | not reached | not reached | 0/2 | not reached | not reached | 0/2 |
| A-n80-k10 | ga | not reached | not reached | 0/2 | 1 | 1.336 | 1/2 |
| A-n80-k10 | pso | not reached | not reached | 0/2 | not reached | not reached | 0/2 |
| A-n80-k10 | qpso | not reached | not reached | 0/2 | not reached | not reached | 0/2 |
| A-n80-k10 | random | not reached | not reached | 0/2 | not reached | not reached | 0/2 |
| A-n80-k10 | sa | not reached | not reached | 0/2 | not reached | not reached | 0/2 |

Medians are over the runs that reached the target; the last column of each block gives how many of the measurable seeds did. "not reached" means no seed reached that target within the time budget, which is a result; an em dash with a k/0 count means the target could not be evaluated for those runs at all.

## Ablation: contribution of each component

| Variant | What changed | Instances | Runs | Mean gap % | Change vs full method | Runs at best known | Mean seconds |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| qpso | full method: quantum-behaved swarm with local search | 4 | 8 | 3.34 | — | 0/8 | 3.237 |
| random | control: random restart with the same local search | 4 | 8 | 3.22 | -0.12 | 0/8 | 3.065 |

All arms are compared on the 4 instance(s) that every arm solved, with the same seeds and the same time budget. "Change vs full method" is the difference in mean gap against qpso; a positive number means removing that component made the result worse.

## Per-instance detail

| Instance | Size | Best known | Algorithm | Runs | Feasible runs | Best cost | Mean cost | Median cost | Std dev | Worst cost | Best gap % | Mean gap % |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-n60-k9 | 60 | 1354 | ga | 2 | 2 | 1358.00 | 1358.50 | 1358.50 | 0.71 | 1359.00 | 0.30 | 0.33 |
| A-n60-k9 | 60 | 1354 | pso | 2 | 2 | 1363.00 | 1365.50 | 1365.50 | 3.54 | 1368.00 | 0.66 | 0.85 |
| A-n60-k9 | 60 | 1354 | qpso | 2 | 2 | 1366.00 | 1377.50 | 1377.50 | 16.26 | 1389.00 | 0.89 | 1.74 |
| A-n60-k9 | 60 | 1354 | random | 2 | 2 | 1366.00 | 1367.00 | 1367.00 | 1.41 | 1368.00 | 0.89 | 0.96 |
| A-n60-k9 | 60 | 1354 | sa | 2 | 2 | 1364.00 | 1367.00 | 1367.00 | 4.24 | 1370.00 | 0.74 | 0.96 |
| B-n64-k9 | 64 | 861 | ga | 2 | 2 | 887.00 | 948.50 | 948.50 | 86.97 | 1010.00 | 3.02 | 10.16 |
| B-n64-k9 | 64 | 861 | pso | 2 | 2 | 889.00 | 900.50 | 900.50 | 16.26 | 912.00 | 3.25 | 4.59 |
| B-n64-k9 | 64 | 861 | qpso | 2 | 2 | 906.00 | 907.00 | 907.00 | 1.41 | 908.00 | 5.23 | 5.34 |
| B-n64-k9 | 64 | 861 | random | 2 | 2 | 915.00 | 920.50 | 920.50 | 7.78 | 926.00 | 6.27 | 6.91 |
| B-n64-k9 | 64 | 861 | sa | 2 | 2 | 887.00 | 900.00 | 900.00 | 18.38 | 913.00 | 3.02 | 4.53 |
| P-n76-k5 | 76 | 627 | ga | 2 | 2 | 630.00 | 631.50 | 631.50 | 2.12 | 633.00 | 0.48 | 0.72 |
| P-n76-k5 | 76 | 627 | pso | 2 | 2 | 643.00 | 646.00 | 646.00 | 4.24 | 649.00 | 2.55 | 3.03 |
| P-n76-k5 | 76 | 627 | qpso | 2 | 2 | 645.00 | 645.50 | 645.50 | 0.71 | 646.00 | 2.87 | 2.95 |
| P-n76-k5 | 76 | 627 | random | 2 | 2 | 633.00 | 635.00 | 635.00 | 2.83 | 637.00 | 0.96 | 1.28 |
| P-n76-k5 | 76 | 627 | sa | 2 | 2 | 640.00 | 641.00 | 641.00 | 1.41 | 642.00 | 2.07 | 2.23 |
| A-n80-k10 | 80 | 1763 | ga | 2 | 2 | 1788.00 | 1800.50 | 1800.50 | 17.68 | 1813.00 | 1.42 | 2.13 |
| A-n80-k10 | 80 | 1763 | pso | 2 | 2 | 1818.00 | 1828.50 | 1828.50 | 14.85 | 1839.00 | 3.12 | 3.72 |
| A-n80-k10 | 80 | 1763 | qpso | 2 | 2 | 1814.00 | 1821.50 | 1821.50 | 10.61 | 1829.00 | 2.89 | 3.32 |
| A-n80-k10 | 80 | 1763 | random | 2 | 2 | 1823.00 | 1829.00 | 1829.00 | 8.49 | 1835.00 | 3.40 | 3.74 |
| A-n80-k10 | 80 | 1763 | sa | 2 | 2 | 1840.00 | 1844.00 | 1844.00 | 5.66 | 1848.00 | 4.37 | 4.59 |

Costs are the final cost of each run, re-scored by the reference evaluator. The standard deviation is the sample standard deviation over seeds; it is 0.00 for a single run and an em dash when no run produced a cost.
