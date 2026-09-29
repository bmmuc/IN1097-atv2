# Atividade 2 -- comparison summary (lodo)
N = 50 datasets.
## Per-treatment summary
| treatment | approach | variant | n_features | mean_spearman | std_spearman | mean_auc_loss | std_auc_loss | loss_t1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR | AR | AR | 0 | 0.7484 | 0.2997 | 0.0021 | 0.0034 | 0.0164 |
| RegP|base | RegP | base | 36 | 0.8383 | 0.1923 | 0.001 | 0.002 | 0.0068 |
| RegP|ext | RegP | ext | 82 | 0.8145 | 0.2323 | 0.0015 | 0.0025 | 0.0117 |
| RegP|filt | RegP | filt | 38 | 0.7988 | 0.2425 | 0.0014 | 0.0024 | 0.0099 |
| RegP|sel | RegP | sel | 36 | 0.8368 | 0.2205 | 0.0013 | 0.0021 | 0.0102 |
| RegP|new | RegP | new | 46 | 0.8097 | 0.2491 | 0.0015 | 0.0025 | 0.0114 |
| RegP|cheap | RegP | cheap | 16 | 0.7995 | 0.2327 | 0.0012 | 0.0021 | 0.0088 |
| RegR|base | RegR | base | 36 | 0.8403 | 0.1725 | 0.001 | 0.0017 | 0.0077 |
| RegR|ext | RegR | ext | 82 | 0.8564 | 0.1665 | 0.0011 | 0.0022 | 0.0082 |
| RegR|filt | RegR | filt | 38 | 0.8614 | 0.1515 | 0.001 | 0.002 | 0.0074 |
| RegR|sel | RegR | sel | 36 | 0.8573 | 0.1783 | 0.0011 | 0.0023 | 0.008 |
| RegR|new | RegR | new | 46 | 0.863 | 0.1633 | 0.0011 | 0.0023 | 0.0076 |
| RegR|cheap | RegR | cheap | 16 | 0.8379 | 0.1734 | 0.001 | 0.0016 | 0.0086 |
| Harris|base | Harris | base | 36 | 0.8398 | 0.1358 | 0.001 | 0.0016 | 0.0076 |
| Harris|ext | Harris | ext | 82 | 0.8451 | 0.1492 | 0.001 | 0.0016 | 0.0081 |
| Harris|filt | Harris | filt | 38 | 0.8398 | 0.149 | 0.0009 | 0.0016 | 0.0081 |
| Harris|sel | Harris | sel | 36 | 0.8541 | 0.1377 | 0.0009 | 0.0016 | 0.0075 |
| Harris|new | Harris | new | 46 | 0.8383 | 0.1792 | 0.0013 | 0.0023 | 0.0102 |
| Harris|cheap | Harris | cheap | 16 | 0.8419 | 0.1745 | 0.001 | 0.0016 | 0.0078 |


## Wilcoxon vs base (X-dependent approaches, variant != base)
| approach | variant | mean_diff_spearman | mean_diff_auc_loss | wins_spearman | ties_spearman | losses_spearman | wilcoxon_p_spearman | wilcoxon_p_auc_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RegP | ext | -0.0238 | 0.0005 | 21 | 8 | 21 | 0.4343 | 0.4209 |
| RegP | filt | -0.0395 | 0.0003 | 17 | 9 | 24 | 0.0982 | 0.357 |
| RegP | sel | -0.0015 | 0.0003 | 25 | 11 | 14 | 0.1711 | 0.2146 |
| RegP | new | -0.0286 | 0.0005 | 20 | 4 | 26 | 0.7388 | 0.0875 |
| RegP | cheap | -0.0388 | 0.0002 | 17 | 4 | 29 | 0.0671 | 0.8107 |
| RegR | ext | 0.0161 | 0.0001 | 22 | 11 | 17 | 0.4939 | 0.3812 |
| RegR | filt | 0.0211 | -0.0 | 27 | 6 | 17 | 0.0916 | 0.6192 |
| RegR | sel | 0.017 | 0.0001 | 23 | 7 | 20 | 0.4046 | 0.4631 |
| RegR | new | 0.0227 | 0.0001 | 24 | 6 | 20 | 0.2575 | 0.6274 |
| RegR | cheap | -0.0024 | 0.0 | 19 | 6 | 25 | 0.4072 | 0.6791 |
| Harris | ext | 0.0053 | 0.0 | 17 | 11 | 22 | 0.8834 | 0.4838 |
| Harris | filt | -0.0 | -0.0 | 15 | 12 | 23 | 0.4815 | 0.7989 |
| Harris | sel | 0.0143 | -0.0 | 26 | 7 | 17 | 0.0643 | 0.8658 |
| Harris | new | -0.0015 | 0.0004 | 23 | 8 | 19 | 0.5733 | 0.3281 |
| Harris | cheap | 0.0022 | -0.0 | 22 | 4 | 24 | 0.9434 | 0.925 |


## English vs non-English mean Spearman
| treatment | mean_spearman_english | mean_spearman_non_english | n_english | n_non_english |
| --- | --- | --- | --- | --- |
| AR | 0.8512 | 0.1169 | 43 | 7 |
| RegP|base | 0.8746 | 0.6156 | 43 | 7 |
| RegP|ext | 0.8763 | 0.4355 | 43 | 7 |
| RegP|filt | 0.8726 | 0.3455 | 43 | 7 |
| RegP|sel | 0.8861 | 0.5342 | 43 | 7 |
| RegP|new | 0.8754 | 0.4061 | 43 | 7 |
| RegP|cheap | 0.8613 | 0.4199 | 43 | 7 |
| RegR|base | 0.8809 | 0.5909 | 43 | 7 |
| RegR|ext | 0.8911 | 0.6433 | 43 | 7 |
| RegR|filt | 0.8929 | 0.6675 | 43 | 7 |
| RegR|sel | 0.8927 | 0.6398 | 43 | 7 |
| RegR|new | 0.8931 | 0.6779 | 43 | 7 |
| RegR|cheap | 0.8676 | 0.6554 | 43 | 7 |
| Harris|base | 0.8703 | 0.6519 | 43 | 7 |
| Harris|ext | 0.8799 | 0.6312 | 43 | 7 |
| Harris|filt | 0.8777 | 0.6069 | 43 | 7 |
| Harris|sel | 0.8864 | 0.6554 | 43 | 7 |
| Harris|new | 0.8763 | 0.6052 | 43 | 7 |
| Harris|cheap | 0.8788 | 0.6156 | 43 | 7 |


## Friedman + Nemenyi -- Spearman (19 treatments)
statistic = 78.4577, p = 1.59294e-09, CD = 3.9583
Mean ranks: RegR|filt=7.580, RegR|new=7.610, RegR|sel=7.930, RegR|ext=8.040, Harris|sel=8.830, RegP|sel=9.090, RegR|base=9.410, Harris|cheap=9.840, Harris|new=10.220, Harris|base=10.230, Harris|ext=10.310, RegP|base=10.400, RegR|cheap=10.440, Harris|filt=10.570, RegP|ext=10.730, RegP|new=11.100, RegP|filt=11.920, RegP|cheap=11.940, AR=13.810


## Friedman + Nemenyi -- AUC_loss (19 treatments)
statistic = 16.7465, p = 0.540595, CD = 3.9583
Mean ranks: Harris|filt=9.100, Harris|base=9.480, Harris|ext=9.490, Harris|sel=9.490, RegR|base=9.580, RegP|cheap=9.600, Harris|new=9.710, Harris|cheap=9.750, RegR|cheap=9.850, RegR|filt=9.890, RegP|ext=9.960, RegP|base=10.180, RegR|new=10.180, RegP|sel=10.350, RegR|sel=10.410, RegR|ext=10.460, RegP|filt=10.620, AR=10.760, RegP|new=11.140
