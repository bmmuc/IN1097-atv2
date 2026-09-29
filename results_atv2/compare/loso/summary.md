# Atividade 2 -- comparison summary (loso)
N = 50 datasets.
## Per-treatment summary
| treatment | approach | variant | n_features | mean_spearman | std_spearman | mean_auc_loss | std_auc_loss | loss_t1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR | AR | AR | 0 | 0.7348 | 0.2945 | 0.0022 | 0.0034 | 0.0164 |
| RegP|base | RegP | base | 36 | 0.8012 | 0.2639 | 0.0013 | 0.0021 | 0.0085 |
| RegP|ext | RegP | ext | 82 | 0.7947 | 0.2641 | 0.0016 | 0.0024 | 0.0125 |
| RegP|filt | RegP | filt | 38 | 0.7762 | 0.2852 | 0.0016 | 0.0025 | 0.0113 |
| RegP|sel | RegP | sel | 36 | 0.8133 | 0.2355 | 0.0015 | 0.0017 | 0.0121 |
| RegP|new | RegP | new | 46 | 0.7886 | 0.2732 | 0.0017 | 0.0025 | 0.0132 |
| RegP|cheap | RegP | cheap | 16 | 0.7644 | 0.2812 | 0.0014 | 0.0021 | 0.0111 |
| RegR|base | RegR | base | 36 | 0.7797 | 0.2258 | 0.0017 | 0.0027 | 0.0104 |
| RegR|ext | RegR | ext | 82 | 0.8208 | 0.187 | 0.0018 | 0.0032 | 0.0099 |
| RegR|filt | RegR | filt | 38 | 0.8026 | 0.1878 | 0.0017 | 0.0032 | 0.0102 |
| RegR|sel | RegR | sel | 36 | 0.828 | 0.1905 | 0.0016 | 0.0029 | 0.0092 |
| RegR|new | RegR | new | 46 | 0.8258 | 0.1751 | 0.0016 | 0.0025 | 0.0096 |
| RegR|cheap | RegR | cheap | 16 | 0.8177 | 0.1893 | 0.0016 | 0.0021 | 0.0114 |
| Harris|base | Harris | base | 36 | 0.7981 | 0.2052 | 0.0013 | 0.0018 | 0.0099 |
| Harris|ext | Harris | ext | 82 | 0.8073 | 0.1772 | 0.0012 | 0.0017 | 0.0089 |
| Harris|filt | Harris | filt | 38 | 0.7867 | 0.2095 | 0.0016 | 0.0019 | 0.0125 |
| Harris|sel | Harris | sel | 36 | 0.8206 | 0.1709 | 0.0011 | 0.0016 | 0.0085 |
| Harris|new | Harris | new | 46 | 0.809 | 0.2174 | 0.0015 | 0.0022 | 0.0107 |
| Harris|cheap | Harris | cheap | 16 | 0.8051 | 0.2091 | 0.0011 | 0.0017 | 0.0089 |


## Wilcoxon vs base (X-dependent approaches, variant != base)
| approach | variant | mean_diff_spearman | mean_diff_auc_loss | wins_spearman | ties_spearman | losses_spearman | wilcoxon_p_spearman | wilcoxon_p_auc_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RegP | ext | -0.0065 | 0.0003 | 19 | 7 | 24 | 0.3274 | 0.2868 |
| RegP | filt | -0.025 | 0.0003 | 21 | 3 | 26 | 0.0734 | 0.3458 |
| RegP | sel | 0.0121 | 0.0002 | 18 | 12 | 20 | 0.2796 | 0.3318 |
| RegP | new | -0.0126 | 0.0005 | 18 | 4 | 28 | 0.2312 | 0.1089 |
| RegP | cheap | -0.0368 | 0.0001 | 12 | 6 | 32 | 0.0157 | 0.9479 |
| RegR | ext | 0.041 | 0.0 | 28 | 6 | 16 | 0.0322 | 0.6766 |
| RegR | filt | 0.0228 | 0.0 | 19 | 9 | 22 | 0.2705 | 0.9494 |
| RegR | sel | 0.0482 | -0.0001 | 32 | 4 | 14 | 0.0057 | 0.7751 |
| RegR | new | 0.0461 | -0.0002 | 33 | 5 | 12 | 0.0084 | 0.6702 |
| RegR | cheap | 0.0379 | -0.0001 | 25 | 6 | 19 | 0.2204 | 0.8036 |
| Harris | ext | 0.0092 | -0.0 | 18 | 14 | 18 | 0.7002 | 0.9176 |
| Harris | filt | -0.0114 | 0.0003 | 17 | 14 | 19 | 0.3337 | 0.1024 |
| Harris | sel | 0.0225 | -0.0002 | 23 | 16 | 11 | 0.0298 | 0.5067 |
| Harris | new | 0.0109 | 0.0002 | 24 | 7 | 19 | 0.4395 | 0.3259 |
| Harris | cheap | 0.007 | -0.0001 | 17 | 8 | 25 | 0.7639 | 1.0 |


## English vs non-English mean Spearman
| treatment | mean_spearman_english | mean_spearman_non_english | n_english | n_non_english |
| --- | --- | --- | --- | --- |
| AR | 0.8354 | 0.1169 | 43 | 7 |
| RegP|base | 0.8774 | 0.3333 | 43 | 7 |
| RegP|ext | 0.8709 | 0.3264 | 43 | 7 |
| RegP|filt | 0.8644 | 0.2346 | 43 | 7 |
| RegP|sel | 0.8751 | 0.4338 | 43 | 7 |
| RegP|new | 0.8644 | 0.3229 | 43 | 7 |
| RegP|cheap | 0.8506 | 0.2346 | 43 | 7 |
| RegR|base | 0.8468 | 0.368 | 43 | 7 |
| RegR|ext | 0.8722 | 0.5048 | 43 | 7 |
| RegR|filt | 0.8556 | 0.4771 | 43 | 7 |
| RegR|sel | 0.8781 | 0.5203 | 43 | 7 |
| RegR|new | 0.8712 | 0.5472 | 43 | 7 |
| RegR|cheap | 0.8596 | 0.5597 | 43 | 7 |
| Harris|base | 0.8593 | 0.4216 | 43 | 7 |
| Harris|ext | 0.8605 | 0.4805 | 43 | 7 |
| Harris|filt | 0.8509 | 0.3922 | 43 | 7 |
| Harris|sel | 0.8717 | 0.5065 | 43 | 7 |
| Harris|new | 0.8684 | 0.4442 | 43 | 7 |
| Harris|cheap | 0.861 | 0.4615 | 43 | 7 |


## Friedman + Nemenyi -- Spearman (19 treatments)
statistic = 54.9417, p = 1.30765e-05, CD = 3.9583
Mean ranks: RegR|sel=7.960, RegR|new=8.160, Harris|sel=8.910, RegP|base=8.940, RegP|sel=9.020, RegR|ext=9.040, Harris|new=9.280, RegR|cheap=9.560, RegP|ext=9.560, Harris|base=9.880, RegP|new=10.000, Harris|cheap=10.300, RegR|filt=10.400, Harris|ext=10.420, RegP|filt=10.570, RegR|base=11.190, Harris|filt=11.270, RegP|cheap=11.930, AR=13.610


## Friedman + Nemenyi -- AUC_loss (19 treatments)
statistic = 31.6316, p = 0.0242996, CD = 3.9583
Mean ranks: Harris|sel=8.740, Harris|cheap=8.900, Harris|base=8.910, RegP|base=9.180, RegP|cheap=9.180, Harris|ext=9.490, Harris|new=9.560, RegP|ext=9.750, RegP|filt=9.840, RegP|sel=10.080, Harris|filt=10.430, RegP|new=10.440, AR=10.460, RegR|sel=10.510, RegR|filt=10.540, RegR|cheap=10.610, RegR|new=10.750, RegR|base=11.290, RegR|ext=11.340
