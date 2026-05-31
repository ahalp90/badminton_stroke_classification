# Per-class arc tables (serial-mean across 5 serials)

Source: parse_arcs.py over the taxon_pinned_w_preds 6-cell batch. a=alpha (renorm mean 1), v=val F1, t=train F1, gap=t_max-v_max. Labels are advisory (FLOOR/memorise/ceiling/mid/healthy).

### shuttleset_18_v2  (shuttleset_18 / split_v2)  run run_20260530_161525_131279

macro plateau ~epoch 31 (run-max 0.692); macro 10/25/40/80 = 0.646 / 0.670 / 0.683 / 0.692; best-macro epochs [19, 31, 36, 38, 62]
alpha-vs-valF1max corr = -0.91; above-mean alpha budget = 3.81, of which 2.00 (53%) sits on classes already plateaued by epoch 31 (floor classes alone: 0.83)

| class                    | n_tr | n_val | a_fin | a_pk(ep) | v_max(ep) | v_fin | t_max | gap | dV>plat | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| driven_flight            |   42 |     9 |  1.83 |  2.46(27) |  0.157(26) | 0.000 | 0.681 |  0.52 |    flat | FLOOR    |
| wrist_smash              |  978 |   331 |  1.68 |  1.69(76) |  0.503(79) | 0.499 | 0.668 |  0.16 |    0.06 | ceiling  |
| defensive_return_drive   |  269 |    55 |  1.66 |  1.71(39) |  0.510(80) | 0.510 | 0.691 |  0.18 |    0.07 | ceiling  |
| drive                    |  467 |    85 |  1.40 |  1.43(73) |  0.638(24) | 0.615 | 0.744 |  0.11 |    flat | mid      |
| push                     | 1882 |   389 |  1.31 |  1.31(79) |  0.647(71) | 0.631 | 0.740 |  0.09 |    flat | mid      |
| passive_drop             |  796 |   224 |  1.30 |  1.30(79) |  0.639(20) | 0.595 | 0.745 |  0.11 |    flat | mid      |
| back_court_drive         |  263 |    77 |  1.28 |   1.37(2) |  0.612(79) | 0.609 | 0.760 |  0.15 |    0.03 | ceiling  |
| cross_court_net_shot     |  847 |   267 |  1.19 |   1.80(5) |  0.808(79) | 0.807 | 0.780 | -0.03 |    0.06 | healthy  |
| defensive_return_lob     |  200 |    30 |  1.13 |   1.25(3) |  0.574(31) | 0.551 | 0.788 |  0.21 |    flat | memorise |
| drop                     | 1465 |   304 |  1.03 |  1.03(79) |  0.698(40) | 0.693 | 0.800 |  0.10 |    flat | mid      |
| smash                    | 1786 |   299 |  0.94 |  0.94(76) |  0.674(64) | 0.670 | 0.813 |  0.14 |    0.02 | mid      |
| return_net               | 2387 |   536 |  0.79 |  0.80(76) |  0.860(80) | 0.860 | 0.847 | -0.01 |    0.02 | healthy  |
| rush                     |  335 |    67 |  0.76 |   1.29(2) |  0.759(62) | 0.732 | 0.860 |  0.10 |    0.03 | mid      |
| lob                      | 3417 |   915 |  0.72 |  0.73(79) |  0.846(51) | 0.822 | 0.856 |  0.01 |    flat | healthy  |
| net_shot                 | 4139 |   956 |  0.49 |   0.58(1) |  0.922(80) | 0.922 | 0.904 | -0.02 |    0.02 | healthy  |
| long_service             |  252 |    33 |  0.22 |   1.13(1) |  0.985(79) | 0.985 | 0.970 | -0.01 |    flat | healthy  |
| clear                    | 1897 |   382 |  0.16 |   0.55(1) |  0.970(73) | 0.965 | 0.970 | -0.00 |    flat | healthy  |
| short_service            | 1312 |   291 |  0.10 |   0.62(1) |  0.995(79) | 0.995 | 0.985 | -0.01 |    flat | healthy  |

### bst_24_v2  (bst_24 / split_v2)  run run_20260530_174818_410060

macro plateau ~epoch 26 (run-max 0.845); macro 10/25/40/80 = 0.796 / 0.832 / 0.836 / 0.844; best-macro epochs [35, 42, 49, 70, 73]
alpha-vs-valF1max corr = -0.94; above-mean alpha budget = 6.33, of which 4.84 (76%) sits on classes already plateaued by epoch 26 (floor classes alone: 0.00)

| class                    | n_tr | n_val | a_fin | a_pk(ep) | v_max(ep) | v_fin | t_max | gap | dV>plat | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Top_drive                |  491 |   102 |  2.17 |  2.19(61) |  0.622(17) | 0.582 | 0.746 |  0.12 |    flat | mid      |
| Bottom_push              |  927 |   214 |  2.08 |  2.08(80) |  0.637(20) | 0.610 | 0.750 |  0.11 |    flat | mid      |
| Top_push                 |  955 |   175 |  1.94 |  1.94(80) |  0.656(45) | 0.649 | 0.767 |  0.11 |    flat | mid      |
| Bottom_drive             |  550 |   124 |  1.93 |  1.99(56) |  0.657(34) | 0.649 | 0.773 |  0.12 |    flat | mid      |
| Top_cross_court_net_shot |  405 |   141 |  1.85 |  2.57(11) |  0.773(47) | 0.760 | 0.785 |  0.01 |    0.05 | mid      |
| Bottom_cross_court_net_shot |  442 |   126 |  1.55 |   2.30(8) |  0.835(58) | 0.825 | 0.826 | -0.01 |    0.06 | healthy  |
| Top_return_net           | 1136 |   250 |  1.30 |  1.30(80) |  0.855(57) | 0.846 | 0.845 | -0.01 |    flat | healthy  |
| Top_lob                  | 1716 |   443 |  1.16 |  1.16(78) |  0.876(26) | 0.854 | 0.861 | -0.01 |    flat | healthy  |
| Bottom_return_net        | 1251 |   286 |  1.15 |  1.15(78) |  0.857(53) | 0.851 | 0.861 |  0.00 |    flat | healthy  |
| Bottom_lob               | 1901 |   502 |  1.11 |  1.11(75) |  0.839(23) | 0.814 | 0.867 |  0.03 |    flat | healthy  |
| Top_rush                 |  198 |    35 |  1.05 |   1.40(4) |  0.766(59) | 0.752 | 0.883 |  0.12 |    0.04 | mid      |
| Bottom_rush              |  137 |    32 |  1.05 |   1.63(7) |  0.760(64) | 0.750 | 0.892 |  0.13 |    0.02 | mid      |
| Top_drop                 | 1145 |   313 |  0.89 |  0.89(78) |  0.919(27) | 0.916 | 0.894 | -0.02 |    flat | healthy  |
| Bottom_drop              | 1116 |   215 |  0.81 |   0.95(1) |  0.904(54) | 0.902 | 0.903 | -0.00 |    flat | healthy  |
| Top_net_shot             | 2124 |   495 |  0.75 |   0.88(1) |  0.924(59) | 0.921 | 0.911 | -0.01 |    flat | healthy  |
| Bottom_net_shot          | 2015 |   461 |  0.73 |   0.91(1) |  0.923(58) | 0.919 | 0.912 | -0.01 |    0.03 | healthy  |
| Bottom_smash             | 1252 |   277 |  0.68 |   0.83(1) |  0.922(68) | 0.921 | 0.920 | -0.00 |    flat | healthy  |
| Top_smash                | 1512 |   353 |  0.61 |   0.69(1) |  0.918(28) | 0.912 | 0.927 |  0.01 |    flat | healthy  |
| Bottom_long_service      |  142 |    21 |  0.29 |   1.24(1) |  0.977(32) | 0.965 | 0.973 | -0.00 |    flat | healthy  |
| Top_clear                |  941 |   179 |  0.27 |   0.78(1) |  0.957(10) | 0.954 | 0.971 |  0.01 |    flat | healthy  |
| Bottom_clear             |  956 |   203 |  0.21 |   0.84(1) |  0.983(60) | 0.981 | 0.976 | -0.01 |    flat | healthy  |
| Top_long_service         |  110 |    12 |  0.20 |   1.28(1) |  0.959(38) | 0.926 | 0.983 |  0.02 |    0.02 | healthy  |
| Bottom_short_service     |  674 |   150 |  0.13 |   0.88(1) |  0.994(76) | 0.994 | 0.986 | -0.01 |    flat | healthy  |
| Top_short_service        |  638 |   141 |  0.10 |   0.85(1) |  0.995(76) | 0.993 | 0.990 | -0.01 |    flat | healthy  |

### bst_12_v2  (bst_12 / split_v2)  run run_20260530_192738_970644

macro plateau ~epoch 27 (run-max 0.847); macro 10/25/40/80 = 0.802 / 0.835 / 0.845 / 0.847; best-macro epochs [34, 37, 44, 46, 56]
alpha-vs-valF1max corr = -0.94; above-mean alpha budget = 3.10, of which 2.42 (78%) sits on classes already plateaued by epoch 27 (floor classes alone: 0.00)

| class                    | n_tr | n_val | a_fin | a_pk(ep) | v_max(ep) | v_fin | t_max | gap | dV>plat | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| push                     | 1882 |   389 |  2.00 |  2.00(80) |  0.636(13) | 0.625 | 0.763 |  0.13 |    flat | mid      |
| drive                    | 1041 |   226 |  2.00 |  2.07(50) |  0.632(43) | 0.630 | 0.768 |  0.14 |    flat | mid      |
| cross_court_net_shot     |  847 |   267 |  1.68 |   2.52(8) |  0.809(47) | 0.796 | 0.806 | -0.00 |    0.04 | healthy  |
| return_net               | 2387 |   536 |  1.23 |  1.23(80) |  0.853(80) | 0.853 | 0.856 |  0.00 |    flat | healthy  |
| lob                      | 3617 |   945 |  1.12 |  1.12(80) |   0.848(5) | 0.833 | 0.866 |  0.02 |    flat | healthy  |
| rush                     |  335 |    67 |  1.07 |   1.64(4) |  0.755(44) | 0.745 | 0.880 |  0.12 |    flat | mid      |
| drop                     | 2261 |   528 |  0.86 |   0.97(1) |  0.908(57) | 0.905 | 0.900 | -0.01 |    flat | healthy  |
| net_shot                 | 4139 |   956 |  0.75 |  0.75(80) |  0.920(57) | 0.918 | 0.911 | -0.01 |    flat | healthy  |
| smash                    | 2764 |   630 |  0.64 |   0.78(1) |  0.916(59) | 0.914 | 0.925 |  0.01 |    flat | healthy  |
| long_service             |  252 |    33 |  0.28 |   1.26(1) |  0.988(59) | 0.981 | 0.976 | -0.01 |    0.02 | healthy  |
| clear                    | 1897 |   382 |  0.24 |   0.65(1) |  0.971(37) | 0.970 | 0.973 |  0.00 |    flat | healthy  |
| short_service            | 1312 |   291 |  0.12 |   0.69(1) |  0.993(52) | 0.991 | 0.987 | -0.01 |    flat | healthy  |

### bst_25_baseline  (bst_25 / split_bst_baseline)  run run_20260530_210600_435552

macro plateau ~epoch 31 (run-max 0.820); macro 10/25/40/80 = 0.777 / 0.805 / 0.813 / 0.819; best-macro epochs [41, 49, 57, 58, 74]
alpha-vs-valF1max corr = -0.95; above-mean alpha budget = 5.67, of which 3.68 (65%) sits on classes already plateaued by epoch 31 (floor classes alone: 0.00)

| class                    | n_tr | n_val | a_fin | a_pk(ep) | v_max(ep) | v_fin | t_max | gap | dV>plat | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Top_drive                |  548 |    81 |  2.15 |  2.17(62) |  0.611(19) | 0.567 | 0.742 |  0.13 |    flat | ceiling  |
| Bottom_push              | 1030 |   169 |  2.08 |  2.08(80) |  0.607(31) | 0.603 | 0.748 |  0.14 |    flat | ceiling  |
| Top_push                 | 1004 |   156 |  1.96 |  1.96(80) |  0.686(64) | 0.678 | 0.761 |  0.08 |    0.02 | mid      |
| Bottom_drive             |  594 |    98 |  1.95 |  1.99(53) |  0.658(69) | 0.651 | 0.772 |  0.11 |    flat | mid      |
| Top_cross_court_net_shot |  503 |    60 |  1.61 |  2.36(10) |  0.630(45) | 0.626 | 0.811 |  0.18 |    0.03 | mid      |
| Bottom_cross_court_net_shot |  498 |    61 |  1.33 |   2.25(8) |  0.754(49) | 0.747 | 0.848 |  0.09 |    0.02 | mid      |
| Top_return_net           | 1254 |   179 |  1.22 |  1.22(80) |  0.819(53) | 0.812 | 0.853 |  0.03 |    flat | healthy  |
| Bottom_lob               | 2082 |   379 |  1.11 |  1.11(80) |   0.812(3) | 0.787 | 0.866 |  0.05 |    flat | healthy  |
| Bottom_return_net        | 1374 |   207 |  1.10 |  1.10(79) |  0.834(52) | 0.825 | 0.867 |  0.03 |    flat | healthy  |
| Top_rush                 |  203 |    33 |  1.10 |   1.45(8) |  0.777(48) | 0.731 | 0.874 |  0.10 |    0.02 | mid      |
| Top_lob                  | 1954 |   281 |  1.07 |  1.07(80) |  0.811(37) | 0.791 | 0.870 |  0.06 |    flat | healthy  |
| Bottom_rush              |  154 |    31 |  0.98 |   1.73(6) |  0.784(63) | 0.771 | 0.889 |  0.11 |    0.05 | mid      |
| unknown                  |  843 |   236 |  0.85 |  0.86(42) |  0.915(53) | 0.910 | 0.899 | -0.02 |    flat | healthy  |
| Top_drop                 | 1245 |   247 |  0.81 |   0.83(1) |  0.884(25) | 0.873 | 0.904 |  0.02 |    flat | healthy  |
| Bottom_drop              | 1193 |   177 |  0.80 |   0.90(1) |  0.881(63) | 0.875 | 0.904 |  0.02 |    flat | healthy  |
| Top_net_shot             | 2402 |   357 |  0.69 |   0.88(1) |  0.907(45) | 0.905 | 0.918 |  0.01 |    flat | healthy  |
| Bottom_net_shot          | 2282 |   338 |  0.68 |   0.92(1) |  0.914(58) | 0.912 | 0.919 |  0.01 |    flat | healthy  |
| Bottom_smash             | 1368 |   210 |  0.65 |   0.79(1) |  0.903(53) | 0.901 | 0.923 |  0.02 |    flat | healthy  |
| Top_smash                | 1585 |   305 |  0.63 |   0.70(1) |  0.918(14) | 0.907 | 0.923 |  0.01 |    flat | healthy  |
| Bottom_long_service      |  114 |    27 |  0.55 |   1.29(1) |  0.968(34) | 0.955 | 0.940 | -0.03 |    flat | healthy  |
| Top_long_service         |   97 |    19 |  0.51 |   1.35(2) |  0.831(50) | 0.809 | 0.946 |  0.12 |    0.04 | healthy  |
| Top_short_service        |  733 |    84 |  0.33 |   0.86(1) |  0.966(58) | 0.962 | 0.961 | -0.01 |    flat | healthy  |
| Bottom_short_service     |  763 |    90 |  0.33 |   0.84(1) |  0.961(49) | 0.955 | 0.962 |  0.00 |    flat | healthy  |
| Top_clear                |  925 |   197 |  0.30 |   0.82(1) |  0.962(31) | 0.959 | 0.966 |  0.00 |    flat | healthy  |
| Bottom_clear             |  946 |   214 |  0.22 |   0.82(1) |  0.973(60) | 0.971 | 0.974 |  0.00 |    flat | healthy  |

### bst_24_baseline  (bst_24 / split_bst_baseline)  run run_20260530_225714_593038

macro plateau ~epoch 31 (run-max 0.829); macro 10/25/40/80 = 0.786 / 0.816 / 0.821 / 0.829; best-macro epochs [32, 39, 44, 44, 53]
alpha-vs-valF1max corr = -0.98; above-mean alpha budget = 6.24, of which 3.87 (62%) sits on classes already plateaued by epoch 31 (floor classes alone: 0.00)

| class                    | n_tr | n_val | a_fin | a_pk(ep) | v_max(ep) | v_fin | t_max | gap | dV>plat | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Top_drive                |  548 |    81 |  2.24 |  2.25(65) |  0.605(33) | 0.590 | 0.743 |  0.14 |    flat | ceiling  |
| Bottom_push              | 1030 |   169 |  2.15 |  2.16(77) |  0.611(44) | 0.591 | 0.749 |  0.14 |    flat | ceiling  |
| Bottom_drive             |  594 |    98 |  2.02 |  2.05(47) |  0.638(62) | 0.638 | 0.766 |  0.13 |    0.03 | mid      |
| Top_push                 | 1004 |   156 |  2.01 |  2.02(76) |  0.687(50) | 0.652 | 0.766 |  0.08 |    flat | mid      |
| Top_cross_court_net_shot |  503 |    60 |  1.67 |   2.46(9) |  0.651(80) | 0.651 | 0.814 |  0.16 |    0.07 | mid      |
| Bottom_cross_court_net_shot |  498 |    61 |  1.38 |   2.27(7) |  0.737(44) | 0.735 | 0.846 |  0.11 |    0.03 | mid      |
| Top_return_net           | 1254 |   179 |  1.26 |  1.26(79) |  0.815(80) | 0.815 | 0.854 |  0.04 |    0.02 | healthy  |
| Bottom_return_net        | 1374 |   207 |  1.15 |  1.15(79) |  0.836(45) | 0.828 | 0.865 |  0.03 |    flat | healthy  |
| Bottom_lob               | 2082 |   379 |  1.14 |  1.15(77) |  0.800(53) | 0.798 | 0.867 |  0.07 |    flat | healthy  |
| Top_lob                  | 1954 |   281 |  1.10 |  1.10(80) |  0.811(20) | 0.803 | 0.871 |  0.06 |    flat | healthy  |
| Top_rush                 |  203 |    33 |  1.07 |   1.45(8) |  0.798(33) | 0.756 | 0.889 |  0.09 |    flat | mid      |
| Bottom_rush              |  154 |    31 |  1.03 |   1.70(5) |  0.764(77) | 0.762 | 0.891 |  0.13 |    0.04 | mid      |
| Bottom_drop              | 1193 |   177 |  0.86 |   0.93(1) |  0.884(45) | 0.874 | 0.899 |  0.01 |    flat | healthy  |
| Top_drop                 | 1245 |   247 |  0.84 |   0.85(1) |  0.886(48) | 0.886 | 0.905 |  0.02 |    flat | healthy  |
| Top_net_shot             | 2402 |   357 |  0.72 |   0.87(1) |  0.914(80) | 0.914 | 0.917 |  0.00 |    0.02 | healthy  |
| Bottom_net_shot          | 2282 |   338 |  0.69 |   0.90(1) |  0.912(44) | 0.911 | 0.919 |  0.01 |    flat | healthy  |
| Bottom_smash             | 1368 |   210 |  0.69 |   0.80(1) |  0.907(39) | 0.896 | 0.920 |  0.01 |    flat | healthy  |
| Top_smash                | 1585 |   305 |  0.66 |   0.69(1) |  0.921(64) | 0.919 | 0.923 |  0.00 |    flat | healthy  |
| Top_clear                |  925 |   197 |  0.31 |   0.76(1) |  0.958(46) | 0.955 | 0.967 |  0.01 |    flat | healthy  |
| Bottom_long_service      |  114 |    27 |  0.31 |   1.25(1) |  0.995(75) | 0.988 | 0.972 | -0.02 |    flat | healthy  |
| Bottom_clear             |  946 |   214 |  0.24 |   0.80(1) |   0.974(9) | 0.971 | 0.974 |  0.00 |    flat | healthy  |
| Top_long_service         |   97 |    19 |  0.23 |   1.36(2) |  1.000(26) | 0.972 | 0.983 | -0.02 |    flat | healthy  |
| Bottom_short_service     |  763 |    90 |  0.13 |   0.85(1) |  0.998(23) | 0.996 | 0.987 | -0.01 |    flat | healthy  |
| Top_short_service        |  733 |    84 |  0.10 |   0.85(1) |  0.996(46) | 0.994 | 0.990 | -0.01 |    flat | healthy  |

### une_v1_14_v2  (une_v1_14 / split_v2)  run run_20260531_005535_005154

macro plateau ~epoch 28 (run-max 0.768); macro 10/25/40/80 = 0.717 / 0.754 / 0.764 / 0.765; best-macro epochs [28, 43, 54, 62, 66]
alpha-vs-valF1max corr = -0.95; above-mean alpha budget = 3.20, of which 2.70 (84%) sits on classes already plateaued by epoch 28 (floor classes alone: 0.00)

| class                    | n_tr | n_val | a_fin | a_pk(ep) | v_max(ep) | v_fin | t_max | gap | dV>plat | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wrist_smash              |  978 |   331 |  1.97 |  1.99(60) |  0.479(13) | 0.412 | 0.679 |  0.20 |    flat | memorise |
| push                     | 1882 |   389 |  1.56 |  1.56(80) |  0.637(45) | 0.627 | 0.741 |  0.10 |    flat | mid      |
| drive                    | 1041 |   226 |  1.51 |  1.54(47) |  0.640(34) | 0.630 | 0.756 |  0.12 |    flat | mid      |
| passive_drop             |  796 |   224 |  1.49 |  1.49(80) |  0.647(28) | 0.600 | 0.755 |  0.11 |    flat | mid      |
| cross_court_net_shot     |  847 |   267 |  1.41 |   2.11(7) |  0.781(66) | 0.777 | 0.771 | -0.01 |    0.05 | mid      |
| drop                     | 1465 |   304 |  1.17 |  1.17(78) |  0.698(59) | 0.690 | 0.808 |  0.11 |    flat | mid      |
| smash                    | 1786 |   299 |  1.09 |  1.10(58) |  0.661(75) | 0.657 | 0.822 |  0.16 |    0.02 | mid      |
| return_net               | 2387 |   536 |  0.94 |  0.94(77) |  0.852(79) | 0.850 | 0.844 | -0.01 |    flat | healthy  |
| lob                      | 3617 |   945 |  0.87 |  0.87(80) |  0.848(62) | 0.840 | 0.856 |  0.01 |    flat | healthy  |
| rush                     |  335 |    67 |  0.87 |   1.47(3) |  0.774(71) | 0.769 | 0.866 |  0.09 |    0.03 | mid      |
| net_shot                 | 4139 |   956 |  0.58 |   0.63(1) |  0.914(47) | 0.912 | 0.904 | -0.01 |    flat | healthy  |
| long_service             |  252 |    33 |  0.24 |   1.25(1) |  0.992(71) | 0.989 | 0.969 | -0.02 |    0.03 | healthy  |
| clear                    | 1897 |   382 |  0.19 |   0.59(1) |  0.971(55) | 0.969 | 0.969 | -0.00 |    flat | healthy  |
| short_service            | 1312 |   291 |  0.10 |   0.65(1) |  0.993(44) | 0.991 | 0.986 | -0.01 |    flat | healthy  |
