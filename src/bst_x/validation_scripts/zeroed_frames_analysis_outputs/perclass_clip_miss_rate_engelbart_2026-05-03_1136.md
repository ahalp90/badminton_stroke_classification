# Per-class whole-clip shuttle miss rate

- shuttle-dir: `/scratch/comp320a/ShuttleSet/shuttle_npy_flat`
- clips-csv: `/home/ahalperi/badminton_stroke_classifier/notebooks/clips_master.csv`
- host: `engelbart`
- timestamp: 2026-05-03_1136
- edge trim for central window: 15
- clips processed (non-unknown): 32203

## Whole-clip shuttle miss rate

Per-clip metric: `(visibility == 0).sum() / len(visibility)`. Per class: median, 1 SD, 2 SD across that class's clips. Sorted by median descending.

| Class | n_clips | median | mean | sd1 | sd2 | min | max |
|---|---|---|---|---|---|---|---|
| push | 2652 | 0.0000 | 0.0104 | 0.0435 | 0.0870 | 0.0000 | 0.5116 |
| back_court_drive | 435 | 0.0000 | 0.0154 | 0.0503 | 0.1007 | 0.0000 | 0.3514 |
| return_net | 3374 | 0.0000 | 0.0111 | 0.0481 | 0.0962 | 0.0000 | 0.8261 |
| net_shot | 5824 | 0.0000 | 0.0077 | 0.0368 | 0.0737 | 0.0000 | 0.5424 |
| lob | 4879 | 0.0000 | 0.0910 | 0.1652 | 0.3304 | 0.0000 | 0.8571 |
| smash | 2362 | 0.0000 | 0.1374 | 0.2034 | 0.4069 | 0.0000 | 0.9464 |
| clear | 2661 | 0.0000 | 0.1194 | 0.1656 | 0.3312 | 0.0000 | 0.8367 |
| drop | 1979 | 0.0000 | 0.0811 | 0.1509 | 0.3019 | 0.0000 | 0.6333 |
| defensive_return_drive | 382 | 0.0000 | 0.0205 | 0.0741 | 0.1482 | 0.0000 | 0.7500 |
| cross_court_net_shot | 1226 | 0.0000 | 0.0050 | 0.0256 | 0.0512 | 0.0000 | 0.4167 |
| wrist_smash | 1559 | 0.0000 | 0.0849 | 0.1640 | 0.3280 | 0.0000 | 0.8833 |
| passive_drop | 1198 | 0.0000 | 0.0228 | 0.0778 | 0.1555 | 0.0000 | 0.5714 |
| rush | 471 | 0.0000 | 0.0289 | 0.0935 | 0.1870 | 0.0000 | 0.5263 |
| short_service | 1858 | 0.0000 | 0.0163 | 0.0864 | 0.1727 | 0.0000 | 1.0000 |
| defensive_return_lob | 278 | 0.0000 | 0.0248 | 0.0656 | 0.1312 | 0.0000 | 0.5797 |
| drive | 654 | 0.0000 | 0.0181 | 0.0704 | 0.1408 | 0.0000 | 0.6458 |
| long_service | 359 | 0.0000 | 0.2474 | 0.2963 | 0.5926 | 0.0000 | 0.9701 |
| driven_flight | 52 | 0.0000 | 0.0090 | 0.0313 | 0.0625 | 0.0000 | 0.1852 |

## Share of missing frames in central window [15, len - 15)

Per-clip metric: `(visibility[15:-15] == 0).sum() / (visibility == 0).sum()`. Per class: median, 1 SD, 2 SD across that class's clips. Excludes clips shorter than 30 frames (no central window) and clips with zero missing frames (ratio undefined). Same class ordering as the miss-rate table.

| Class | n_clips | median | mean | sd1 | sd2 | min | max | excl_short | excl_nomiss |
|---|---|---|---|---|---|---|---|---|---|
| push | 281 | 0.0000 | 0.0746 | 0.2347 | 0.4694 | 0.0000 | 1.0000 | 118 | 2253 |
| back_court_drive | 66 | 0.0000 | 0.1511 | 0.3093 | 0.6186 | 0.0000 | 1.0000 | 3 | 366 |
| return_net | 363 | 0.0000 | 0.0121 | 0.0836 | 0.1673 | 0.0000 | 1.0000 | 361 | 2650 |
| net_shot | 599 | 0.0000 | 0.0113 | 0.0692 | 0.1384 | 0.0000 | 0.7600 | 58 | 5167 |
| lob | 1684 | 0.7375 | 0.6097 | 0.3975 | 0.7949 | 0.0000 | 1.0000 | 24 | 3171 |
| smash | 985 | 0.5417 | 0.4824 | 0.2972 | 0.5944 | 0.0000 | 1.0000 | 53 | 1324 |
| clear | 1267 | 0.7931 | 0.6656 | 0.3630 | 0.7260 | 0.0000 | 1.0000 | 6 | 1388 |
| drop | 621 | 0.5161 | 0.4600 | 0.3002 | 0.6005 | 0.0000 | 1.0000 | 7 | 1351 |
| defensive_return_drive | 47 | 0.0000 | 0.0213 | 0.1443 | 0.2886 | 0.0000 | 1.0000 | 95 | 240 |
| cross_court_net_shot | 80 | 0.0000 | 0.0094 | 0.0562 | 0.1125 | 0.0000 | 0.4400 | 32 | 1114 |
| wrist_smash | 460 | 0.5000 | 0.4389 | 0.3132 | 0.6264 | 0.0000 | 1.0000 | 32 | 1067 |
| passive_drop | 185 | 0.0000 | 0.2408 | 0.3000 | 0.6001 | 0.0000 | 1.0000 | 6 | 1007 |
| rush | 44 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 114 | 313 |
| short_service | 141 | 0.0000 | 0.0354 | 0.1050 | 0.2101 | 0.0000 | 0.7045 | 67 | 1650 |
| defensive_return_lob | 57 | 0.0000 | 0.1266 | 0.3183 | 0.6365 | 0.0000 | 1.0000 | 27 | 194 |
| drive | 53 | 0.0000 | 0.0210 | 0.1001 | 0.2003 | 0.0000 | 0.5484 | 244 | 357 |
| long_service | 163 | 0.7209 | 0.7210 | 0.2347 | 0.4694 | 0.0000 | 1.0000 | 14 | 182 |
| driven_flight | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9 | 40 |
