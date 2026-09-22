```
UNSUPERVISED FRAUD/ANOMALY DETECTION - RESULTS REPORT
=======================================================

1. Model & split configuration
==============================
Model:                 isolation_forest
Contamination target:  0.02
Operating threshold:   0.710495
Feature count:         85

Temporal split:
  train      n= 2431  2023-01-02T16:00:06 -> 2023-10-03T17:34:16
  validation n=  302  2023-10-04T04:08:46 -> 2023-11-02T19:19:15
  test       n=  560  2023-11-03T16:08:00 -> 2024-01-01T18:21:50


2. Anomaly-score metrics per window
===================================
window          n  flagged     rate     mean      std      p95      p99
train        2431       49    2.02%   0.3610   0.1338   0.6251   0.7761
validation    302        7    2.32%   0.4244   0.1309   0.6386   0.8551
test          560       38    6.79%   0.4229   0.1592   0.7455   0.9574


3. Heuristic alignment (sanity check, not ground truth)
=======================================================
Spearman correlation vs. hand-built heuristic: 0.4734
Overlap among top-flagged transactions:        9.09%
Note: Sanity check against a transparent hand rule, not ground truth. The dataset is unlabelled.


4. Top feature importance (SHAP mean |value|)
=============================================
Method: shap_tree_explainer (sample_size=560)

  #  feature                             importance
  1  Month                                   0.1469
  2  USState_Texas                           0.1428
  3  TransactionType_Debit                   0.1297
  4  Amount_Dev_Account                      0.1134
  5  USState_California                      0.1130
  6  CustomerOccupation_Student              0.0911
  7  CustomerOccupation_Retired              0.0845
  8  CustomerOccupation_Engineer             0.0830
  9  USState_Frequency                       0.0828
 10  USState_Arizona                         0.0806
 11  Account_Txn_Count                       0.0796
 12  Channel_Branch                          0.0768
 13  Is_Night                                0.0660
 14  USState_Florida                         0.0624
 15  Amount_Dev_MerchantID                   0.0593


5. Flagged vs. normal transactions (test window, largest gaps)
==============================================================
feature                          flagged_mean    normal_mean    std_gap
TransactionAmount                    877.5966       272.9757     1.9386
Amount_Dev_Hour                        2.1802        -0.0665     1.9349
Amount_Dev_USState                     2.1237        -0.0754     1.9205
Amount_Dev_MerchantID                  2.7165        -0.0509     1.8745
Amount_To_Balance_Ratio                1.2645         0.1267     1.8281
Is_Weekend                             0.3421         0.0383     1.2900
Log_Transaction_Amount                 6.3506         5.0734     1.0015
Amount_Dev_Account                     4.9407         0.5920     0.9870
CustomerOccupation_Student             0.5789         0.2280     0.8086
Is_Night                               0.2368         0.0498     0.7727
LoginAttempts                          1.4474         1.0670     0.7522
Hour                                  12.5789        15.5728    -0.7419
Log_Account_Balance                    7.4763         8.1974    -0.6838
USState_Connecticut                    0.0526         0.0038     0.5795
USState_Maine                          0.0526         0.0038     0.5795


6. Full-dataset overview (all scored transactions)
==================================================
Total transactions:     3293
Total accounts:         654
States covered:         52
Period:                 2023-01-02T16:00:06 -> 2024-01-01T18:21:50
Flagged transactions:   108 (3.28%)
Mean anomaly score:     0.3847
Total amount:           $981,758.06
Flagged amount:         $93,262.77
Avg transaction amount: $298.13
Avg account balance:    $5,145.78


7. Geo (state) reconciliation
=============================
Dataset states:         52
Matched to GeoJSON:     52
Unmatched:              []
Reconciled OK:          True


8. Top 10 states by anomaly rate (min-support filtered)
=======================================================
state                          n  flagged     rate mean_score
Connecticut                   36        6   16.67%     0.4904
Alaska                        20        3   15.00%     0.4568
West Virginia                 20        3   15.00%     0.5004
Kansas                        29        4   13.79%     0.5313
Minnesota                     57        6   10.53%     0.5004
Arkansas                      30        3   10.00%     0.5021
Maine                         20        2   10.00%     0.4592
Montana                       20        2   10.00%     0.4778
New Hampshire                 20        2   10.00%     0.4291
South Dakota                  20        2   10.00%     0.5318


9. Multi-seed stability analysis (Isolation Forest robustness)
==============================================================
Model: isolation_forest  |  runs: 10  |  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
Test window size: 560

CAUTION: the mean stability score over ALL test transactions is
diluted by the majority never flagged in any run and is NOT a
meaningful headline number on its own. Read the conditional
statistics below instead.

Mean stability (all 560 test rows):        0.0554
  -> flagged in >=1 run:    52 (9.29% of test set)

Among rows flagged at least once (the only rows for which
'stability' is a meaningful question):
  mean stability given flagged>=1:   0.5962
  median stability given flagged>=1: 0.7500

Breakdown over the full test set:
  stability == 100%  (flagged every run):    16 (2.86%)
  stability >=  80%:                         26 (4.64%)
  stability >=  50%:                         31 (5.54%)
  stability >   0%   (ever flagged):         52 (9.29%)

Jaccard similarity of the flagged SET, across the 45 seed pairs:
  mean=0.6886  median=0.6857  std=0.0652  min=0.5676  max=0.8621

Spearman rank correlation of the raw SCORE (threshold-independent):
  mean=0.8991  median=0.9012  std=0.0135  min=0.8693  max=0.9256
  -> The ranking is far more stable than the binary flag: seeds agree
     on relative anomalousness (~0.90 correlation) much more than they
     agree on which rows cross a hard threshold near the boundary.
     Volatility concentrates at the cutoff, not in the ranking itself.

  seed  threshold  n_flagged     rate  score_mean
    42     0.7105         38    6.79%      0.4229
    43     0.7019         28    5.00%      0.3819
    44     0.6740         29    5.18%      0.3818
    45     0.7619         33    5.89%      0.4464
    46     0.6962         34    6.07%      0.4031
    47     0.7742         29    5.18%      0.4424
    48     0.7208         26    4.64%      0.3907
    49     0.7439         31    5.54%      0.4372
    50     0.7778         28    5.00%      0.4550
    51     0.7517         34    6.07%      0.4355

Threshold-generalisation note: the operating threshold is the 98% quantile of TRAIN scores; applied to test it flags 6.79% of rows, 3.4x the 2.00% contamination target. This is a score-distribution shift between the train and test periods, not a bug in the threshold — worth stating explicitly in the paper as a limitation of a fixed, train-derived cutoff.

```
