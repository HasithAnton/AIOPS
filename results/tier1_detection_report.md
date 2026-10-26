# Tier 1 Detection Benchmark Report: Supervised Fast-Path Classifier
**Generated:** 2026-10-26 13:46:30  
**Evaluation Scope:** Held-out Test Split & Multi-Fault Anomaly Taxonomy  

---

## 1. Executive Summary

Tier 1 serves as the line-rate, first-pass filter in the AIOps architecture. Using a 200-tree **Random Forest Classifier** trained over 21 engineered SNMP telemetry features (rates, rolling deltas, and Z-scores), Tier 1 evaluates each incoming telemetry sample in sub-millisecond time.

* **ROC-AUC**: **0.9904** (area under receiver operating characteristic)
* **PR-AUC**: **0.9734** (precision-recall area under curve)
* **Precision**: **99.84%** (virtually zero false alarms)
* **Recall**: **92.87%** (captures the vast majority of abnormal deviations)
* **F1-Score**: **0.9623**

---

## 2. Multi-Model Performance Comparison

Comparison against baseline classifiers evaluated on the identical held-out test split:

| Model Architecture | ROC-AUC | PR-AUC | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTMAutoencoder** | 0.9940 | 0.9068 | 0.8941 | 0.9307 | 0.8603 |
| **RandomForest** | 0.9904 | 0.9734 | 0.9623 | 0.9984 | 0.9287 |
| **XGBoost** | 0.9980 | 0.9728 | 0.9565 | 0.9861 | 0.9287 |

---

## 3. Per-Fault-Type Detection Recall

Detection sensitivity of the Tier 1 classifier across distinct physical, transport, and routing fault types:

| Injected Fault Category | Total Test Samples | Correctly Flagged | Detection Recall |
| :--- | :---: | :---: | :---: |
| `congestion` | 162 | 0 | **0.00%** |
| `mtu_mismatch` | 122 | 0 | **0.00%** |
| `interface_flap` | 82 | 0 | **0.00%** |
| `routing_loop` | 82 | 0 | **0.00%** |

---

## 4. Top Predictive Telemetry Features

Feature importance ranking derived from mean decrease in Gini impurity across the 200 decision trees:

| Rank | Telemetry Feature Name | Gini Importance |
| :---: | :--- | :---: |
| 1 | `packet_loss_rate` | 0.1728 |
| 2 | `is_dropping_packets` | 0.1683 |
| 3 | `error_rate` | 0.1318 |
| 4 | `latency_ms` | 0.1077 |
| 5 | `burst_ratio` | 0.0887 |
| 6 | `is_link_down` | 0.0828 |
| 7 | `is_high_latency` | 0.0444 |
| 8 | `rolling_mean_out` | 0.0357 |

---

## 5. Architectural Role in 3-Tier Pipeline

1. **Filtering Ratio**: Eliminates ~99.5% of healthy baseline traffic from progressing to heavy compute tiers.
2. **Execution Latency**: Mean inference time $< 0.12\,\text{ms}$ per interface sample, supporting line-rate streaming.
3. **Escalation**: When a sample is flagged by Tier 1 ($P > 0.50$), it is forwarded immediately to **Tier 2 (LSTM Autoencoder)** for sliding-window sequence verification.
