# Phase 5: groundwork for hard cases, embeddings and the anomaly baseline

Base model (`genco-pf-datakit/case14_ieee/base/seed0`) on the 1000-sample case14 test subset (all N-1).
Scripts: graphkit `predict --get_embeddings` (command below) + `scripts/phase5_analysis.py`.
The **datakit/Julia N-k smoke test is deferred**: PFΔ covers the data needs (see README).

```bash
gridfm_graphkit predict --config configs/case14_base_infer.yaml \
  --model_path checkpoints/genco-pf-datakit/case14_ieee/base/seed0/best_model_state_dict.pt \
  --normalizer_stats checkpoints/genco-pf-datakit/case14_ieee/base/seed0/normalizer_stats.pt \
  --data_path data/case14_test1000 --log_dir mlruns --exp_name phase5_embeddings --run_name base_n1000 \
  --num_workers 0 --get_embeddings --output_path data/predictions/base_n1000      # 17 s
./venv/Scripts/python.exe scripts/phase5_analysis.py                              # 3 s
```
`predict` and Phase 4's `evaluate` give **bit-identical** predictions (max abs difference 0).

## 1. Error distribution: median / p95 / p99 next to RMSE

Absolute error of the unknowns, base model, 1000 samples (`inference_percentiles_by_bus_type.csv`,
`inference_percentiles_by_outage_type.csv`):

| Variable | Bus type | median | p95 | p99 | max | RMSE |
|---|---|---|---|---|---|---|
| Vm (p.u.) | PQ | 1.31e-5 | 1.31e-4 | 3.17e-4 | 3.67e-3 | 8.78e-5 |
| Vm (p.u.) | REF | 3.07e-4 | 1.17e-3 | 2.01e-3 | 4.81e-3 | 6.21e-4 |
| Va (rad) | PQ | 1.26e-4 | 4.85e-4 | 1.03e-3 | 2.04e-2 | 7.06e-4 |
| Va (rad) | PV | 1.31e-4 | 5.15e-4 | 1.06e-3 | 2.12e-2 | 7.48e-4 |
| Qg (MVAr) | PV | 0.030 | 1.21 | 2.47 | 9.34 | 0.606 |
| Qg (MVAr) | REF | 0.626 | 2.43 | 3.91 | 9.85 | 1.26 |
| Pg (MW) | REF | 0.0175 | 0.075 | 0.164 | 3.37 | 0.118 |

- The typical error is far below RMSE: PQ Vm median is **7× below RMSE**, PV Qg median **20× below**. RMSE is set by the tail.
- For Va, RMSE (7e-4) is above the p95 (5e-4). A handful of samples drive the RMSE.
- By outage type the medians are similar (e.g. Va PQ median 1.3e-4 branch vs 1.1e-4 gen). **Branch outages differ in
  the tail** (Va PQ p99 1.27e-3 vs 5.4e-4, max 2.0e-2 vs 9.8e-4). Generator outages have a higher median for PQ Vm
  (2.6e-5 vs 1.1e-5) and PV Qg (0.049 vs 0.027).
- Per-sample overall score: median 0.36, p95 1.15, p99 2.00, max **17.3** (sample 438).

## 2. Hard cases (top 5 %, `results/hard_cases_top5pct.csv`)

- 50 samples: **45 branch / 5 generator outages** (subset: 81 % / 19 %).
- Outaged elements over-represented: branch 3 (bus 1–3) ×9, branch 5 (2–3) ×7, branch 9 (4–5 transformer) ×5,
  generator 3 (bus 5) ×4, branch 0 (0–1) ×3.
- **Where the largest error sits**: Vm, Qg, Pg → almost always **bus 0 (slack)** (48 / 46 / 50 of 50); Va → **bus 2**
  (25), bus 1 (11), bus 11 (6). The slack absorbs all power mismatch, so its unknowns are the hardest.
- **Hard ≠ anomalous**: only 7 of the 50 hardest samples have a true voltage or overload violation (6.4 % base rate, so
  about the same). Spearman(overall score, y_voltage) = 0.12, with y_overload = 0.07. **Reconstruction error alone would be a
  poor anomaly score on this data** (relevant for approach A).

## 3. Embeddings

| Item | Value |
|---|---|
| What | final-layer bus hidden state `h_bus`, exactly the input of the Vm/Va output head (`return_embeddings=True` in graphkit 0.9.0.3) |
| Shape | **384 per bus** (hidden 48 × 8 attention heads); gen embeddings are not exported for PF models |
| Saved | `data/embeddings/base_bus_embeddings.npy` (1000 × 14 × 384, float32), `base_graph_embeddings_meanpool.npy` (1000 × 384), `sample_ids.npy` (subset ids 0..999, row order of both arrays; map to original scenarios via `test_subset_1000.csv`) |
| Values | range −0.16…21.7 (post-activation, mostly ≥ 0), no dead dimensions in the pooled embedding |
| PCA (mean-pooled, centered) | PC1 82 %, PC2 10 %, PC3 2 %; 2 PCs explain > 90 % |

**What the PCA shows** (`figures/fig3_embeddings_pca.png`):
- (b) Outage type: generator outages lie on one edge of the main cloud. The two types overlap but differ systematically.
- (c) **Overloads form a separate arm** along PC1. **−PC1 alone separates y_overload perfectly (AUC 1.00)**, while total
  system load alone gives AUC 0.51. The embedding encodes branch stress, not just load level.
- Voltage violations form a band at the lower edge of the main cloud: −PC2 gives AUC 0.87. **But much of that is load
  level**: total load alone gives AUC 0.83 (inverted: the over-voltages in this data occur at *light* load). PC1 and PC2
  correlate with total load (Spearman −0.62 / 0.72).
- (a) High reconstruction error is *not* concentrated in the anomaly regions. Sample 438 (worst error) is the far end of
  the overload arm and the only sample with both violations.

**Caveat for approach B**: the embedding is the representation the model decodes into Vm/Va. On a grid where
predictions are this accurate, a classifier on embeddings may simply relearn "predict the state, then check limits".
The interesting test is where predictions degrade (PFΔ N-2, near-infeasible).

## 4. Predict-then-check baseline

Labels from the model's predicted state vs limits, compared with ground truth (`predict_then_check_confusion.csv`,
`predict_then_check_voltage_variants.csv`, per-sample `labels_true_vs_pred_base_n1000.csv`):
- `y_voltage_pred` = any bus with predicted Vm outside [Vm_min, Vm_max]
- `y_overload_pred` = any directed edge with graphkit's `thermal_excess > 0` (predicted flow from the predicted state)

**thermal_excess check.** `thermal_excess = relu(|S_ft| − rate_a)` on every *directed* edge (both branch ends), in MVA.
`thermal_excess_target` (from the true state via the admittances) agrees with `relu(|S_raw| − rate_a)` from the raw
datakit flows to within **5e-5 MVA** on all 38,372 directed edges. Target labels agree with raw-data labels on all 1000
samples. **So `any(thermal_excess > 0)` is exactly `y_overload`**, and graphkit's branch predictions can be used directly.

| Target | Rule | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|---|
| y_overload | predicted flows vs rate_a | 37 | 0 | 0 | 963 | **1.00** | **1.00** |
| y_voltage | (a) naive: predicted Vm incl. slack | 27 | **415** | 1 | 557 | 0.06 | 0.96 |
| y_voltage | (b) tolerance 1e-3 p.u. | 24 | 19 | 4 | 953 | 0.56 | 0.86 |
| y_voltage | (b) tolerance 2e-3 p.u. | 20 | 1 | 8 | 971 | 0.95 | 0.71 |
| y_voltage | (c) **slack setpoint known** | 27 | 0 | 1 | 972 | **1.00** | **0.96** |

**Why (a) fails.** All 415 false positives are at **bus 0 (slack)**. The true slack Vm is exactly **1.06 = the upper
limit in 99.2 %** of scenarios. graphkit masks slack Vm (data_format.md §2), so the model predicts it (median error
3e-4), and any positive error reads as over-voltage. In real operation the slack voltage is a known setpoint, so rule
(c) is the fair one. A tolerance (b) cannot fix it without losing recall. Under (c) the only miss is sample 119, 5.5e-5
p.u. over the limit (a hair).

**Interpretation.** On in-distribution N-1 data the model is accurate enough that predict-then-check is near-perfect.
Anomaly detection only becomes a real problem when the state prediction degrades: PFΔ (different load distribution,
heavy under-voltage), N-2, near-infeasible. This baseline is what approaches A and B must beat there.

## Figures (`results/figures/`, 200 dpi PNG)

| File | Content |
|---|---|
| `fig1_error_distribution.png` | histogram of the per-sample overall score (log x), median / p95 / p99 lines, sample 438 marked |
| `fig2_error_by_outage_type.png` | absolute error by outage type for Vm (PQ), Va (PQ), Qg (PV), Pg (slack); log-scale boxes 25–75 %, whiskers 5–95 % |
| `fig3_embeddings_pca.png` | PCA of mean-pooled embeddings, colored by (a) overall error, (b) outage type, (c) true violation |

Colors from a validated categorical palette (CVD-safe for all pairs). The "both" class also has its own marker shape,
because aqua is below 3:1 contrast.
