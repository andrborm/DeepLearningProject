# PFΔ distribution-shift test (base model, 1000 PFΔ case14 N-0/N-1 test samples)

First test of the datakit-trained GENCO case14 base model on PFΔ data. **N-2 was not converted, read or run.**
Scripts: `scripts/vendor/batch_convert_pfdelta.py` (conversion), `scripts/make_pfdelta_subset.py`,
`scripts/run_inference.py --dataset pfdelta`, `scripts/pfdelta_shift_eval.py`.

## Data

| Step | Detail |
|---|---|
| Conversion | datakit repo's `pfdelta/batch_convert_pfdelta.py` (vendored unmodified, commit `5a46ffd`) on PFΔ case14 **feasible N-0 (56,000) and N-1 (29,000)** → `data/pfdelta_case14/{n,n-1}/raw/` (87 MB + 46 MB, 2 min + 1 min). Near-infeasible and N-2 not converted |
| Schema | Same as datakit except missing `load_scenario_idx` and DC-baseline columns (not needed). Grid parameters (`rate_a`, limits, gen limits) identical to the checkpoint's training data. PFΔ also turns an outaged generator's bus into PQ, like datakit |
| Test subset | PFΔ's fixed test split = last 2,000 positions of each topology's shuffled list. Seeded sample (seed 42) of **500 N-0 + 500 N-1** (N-1: 216 branch, 284 generator outages) → `data/pfdelta_test1000/`, mapping `pfdelta_test_subset_1000.csv` |
| Conversion checks | graphkit's true flows (recomputed from true Vm/Va via admittances) match the converted flows within 5e-4 MW / MVAr on all 39,568 directed edges. Labels from the converted parquet match labels computed from the original JSON on 100 % of samples |

**The shift in inputs.** Total load: datakit test 172–315 MW (median 224), PFΔ test 180–399 MW (median 339).
**626 of 1000 PFΔ samples (485 N-0, 141 N-1) have more total load than any datakit test sample.** Reactive load is also
higher (median 103 vs 64 MVAr). The true voltages are much lower (median min Vm 0.94 vs 1.01), and generator voltage
setpoints differ (median max Vm 1.01 vs 1.06).

## 1. Errors: PFΔ vs datakit

Absolute error of the unknowns (`pfdelta_vs_datakit_errors.csv`, `pfdelta_errors_by_topology.csv`):

| Variable | Bus type | datakit RMSE | PFΔ RMSE | PFΔ median | PFΔ p95 | PFΔ p99 | RMSE ratio | median ratio |
|---|---|---|---|---|---|---|---|---|
| Vm (p.u.) | PQ | 8.8e-5 | **0.067** | 0.0086 | 0.167 | 0.305 | 760× | 650× |
| Vm (p.u.) | REF | 6.2e-4 | 0.027 | 0.022 | 0.047 | 0.057 | 43× | 72× |
| Va (rad) | PQ | 7.1e-4 | 0.062 | 0.027 | 0.135 | 0.196 | 88× | 210× |
| Va (rad) | PV | 7.5e-4 | 0.048 | 0.018 | 0.110 | 0.165 | 64× | 140× |
| Qg (MVAr) | PV | 0.61 | 74.6 | 11.6 | 182 | 336 | 123× | 390× |
| Qg (MVAr) | REF | 1.26 | 55.2 | 42.6 | 93.5 | 107 | 44× | 68× |
| Pg (MW) | REF | 0.12 | **34.5** | 19.9 | 70.7 | 86.1 | 290× | 1,130× |

- **The model breaks down under the PFΔ shift.** Errors grow 40–1,000×. The *median* grows even more than the RMSE, so
  the whole distribution moves, not just the tail. PQ Vm errors up to 0.3 p.u. (p99) are physically meaningless.
- **By topology**, N-0 is far worse than N-1: PQ Vm RMSE 0.091 vs 0.029, slack Pg 46 vs 15 MW, median overall score 183
  vs 43 × the datakit level. This matches the PFΔ confound (N-0 samples are the most heavily loaded) and the fact that
  485 of the 500 N-0 samples lie above the datakit load range. **N-0 vs N-1 differences here are load level, not topology.**
- **Systematic bias**: predicted PQ voltages are too low. Mean error −0.030 p.u. (N-0: −0.055, and 98.7 % of errors are
  negative; N-1: −0.0075, 76 % negative). On datakit the error was unbiased (mean −1e-5).
- Graphkit's own RMSE table shows the same picture (ratio to model card 40–580×). Power-balance residual: PBE mean 17.9
  MVA (datakit 0.0026), max 781 MVA.

## 2. Predict-then-check (slack Vm treated as known setpoint)

`pfdelta_predict_then_check.csv`

| Scope | Target | TP | FP | FN | TN | Precision | Recall | Base rate |
|---|---|---|---|---|---|---|---|---|
| all | y_voltage | 508 | 161 | 7 | 324 | **0.76** | **0.99** | 51.5 % |
| all | y_overload | 317 | 173 | 4 | 506 | **0.65** | **0.99** | 32.1 % |
| all | any violation | 606 | 123 | 7 | 264 | 0.83 | 0.99 | 61.3 % |
| N-0 | y_voltage | 316 | 115 | 0 | 69 | 0.73 | 1.00 | 63.2 % |
| N-0 | y_overload | 293 | 146 | 4 | 57 | 0.67 | 0.99 | 59.4 % |
| N-1 | y_voltage | 192 | 46 | 7 | 255 | 0.81 | 0.97 | 39.8 % |
| N-1 | y_overload | 24 | 27 | 0 | 449 | 0.47 | 1.00 | 4.8 % |
| datakit reference | y_voltage / y_overload | | | | | 1.00 / 1.00 | 0.96 / 1.00 | 2.8 % / 3.7 % |

- **Recall stays near 1, precision drops.** The degraded model over-predicts violations, it does not miss them.
- **Voltage false positives are not borderline.** All 718 triggering buses in the 161 FP samples are PQ buses predicted
  *below 0.94*, while their true Vm is on median **0.048 p.u. above** the limit. This is the negative Vm bias from §1.
- **Overload false positives are near-misses.** True loading of FP samples is median 95.5 % of `rate_a` (TN: 76.5 %).
  Lower predicted voltages at the same load mean larger predicted flows, which push near-limit branches over.
- The 7 voltage FNs are all N-1 and close to the limit (true min Vm median 0.0025 p.u. below 0.94). The 4 overload
  FNs are N-0 (true loading median 104 % of `rate_a`).
- In N-1 the overload base rate is only 4.8 %, so precision 0.47 means about 1 false alarm per true alarm.

## 3. Reconstruction error and residual on wrong vs right labels

`pfdelta_wrong_vs_right_auc.csv`, per-sample `pfdelta_labels_true_vs_pred.csv`. Scores per sample: reconstruction error
= `overall_score_vs_datakit` (1.0 = datakit RMSE level), power-balance residual = `pbe_mean`, `pbe_max` (MVA).

Medians per confusion cell:

| Label | Cell | n | Overall score | PBE mean | PBE max |
|---|---|---|---|---|---|
| voltage | TN | 324 | 44 | 0.90 | 2.4 |
| voltage | TP | 508 | 109 | 3.9 | 12.6 |
| voltage | **FP** | 161 | **241** | **39.7** | **178** |
| voltage | FN | 7 | 25 | 0.41 | 1.6 |
| overload | TN | 506 | 44 | 0.85 | 2.4 |
| overload | TP | 317 | 217 | 25.2 | 111 |
| overload | **FP** | 173 | 138 | 5.0 | 21.3 |
| overload | FN | 4 | 285 | 23.4 | 94 |

How well does each score flag a **wrong** predict-then-check label? (AUC; 0.5 = no signal)

| Scope | Wrong label | n wrong / right | AUC overall score | AUC PBE mean | AUC PBE max |
|---|---|---|---|---|---|
| all | voltage | 168 / 832 | 0.78 | **0.80** | 0.80 |
| all | overload | 177 / 823 | 0.68 | 0.69 | 0.68 |
| all | either | 296 / 704 | 0.74 | 0.76 | 0.76 |
| N-0 | voltage | 115 / 385 | 0.77 | 0.80 | 0.79 |
| N-0 | overload | 150 / 350 | **0.39** | **0.43** | 0.44 |
| N-1 | voltage | 53 / 447 | 0.76 | 0.79 | 0.79 |
| N-1 | overload | 27 / 473 | 0.80 | **0.83** | 0.79 |

- **Voltage errors are flagged fairly well by both scores** (AUC ≈ 0.8, stable across N-0 and N-1). Voltage FPs have by
  far the largest residual (median PBE 40 MVA vs 0.9 for TN): when the model invents an under-voltage, its own physics
  check disagrees.
- **Overload errors are only flagged in N-1** (AUC 0.80–0.83). In N-0 the scores are *inverted* (AUC 0.39–0.44):
  overload FPs have lower error and residual than the correctly labeled, heavily loaded TPs. Error and residual track
  "how stressed is the sample" more than "is this label wrong".
- **Under shift the residual tracks reconstruction error closely** (Spearman 0.91; on datakit only 0.25). The residual
  is available at inference time without ground truth, so it is the usable one of the two.
- The residual works as a **confidence signal**, not as an anomaly score: TPs (true violations) also have high residuals
  (voltage TP 3.9, overload TP 25 MVA vs TN ≈ 0.9). A high residual means "this prediction and therefore this label is
  unreliable", and it also correlates with real stress.

## Takeaways for the next stages

1. **Stage 2 fine-tuning is clearly needed** before any PFΔ-based anomaly result is meaningful. The base model
   extrapolates beyond its training load range and is biased low in voltage. The fine-tuning plan stands (N-0/N-1
   feasible training pool outside the test positions; N-2 held out).
2. **Predict-then-check after fine-tuning is the baseline to beat.** Even degraded, it keeps recall ≈ 0.99. Its weakness is
   false alarms (precision 0.65–0.76).
3. **The residual is a candidate for approach A in a "flag uncertain predictions" role.** Combining predict-then-check with
   a residual threshold could cut voltage false alarms (voltage-FP residuals are ≈ 10× those of TPs). The N-0 overload
   inversion shows it must be evaluated per topology / load level.
4. Report PFΔ results per topology and ideally per load bin. With the N-0 load confound, pooled numbers hide the story.
