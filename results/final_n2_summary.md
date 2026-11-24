# Final N-2 test (run once, frozen pipeline)

The setup frozen in `stage3_summary.md` was run once on unseen PFΔ case14 **N-2** contingencies. No design choice was
changed afterwards. Script: `scripts/final_n2_test.py` (`extract --model <m>`, `evaluate`).

| Item | Value |
|---|---|
| Conversion | datakit's `batch_convert_pfdelta.py` (vendored) on PFΔ case14 **feasible N-2**: 20,000 files → `data/pfdelta_case14/n-2/raw` (1.5 min) |
| Test subset | PFΔ's N-2 test split = last 2,000 positions of `raw_shuffle.json`. Seeded sample (seed 42) of **1,000** → `data/pfdelta_n2_test1000/`, mapping `pfdelta_n2_test_subset_1000.csv`. Outages: 263 branch–branch, 385 gen–gen, 352 branch–gen |
| Models | replay (final) and original. One `predict --get_embeddings` run per model (1,000 samples), checkpoint normalizer |
| Frozen parts | labeling rule, PtC, gate t (re-derived with the frozen rule on the same val data and asserted equal: original 1.27 MVA, replay ∞), B probes + decision thresholds loaded from `data/stage3_probes/` (no refit) |
| Load level | "within" = total Pd ≤ 321.8 MW, the largest total load in the checkpoint's TRAIN split; "above" = more. N-2: 836 within / 164 above. PFΔ test: 407 / 593. datakit test: 1,000 / 0. (The replay model has also trained on PFΔ N-0/N-1 loads.) |

Base rates on N-2: **61.8 % voltage, 10.7 % overload** (gen–gen outages: 70 % voltage, 0 % overload).
For comparison, PFΔ N-0/N-1 has 51.5 % / 32.1 %.

## Errors (unknowns only, 1,000 samples each)

`final_n2_errors.csv`, `final_n2_scores.csv`. Score = mean over Vm/Va/Pg/Qg of sample RMSE ÷ datakit base RMSE (1 = datakit level).

| Variable | Bus | Data | orig RMSE | **replay RMSE** | orig median | replay median | orig p95 | replay p95 | orig p99 | replay p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| Vm (p.u.) | PQ | datakit | 8.8e-5 | 2.4e-4 | 1.3e-5 | 4.1e-5 | 1.3e-4 | 4.7e-4 | 3.2e-4 | 1.0e-3 |
| Vm (p.u.) | PQ | PFΔ N-0/N-1 | 0.067 | 9.5e-4 | 0.0086 | 1.5e-4 | 0.167 | 0.0017 | 0.305 | 0.0039 |
| Vm (p.u.) | PQ | **N-2** | 0.036 | **0.0088** | 0.0052 | 0.0011 | 0.056 | 0.017 | 0.193 | 0.033 |
| Pg (MW) | REF | datakit | 0.12 | 0.21 | 0.018 | 0.079 | 0.075 | 0.27 | 0.16 | 0.62 |
| Pg (MW) | REF | PFΔ N-0/N-1 | 34.5 | 1.19 | 19.9 | 0.69 | 70.7 | 2.07 | 86.1 | 3.97 |
| Pg (MW) | REF | **N-2** | 15.5 | **5.90** | 6.1 | 2.39 | 33.7 | 9.3 | 58.2 | 24.7 |
| Qg (MVAr) | PV | **N-2** | 43.0 | **11.3** | 5.1 | 1.31 | 87.4 | 22.6 | 159 | 46.4 |
| Va (rad) | PQ | **N-2** | 0.042 | **0.019** | 0.015 | 0.0044 | 0.068 | 0.025 | 0.127 | 0.068 |

| Overall score (median / p95 / p99) | original | replay |
|---|---|---|
| datakit | 0.35 / 1.15 / 2.0 | 1.13 / 3.8 / 5.9 |
| PFΔ N-0/N-1 | 87 / 396 / 473 | 3.9 / 12 / 21 |
| **N-2** | 44 / 181 / 371 | **12.8 / 55 / 115** |
| N-2, load within / above | 44 / 49 (median) | 12.3 / 14.6 (median) |

The replay model is 3–4× more accurate than the original on N-2, but **3× worse on N-2 than on the PFΔ N-0/N-1 test**
(median score 12.8 vs 3.9; PQ Vm RMSE 0.0088 vs 0.00095). Unseen topologies cost accuracy. On N-2, load level matters
much less than topology: within-range and above-range samples have similar scores (12.3 vs 14.6).

## Comparison (same format as `stage3_comparison.csv`)

`final_n2_detectors.csv` (all scopes: all, load within/above, N-0/N-1, N-2 branch/gen/mixed). P = precision, R = recall.

| Data | Label | Method | orig P | orig R | orig F1 | orig FP / FN | **replay P** | **replay R** | **replay F1** | **replay FP / FN** | PR-AUC orig / replay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **N-2** | voltage | Baseline PtC | 0.897 | 0.963 | 0.929 | 68 / 23 | **0.998** | **0.973** | **0.985** | **1 / 17** | – |
| N-2 | voltage | A gated (solver) | 0.985 | 0.968 | 0.976 | 9 / 20 (45 % solved) | 0.998 | 0.973 | 0.985 | 1 / 17 (0 %) | – |
| N-2 | voltage | A gated (drop) | 0.970 | 0.474 | 0.637 | 9 / 325 | 0.998 | 0.973 | 0.985 | 1 / 17 | – |
| N-2 | voltage | B logreg | 0.942 | 0.833 | 0.884 | 32 / 103 | 0.931 | 0.898 | 0.914 | 41 / 63 | 0.973 / 0.974 |
| N-2 | voltage | B MLP | 0.935 | 0.814 | 0.870 | 35 / 115 | 0.917 | 0.927 | 0.922 | 52 / 45 | 0.967 / 0.983 |
| **N-2** | overload | Baseline PtC | 0.635 | 0.944 | 0.759 | 58 / 6 | **0.990** | **0.879** | **0.931** | **1 / 13** | – |
| N-2 | overload | A gated (solver) | 0.929 | 0.981 | **0.955** | 8 / 2 (45 % solved) | 0.990 | 0.879 | 0.931 | 1 / 13 (0 %) | – |
| N-2 | overload | A gated (drop) | 0.742 | 0.215 | 0.333 | 8 / 84 | 0.990 | 0.879 | 0.931 | 1 / 13 | – |
| N-2 | overload | B logreg | 0.868 | 0.617 | 0.721 | 10 / 41 | 0.773 | 0.794 | 0.783 | 25 / 22 | 0.863 / 0.852 |
| N-2 | overload | B MLP | 0.829 | 0.542 | 0.655 | 12 / 49 | 0.825 | 0.748 | 0.784 | 17 / 27 | 0.777 / 0.878 |
| PFΔ N-0/N-1 | voltage | Baseline PtC | 0.759 | 0.986 | 0.858 | 161 / 7 | 1.000 | 0.988 | 0.994 | 0 / 6 | – |
| PFΔ N-0/N-1 | voltage | A gated (solver) | 0.985 | 0.986 | 0.986 | 8 / 7 (66 %) | 1.000 | 0.988 | 0.994 | 0 / 6 (0 %) | – |
| PFΔ N-0/N-1 | voltage | B logreg / MLP | 0.907 / 0.903 | 0.893 / 0.891 | 0.900 / 0.897 | 47/55, 49/56 | 0.944 / 0.907 | 0.915 / 0.934 | 0.929 / 0.921 | 28/44, 49/34 | 0.973 / 0.987 |
| PFΔ N-0/N-1 | overload | Baseline PtC | 0.647 | 0.988 | 0.782 | 173 / 4 | 1.000 | 0.991 | 0.995 | 0 / 3 | – |
| PFΔ N-0/N-1 | overload | A gated (solver) | 0.982 | 1.000 | 0.991 | 6 / 0 (66 %) | 1.000 | 0.991 | 0.995 | 0 / 3 (0 %) | – |
| PFΔ N-0/N-1 | overload | B logreg / MLP | 0.913 / 0.859 | 0.888 / 0.872 | 0.900 / 0.866 | 27/36, 46/41 | 0.955 / 0.943 | 0.922 / 0.928 | 0.938 / 0.936 | 14/25, 18/23 | 0.971 / 0.988 |
| datakit | voltage | Baseline PtC (= A) | 1.000 | 0.964 | 0.982 | 0 / 1 | 1.000 | 0.964 | 0.982 | 0 / 1 | – |
| datakit | voltage | B logreg / MLP | 0.767 / 0.889 | 0.821 / 0.571 | 0.793 / 0.696 | 7/5, 2/12 | 0.857 / 0.828 | 0.857 / 0.857 | 0.857 / 0.842 | 4/4, 5/4 | 0.90 / 0.92 |
| datakit | overload | all methods | 1.000 | 1.000 | 1.000 | 0 / 0 | 1.000 | 1.000 | 1.000 | 0 / 0 | 1.00 |

The PFΔ and datakit rows were recomputed from this run's predict outputs and match `stage3_comparison.csv` exactly.

### By load level (N-2 and PFΔ N-0/N-1), F1 and FP / FN

| Data | Scope (n) | Label | orig PtC | orig A solver (share) | orig B logreg | **replay PtC** | replay B logreg |
|---|---|---|---|---|---|---|---|
| N-2 | within (836) | voltage | 0.934, 48 / 23 | 0.976 (43 %) | 0.891 | **0.986, 1 / 14** | 0.918 |
| N-2 | above (164) | voltage | 0.906, 20 / 0 | 0.980 (56 %) | 0.844 | **0.984, 0 / 3** | 0.892 |
| N-2 | within (836) | overload | 0.761, 43 / 6 | 0.954 (43 %) | 0.717 | **0.917, 1 / 12** | 0.767 |
| N-2 | above (164) | overload | 0.754, 15 / 0 | 0.958 (56 %) | 0.737 | **0.978, 0 / 1** | 0.844 |
| PFΔ N-0/N-1 | within (407) | voltage | 0.910, 26 / 7 | 0.965 (31 %) | 0.836 | 0.985, 0 / 5 | 0.880 |
| PFΔ N-0/N-1 | above (593) | voltage | 0.835, 135 / 0 | 0.996 (89 %) | 0.932 | 0.998, 0 / 1 | 0.953 |
| PFΔ N-0/N-1 | within (407) | overload | 0.716, 19 / 0 | 0.941 (31 %) | 0.780 | 1.000, 0 / 0 | 0.884 |
| PFΔ N-0/N-1 | above (593) | overload | 0.788, 154 / 4 | 0.995 (89 %) | 0.909 | 0.995, 0 / 3 | 0.942 |

## A: residual as a trust signal on N-2 (`final_n2_A_trust.csv`)

| Model | Data | Spearman(pbe, error) | Wrong labels V / O | AUROC wrong V | AUROC wrong O | PR-AUC wrong V (chance) | PR-AUC wrong O (chance) |
|---|---|---|---|---|---|---|---|
| original | PFΔ N-0/N-1 | 0.91 | 168 / 177 | 0.80 | 0.68 | 0.49 (0.17) | 0.24 (0.18) |
| original | **N-2** | 0.63 | 91 / 64 | 0.70 | 0.78 | 0.19 (0.09) | 0.17 (0.06) |
| replay | PFΔ N-0/N-1 | 0.51 | 6 / 3 | 0.44 | 0.35 | 0.006 (0.006) | 0.003 (0.003) |
| replay | **N-2** | **0.80** | 18 / 14 | 0.60 | **0.86** | 0.025 (0.018) | **0.13 (0.014)** |
| replay | N-2 within / above load | 0.80 / 0.79 | 15+13 / 3+1 | 0.62 / 0.44 | 0.87 / 0.77 | – | – |

Gate sweep on N-2 (`final_n2_A_gate_sweep.csv`; thresholds = the frozen val quantiles, *not* the chosen t):

| Model | Threshold (MVA) | Solver share N-2 | Voltage P / R | Overload P / R |
|---|---|---|---|---|
| replay | ∞ (chosen) | 0 % | 0.998 / 0.973 | 0.990 / 0.879 |
| replay | 0.173 (val 1 %) | 51 % | 1.000 / 0.992 | 1.000 / 1.000 |
| replay | 0.103 (val 5 %) | 78 % | 1.000 / 0.998 | 1.000 / 1.000 |
| original | 37.3 (val 10 %) | 6 % | 0.92 / 0.96 | 0.69 / 0.94 |
| original | 5.1 (val 20 %) | 19 % | 0.96 / 0.96 | 0.78 / 0.94 |
| original | 1.27 (chosen) | 45 % | 0.985 / 0.968 | 0.929 / 0.981 |

## Findings

1. **Replay + predict-then-check is still the best detector on N-2, but it degrades.** Voltage F1 0.985 (0.994 on
   N-0/N-1) with a single false alarm. Overload recall falls to **0.879**: 13 of 107 overloads are missed (FN 3 on
   N-0/N-1). 10 of the 13 misses are branch–branch outages (the other 3 branch–gen), where losing two lines reroutes flow in ways the model has
   never seen. Precision stays ≈ 1, so PtC under-predicts flows rather than raising false alarms.
2. **On N-2 the residual becomes a trust signal for the replay model.** On N-0/N-1 it carried no information about the
   few wrong labels (AUROC 0.35–0.44). On N-2 it tracks the error (Spearman 0.80) and flags missed overloads (AUROC
   0.86, PR-AUC 0.13 vs 0.014 chance). The median residual is 3.6× higher than on N-0/N-1 (0.18 vs 0.05 MVA). The frozen gate (t = ∞)
   therefore sent nothing. Any val-quantile threshold would have caught all 107 overloads, but only by sending 51 % or
   more of N-2 to the solver. The val data (N-0/N-1) has no N-2 residual levels, so a val-chosen threshold either never
   fires or fires on half the samples. **The residual is a useful out-of-distribution alarm, but its threshold cannot be
   tuned on in-distribution data.**
3. **The original model + solver gate transfers well**: it sends 45 % to the solver (66 % on N-0/N-1), reaches overload
   F1 0.955 and voltage F1 0.976, and **beats replay PtC on overload recall** (0.981 vs 0.879). The price is 450 solver calls
   per 1,000 samples, vs none for replay. Dropping high-residual alarms is again harmful (voltage recall 0.47,
   overload 0.22).
4. **B (embedding probes) degrades most under unseen topologies.** For overload, F1 falls from 0.94 to 0.78 (replay)
   and from 0.90 to 0.72/0.66 (original). PR-AUC also falls (0.988 → 0.85–0.88). Voltage holds better (F1 0.91–0.92,
   PR-AUC 0.97–0.98). The probes learned what N-0/N-1 overloads look like in embedding space, and N-2 overloads lie
   elsewhere. B is below PtC on every set.
5. **Load level is not the main driver on N-2.** For replay, errors and detector quality are similar within vs above
   datakit's training load range: most failures (14 of 17 voltage FN, 12 of 13 overload FN) are within range, which
   points to topology. On PFΔ N-0/N-1 with the *original* model, load level dominated: PtC FP 135 above vs 26 within
   for voltage, and the solver gate sends 89 % of above-range samples vs 31 % within.
6. **Overall ranking on the final test:** replay PtC ≥ original + gated solver (better overload recall, 45 % solver)
   > B probes > original PtC. Main conclusion: a fine-tuned state model with physics-based checking is the most
   reliable detector. Its residual tells when to stop trusting it, but the residual threshold must be set from
   out-of-distribution data.

Caveats: one test draw of 1,000 samples (overload has 107 positives, so ±0.03 on recall), one training seed, and N-2
was never used for any choice.
