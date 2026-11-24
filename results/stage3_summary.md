# Task 3 detectors: predict-then-check, residual trust signal, embedding probes (no N-2)

Models: **replay** (final model, `finetune_summary.md`) and the **original** checkpoint for comparison.
Labels: multi-label `y_voltage`, `y_overload`, truth from the solved power flow; slack Vm treated as a known setpoint.
Script: `scripts/stage3_detectors.py` (`extract`, `analyze --with-datakit`, `compare`).
**N-2 was not converted, read or run.**

## Data

| Split | Samples | Use |
|---|---|---|
| train (probes) | `pf_probe` 2,000 PFΔ train positions (1,000 N-0 + 1,000 N-1, not in any fine-tuning set) + `dk_probe` 2,000 datakit checkpoint-TRAIN samples (not in the replay set) | fit B probes |
| val | `pf_val` 1,000 PFΔ val positions + `dk_val` 1,000 datakit checkpoint-VAL samples | **all tuning**: probe hyperparameters, features, decision thresholds, gate threshold |
| test | `pfdelta_test1000` (500 N-0 + 500 N-1), `case14_test1000` (datakit) | reported numbers only |

Datakit is included in probe train/val because replay was chosen. The original model's probes use the same
data, so the two models differ only in their embeddings. All inference runs are ≤ 1,000 samples (8 `predict`
runs per model). Half of `pf_val`/`dk_val` was also the replay run's validation set. It was used only for monitoring
there, because the epoch-40 weights were used.

## Methods

- **Baseline PtC:** labels from the predicted Vm (slack = setpoint) and graphkit's predicted `thermal_excess`.
- **A, trust signal:** per-sample mean power-balance residual `pbe_mean` (MVA, computed from the prediction alone).
  - *Gated PtC (solver):* samples with `pbe_mean > t` are sent to a classical solver (exact labels). The rest keep PtC.
  - *Gated PtC (drop):* the literal "keep an alarm only if the residual is below t". High-residual alarms are removed, no solver.
  - **t chosen on val:** the fewest solver calls for which gated PtC reaches precision and recall ≥ 0.98 on both
    labels (combined val). Original: **t = 1.27 MVA** (33 % of val sent). Replay: **t = ∞** (PtC already meets the
    target, so nothing is sent).
- **B, embedding probes:** frozen 384-d bus embeddings from graphkit `predict --get_embeddings`. Features are either
  *pooled* (mean, max, min over the 14 buses = 1,152-d) or *flat* (14 × 384 = 5,376-d; case14 has a fixed bus order).
  - Logistic regression (balanced class weights, C ∈ {1e-3, 1e-2, 0.1, 1}) and an MLP (1 hidden layer
    {64, 256}, alpha ∈ {1e-4, 1e-2, 1}, early stopping). One probe per label; inputs standardized.
  - The feature type and hyperparameters maximize val PR-AUC. The decision threshold maximizes val F1.
  - Chosen settings: `stage3_<model>_B_chosen.csv`.

## Comparison (test subsets)

`stage3_comparison.csv`. P = precision, R = recall. Solver share = fraction of test samples sent to the solver.

| Data | Label | Method | orig P | orig R | orig F1 | orig FP / FN | **replay P** | **replay R** | **replay F1** | **replay FP / FN** | PR-AUC orig / replay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PFΔ | voltage | Baseline PtC | 0.759 | 0.986 | 0.858 | 161 / 7 | **1.000** | **0.988** | **0.994** | **0 / 6** | – |
| PFΔ | voltage | A gated (solver) | 0.985 | 0.986 | 0.986 | 8 / 7 (66 % solved) | 1.000 | 0.988 | 0.994 | 0 / 6 (0 %) | – |
| PFΔ | voltage | A gated (drop) | 0.938 | 0.235 | 0.376 | 8 / 394 | 1.000 | 0.988 | 0.994 | 0 / 6 | – |
| PFΔ | voltage | B logreg | 0.907 | 0.893 | 0.900 | 47 / 55 | 0.944 | 0.915 | 0.929 | 28 / 44 | 0.973 / 0.987 |
| PFΔ | voltage | B MLP | 0.903 | 0.891 | 0.897 | 49 / 56 | 0.907 | 0.934 | 0.921 | 49 / 34 | 0.971 / 0.986 |
| PFΔ | overload | Baseline PtC | 0.647 | 0.988 | 0.782 | 173 / 4 | **1.000** | **0.991** | **0.995** | **0 / 3** | – |
| PFΔ | overload | A gated (solver) | 0.982 | 1.000 | 0.991 | 6 / 0 (66 % solved) | 1.000 | 0.991 | 0.995 | 0 / 3 (0 %) | – |
| PFΔ | overload | A gated (drop) | 0.500 | 0.019 | 0.036 | 6 / 315 | 1.000 | 0.991 | 0.995 | 0 / 3 | – |
| PFΔ | overload | B logreg | 0.913 | 0.888 | 0.900 | 27 / 36 | 0.955 | 0.922 | 0.938 | 14 / 25 | 0.971 / 0.988 |
| PFΔ | overload | B MLP | 0.859 | 0.872 | 0.866 | 46 / 41 | 0.943 | 0.928 | 0.936 | 18 / 23 | 0.956 / 0.988 |
| datakit | voltage | Baseline PtC | 1.000 | 0.964 | 0.982 | 0 / 1 | **1.000** | **0.964** | **0.982** | **0 / 1** | – |
| datakit | voltage | A gated (solver / drop) | 1.000 | 0.964 | 0.982 | 0 / 1 (0 %) | 1.000 | 0.964 | 0.982 | 0 / 1 (0 %) | – |
| datakit | voltage | B logreg | 0.767 | 0.821 | 0.793 | 7 / 5 | 0.857 | 0.857 | 0.857 | 4 / 4 | 0.897 / 0.924 |
| datakit | voltage | B MLP | 0.889 | 0.571 | 0.696 | 2 / 12 | 0.828 | 0.857 | 0.842 | 5 / 4 | 0.874 / 0.934 |
| datakit | overload | all methods | 1.000 | 1.000 | 1.000 | 0 / 0 | 1.000 | 1.000 | 1.000 | 0 / 0 | 1.00 / 1.00 (B) |

Base rates: PFΔ test 51.5 % voltage / 32.1 % overload; datakit test 2.8 % / 3.7 %.

### A: residual as a trust signal (`stage3_<model>_A_trust.csv`)

| Model | Data | Spearman(pbe, true error) | Wrong labels (V / O) | AUROC wrong V | AUROC wrong O | PR-AUC wrong V (chance) | PR-AUC wrong O (chance) |
|---|---|---|---|---|---|---|---|
| original | PFΔ | **0.91** | 168 / 177 | **0.80** | 0.68 | 0.49 (0.17) | 0.24 (0.18) |
| original | datakit | 0.27 | 1 / 0 | 0.94 | – | 0.015 (0.001) | – |
| replay | PFΔ | 0.51 | 6 / 3 | 0.44 | 0.35 | 0.006 (0.006) | 0.003 (0.003) |
| replay | datakit | 0.34 | 1 / 0 | 0.96 | – | 0.025 (0.001) | – |

Datakit AUROCs rest on a single wrong label and mean little.

Gate sweep on PFΔ test, original model (thresholds = combined-val quantiles; `stage3_original_A_gate_sweep.csv`):

| Solver share (PFΔ test) | Voltage P / R (solver) | Overload P / R (solver) | Voltage P / R (drop) | Overload P / R (drop) |
|---|---|---|---|---|
| 0 % | 0.76 / 0.99 | 0.65 / 0.99 | 0.76 / 0.99 | 0.65 / 0.99 |
| 7 % | 0.81 / 0.99 | 0.66 / 0.99 | 0.80 / 0.93 | 0.62 / 0.81 |
| 19 % | 0.88 / 0.99 | 0.71 / 0.99 | 0.85 / 0.80 | 0.59 / 0.58 |
| 37 % | 0.94 / 0.99 | 0.79 / 1.00 | 0.90 / 0.55 | 0.49 / 0.26 |
| 66 % (chosen t) | 0.98 / 0.99 | 0.98 / 1.00 | 0.94 / 0.24 | 0.50 / 0.02 |

## Findings

1. **With the replay model, plain predict-then-check is the best detector**: zero false alarms on both datasets, F1
   0.994 / 0.995 on PFΔ and 0.982 / 1.000 on datakit. Its 9 PFΔ misses (6 voltage, 3 overload) and 1 datakit miss
   are the only errors. Neither A nor B improves on it.
2. **A is useful only when the state model is wrong.** With the original model under the PFΔ shift, the residual
   tracks the true error (Spearman 0.91) and flags wrong voltage labels well (AUROC 0.80). Solver gating can
   fix PtC (precision 0.65–0.76 → 0.98), but only by sending 66 % of PFΔ samples (and 0 % of datakit) to the
   solver, so most of the gain is "solve the hard half". With the replay model the residual carries no signal about
   the few remaining wrong labels (AUROC 0.35–0.44), and the val rule sends nothing.
3. **"Drop high-residual alarms" is harmful.** High residual marks stressed samples, and those are mostly *true*
   violations. Dropping their alarms kills recall (original PFΔ: voltage recall 0.99 → 0.24 at the chosen t, overload
   → 0.02). The residual is a "check this" signal, not a "this alarm is false" signal.
4. **B (embedding probes) is clearly below PtC on both models** (PFΔ F1 0.87–0.94 vs PtC 0.99 for replay), but it is
   not tied to the predicted state. It beats PtC of the *original* model on PFΔ precision (0.86–0.91 vs 0.65–0.76),
   at 5–10× the false negatives. Replay embeddings give better probes than original ones (PR-AUC 0.986–0.988 vs
   0.956–0.973 on PFΔ). Logistic regression and the MLP are similar; logreg is the more stable of the two (the
   original-model MLP misses 12 of 28 datakit voltage violations).
5. **Probes need datakit in training to work on datakit.** In a side run with the random (PFΔ-only) model and
   PFΔ-only probes (`stage3_random_*`), the voltage probes found 0–1 of 28 datakit voltage violations. datakit's
   violations are over-voltages and PFΔ's are under-voltages, so a PFΔ-only probe never sees the datakit pattern.

## Frozen for the final N-2 test

No further changes after this point. N-2 is evaluated once with exactly this setup:

- **State model:** replay, `mlruns/657056988967417188/f061c2d3f30c40169fef7f4bc0845665/artifacts/model/last_epoch_state_dict.pt`,
  checkpoint `normalizer_stats.pt`. The original model is kept as the reference.
- **Labels:** `y_voltage` = any bus Vm outside [Vm_min, Vm_max] by > 1e-6 (slack Vm = setpoint); `y_overload` = any
  in-service branch with thermal excess > 1e-6. Truth from the solved PF (`true_labels_from_raw`).
- **Baseline:** PtC as above.
- **A:** `pbe_mean`, gate t = ∞ for replay (no solver calls) and t = 1.27 MVA for the original model. The gate sweep is
  also reported at the same val-quantile thresholds (replay: 0.173 / 0.138 / 0.103 / 0.082 / 0.060 MVA).
- **B:** the fitted probes and their decision thresholds are saved in `data/stage3_probes/<model>_<probe>_<label>.joblib`
  (gitignored; settings in `stage3_<model>_B_chosen.csv`, trained on pf_probe + dk_probe). N-2 must load these, with
  no refit or re-tuning. Refitting gives the same table; val PR-AUCs differ only in the 4th digit across thread counts.
- **Metrics:** precision, recall, F1, PR-AUC, FP/FN, solver share; Spearman and AUROC/PR-AUC for wrong-label flagging;
  per topology where it applies.

Open for the N-2 run: `stage3_detectors.py` still needs an N-2 test block (a conversion step plus a 1,000-sample
seeded subset like `make_pfdelta_subset.py`). That is plumbing only, with no design choice.
