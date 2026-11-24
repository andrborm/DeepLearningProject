# Facts for the report (read-only lookup, no new experiments)

Every answer gives its source. "code" = the frozen script, "result" = an existing file in `results/`.

## 1. Which samples the probes were trained and validated on

The probes were fitted with `stage3_detectors.py analyze --with-datakit`, so `train_groups = pf_probe + dk_probe` and
`val_groups = pf_val + dk_val`. Sources: code `scripts/stage3_detectors.py` (`stage_analyze`, `blocks`);
ids in result `stage3_mapping.csv` (column `role`, folder `data/stage3`); selection in code `scripts/make_stage3_data.py` (rng seed 7).

| Role | n | Samples |
|---|---|---|
| train `pf_probe` | 2,000 | PFΔ case14 **train positions** from the 15,000-sample mining pool (`pfdelta_ft_selection.csv`) that are **not** in the random fine-tuning set: 1,000 N-0 + 1,000 N-1, drawn per topology with seed 7 |
| train `dk_probe` | 2,000 | datakit, checkpoint **TRAIN** split: `permutation(train)[5000:7000]` (seed 7), disjoint from the 5,000 replay samples `dk_train` |
| val `pf_val` | 1,000 | the PFΔ **val positions** (500 N-0 + 500 N-1, last 10 % of the non-test pool), the same as `data/pfdelta_ft` val |
| val `dk_val` | 1,000 | datakit, checkpoint **VAL** split, 1,000 drawn with seed 7 |

All tuning (feature type, hyperparameters, decision threshold) used val only. N-2 was never used.
**Note (mismatch in an existing text):** `stage3_summary.md` calls `pf_probe` "not in any fine-tuning set". The code
excludes only the *random* set. **657 of the 2,000 `pf_probe` samples (326 N-0 + 331 N-1) are in the *hard*
fine-tuning set** (checked against `pfdelta_ft_selection.csv`). The frozen probes use the original and replay
embeddings. Replay was trained on the random set + datakit, and its overlap with `pf_probe` is 0, so the reported
probe results are not affected. Only the sentence is too strong.

## 2. Probe input format and MLP architecture

Source: code `stage3_detectors.py` (`features`, `make_clf`, `approach_b`); result `stage3_<model>_B_chosen.csv`.

- Input = frozen GENCO **bus embeddings, 14 buses × 384 dims**, from graphkit `predict --get_embeddings`. Two
  candidate feature types, chosen per probe and label by val PR-AUC:
  - **pooled**: concat(mean, max, min over the 14 buses) = **1,152-d**
  - **flat**: the 14 × 384 matrix flattened = **5,376-d** (case14 has a fixed bus order)
- Every probe: `StandardScaler` → classifier, one probe per label.
- Logistic regression: `LogisticRegression(C, class_weight="balanced", max_iter=3000)`, C ∈ {1e-3, 1e-2, 0.1, 1}.
- **MLP**: sklearn `MLPClassifier` with **one hidden layer** (width ∈ {64, 256}), ReLU (sklearn default), Adam,
  L2 `alpha` ∈ {1e-4, 1e-2, 1}, `early_stopping=True` (10 % internal split), `max_iter=300`, `random_state=0`,
  sigmoid output.
- Frozen choices:

| Model | Probe | Label | Features | Hyperparameters | Decision threshold |
|---|---|---|---|---|---|
| replay | logreg | voltage | pooled | C = 0.1 | 0.6775 |
| replay | logreg | overload | flat | C = 1 | 0.885 |
| replay | MLP | voltage | pooled | 64 hidden, alpha = 1 | 0.4729 |
| replay | MLP | overload | pooled | 256 hidden, alpha = 1 | 0.5791 |
| original | logreg | voltage / overload | flat / flat | C = 0.01 / 0.01 | 0.6468 / 0.7122 |
| original | MLP | voltage / overload | flat / pooled | 256, 1e-4 / 64, 1e-4 | 0.585 / 0.4817 |

## 3. Integrated Gradients: baseline input and number of steps

Source: code `scripts/explain_ig.py` (docstring, `STEPS = 64`, `baseline_embeddings`); result `explain_summary.md`.

- **Baseline:** for each label, the **per-bus mean embedding (14 × 384) of the probe-TRAIN samples (`pf_probe` + `dk_probe`)
  that are negative for that label** ("typical grid without that violation"). It is not a zero baseline.
- **Steps: 64**, trapezoid rule over the straight-line path (65 points). Completeness error ≤ 1.6 % (median 0.2 %).
- Applied to the replay-model probes only.

## 4. Epoch caps and the epoch whose weights were used

Sources: MLflow `mlruns/657056988967417188/<run>/metrics/Validation loss` (read by `scripts/extra/loss_curves.py`,
`results/extra/finetune_loss_curves.csv`); `finetune_summary.md`; `SETUP_LOG.md` (commands with `--epochs`);
code `finetune_pipeline.py` (`MODEL_FILE`).

| Run | Epoch cap | Epochs run | Early stop | Val-loss minimum | Weights used |
|---|---|---|---|---|---|
| hard | 30 (patience 5) | 30 | no | epoch 30 (1.84e-4) | `best_model_state_dict.pt` = **epoch 30** (last) |
| random | 30 (patience 5) | 30 | no | epoch 30 (1.82e-4) | `best_model_state_dict.pt` = **epoch 30** (last) |
| replay | 40 (patience 5) | 40 | no | epoch 39 (8.57e-5) | `last_epoch_state_dict.pt` = **epoch 40** (best-checkpoint bug: the saved "best" was the unchanged original model) |

## 5. Test subset composition

Sources: result `test_subset_1000.csv`, `pfdelta_test_subset_1000.csv`, `pfdelta_n2_test_subset_1000.csv`
(counts recomputed from those files).

| Subset | n | Composition |
|---|---|---|
| datakit (`data/case14_test1000`) | 1,000 | all N-1: **814 branch outages, 186 generator outages** (no N-0 in datakit) |
| PFΔ N-0/N-1 (`data/pfdelta_test1000`) | 1,000 | **500 N-0 + 500 N-1** (N-1: 216 branch, 284 generator) |
| PFΔ N-2 (`data/pfdelta_n2_test1000`) | 1,000 | **263 double-branch, 385 double-generator, 352 mixed (branch + generator)** |

## 6. N-2 numbers vs `final_n2_summary.md` (and `final_n2_detectors.csv`, scope "all")

| Claim | Summary / CSV | Match |
|---|---|---|
| residual routing, original, voltage F1 0.976 (FP 9 / FN 20) | 0.976, 9 / 20 (CSV 0.9763) | ✅ |
| residual routing, original, overload F1 0.955 (8 / 2) | 0.955, 8 / 2 (CSV 0.9545) | ✅ |
| logistic probe, replay, voltage 0.914 (41 / 63) | 0.914, 41 / 63 (CSV 0.9143) | ✅ |
| logistic probe, replay, overload 0.783 (25 / 22) | 0.783, 25 / 22 (CSV 0.7834) | ✅ |
| MLP probe, replay, voltage 0.922 (52 / 45) | 0.922, 52 / 45 (CSV 0.9220) | ✅ |
| MLP probe, replay, overload 0.784 (17 / 27) | 0.784, 17 / 27 (CSV 0.7843) | ✅ |

**No mismatches.** Residual routing sent 45 % of N-2 samples to the solver (`solver share` 0.45).

## 7. Original model on PFΔ N-0/N-1

Source: result `final_n2_detectors.csv` (dataset `test_pfdelta`, scope "all"), identical to `stage3_comparison.csv`.

| Method | Label | Precision | Recall | F1 | FP / FN | Share sent to solver |
|---|---|---|---|---|---|---|
| limit check (PtC) | voltage | 0.759 | 0.986 | **0.858** | 161 / 7 | 0 % |
| limit check (PtC) | overload | 0.647 | 0.988 | **0.782** | 173 / 4 | 0 % |
| residual routing (t = 1.27 MVA) | voltage | 0.985 | 0.986 | **0.986** | 8 / 7 | **65.6 %** |
| residual routing (t = 1.27 MVA) | overload | 0.982 | 1.000 | **0.991** | 6 / 0 | **65.6 %** |

## 8. Spearman correlation, residual vs error

Spearman(per-sample mean power-balance residual `pbe_mean`, per-sample error score) over all 1,000 samples.
Source: result `final_n2_A_trust.csv` (scope "all"); same values in `stage3_<model>_A_trust.csv` for the two N-0/N-1 sets.

| Model | datakit | PFΔ N-0/N-1 | PFΔ N-2 |
|---|---|---|---|
| original | 0.27 | **0.91** | 0.63 |
| replay | 0.34 | 0.51 | **0.80** |
