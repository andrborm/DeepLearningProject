# Stage 2 fine-tuning on PFΔ N-0/N-1 (Task 2.2–2.3)

Base GENCO case14 checkpoint, fine-tuned on PFΔ case14 **N-0/N-1 train positions**. Two runs: **hard cases** and an
equal-size **random** control. Evaluated on the datakit 1,000 test subset (forgetting) and the PFΔ 1,000 test subset.
**N-2 was not converted, read or run.** Scripts: `scripts/make_pfdelta_finetune_data.py`,
`scripts/finetune_pipeline.py`.

## Setup

| Item | Choice |
|---|---|
| Model | **base** (20.1 M params). Timing: 1 epoch on 5,000 samples = **82 s** on CPU, incl. validation on 1,000 (60–70 s when two runs share the CPU, 13 threads each). 30 epochs ≈ 35–40 min, far below the 3 h limit, so no switch to small/tiny |
| PFΔ positions | per topology: test = last 2,000 of `raw_shuffle.json` (never read); pool = the rest; **val = last 10 % of the pool**, train = first 90 % |
| Mining pool | 7,500 N-0 + 7,500 N-1 seeded train positions, scored with the base model in 15 `evaluate` runs of 1,000 |
| Hard score | per sample, mean over Vm/Va/Pg/Qg of (sample RMSE / datakit base RMSE), as in `inference_per_sample*.csv`. Pool median 185 (N-0) / 44 (N-1), the same as on the test subset (183 / 43) |
| Hard set | **top 2,500 per topology** = 5,000 (N-0 score ≥ 244, N-1 ≥ 56). Ranked per topology, because a global ranking would be almost all N-0 |
| Random set | 2,500 per topology from the same pool (seed 1). 1,679 of its 5,000 samples are also in the hard set |
| Validation | 500 N-0 + 500 N-1 from the PFΔ val positions, used for early stopping and best-checkpoint selection |
| Optimizer | AdamW, **lr 5e-5** (base training used 5e-4), ReduceLROnPlateau (factor 0.5, patience 2), early-stopping patience 5, max 30 epochs, batch 64, same loss as the checkpoint (0.1 physics + 0.9 masked bus MSE) |
| Normalizer | the checkpoint's `normalizer_stats.pt` (baseMVA 84.6). graphkit's `finetune` CLI has no flag for this and would refit baseMVA on the PFΔ loads, so the script calls `main_cli` directly |

Training: both runs used all **30 epochs** (34 min each) and were still improving. Validation loss went from 1.34e-3
after epoch 1 to **1.84e-4 (hard) / 1.82e-4 (random)**, and the best checkpoint is the last epoch. Early stopping
never triggered, so more epochs would probably help a little more.

## Results (unknowns only, absolute error)

| Data | Variable | Bus | RMSE orig | RMSE hard | RMSE random | median orig | median hard | median random | p95 orig | p95 hard | p95 random | p99 orig | p99 hard | p99 random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **PFΔ** | Vm (p.u.) | PQ | 0.0668 | **0.00145** | 0.00152 | 0.0086 | 0.00024 | **0.00020** | 0.167 | 0.0027 | 0.0027 | 0.305 | **0.0065** | 0.0070 |
| PFΔ | Vm (p.u.) | REF | 0.0269 | 0.0067 | **0.0062** | 0.022 | 0.0039 | **0.0030** | 0.047 | 0.014 | **0.013** | 0.057 | **0.021** | 0.023 |
| PFΔ | Va (rad) | PQ | 0.0622 | **0.0042** | 0.0044 | 0.027 | 0.0021 | **0.0018** | 0.135 | 0.0080 | 0.0081 | 0.196 | **0.014** | 0.017 |
| PFΔ | Va (rad) | PV | 0.0477 | **0.0040** | 0.0042 | 0.018 | 0.0020 | **0.0017** | 0.110 | 0.0077 | **0.0073** | 0.165 | **0.013** | 0.015 |
| PFΔ | Qg (MVAr) | PV | 74.6 | 6.33 | **5.55** | 11.6 | 0.43 | **0.36** | 182 | 14.3 | **12.0** | 336 | 27.6 | **23.9** |
| PFΔ | Qg (MVAr) | REF | 55.2 | 12.0 | **10.4** | 42.6 | 7.07 | **5.63** | 93.6 | 23.8 | **21.5** | 107 | 38.0 | **34.9** |
| PFΔ | Pg (MW) | REF | 34.5 | **1.28** | 1.45 | 19.9 | 0.58 | **0.53** | 70.7 | **2.28** | 2.29 | 86.1 | **4.48** | 5.68 |
| **datakit** | Vm (p.u.) | PQ | **8.8e-5** | 1.95e-3 | 1.83e-3 | **1.3e-5** | 6.7e-4 | 6.7e-4 | **1.3e-4** | 4.3e-3 | 3.9e-3 | **3.2e-4** | 6.3e-3 | 6.2e-3 |
| datakit | Vm (p.u.) | REF | **6.2e-4** | 0.0169 | 0.0145 | **3.1e-4** | 0.0164 | 0.0142 | **1.2e-3** | 0.0230 | 0.0192 | **2.0e-3** | 0.0260 | 0.0220 |
| datakit | Va (rad) | PQ | **7.1e-4** | 8.9e-3 | 8.6e-3 | **1.3e-4** | 7.0e-3 | 6.7e-3 | **4.8e-4** | 0.0131 | 0.0132 | **1.0e-3** | 0.0198 | 0.0222 |
| datakit | Va (rad) | PV | **7.5e-4** | 8.9e-3 | 8.6e-3 | **1.3e-4** | 7.2e-3 | 6.7e-3 | **5.1e-4** | 0.0128 | 0.0123 | **1.1e-3** | 0.0190 | 0.0223 |
| datakit | Qg (MVAr) | PV | **0.61** | 17.4 | 15.0 | **0.03** | 1.43 | 1.32 | **1.21** | 39.7 | 34.4 | **2.47** | 49.7 | 41.3 |
| datakit | Qg (MVAr) | REF | **1.26** | 35.6 | 30.5 | **0.63** | 34.3 | 29.4 | **2.43** | 49.8 | 41.9 | **3.91** | 61.0 | 51.3 |
| datakit | Pg (MW) | REF | **0.12** | 2.18 | 2.29 | **0.018** | 0.99 | 1.24 | **0.075** | 4.01 | 4.25 | **0.16** | 7.05 | 8.27 |

Per-sample overall score (1.0 = datakit base RMSE level), median: PFΔ **87 → 5.6 (hard) / 4.7 (random)**
(N-0: 183 → 4.0 / 3.6, N-1: 43 → 7.7 / 6.1); datakit **0.35 → 18 / 17**.

### Predict-then-check (slack Vm = known setpoint)

| Data | Label | Precision orig | Precision hard | Precision random | Recall orig | Recall hard | Recall random |
|---|---|---|---|---|---|---|---|
| PFΔ | y_voltage | 0.759 | **0.998** | **0.998** | 0.986 | 0.979 | **0.986** |
| PFΔ | y_overload | 0.647 | 0.988 | **0.997** | 0.988 | **1.000** | 0.997 |
| PFΔ N-0 | y_voltage | 0.733 | **1.000** | **1.000** | **1.000** | 0.997 | 0.997 |
| PFΔ N-0 | y_overload | 0.667 | 0.987 | **0.997** | 0.987 | **1.000** | 0.997 |
| PFΔ N-1 | y_voltage | 0.807 | 0.995 | 0.995 | 0.965 | 0.950 | **0.970** |
| PFΔ N-1 | y_overload | 0.471 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| datakit | y_voltage | **1.000** | **1.000** | **1.000** | **0.964** | 0.786 | 0.821 |
| datakit | y_overload | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

Counts in `finetune_predict_then_check_base.csv`. False alarms on PFΔ: 161 → 1 (voltage), 173 → 4 / 1 (overload).
datakit voltage misses: 1 → 6 (hard) / 5 (random) of 28.

## Findings

1. **Fine-tuning fixes the PFΔ shift.** Errors drop 10–50× (PQ Vm RMSE 0.067 → 0.0015 p.u., slack Pg 34.5 → 1.3 MW). The
   predict-then-check false alarms almost disappear: precision 0.76 → 0.998 (voltage) and 0.65 → 0.99 (overload),
   with recall unchanged. It is now near-perfect on PFΔ N-0/N-1.
2. **Hard-case selection does not beat random.** On PFΔ, random is equal or better on most medians and RMSEs.
   Hard is slightly better on PQ Vm/Va RMSE, slack Pg and some p99 values. The differences (≤ 20 %) are small next
   to the ~30× gain from either set, and come from a single seed. A plausible reason: base-model error mostly tracks
   how stressed a sample is (`pfdelta_shift_summary.md` §3), so "hard" mostly means "most heavily loaded". A random
   sample covers the whole new load range just as well.
3. **Strong forgetting on datakit.** datakit errors grow 15–40× (PQ Vm RMSE 8.8e-5 → 1.9e-3 p.u., slack Qg 1.3 →
   36 MVAr). That is still 20–40× smaller than the original model's errors on PFΔ. The predict-then-check impact is
   limited: overload stays 1.00/1.00, voltage precision stays 1.00, but **voltage recall drops 0.96 → 0.79 / 0.82**
   (5–6 extra misses of 28, presumably near-limit over-voltages; not inspected). Hard forgets slightly more than random.
4. **The slack bus is the least improved** on both datasets (REF Vm/Qg have the largest remaining errors). The slack
   Vm is masked as an input by graphkit convention, and PFΔ generator setpoints differ from datakit's.

## Caveats

- One seed per run. 30 epochs were not enough for convergence, and both runs stopped at the epoch cap, not by early stopping.
- The hard set was mined from 15,000 train-position samples (15 inference runs of ≤ 1,000), not from ≤ 1,000 as the
  earlier plan in `inference_summary.md` suggested. This gives a 5,000-sample hard set without reusing samples.
- "PFΔ validation split" = the last 10 % of the non-test shuffled positions. This is our reading of the PFΔ
  convention noted in `inference_summary.md` and was not re-checked against the PFΔ code.
- The fine-tuned model was selected on PFΔ val only. Nothing datakit-based was used for model selection, so the
  forgetting numbers are clean, but nothing limited forgetting either (no replay, no datakit in val).

## Next

- Mixing in some datakit replay (or a datakit val term) should cut the forgetting at little PFΔ cost.
- The PFΔ N-2 final test should use the **random** fine-tuned model (or both), because hard selection gave no gain.

---

# Replay fine-tuning against forgetting (and final model choice)

Scripts: `scripts/make_stage3_data.py` (data folder `data/stage3`, mapping `stage3_mapping.csv`),
`scripts/finetune_pipeline.py finetune --set replay`. All four models in `finetune_errors_base.csv` and
`finetune_predict_then_check_base.csv`.

| Item | Choice |
|---|---|
| Start | original checkpoint, same normalizer handling (checkpoint `normalizer_stats.pt`), lr 5e-5, same scheduler and loss |
| Train | **5,000 random PFΔ N-0/N-1** (the same samples as the random run) **+ 5,000 datakit** from the checkpoint's TRAIN split (`scenario_splits.json`; seeded, disjoint from the datakit test1000 subset) |
| Validation | 250 N-0 + 250 N-1 PFΔ val positions + 500 datakit from the checkpoint's VAL split (1,000 total) |
| Epochs | cap 40. Ran all 40 (93 min, ~2.1 min/epoch). Val loss 7.8e-4 → 9.5e-5 at epoch 40 (best 8.6e-5 at epoch 39) |

**Bug found and worked around.** graphkit's `SaveBestModelStateDict` only skips Lightning's pre-training sanity check
when the metric is missing. Here the metric was present: the sanity check runs 2 validation batches, and with sorted
ids those were the datakit samples (ids 5000+, before PFΔ val at 13000+). Their tiny loss became "best", and no
later epoch beat it, so `best_model_state_dict.pt` is the **unchanged original model**. The replay model used below
is the **epoch-40 weights** from Lightning's `checkpoints/last.ckpt`, saved as `artifacts/model/last_epoch_state_dict.pt`.
The hard/random runs were not affected: their best file was written at the last epoch. `finetune_pipeline.py` now
patches the callback to always skip the sanity check.

## Results (unknowns only)

| Data | Variable | Bus | RMSE orig | RMSE random | **RMSE replay** | median orig | median random | **median replay** | p95 orig | p95 random | **p95 replay** | p99 orig | p99 random | **p99 replay** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PFΔ | Vm (p.u.) | PQ | 0.0668 | 0.00152 | **0.00095** | 0.0086 | 0.00020 | **0.00015** | 0.167 | 0.0027 | **0.0017** | 0.305 | 0.0070 | **0.0039** |
| PFΔ | Vm (p.u.) | REF | 0.0269 | 0.0062 | **0.0044** | 0.022 | 0.0030 | **0.0020** | 0.047 | 0.013 | **0.0091** | 0.057 | 0.023 | **0.017** |
| PFΔ | Va (rad) | PQ | 0.0622 | 0.0044 | **0.0034** | 0.027 | 0.0018 | **0.0015** | 0.135 | 0.0081 | **0.0060** | 0.196 | 0.017 | **0.012** |
| PFΔ | Va (rad) | PV | 0.0477 | 0.0042 | **0.0032** | 0.018 | 0.0017 | **0.0014** | 0.110 | 0.0073 | **0.0055** | 0.165 | 0.015 | **0.012** |
| PFΔ | Qg (MVAr) | PV | 74.6 | 5.55 | **4.21** | 11.6 | 0.36 | **0.28** | 182 | 12.0 | **8.8** | 336 | 23.9 | **18.2** |
| PFΔ | Qg (MVAr) | REF | 55.2 | 10.4 | **7.69** | 42.6 | 5.63 | **3.97** | 93.6 | 21.5 | **16.0** | 107 | 34.9 | **28.8** |
| PFΔ | Pg (MW) | REF | 34.5 | 1.45 | **1.19** | 19.9 | **0.53** | 0.69 | 70.7 | 2.29 | **2.07** | 86.1 | 5.68 | **3.97** |
| datakit | Vm (p.u.) | PQ | **8.8e-5** | 1.83e-3 | 2.4e-4 | **1.3e-5** | 6.7e-4 | 4.1e-5 | **1.3e-4** | 3.9e-3 | 4.7e-4 | **3.2e-4** | 6.2e-3 | 1.0e-3 |
| datakit | Vm (p.u.) | REF | **6.2e-4** | 0.0145 | 0.0019 | **3.1e-4** | 0.0142 | 9.6e-4 | **1.2e-3** | 0.0192 | 0.0036 | **2.0e-3** | 0.0220 | 0.0054 |
| datakit | Va (rad) | PQ | **7.1e-4** | 8.6e-3 | 9.2e-4 | **1.3e-4** | 6.7e-3 | 4.0e-4 | **4.8e-4** | 0.0132 | 1.6e-3 | **1.0e-3** | 0.0222 | 2.9e-3 |
| datakit | Va (rad) | PV | **7.5e-4** | 8.6e-3 | 9.7e-4 | **1.3e-4** | 6.7e-3 | 4.0e-4 | **5.1e-4** | 0.0123 | 1.6e-3 | **1.1e-3** | 0.0223 | 2.9e-3 |
| datakit | Qg (MVAr) | PV | **0.61** | 15.0 | 1.93 | **0.03** | 1.32 | 0.09 | **1.21** | 34.4 | 4.0 | **2.47** | 41.3 | 8.0 |
| datakit | Qg (MVAr) | REF | **1.26** | 30.5 | 3.97 | **0.63** | 29.4 | 1.85 | **2.43** | 41.9 | 7.7 | **3.91** | 51.3 | 12.1 |
| datakit | Pg (MW) | REF | **0.12** | 2.29 | 0.21 | **0.018** | 1.24 | 0.079 | **0.075** | 4.25 | 0.27 | **0.16** | 8.27 | 0.62 |

Per-sample overall score, median (1.0 = datakit base RMSE level): PFΔ 87 (orig) / 4.7 (random) / **3.9 (replay)**;
datakit 0.35 / 16.7 / **1.13**.

| Data | Label | orig P / R | hard P / R | random P / R | **replay P / R** |
|---|---|---|---|---|---|
| PFΔ | y_voltage | 0.759 / 0.986 | 0.998 / 0.979 | 0.998 / 0.986 | **1.000 / 0.988** |
| PFΔ | y_overload | 0.647 / 0.988 | 0.988 / 1.000 | 0.997 / 0.997 | **1.000 / 0.991** |
| datakit | y_voltage | 1.000 / 0.964 | 1.000 / 0.786 | 1.000 / 0.821 | **1.000 / 0.964** |
| datakit | y_overload | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 / 1.000** |

Replay FP/FN on PFΔ: voltage 0 / 6, overload 0 / 3 (random: 1 / 7 and 1 / 1).

## Final model: **replay**

- **PFΔ:** replay is better than random and hard on every RMSE, p95 and p99 (and every median except slack Pg),
  by 20–45 %. Predict-then-check has zero false alarms on both labels. It misses 2 more overloads than random
  (3 vs 1 of 321).
- **datakit (forgetting):** errors are 1.3–3.6× the original's, instead of 15–40× for random/hard. Predict-then-check is
  **identical to the original** (voltage recall 0.964, overload 1.00/1.00). The 4–5 extra voltage misses of the
  PFΔ-only models are gone.
- So replay is the only model that is good on both datasets at once. Caveat: it also trained longer (40 vs 30
  epochs, 157 vs 79 batches per epoch) and was selected by a different validation mix. Part of its PFΔ gain may be
  more training, not replay itself. One seed.
- **Frozen for the N-2 test:** replay model, `mlruns/657056988967417188/f061c2d3…/artifacts/model/last_epoch_state_dict.pt`.
