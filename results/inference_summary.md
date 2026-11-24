# Inference summary (Phase 4)

Checkpoint `gridfm/genco-pf-datakit` → `case14_ieee/{base,tiny}/seed0`, graphkit 0.9.0.3, CPU.
Data: 1000-scenario seeded subset (`seed 42`) of the checkpoint's own **test split** (`data/case14_test1000`,
see `test_subset_1000.csv`). All scenarios are N-1: 814 branch outages, 186 generator outages (there is no N-0 in
this dataset). Produced by `scripts/run_inference.py`.

## How it was run

The repo-intended path, the CLI, called from the script:
```bash
gridfm_graphkit evaluate --config configs/case14_base_infer.yaml \
  --model_path checkpoints/genco-pf-datakit/case14_ieee/base/seed0/best_model_state_dict.pt \
  --normalizer_stats checkpoints/genco-pf-datakit/case14_ieee/base/seed0/normalizer_stats.pt \
  --data_path data/case14_test1000 --log_dir mlruns --exp_name phase4_inference --run_name base_n1000 \
  --num_workers 0 --save_output
```
`configs/case14_{base,tiny}_infer.yaml` = the checkpoint YAML converted v0→v1, plus
`data.split_from_existing_files` (subset split files), `data.split_by_load_scenario_idx: false`, `workers: 0`,
`training.accelerator: cpu`. The env var `MLFLOW_ALLOW_FILE_STORE=true` is needed (MLflow 3.16 refuses `./mlruns` otherwise).
`evaluate --save_output` writes graphkit's RMSE table (same code as the model card) and per-bus / per-branch prediction
parquets under `mlruns/<exp>/<run>/artifacts/test/`. Runtime: **~40 s per 1000 samples** on CPU
(most of it is process start-up and data loading).

| Run | Samples | Result |
|---|---|---|
| base, first 50 | 50 | works; script RMSE = graphkit RMSE exactly (post-processing check) |
| base | 1000 | main result below |
| tiny | 1000 | sanity check below |

## Error metrics (base, 1000 samples)

Only the **unknowns** are scored (PQ: Vm, Va · PV: Va, Qg · REF: Vm, Qg, Pg). All other bus outputs are copies of the
input and have zero error by construction. Units: Vm p.u., Va **radians**, Pg MW, Qg MVAr.

| Variable | Bus type | n (bus-rows) | MAE | RMSE | max abs |
|---|---|---|---|---|---|
| Vm | PQ | 9,186 | 3.49e-5 | 8.78e-5 | 3.67e-3 |
| Vm | REF | 1,000 | 4.21e-4 | 6.21e-4 | 4.81e-3 |
| Va | PQ | 9,186 | 1.98e-4 | 7.06e-4 | 2.04e-2 |
| Va | PV | 3,814 | 2.09e-4 | 7.48e-4 | 2.12e-2 |
| Qg | PV | 3,814 | 0.230 | 0.606 | 9.34 |
| Qg | REF | 1,000 | 0.840 | 1.255 | 9.85 |
| Pg | REF | 1,000 | 0.030 | 0.118 | 3.37 |

For scale: Vm MAE 3.5e-5 p.u. is ~0.004 % of 1 p.u.; Va MAE 2e-4 rad ≈ 0.011°. The model is very accurate on typical
samples. **Errors are heavy-tailed** (RMSE ≫ MAE, max ≈ 100× MAE).

Robust view (added in Phase 5, `inference_percentiles_by_bus_type.csv`):

| Variable | Bus type | median | p95 | p99 | RMSE |
|---|---|---|---|---|---|
| Vm | PQ | 1.31e-5 | 1.31e-4 | 3.17e-4 | 8.78e-5 |
| Vm | REF | 3.07e-4 | 1.17e-3 | 2.01e-3 | 6.21e-4 |
| Va | PQ | 1.26e-4 | 4.85e-4 | 1.03e-3 | 7.06e-4 |
| Va | PV | 1.31e-4 | 5.15e-4 | 1.06e-3 | 7.48e-4 |
| Qg | PV | 0.030 | 1.21 | 2.47 | 0.606 |
| Qg | REF | 0.626 | 2.43 | 3.91 | 1.26 |
| Pg | REF | 0.0175 | 0.075 | 0.164 | 0.118 |

For Va, RMSE even exceeds the p95: a few samples dominate it. Per outage type see `inference_percentiles_by_outage_type.csv`
and `phase5_summary.md` §1.

### By outage type (base)

| Variable | Bus type | RMSE, branch outage | RMSE, gen outage | max abs, branch | max abs, gen |
|---|---|---|---|---|---|
| Vm | PQ | 8.16e-5 | 1.09e-4 | 3.67e-3 | 1.29e-3 |
| Vm | REF | 6.48e-4 | 4.83e-4 | 4.81e-3 | 1.88e-3 |
| Va | PQ | 7.85e-4 | 1.86e-4 | 2.04e-2 | 9.8e-4 |
| Va | PV | 8.05e-4 | 2.05e-4 | 2.12e-2 | 1.17e-3 |
| Qg | PV | 0.629 | 0.449 | 9.34 | 2.73 |
| Qg | REF | 1.333 | 0.834 | 9.85 | 2.88 |
| Pg | REF | 0.130 | 0.025 | 3.37 | 0.10 |

- **Branch outages are harder** for every variable except Vm at PQ buses: ~4× higher Va RMSE, 5× higher slack-Pg RMSE,
  and all of the extreme errors.
- **Generator outages give slightly larger Vm errors at PQ buses** (1.09e-4 vs 8.2e-5). When a generator trips, its bus
  loses voltage control (PV→PQ), so the model must predict a voltage it normally gets as input.
- Worst 50 samples: 45 branch outages (90 %, vs 81 % in the subset). Outages of branch 3 (bus 1–3) and branch 5 (2–3)
  are over-represented (9 and 7 of 50). **One extreme outlier**: sample 438 (outage of branch 0, bus 0–1, the
  472 MVA line leaving the slack bus) has overall score 17.3, next worst 4.8. It alone sets the Va max error.

## Comparison with the model card (graphkit's own RMSE table, same code path)

| RMSE | this run, base | card, base | ratio | this run, tiny | card, tiny | ratio |
|---|---|---|---|---|---|---|
| PQ Vm (p.u.) | 8.78e-5 | 1.16e-4 | 0.76 | 1.04e-4 | 1.37e-4 | 0.76 |
| PQ Va (rad) | 7.07e-4 | 1.03e-3 | 0.69 | 6.88e-4 | 9.39e-4 | 0.73 |
| PV Va (rad) | 7.46e-4 | 1.11e-3 | 0.67 | 7.18e-4 | 9.92e-4 | 0.72 |
| PV Qg (MVAr) | 0.606 | 0.579 | 1.05 | 0.841 | 0.809 | 1.04 |
| REF Vm (p.u.) | 6.21e-4 | 6.12e-4 | 1.02 | 8.62e-4 | 8.30e-4 | 1.04 |
| REF Qg (MVAr) | 1.255 | 1.196 | 1.05 | 1.729 | 1.674 | 1.03 |
| REF Pg (MW) | 0.118 | 0.160 | 0.74 | 0.121 | 0.194 | 0.62 |

- **Reproduction is consistent with the card.** Qg and slack Vm are within ±5 %. Va, PQ Vm and slack Pg are 25–35 %
  *lower* on our 1000 samples. These are the heavy-tailed variables, and the card's RMSE is over 21,946 test scenarios,
  which contain more tail samples. On the first 50 samples, Va was 5× below the card; the ratio moves toward 1 as n grows.
- The checkpoint loads in graphkit 0.9.0.3 and gives card-level accuracy, so **the version gap to the 0.0.7-based
  training branch does not change predictions**.
- **tiny vs base**: MAE 30–60 % worse for all variables (e.g. PV Qg MAE 0.311 vs 0.230, REF Qg 1.14 vs 0.84), but RMSE of
  Va is similar. The tail is driven by the same kind of samples, not by model capacity.
- Hard cases are only partly shared: Spearman(base, tiny overall score) = **0.40**, and 25 of the top-50 worst samples are shared.

## Physics consistency (power-balance residual)

graphkit provides it. Per bus, `PBE = |ΔP + jΔQ|` (MVA) of the AC power-balance equations, evaluated on the predicted
state with known quantities clamped to ground truth (0.9.0.3 definition). Per-sample mean/max are in the CSV.

| | base, this run | card (training-branch definition) |
|---|---|---|
| Avg. active residual (MW) | 0.00202 | 0.00211 |
| Avg. reactive residual (MVAr) | 0.00112 | 0.00114 |
| PBE mean (MVA) | 0.00259 | 0.00268 |
| PBE max (MVA) | 0.144 | 0.573 |

The means are close to the card. **PBE max is not comparable** (different definition + 1000 vs 21,946 samples).
Residuals are tiny because the model runs a physics-residual correction in every layer. **The residual is a weak
indicator of accuracy here** (Spearman(overall score, per-sample PBE max) = 0.25). For anomaly scoring, prediction
error and residual carry different information.

## Per-sample table: `results/inference_per_sample.csv` (base)

One row per sample (`sample_id` = subset id 0..999):

| Column | Meaning |
|---|---|
| `orig_scenario` | scenario id in `pf_small_case14_ieee` |
| `outage_type`, `out_branches`, `out_gens` | contingency (branch / gen, element id and buses) |
| `rmse_<var>_<bustype>` | RMSE of that unknown over the sample's buses of that type (Vm_PQ, Vm_REF, Va_PQ, Va_PV, Qg_PV, Qg_REF, Pg_REF) |
| `maxabs_<var>` | max absolute error over the sample's unknown entries of that variable |
| `overall_score` | mean over Vm, Va, Pg, Qg of (sample RMSE / subset RMSE of that variable). Unit-free; 1.0 = at the subset-wide RMSE level. Because of the heavy tail the median is ~0.36 |
| `pbe_mean`, `pbe_max`, `res_P_mean`, `res_Q_mean` | power-balance residuals from graphkit (MVA / MW / MVAr) |

Aggregate tables: `inference_by_bus_type.csv`, `inference_by_outage_type.csv`, `inference_by_variable.csv`,
`inference_rmse_vs_card.csv`, `inference_residuals.csv`. Raw predictions (per bus and per branch,
incl. predicted branch flows and `thermal_excess`) are in `mlruns/.../artifacts/test/*.parquet` (gitignored).

## Notes for the next stages

### PFΔ samples for fine-tuning (not converted or run yet)

PFΔ's fixed test split = the **last 2,000 files of each topology's shuffled list** (`shuffle_files/<topo>/raw_shuffle.json`),
the same for every PFΔ task. Train/val come from the front of the list.

| Use | PFΔ case14 samples |
|---|---|
| **Fine-tuning pool** | N-0 and N-1 **feasible**, excluding the PFΔ test positions: up to 54,000 N-0 + 27,000 N-1. Optional: N-0/N-1 `around_nose/train` (14,400 each) and `nose/train` (3,600 each) for near-infeasible operating points |
| Validation | 10 % carved from the fine-tuning pool (PFΔ convention: shuffled positions 0.9·n…n) |
| In-distribution test | PFΔ test positions of N-0 and N-1 (2,000 each) + `nose/test` N-0/N-1 (200 each) |
| **Strictly held-out test** | **All N-2**: 20,000 feasible + 3,600 nose/train + 200 nose/test + 14,400 around_nose. Never used for fine-tuning, validation, early stopping, threshold tuning or model selection |

Suggested order: start with **N-1 feasible**. It matches the checkpoint's N-1 training prior, so it isolates the PFΔ
load/stress shift. The "hard cases" for Stage 2 should come from base-model inference on ≤ 1000 PFΔ N-0/N-1 *training-pool*
samples, not from the test positions. Mind the PFΔ confound (N-0 is the most stressed topology, `data_format.md` §5), so
report results per topology. PFΔ files need the datakit converter (and a wrapper for nose files) and
contiguous renumbering before graphkit can read them.

### Classifier targets: multi-label

Plan two independent binary targets per sample, **normal = neither**:
- `y_voltage = any bus with Vm < vmin or Vm > vmax` (vmin/vmax = 0.94 / 1.06 in both datasets)
- `y_overload = any in-service branch with max(|S_from|, |S_to|) > rate_a`

This handles the many samples with both violations (27,436 of 105,000 in PFΔ feasible; only 139 of 219,606 in datakit). Both labels can be
computed from ground truth (for training labels) and from the model's predicted state (for "predicted" labels).
graphkit's branch prediction parquet already carries `thermal_excess` and `thermal_excess_target`, which should map
directly onto `y_overload` (to verify). Expected base rates: datakit test subset ~2.8 % voltage / 3.7 % overload;
PFΔ feasible 56 % voltage / 36 % overload.
