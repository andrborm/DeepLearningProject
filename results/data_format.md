# Data format (Phase 3)

Produced with `scripts/explore_data.py` (parts A–E) and `scripts/make_test_subset.py`.
Raw numbers: `results/case14_variable_stats.csv`, `results/case14_violation_counts.csv`,
`results/pfdelta_case14_counts.csv`, `results/pfdelta_case14_feasible_violations.csv`, `results/test_subset_1000.csv`.

## 1. datakit case14 (`gridfm/pf_small_case14_ieee`, the checkpoint's training data)

| Item | Value |
|---|---|
| Scenarios | **219,606** (from 9,986 load scenarios × up to 22 topology variants) |
| Grid | pglib `case14_ieee`: **14 buses, 5 generators, 20 branches** (3 of them transformers, tap ≠ 1) |
| Files used by graphkit | `bus_data.parquet`, `gen_data.parquet`, `branch_data.parquet` (Hive-partitioned by `scenario_partition`, 1,108 partitions). `y_bus_data`, `runtime_data`, `stats` are extra |
| Checkpoint split | train 175,713 / val 21,947 / test 21,946, split by `load_scenario_idx` (no load profile shared between splits) |

### Columns

**bus_data** (one row per scenario × bus)

| Column | Meaning | Model input? |
|---|---|---|
| `Pd`, `Qd` | load active / reactive power (MW, MVAr) | yes |
| `Pg`, `Qg` | generation at the bus, summed over its generators (MW, MVAr). The input tensor uses `Qg` only; `Pg` comes from `gen_data` | Qg: depends on bus type |
| `Vm`, `Va` | voltage magnitude (p.u.), angle (**degrees** in data, radians inside graphkit) | depends on bus type |
| `PQ`, `PV`, `REF` | **one-hot bus type** (exactly one is 1 per row) | yes |
| `min_vm_pu`, `max_vm_pu` | voltage limits: **0.94 / 1.06 for every bus** | masked |
| `GS`, `BS` | shunt conductance / susceptance | yes |
| `vn_kv` | nominal voltage (1.0 in pglib case14) | masked |
| `Va_dc`, `Pg_dc` | DC power flow baseline | no |
| `scenario`, `bus`, `load_scenario_idx` | ids | no |

`min_q_mvar`, `max_q_mvar` are added to the bus features by graphkit (sum over the bus's generators) → 15 bus features.

**gen_data** (scenario × generator): `p_mw`, `q_mvar`, `min/max_p_mw`, `min/max_q_mvar`, cost `cp0/cp1/cp2`,
`in_service`, `is_slack_gen`, `bus`, `p_mw_dc`. Model uses `p_mw, min/max_p_mw, cp0..cp2` (6 features; `in_service` is
used to drop outaged generators).

**branch_data** (scenario × branch): `from_bus`, `to_bus`, flows `pf, qf, pt, qt` (MW / MVAr at from / to end), `r, x, b`,
admittance entries `Yff, Yft, Ytf, Ytt` (real / imag), `tap`, `shift`, `ang_min`, `ang_max`, **`rate_a`**, `br_status`,
`pf_dc`, `pt_dc`. Edge features (10): `P, Q, Y_self_r/i, Y_mutual_r/i, tap, ang_min, ang_max, rate_a`, forward edges use
the from-side quantities and reverse edges the to-side ones. Outaged branches (`br_status = 0`) are removed from the graph.

### Bus types

- Bus 0 is always **REF** (slack, with generator 0).
- Buses 1, 2, 5, 7 carry generators 1–4 and are **PV**, except when their generator is outaged. Then the bus
  becomes **PQ** (all 38,424 generator outages). So **the bus type is not fixed per bus index**; it is ~4 % PQ for these buses.
- The other 9 buses are always **PQ**.

### Contingencies: every scenario is N-1

| | all | checkpoint test split | our test subset (1000) |
|---|---|---|---|
| N-0 | **0** | **0** | **0** |
| N-1 branch outage | 181,182 (82.5 %) | 18,157 (82.7 %) | 814 (81.4 %) |
| N-1 generator outage | 38,424 (17.5 %) | 3,789 (17.3 %) | 186 (18.6 %) |

Datakit's `random` topology generator draws the number of outages from `randint(1, k+1)`, which is always 1 for `k = 1`.
**There are no N-0 scenarios**, so the requested N-0 vs N-1 comparison is not possible on this data. `test_subset_1000.csv`
therefore records the **outage type (branch / gen) and the outaged element** per sample instead. PFΔ provides N-0, N-1
and N-2 if that comparison is needed (section 4).

Outaged branches are spread evenly (~9,500 each) over 19 of the 20 branches. **Branch 13 (bus 6–7) is never outaged**:
bus 7 is a leaf connected only through it, so removing it would island bus 7. That is datakit's islanding filter at
work. The slack generator is never outaged either.

## 2. Masking in graphkit's PowerFlow task

Applied per sample by `AddPFHeteroMask` + `ApplyMasking` (in `PowerFlowTransforms`, after `RemoveInactiveBranches/Generators`).
Masked entries are overwritten with `mask_value = 0.0` in the (normalized) input. Verified on a real scenario:

| Bus type | Hidden from model (bus features) | Given to model | Unknowns the model must predict |
|---|---|---|---|
| PQ | `Vm, Va` + limits | `Pd, Qd, Qg (=0), GS, BS`, type | **Vm, Va** |
| PV | `Va, Qg` + limits | `Pd, Qd, Vm (setpoint), GS, BS`, type; gen `Pg` | **Va, Qg** |
| REF | `Vm, Qg` + limits; slack gen `Pg` | `Pd, Qd, Va (= 0), GS, BS`, type | **Vm, Qg, Pg** |

- "limits" = `min_vm_pu, max_vm_pu, min_q_mvar, max_q_mvar, vn_kv`, **always hidden**.
- Generators: `min/max_p_mw` and cost coefficients are always hidden. `p_mw` is hidden only for the slack generator.
- Branches: `pf, qf` (flows), `ang_min, ang_max`, **`rate_a` are always hidden**. Admittances and tap are visible.
- **Unusual convention at REF**: graphkit hides the slack **Vm** and keeps **Va**. The textbook PF convention fixes both
  Vm and Va at the slack. The reference RMSE table confirms this (non-zero Vm RMSE at REF, zero Va RMSE).

Output: two heads, `Vm, Va` per bus and `Pg` per generator. Unmasked positions are overwritten with the inputs. A physics
decoder computes branch flows and bus injections from the predicted voltages, which gives bus-level `Pg, Qg`.
Final bus output = `[Vm, Va, Pg, Qg]`. Metrics are reported per bus type on the masked entries only.

**Implications for the anomaly work:**
- The model never sees `rate_a` or voltage limits. Anomaly labels must be computed **after** prediction from
  predicted Vm and derived branch flows, compared against the limits from the raw data. graphkit already does this in
  `predict` (`<grid>_branch_predictions.parquet` has thermal loading, after `RemovePFMask` restores the limits).
- Branch flows are not predicted directly. They are derived from predicted voltages, so overload errors come from
  voltage errors.

## 3. Summary statistics and limit violations (all 219,606 scenarios)

Units: MW / MVAr / p.u. / degrees. Full table in `case14_variable_stats.csv`.

| Variable | min | max | mean | std |
|---|---|---|---|---|
| Pd (bus) | 0 | 128.1 | 16.21 | 21.87 |
| Qd (bus) | -5.55 | 26.84 | 4.60 | 5.76 |
| Pg (bus) | 0 | 447.0 | 17.15 | 56.22 |
| Qg (bus) | -42.86 | 218.5 | 5.19 | 11.26 |
| Vm (bus) | **0.859** | **1.073** | 1.038 | 0.0185 |
| Va (bus, deg) | -80.28 | 0 | -12.43 | 7.01 |
| Vm, PQ buses | 0.859 | 1.073 | 1.033 | 0.0162 |
| Vm, PV buses | 0.950 | 1.060 | 1.042 | 0.0208 |
| Vm, REF bus | 1.018 | 1.060 | 1.060 | 0.0021 |
| Pg, REF (slack) | 117.1 | 447.0 | 212.2 | 41.3 |
| Qg, PV buses | -8.26 | 203.3 | 18.47 | 13.73 |
| pf (branch, in service) | -183.8 | 447.0 | 22.24 | 43.26 |
| rate_a | 53 | 664 | 190.5 | 142.4 |

Scenarios with at least one violation (tolerance 1e-6):

| Violation | all (219,606) | checkpoint test split (21,946) | test subset (1000) |
|---|---|---|---|
| Vm outside [0.94, 1.06] | **6,599 (3.0 %)** | 691 (3.1 %) | 28 |
| of which Vm < 0.94 | 209 | 25 | 1 |
| of which Vm > 1.06 | 6,390 | 666 | 27 |
| Line overload (\|S\| > rate_a) | **9,607 (4.4 %)** | 974 (4.4 %) | 37 |
| Voltage **or** overload | 16,067 (7.3 %) | 1,651 (7.5 %) | 64 (6.4 %) |
| Generator Qg outside [Qmin, Qmax] | 125,797 (57 %) | 12,407 | 583 |
| Generator Pg outside [Pmin, Pmax] | 302 | 23 | 1 |

- Violations are real (datakit `mode: pf` does not enforce operating limits). The per-bus `min/max_vm_pu` limits are
  exactly 0.94 / 1.06, so both voltage checks give identical counts.
- Voltage violations in this dataset are mostly **over-voltage** at PQ buses (up to 1.073), not under-voltage.
- **Qg limit violations are not a useful anomaly class**: 57 % of scenarios violate them. PF holds the PV voltage
  setpoint regardless of Q limits (no PV→PQ switching).
- Branch loading: mean 0.18, p99 0.66, max **3.75** × rate_a.
- For a 3-class classifier the datakit test data is **very imbalanced** (≈ 94 % normal). The 1000-sample subset has only 28 voltage and 37 overload cases.

## 4. Branch flow limits (`rate_a`)

- Present for all branches in all scenarios, all > 0, **constant per branch** (static pglib ratings, in MVA, same
  units as `pf/qf`): e.g. branch 0 (0–1) 472, branch 8 (3–8 transformer) 53, branch 19 (12–13) 76.
- Overload label: `loading = max(|pf + j·qf|, |pt + j·qt|) / rate_a > 1` for in-service branches.
- `ang_min/ang_max` are also present (angle-difference limits), as a possible extra check.
- datakit's `stats.parquet` has `n_overloads` / `max_loading`, but **only for 20,000 of the 219,606 scenarios**. Don't
  rely on it; compute from branch flows.

## 5. Original PFΔ case14 (`pfdelta/pfdelta`, `case14.tar.gz`)

Counted by streaming the archive (`explore_data.py --parts E`, 20 s). Extracting 159k small JSON files to disk on
Windows is very slow (more than 1 h, probably antivirus scanning), and the counts don't need it.

### Samples per category

| Topology | Feasible (`raw/`) | Near-infeasible "nose", train | nose, test | Approaching infeasible "around_nose", train | Total |
|---|---|---|---|---|---|
| N-0 | 56,000 | 3,600 | 200 | 14,400 | 74,200 |
| N-1 | 29,000 | 3,600 | 200 | 14,400 | 47,200 |
| N-2 | 20,000 | 3,600 | 200 | 14,400 | 38,200 |
| **Total** | **105,000** | 10,800 | 600 | 43,200 | **159,600** |

All 105,000 feasible samples have `termination_status = LOCALLY_SOLVED`, and each folder contains exactly the outage
count it claims:

| Topology | branch out / gen out | Samples |
|---|---|---|
| N-1 | 1 branch | 13,830 |
| N-1 | 1 gen | 15,170 |
| N-2 | 2 branches | 5,642 |
| N-2 | 1 branch + 1 gen | 6,777 |
| N-2 | 2 gens | 7,581 |

### Violations in the feasible samples (sample-level: any bus / any in-service branch; limits from the JSON: Vm ∈ [0.94, 1.06], |S| ≤ rate_a)

| Topology | Samples | Voltage violation | of which Vm < 0.94 | Vm > 1.06 | Line overload | Both | **Normal** |
|---|---|---|---|---|---|---|---|
| N-0 | 56,000 | 35,795 (64 %) | 35,786 | 11 | 34,213 (61 %) | 25,112 | **11,104 (20 %)** |
| N-1 | 29,000 | 10,908 (38 %) | 10,694 | 294 | 1,681 (6 %) | 743 | **17,154 (59 %)** |
| N-2 | 20,000 | 12,116 (61 %) | 11,984 | 198 | 2,364 (12 %) | 1,581 | **7,101 (36 %)** |
| **Total** | 105,000 | **58,819 (56 %)** | 58,464 | 503 | **38,258 (36 %)** | 27,436 | **35,359 (34 %)** |

Extremes: min Vm **0.483 p.u.**, max branch loading **1.94 × rate_a**.

### Observations

- **PFΔ is heavily stressed compared with datakit** (7 % abnormal in datakit vs 66 % in PFΔ), and its voltage violations are
  almost all **under-voltage** (datakit: almost all over-voltage). This is useful for anomaly classes (plenty of both
  classes, and a "both" class too), but it is a large distribution shift for the checkpoint.
- **Surprise: N-0 is the most stressed topology** (only 20 % normal, 61 % overloaded), more than N-1/N-2. PFΔ most likely
  samples loads differently per topology (contingency cases must stay solvable, which caps their loading). So
  "topology" and "load level" are confounded in PFΔ. **Open question**: check the PFΔ load-sampling code before
  interpreting N-0 vs N-1 vs N-2 error differences.
- Voltage and overload co-occur a lot (27k "both"). A 3-class classifier (normal / voltage / overload) needs a rule
  for "both" (4th class, multi-label, or priority).
- Near-infeasible (nose / around-nose) samples were only counted, not checked for violations. They use the
  `{solved_net, lambda}` schema (see `dataset_overview.md`).
