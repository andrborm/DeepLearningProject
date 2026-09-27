# Dataset overview (Phase 2)

Checked 2026-09-27 against the HuggingFace API (`/api/datasets/...?blobs=true`), dataset cards, `args.log`
files, and source code of gridfm-graphkit, gridfm-datakit and MOSSLab-MIT/pfdelta.

## TL;DR

- **The `gridfm/pfdelta_task*` datasets do not contain case14.** Every one of them holds only `case118/`
  (task 3.1 also includes case57 and case500 as OOD test sets). There is no case14 PFΔ data in graphkit format on HF.
- **The original PFΔ release (`pfdelta/pfdelta`) does include case14** (`case14.tar.gz`, 230 MB, ~159k JSON samples
  with N / N-1 / N-2 and near-infeasible "nose" cases). **Downloaded** to `data/pfdelta_orig/`.
- gridfm-datakit's GitHub repo (not the pip package) contains a **pure-Python converter**
  `pfdelta/batch_convert_pfdelta.py` (PowerModels JSON → graphkit parquet). It needs no Julia.
  The authors' own default path in it is `case14_ieee_n_minus_one`. Caveat: it only reads the `{network, solution}` JSON
  schema used by feasible samples. Nose / around-nose samples use a different schema (`{solved_net, lambda}`) and
  would need a small adapter.
- The checkpoint's own training data, `gridfm/pf_small_case14_ieee`, was **downloaded** to `data/case14_ieee/raw/`.
  It already contains **random N-1** outages (branch or generator).

## 1. The `gridfm/pfdelta_task*` datasets (HF, graphkit-ready parquet)

Layout: `case118/{train,valid,test}/case118/{n,n-1,n-2}/{feasible,nose,around_nose}/raw/{bus,gen,branch,y_bus}_data.parquet`
(+ `stats.parquet` for nose sets). Dataset cards are empty (31 bytes); the task meaning below comes from the PFΔ code
(`core/datasets/pfdelta_dataset.py`, `task_config`) and the paper (arXiv 2510.22048).

Training-sample counts are per topology (N / N-1 / N-2). The **test set is identical for every task**: for each of
N, N-1 and N-2, 2,000 feasible + 200 near-infeasible (nose) samples.

| HF dataset | Grids present | Size | PFΔ task | Train composition (N / N-1 / N-2) | Contingencies | Near-infeasible in train |
|---|---|---|---|---|---|---|
| `gridfm/pfdelta_task1.1` | case118 | 690 MB | 1.1 in-distribution | 54k / 0 / 0 feasible | test only | no |
| `gridfm/pfdelta_task1.2` | case118 | 698 MB | 1.2 in-distribution | 27k / 27k / 0 | N-1 in train | no |
| `gridfm/pfdelta_task1.3` | case118 | 701 MB | 1.3 (= 2.1) in-distribution | 18k / 18k / 18k | N-1 + N-2 | no |
| `gridfm/pfdelta_task2.2` | case118 | 499 MB | 2.2 data efficiency | 12k / 12k / 12k | N-1 + N-2 | no |
| `gridfm/pfdelta_task2.3` | case118 | 292 MB | 2.3 data efficiency | 6k / 6k / 6k | N-1 + N-2 | no |
| `gridfm/pfdelta_task3.1` | case118 (+ case57, case500 test) | 1,048 MB | 3.1 cross-grid OOD | 18k / 18k / 18k on case118 | N-1 + N-2 | no |
| `gridfm/pfdelta_task4.1` | case118 | 706 MB | 4.1 close-to-infeasible | 16.2k feasible + 1.8k nose, each topology | N-1 + N-2 | **yes** (nose) |
| `gridfm/pfdelta_task4.2` | case118 | 695 MB | 4.2 close-to-infeasible | 9k feasible + 1.8k nose + 7.2k around-nose, each | N-1 + N-2 | **yes** |
| `gridfm/pfdelta_task4.3` | case118 | 664 MB | 4.3 close-to-infeasible | 3.6k nose + 14.4k around-nose, each (no feasible) | N-1 + N-2 | **yes** (only) |

**None include case14.** None of these were downloaded.
The matching checkpoints (`gridfm/genco-pfdelta`) are also **case118-only**. They were trained on the
`genco-paper-repro-pfdelta` graphkit branch.

## 2. Original PFΔ release: `pfdelta/pfdelta` (HF), case14

Per-case archives: case14 230 MB, case30 469 MB, case57 1.6 GB, case118 3.8 GB, case500 15.4 GB, case2000 26.9 GB.
**Downloaded:** `case14.tar.gz` (230 MB) and `shuffle_files.tar.gz` (0.8 MB), not extracted.

Contents of `case14.tar.gz` (159,622 entries):

| Topology | Feasible (`raw/`) | Nose (near-infeasible), train / test | Around-nose (approaching infeasible), train |
|---|---|---|---|
| N | 56,000 | 3,600 / 200 | 14,400 |
| N-1 | 29,000 | 3,600 / 200 | 14,400 |
| N-2 | 20,000 | 3,600 / 200 | 14,400 |

- Feasible test split: 2,000 per topology, given by `shuffle_files/{n,n-1,n-2}/raw_shuffle*.json`.
- Base grid: `pglib_opf_case14_ieee.m`. This is **the same base grid** as the checkpoint's training data
  (datakit `source: pglib`). 14 buses, 5 generators, 20 branches, 11 loads.
- Outages (from `data_generation/src/pf_delta_perturbations.jl`): N-1 removes a random line (p=0.5) or a non-slack
  generator (p=0.5). N-2 removes 2 lines, 2 non-slack gens, or 1 line + 1 gen (1/3 each). **The slack generator is never removed.**
- **Islanding is rejected by design.** The sampler loops until `is_connected(graph)`.
- Nose / around-nose: continuation power flow up to the voltage-stability limit (Jacobian singular at the nose).
  Around-nose samples are the last points before the nose (the filename carries `lam_<λ>`).
- Per-sample JSON fields (feasible schema `{network, solution}`):
  - branch: `rate_a` (p.u., 100 MVA base), `br_status`, `br_r/x`, `tap`, `shift`, `angmin/angmax`;
    solution flows `pf, qf, pt, qt`
  - bus: `vmin = 0.94`, `vmax = 1.06`; solution `vm`, `va`
  - gen: `gen_status`, `pg`, `qg`, `pmin/pmax`, `qmin/qmax`
  - solver `termination_status`
- The PFΔ paper states thermal limits are **not enforced** in generation. So overloads remain in the data and can be labeled.

Spot check of one feasible N-2 sample (`n-2/raw/sample_1.json`: branch 19 + gen 3 out):
min Vm = **0.857** (< 0.94), branches 2 and 10 at **109% and 128%** of `rate_a`. Both anomaly types occur.

## 3. gridfm case14 datasets (graphkit-ready parquet)

| HF dataset | Size | Scenarios | Topology perturbation | Notes |
|---|---|---|---|---|
| **`gridfm/pf_small_case14_ieee`** | 396 MB | 219,606 | `random`, k=1, elements [branch, gen], 22 variants × 10k load scenarios | **Training data of the case14 checkpoint.** `mode: pf`, so limits may be violated. Admittance perturbation σ=0.2. **Downloaded** → `data/case14_ieee/raw/` |
| `gridfm/pf_large_case14_ieee` | 950 MB | 1,009,092 | `random`, k=1, 101 variants | Same generator, 4.6× larger |
| `gridfm/opf_small_case14_ieee` | 2,536 MB | – | – | OPF mode (limits satisfied by construction), so not useful for anomalies |
| `gridfm/opfdata_case14_ieee` | 490 MB | – | – | Converted OPFData (OPF), so not useful for anomalies |

`pf_small_case14_ieee` columns include `rate_a`, `br_status`, `pf/qf/pt/qt` (branch), `Vm/Va/Pd/Qd/Pg/Qg`,
`min_vm_pu/max_vm_pu` (bus), `in_service` (gen). So voltage and overload labels are computable (verified in Phase 3).
Datakit also **rejects islanded topologies** (`check_single_connected_component()` in `topology_perturbation.py`).

## 4. Other grids in `gridfm/genco-pf-datakit` (for a later zero-shot cross-topology test)

Listed with `HfApi().list_repo_tree("gridfm/genco-pf-datakit", recursive=True)`. **Nothing downloaded.**
Every run folder has `best_model_state_dict.pt`, `normalizer_stats.pt`, `metrics.csv` and the training YAML
(case10000 also has `last.pt`). Each is trained on the matching `gridfm/pf_small_<grid>` dataset.

| Grid | Buses | Sizes available | Seeds | Weights (base / small / tiny) |
|---|---|---|---|---|
| case14_ieee | 14 | base, small, tiny | 0, 1, 42 | 80.5 / 20.4 / 5.3 MB (seed0 of each downloaded) |
| case30_ieee | 30 | base, small, tiny | 0, 1, 42 | 80.5 / 20.4 / 5.3 MB |
| case57_ieee | 57 | base, small, tiny | 0, 1, 42 | 80.5 / 20.4 / 5.3 MB |
| case118_ieee | 118 | base, small, tiny | 0, 1, 42 | 80.5 / 20.4 / 5.3 MB |
| case500_goc | 500 | base, small, tiny | 0, 1, 42 | 80.5 / 20.4 / 5.3 MB |
| case2000_goc | 2000 | small, tiny | 0, 42 | – / 20.3 / 5.3 MB |
| case10000_goc | 10000 | small, tiny | 0, 1 | – / 20.3 / 5.3 MB |

**The weight file size is identical across grids.** The GNS has no grid-size-dependent parameters (all layers act
per node / per edge), so a checkpoint trained on one grid can be run on another grid's graph as-is. That is what a
zero-shot cross-topology test needs. Things to handle when doing it:
- `normalizer_stats.pt` is keyed by network name (e.g. only `case30_ieee`) and stores a fitted `baseMVA`. Running a
  case30 model on case14 data needs either the case30 baseMVA reused under the `case14_ieee` key (the model's
  "expected" scale) or a refit on case14 (the data's scale). This is a design choice to make and report.
- The same config/version conversion as for case14 (`--config_conversion auto_inline`) will apply.
- Related: `gridfm/genco-pf-transfer` has a checkpoint **pretrained on many subgrids** (no normalizer stats), plus
  case118 fine-tuned vs from-scratch models. It could be a more "foundation-model-like" starting point than a single-grid model.

## 5. Source paper of the genco checkpoints

All six `gridfm/genco-*` model repos are the artifacts of the GENCO paper. The model cards name the sections
(e.g. `genco-pf-datakit` = §5.4.2 "PF on datakit", `genco-pfdelta` = §5.1).

> A. Puech, M. Mazzonelli, T. R. Govindasamy, M. Mngomezulu, H. Maeso-García, T. Tolhurst, J. Bayazi, A. Moeini,
> N. Simumba, C. Cintas, D. Nelischer, R. Kienzler, J. Weiss, A. Varbella, F. Dörfler, G. Hug, M. Mevissen,
> J. Bernabé-Moreno, F. Mirallès, H. F. Hamann, E. Vos, T. Brunschwiler.
> **"GENCO - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis."**
> arXiv:2608.09921 [cs.AI], submitted 10 Aug 2026, v2 20 Aug 2026. https://arxiv.org/abs/2608.09921

```bibtex
@article{puech2026gencounifiedneural,
  title   = {GENCO - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis},
  author  = {Alban Puech and Matteo Mazzonelli and Tamara R. Govindasamy and Mangaliso Mngomezulu and H{\'e}ctor Maeso-Garc{\'i}a and Thomas Tolhurst and Javad Bayazi and Ali Moeini and Naomi Simumba and Celia Cintas and David Nelischer and Romeo Kienzler and Jonas Weiss and Anna Varbella and Florian D{\"o}rfler and Gabriela Hug and Martin Mevissen and Juan Bernab{\'e}-Moreno and Fran{\c{c}}ois Mirall{\`e}s and Hendrik F. Hamann and Etienne Vos and Thomas Brunschwiler},
  journal = {arXiv preprint arXiv:2608.09921},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.09921}
}
```

This is **not** the course's core paper ("Foundation Models for the Electric Power Grid", arXiv 2407.09434, 2024),
which is the earlier GridFM position/vision paper. GENCO is the concrete model and toolkit the checkpoints come from.
Data sources to cite as well:
- gridfm-datakit: Puech et al., arXiv:2512.14658 (2025). Generated `pf_small_case14_ieee`.
- PFΔ: Rivera et al., "PFΔ: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations",
  NeurIPS 2025 Datasets & Benchmarks, arXiv:2510.22048.

## 6. Does PFΔ alone cover the anomaly labeling we need?

| Label | Covered by PFΔ case14? | How |
|---|---|---|
| Normal | Yes | Feasible samples with all Vm in [0.94, 1.06] and all \|S\| ≤ rate_a |
| Voltage violation | **Yes** | `solution.bus.vm` vs `vmin/vmax`. Violations present (seen in spot check) |
| Line overload | **Yes** | `max(\|pf+jqf\|, \|pt+jqt\|)` vs `rate_a`. Limits not enforced, so overloads present |
| N-1 / N-2 contingency (metadata) | **Yes** | `br_status` / `gen_status` per sample, plus separate folders per topology |
| Near-infeasible / voltage collapse | **Yes** (bonus class) | `nose` / `around_nose` samples. Needs a schema adapter for the converter |
| **Islanding** | **No** | Rejected by the generator (connected graphs only). Datakit also rejects it |
| Transformer-specific outages | Partly | Transformers are branches (`transformer` flag) and can be removed as "lines". No separate category |

**Conclusion:** PFΔ case14 alone is sufficient for normal / voltage violation / line overload (+ optional near-infeasible)
**without running datakit generation or Julia**. It is the preferred source. Islanding cannot be a class from either
PFΔ or datakit without modifying the generators. Also, a power-flow solution for an islanded grid needs a slack bus per
island, which neither tool supports. Class balance is not yet known and is computed in Phase 3.

### Caveats for using PFΔ with the case14 checkpoint
1. **Distribution shift, by design useful for "hard cases".** The checkpoint saw datakit's load profiles, cost
   permutation, ±20 % admittance perturbation and N-1 only. PFΔ has different load sampling, **N-2**, and near-infeasible
   operating points. Expect degraded accuracy on N-2 / nose samples.
2. The PFΔ converter lives in the datakit GitHub repo, not in the pip package. It must be copied or run from a clone
   pinned to a commit (checked at `5a46ffd`, 2026-09-24).
3. Nose / around-nose JSONs need a ~10-line wrapper to `{network, solution}` before conversion.
4. graphkit requires contiguous scenario IDs 0..N-1 per dataset folder. A ≤1000-sample subset must be converted into its
   own folder (the converter numbers scenarios from `sample_N.json` → N-1, so files would need renumbering).
