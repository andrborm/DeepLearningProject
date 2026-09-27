# Setup log

Chronological record of every command run and every version installed, for the report appendix.
Shell: Git Bash on Windows unless noted. Paths are relative to `gridfm-anomaly/`.

---

## Phase 1: Project scaffold and environment (2026-09-27)

### 1.1 Machine

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 (`platform.platform()` → `Windows-10-10.0.26200-SP0`) |
| CPU | Intel Core i7-14700KF (20 cores / 28 threads; torch uses 20 threads) |
| RAM | 31.8 GB |
| GPU | AMD Radeon RX 9070 XT (driver 32.0.31041.1004) |
| CUDA | **Not available.** The GPU is AMD, so there is no NVIDIA driver or CUDA; `nvidia-smi` is not found. |
| WSL / Docker | Neither installed |
| git | 2.51.0.windows.1 |
| Python interpreters found (`py -0p`) | 3.13 (default), 3.12, 3.11.0 |

Commands:
```bash
py -0p
nvidia-smi                      # -> command not found
git --version
# PowerShell:
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores
(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB
```

### 1.2 Python version choice

I checked PyPI metadata (`https://pypi.org/pypi/<pkg>/json`) and both READMEs:

| Source | Constraint |
|---|---|
| gridfm-datakit 1.1.0 `requires_python` | `>=3.10,<3.13` |
| gridfm-graphkit 0.9.0.3 `requires_python` | `>=3.10,<3.13` |
| gridfm-datakit README | "Windows users: Python 3.12 is not supported. Use Python 3.10.11 or 3.11.9." |
| gridfm-graphkit README | 3.10 to 3.12, "I highly recommend 3.12" |

**Decision: Python 3.11.** It is the only version that satisfies the datakit Windows restriction and is
available on this machine. The installed patch is 3.11.0 (the README names 3.11.9), and it has not caused
any problems so far. The system default 3.13 is **not** supported by either package.

### 1.3 Scaffold

```bash
mkdir -p gridfm-anomaly/{data,checkpoints,notebooks,scripts,results,configs}
cd gridfm-anomaly
touch notebooks/.gitkeep scripts/.gitkeep results/.gitkeep configs/.gitkeep
# wrote .gitignore (data/, checkpoints/, venv/, __pycache__/, mlruns/, lightning_logs/, *.pt, *.npy, *.parquet in results/, ...)
git init
```
`configs/` was added beyond the requested structure because Phase 5 asks for `configs/smoke_nk.yaml`.

### 1.4 Virtual environment and installs

```bash
py -3.11 -m venv venv
./venv/Scripts/python.exe -m pip install --upgrade pip          # -> pip 26.2.1

# CPU-only PyTorch first, so pip does not choose a build on its own
./venv/Scripts/python.exe -m pip install "torch>=2.10,<2.14" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu            # -> torch 2.13.0+cpu

./venv/Scripts/python.exe -m pip install gridfm-graphkit gridfm-datakit \
    --extra-index-url https://download.pytorch.org/whl/cpu
# FAILED: "Failed to build 'torch-scatter'". graphkit lists torch-scatter as a dependency, and PyPI has
# only an sdist, which needs a C++ compiler (MSVC). The graphkit README says to install it from the PyG wheel index.
```

Checking the PyG wheel index for Windows / cp311 wheels:
```bash
for v in 2.13.0 2.12.0 2.11.0 2.10.0; do
  curl -s "https://data.pyg.org/whl/torch-$v+cpu.html" | grep -oE '(torch_scatter|torch_sparse)-[^"<]*cp311-cp311-win_amd64\.whl'
done
# torch-2.13.0+cpu: no wheels;  torch-2.12.0/2.11.0/2.10.0+cpu: torch_scatter 2.1.2 + torch_sparse 0.6.18 available
```

**Fix:** downgrade to torch 2.12.0, which is still inside graphkit's `torch>=2.10,<2.14` range, then install the prebuilt wheels:
```bash
./venv/Scripts/python.exe -m pip install "torch==2.12.0" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu            # -> torch 2.12.0+cpu, torchvision 0.27.0+cpu
./venv/Scripts/python.exe -m pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.12.0+cpu.html --only-binary torch-scatter,torch-sparse
./venv/Scripts/python.exe -m pip install gridfm-graphkit gridfm-datakit \
    --extra-index-url https://download.pytorch.org/whl/cpu      # -> success
./venv/Scripts/python.exe -m pip check                           # -> No broken requirements found.
./venv/Scripts/python.exe -m pip freeze > requirements.txt       # 158 packages
```

Note: torchaudio stayed at 2.11.0+cpu (pip did not upgrade it with torch 2.12). `pip check` reports no conflict,
and torchaudio is never used by this project.

### 1.5 Verification

Checked with a short Python script: import all packages, print versions, and run one `GCNConv` forward pass plus one `torch_scatter.scatter_add` call.
Both CLIs (`gridfm_graphkit --help`, `gridfm_datakit --help`) start.

| Package | Version |
|---|---|
| Python | 3.11.0 |
| torch | 2.12.0+cpu (`torch.version.cuda = None`, `cuda.is_available() = False`, no ROCm/HIP build) |
| torchvision / torchaudio | 0.27.0+cpu / 2.11.0+cpu |
| torch_geometric | 2.8.0.post1 |
| torch_scatter / torch_sparse | 2.1.2+pt212cpu / 0.6.18+pt212cpu |
| gridfm-graphkit | 0.9.0.3 |
| gridfm-datakit | 1.1.0 |
| lightning | 2.6.6 |
| mlflow | 3.16.1 |
| juliacall | 0.9.36 |
| numpy / pandas / pyarrow | 2.4.6 / 3.0.6 / 25.0.1 |

CLI subcommands found:
- `gridfm_graphkit {train, finetune, evaluate, predict, benchmark}`
- `gridfm_datakit {generate, validate, stats, plots, setup_pm}`

### 1.6 Deferred / open

- `gridfm_datakit setup_pm` (installs Julia + PowerModels.jl + Ipopt through juliacall) has **not** been run yet.
  Only data generation needs it (Phase 5), not inference. It will be run and logged in Phase 5.
- GPU: PyTorch ROCm is not officially supported on Windows, so everything runs on CPU. This should be fine for the
  small case14 model and ≤1000 samples.

### Recreating the environment

`requirements.txt` contains `+cpu` local version tags, so the two extra indexes are required:
```bash
py -3.11 -m venv venv
./venv/Scripts/python.exe -m pip install --upgrade pip
./venv/Scripts/python.exe -m pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
```

---

## Git (2026-09-27)

The remote `https://github.com/andrborm/DeepLearningProject.git` already had one commit (a placeholder `README.md` from GitHub).
Our first commit was built on top of it, so no force-push was needed:
```bash
git remote add origin https://github.com/andrborm/DeepLearningProject.git
git fetch origin
git symbolic-ref HEAD refs/heads/main      # local branch "main" to match the remote
git reset origin/main                       # index = remote tree, working tree untouched
git add -A && git commit -m "Set up Python 3.11 environment for GridFM project"
git push -u origin main                     # a8d9610..3e6788a
```

---

## Phase 2: Checkpoint and data (2026-09-27)

### 2.1 Finding the identifiers

```bash
curl -s "https://huggingface.co/api/models?author=gridfm&limit=100"
curl -s "https://huggingface.co/api/datasets?author=gridfm&limit=200"
curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/gridfm/gridfm-pf-reconstruction   # -> 401
```

- **`gridfm/gridfm-pf-reconstruction` does not exist or is not public** (HTTP 401). The gridfm org has only six model
  repos, all from the GENCO paper (arXiv 2608.09921): `genco-pf-datakit`, `genco-pfdelta`, `genco-pf-transfer`,
  `genco-pf-contingency`, `genco-opfdata`, `genco-opf-datakit`.
- **The case14 PF checkpoint is `gridfm/genco-pf-datakit`, folder `case14_ieee/{base,small,tiny}/seed{0,1,42}/`.**
  `genco-pfdelta` (the PFDelta models) is **case118-only**.

### 2.2 Checkpoint: `gridfm/genco-pf-datakit` / `case14_ieee`

| Item | Value |
|---|---|
| Trained on | `gridfm/pf_small_case14_ieee` (datakit PF data, pglib case14, 219,606 scenarios; the YAML says `scenarios: [250000]` as an upper bound), split 80/10/10 **by load scenario** (`split_by_load_scenario_idx: true`) |
| Task | `PowerFlow` (graphkit `PowerFlowTask`) |
| Architecture | `GNS_heterogeneous`: heterogeneous graph (bus nodes, gen nodes, bus–bus branch edges, gen↔bus edges), 12 message-passing layers with 8 attention heads, LayerNorm and skip connections. After every layer the AC power-balance residual is computed and fed back into the bus hidden state via `physics_mlp` (physics-informed "graph neural solver"). |
| Sizes (seed0) | base: hidden 48, 20.07 M params, 80 MB · small: hidden 24, 5.05 M, 20 MB · tiny: hidden 12, 1.28 M, 5 MB |
| Inputs | bus 15 features, gen 6, edge 10 (see 2.3) |
| Outputs | per bus: **Vm, Va** (2-dim head). Per gen: **Pg** (1-dim head). Known (unmasked) values are overwritten with inputs. A physics decoder then derives bus-level **Pg, Qg** from predicted voltages, giving bus output `[Vm, Va, Pg, Qg]` |
| Loss | 0.1 × `LayeredWeightedPhysics` (per-layer power-balance residual) + 0.9 × `MaskedBusMSE` |
| Normalizer | `HeteroDataMVANormalizer`: divides powers by a fitted `baseMVA = 84.596` (stored in `normalizer_stats.pt`), angles deg→rad, `rate_a` / baseMVA. Not a per-feature mean/std. |
| Training config | AdamW lr 5e-4, ReduceLROnPlateau (0.7, patience 5), batch 64, 200 epochs, seeds 0 / 1 / 42 |
| Training code | graphkit branch `genco-paper-repro` = `v0.0.7-35-g6edf7eb` (pyproject `version = "0.0.7"`). This branch **diverged** from main; it is not an ancestor of v0.9.0.3. The MLflow runs record **no** package version or git commit (tags and params are empty). Eval run date 2026-05-08. |

Reference test metrics (from the checkpoint's MLflow eval run on the 21,946-scenario test split, seed0):

| Model | RMSE Vm PQ (p.u.) | RMSE Va PQ (rad) | RMSE Qg PV (MVar) | RMSE Pg REF (MW) | RMSE Qg REF (MVar) | PBE mean | PBE max |
|---|---|---|---|---|---|---|---|
| base | 1.16e-4 | 1.03e-3 | 0.579 | 0.160 | 1.196 | 2.68e-3 | 0.573 |
| small | 1.48e-4 | 1.23e-3 | 0.665 | 0.164 | 1.362 | 3.03e-3 | 1.646 |
| tiny | 1.37e-4 | 0.94e-3 | 0.809 | 0.194 | 1.674 | 5.13e-3 | 4.285 |

Download:
```bash
./venv/Scripts/hf.exe download gridfm/genco-pf-datakit \
    --include "case14_ieee/base/seed0/*" --include "case14_ieee/small/seed0/*" --include "case14_ieee/tiny/seed0/*" \
    --local-dir checkpoints/genco-pf-datakit
./venv/Scripts/hf.exe download gridfm/genco-pf-datakit \
    --include "mlflow/eval/case14_ieee/*/seed0/*/artifacts/test/*" \
    --include "mlflow/eval/case14_ieee/base/seed0/*/artifacts/stats/*" --include README.md \
    --local-dir checkpoints/genco-pf-datakit
```
Note: with `hf` (huggingface_hub 2.0.0, already installed as a dependency) **each pattern needs its own `--include`**.
`--include a b c` silently dropped patterns.
`artifacts/stats/case14_ieee_scenario_splits.json` gives the train/val/test scenario IDs (175,713 / 21,947 / 21,946).

### 2.3 Compatibility of the checkpoint with graphkit 0.9.0.3

Method: cloned `gridfm-graphkit` main (v0.9.0.3 = `f70230c`) and the `genco-paper-repro` branch (`6edf7eb`), diffed the
relevant modules, then loaded all three checkpoints through the same code path the CLI uses
(`resolve_config` → `LitGridHeteroDataModule` → `get_task` → `model.load_state_dict`, strict).

| Check | Result |
|---|---|
| Config version | Checkpoint YAML has no `version:` key, so it is treated as **v0**. 0.9.0.3 refuses it by default (`ValueError: ... is version 0, but this version ... requires config version 1`). **Fix:** `--config_conversion auto_inline` (only rewrites the optimizer block into the v1 schema, which is irrelevant for inference). |
| State dict keys | **0 missing, 0 unexpected, all shapes match** for base / small / tiny. Strict load OK. |
| Warnings on load | None about keys. Only `torch.jit.script` DeprecationWarnings from inside torch/graphkit (harmless). |
| Bus/gen/branch feature lists | **Identical.** Bus: `Pd, Qd, Qg, Vm, Va, PQ, PV, REF, min_vm_pu, max_vm_pu, min_q_mvar, max_q_mvar, GS, BS, vn_kv`. Gen: `p_mw, min_p_mw, max_p_mw, cp0, cp1, cp2, in_service`. Branch fwd: `pf, qf, Yff_r, Yff_i, Yft_r, Yft_i, tap, ang_min, ang_max, rate_a, br_status` (reverse uses `pt, qt, Ytt, Ytf`). `in_service` / `br_status` are used to drop inactive elements, giving gen dim 6 and edge dim 10. |
| Index constants (`globals.py`) | Identical, plus two new constants (`PD_OUT`, `QD_OUT`) used only for random masking. |
| Normalizer (`HeteroDataMVANormalizer`) | Same scaling logic. 0.9.0.3 adds Pd/Qd output rescaling, but only when the output has >4 columns (not our case). |
| PF masking (`AddPFHeteroMask`) | **Unchanged.** New `AddRandomHeteroMask` is only used when `data.mask_type: rnd` (not set). |
| Model forward | Numerically equivalent predictions. Signature changed to `forward(batch, return_embeddings=False)`. It skips a last-layer `physics_mlp` update whose result was never used, and can now return the final bus/gen hidden states (embeddings). |
| **Metric definition** | **Changed.** 0.9.0.3 overwrites known quantities with ground truth (`_clamp_known_to_ground_truth`) before computing power-balance residuals / PBE. The training branch used raw model output. **Residual/PBE numbers from 0.9.0.3 are therefore not directly comparable** to the model card's `metrics.csv`. Per-bus-type RMSE is computed the same way. |

**Verdict: the checkpoint is usable with graphkit 0.9.0.3** (with `--config_conversion auto_inline` and
`--normalizer_stats`). The only mismatch that affects reported numbers is the residual/PBE metric definition.

### 2.4 Datasets

```bash
# checkpoint's own training/test data (graphkit-ready), layout expected by graphkit: data/<network>/raw/
mkdir -p data/case14_ieee/raw
./venv/Scripts/hf.exe download gridfm/pf_small_case14_ieee --repo-type dataset --local-dir data/case14_ieee/raw

# original PFΔ case14 (N / N-1 / N-2 + near-infeasible), not extracted
./venv/Scripts/hf.exe download pfdelta/pfdelta case14.tar.gz        --repo-type dataset --local-dir data/pfdelta_orig
./venv/Scripts/hf.exe download pfdelta/pfdelta shuffle_files.tar.gz --repo-type dataset --local-dir data/pfdelta_orig
```

Download of `pf_small_case14_ieee` took 13 min (unauthenticated, hit one HF rate limit, then retried automatically).
Verified with pyarrow: 7,262 files (matches the HF listing), 427 MB on disk, Hive-partitioned (1,108 `scenario_partition=*`
folders per table). bus / gen / branch rows = 3,074,484 / 1,098,030 / 4,392,120 = 219,606 scenarios × 14 / 5 / 20.
Scenario IDs are contiguous 0..219,605 and match `n_scenarios.txt`.

Source repos cloned read-only (in the scratchpad, not in this project) to read code:
- `gridfm/gridfm-graphkit` main @ `f70230c` (v0.9.0.3) and `genco-paper-repro` @ `6edf7eb`
- `gridfm/gridfm-datakit` main @ `5a46ffd` (has `pfdelta/batch_convert_pfdelta.py`)
- `MOSSLab-MIT/pfdelta` main @ `6fcdf2b`

See `results/dataset_overview.md` for the full dataset comparison.

```bash
git add -A && git commit -m "Document Phase 2: case14 checkpoint and dataset survey" && git push   # 1aae92b
```

---

## Phase 3: Data format exploration (2026-09-27)

### 3.1 Extra listings (no downloads)

```python
from huggingface_hub import HfApi
HfApi().list_repo_tree("gridfm/genco-pf-datakit", path_in_repo="<grid>", recursive=True)
```
(The raw `/api/models/.../tree` endpoint started returning non-JSON after many calls, likely rate limiting, so the
Python API was used instead.) GENCO paper metadata verified at https://arxiv.org/abs/2608.09921. The course paper
2407.09434 was verified to be the 2024 GridFM position paper. Both are in `results/dataset_overview.md` §4–5.

### 3.2 Test subset (1000 scenarios)

```bash
./venv/Scripts/python.exe scripts/make_test_subset.py --n 1000 --seed 42    # ~30 s
```
- Samples 1000 of the checkpoint's 21,946 test scenario ids with `numpy.random.default_rng(42)`, renumbers them 0..999
  and writes flat parquet to `data/case14_test1000/case14_ieee/raw/`, split files `data/case14_test1000/splits/{train,val,test}.pt`
  (train/val empty, test = all 1000), and `results/test_subset_1000.csv` (original id, load scenario, outage type/element).
- Loaded successfully through `LitGridHeteroDataModule` with the base checkpoint config (setup 13 s, 1000 test samples,
  normalizer restored from `normalizer_stats.pt`, baseMVA 84.596). Two config overrides are **required**:
  `data.split_from_existing_files: data/case14_test1000/splits` **and** `data.split_by_load_scenario_idx: false`
  (graphkit asserts that only one split mode is set).
- First load creates `data/case14_test1000/case14_ieee/processed/` (1000 `.pt` files).

### 3.3 Exploration

```bash
tar -xzf data/pfdelta_orig/case14.tar.gz -C data/pfdelta_orig        # 2.2 GB, 159,622 entries; slow on Windows (many small files)
tar -xzf data/pfdelta_orig/shuffle_files.tar.gz -C data/pfdelta_orig
./venv/Scripts/python.exe scripts/explore_data.py --parts ABCD         # ~15 s
./venv/Scripts/python.exe scripts/explore_data.py --parts E            # PFΔ part, 20 s
```
- Extraction ran at only ~50–500 files/s (Windows + many small files; likely antivirus scanning), so it would take more
  than 1 h. Part E therefore **streams the JSONs directly from `case14.tar.gz`** with Python `tarfile` + a multiprocessing
  pool (105k feasible JSONs parsed in 20 s). The extraction was left running in the background for later use
  (the datakit converter needs files on disk).
- First run of part E crashed with `UnicodeEncodeError` (Windows console cp1252 can't print "Δ"). Fixed with
  `sys.stdout.reconfigure(encoding="utf-8")` in the script.
- Datakit's `stats.parquet` only covers 20,000 of 219,606 scenarios, so violations are computed from the raw tables instead.

```bash
git add -A && git commit -m "Add Phase 3 data exploration and 1000-sample test subset" && git push   # 2adce4f
```

---

## Phase 4: First inference run (2026-09-27)

```bash
./venv/Scripts/python.exe scripts/run_inference.py --size base --n 50      # ~40 s, check run
./venv/Scripts/python.exe scripts/run_inference.py --size base --n 1000    # ~43 s, main run
./venv/Scripts/python.exe scripts/run_inference.py --size tiny --n 1000    # ~39 s, sanity check
```
Each call writes `configs/case14_<size>_infer.yaml` and then runs the graphkit CLI:
```bash
gridfm_graphkit evaluate --config configs/case14_<size>_infer.yaml \
  --model_path checkpoints/genco-pf-datakit/case14_ieee/<size>/seed0/best_model_state_dict.pt \
  --normalizer_stats checkpoints/genco-pf-datakit/case14_ieee/<size>/seed0/normalizer_stats.pt \
  --data_path data/case14_test1000 --log_dir mlruns --exp_name phase4_inference --run_name <size>_n<N> \
  --num_workers 0 --save_output
```
For n < 1000 the script writes `data/case14_test1000/splits_first<n>/` (test = first n subset ids) so the 50-sample
check uses the same processed data.

Problems hit and fixes:
- **MLflow 3.16 refuses the file-store backend**: `MlflowException: The filesystem tracking backend (e.g., './mlruns') is in
  maintenance mode ... set MLFLOW_ALLOW_FILE_STORE=true to opt out`. graphkit's CLI always creates an `MLFlowLogger` on a local
  folder. Fixed by setting `MLFLOW_ALLOW_FILE_STORE=true` in the subprocess environment (alternative: pin `mlflow<3.x`).
- `accelerator: auto` makes the CLI choose `DDPStrategy` on any non-MPS machine (`cli.py`), so the inference config sets
  `training.accelerator: cpu` explicitly.
- Harmless warnings: `Requested number of scenarios (250000) exceeds dataset size (1000)`, `data.scenarios is ignored when
  split_from_existing_files is set`, numpy `invalid value encountered in divide` (graphkit's correlation plot for constant
  targets, e.g. Va at REF = 0), Lightning "few workers" hints.

MLflow run ids (experiment `664988715015135692`, gitignored `mlruns/`): base n50 `04f66457…`, base n1000 `cd767993…`,
tiny n1000 `989a6264…`.

```bash
git add -A && git commit -m "Add Phase 4 inference on 1000-sample case14 test subset" && git push   # 04bf3af
```

---

## Phase 5: Hard cases, embeddings, predict-then-check (2026-09-27)

Scope changed by request: **the datakit/Julia N-k smoke test is deferred** (`gridfm_datakit setup_pm` still not run).

```bash
export MLFLOW_ALLOW_FILE_STORE=true
gridfm_graphkit predict --config configs/case14_base_infer.yaml \
  --model_path checkpoints/genco-pf-datakit/case14_ieee/base/seed0/best_model_state_dict.pt \
  --normalizer_stats checkpoints/genco-pf-datakit/case14_ieee/base/seed0/normalizer_stats.pt \
  --data_path data/case14_test1000 --log_dir mlruns --exp_name phase5_embeddings --run_name base_n1000 \
  --num_workers 0 --get_embeddings --output_path data/predictions/base_n1000        # 17 s
./venv/Scripts/python.exe scripts/phase5_analysis.py > results/phase5_analysis_log.txt   # 3 s
```
- `predict --get_embeddings` writes `case14_ieee_{predictions,branch_predictions,bus_embeddings}.parquet` into
  `--output_path` (gitignored `data/predictions/`). Embeddings are 384-d per bus (14,000 rows × 386 columns).
- Figures follow the dataviz reference palette. The categorical set was validated with the skill's
  `validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all` (Node v24.14.0): all checks pass, with a contrast WARN
  on aqua, handled by a distinct marker shape and count labels.
- The background `tar -xzf case14.tar.gz` finished after **23 min** (159,600 files, 2.5 GB in `data/pfdelta_orig/case14/`).
