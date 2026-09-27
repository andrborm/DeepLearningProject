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
