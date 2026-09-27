# gridfm-anomaly

IMT4392 (NTNU) course project: anomaly detection on the electric power grid using GridFM foundation models
("Foundation Models for the Electric Power Grid", arXiv 2407.09434).

Status: **Phase 1 (environment) done.** Phases 2–5 still to do.

## Layout

```
data/          downloaded / generated datasets (gitignored)
checkpoints/   model weights (gitignored)
configs/       YAML configs (datakit generation, graphkit inference)
notebooks/     exploration notebooks
scripts/       small standalone scripts
results/       markdown summaries, CSVs, plots
SETUP_LOG.md   every command and version, for the report appendix
```

## Environment

Windows, Python **3.11** (datakit does not support 3.12 on Windows; neither package supports 3.13), CPU-only PyTorch 2.12.

```bash
py -3.11 -m venv venv
./venv/Scripts/python.exe -m pip install --upgrade pip
./venv/Scripts/python.exe -m pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
```

See `SETUP_LOG.md` for why torch is pinned to 2.12 (prebuilt `torch-scatter` wheels for Windows).
