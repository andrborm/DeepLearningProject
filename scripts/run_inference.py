"""Run the case14 GENCO checkpoint on the seeded test subset and compute error metrics.

Uses the repo's intended path, the CLI `gridfm_graphkit evaluate --save_output`, which
  - writes graphkit's own RMSE table (same code/format as the model card), and
  - saves per-bus / per-branch predictions with targets and power-balance residuals.
This script then post-processes the predictions into per-sample errors.

Steps:
  1. build an inference config (checkpoint YAML -> v1, CPU, subset split files) in configs/
  2. run `gridfm_graphkit evaluate` on the first --n scenarios of data/case14_test1000
  3. per-variable MAE / RMSE / max-abs error per bus type and per outage type
  4. per-sample table: results/inference_per_sample[_<size>].csv
  5. residual (physics consistency) metrics from graphkit's per-bus PBE

Usage:
  python scripts/run_inference.py --size base --n 50
  python scripts/run_inference.py --size base --n 1000
  python scripts/run_inference.py --size tiny --n 1000
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
# dataset -> (graphkit data folder, subset mapping csv, output suffix)
DATASETS = {
    "datakit": (ROOT / "data/case14_test1000", ROOT / "results/test_subset_1000.csv", ""),
    "pfdelta": (ROOT / "data/pfdelta_test1000", ROOT / "results/pfdelta_test_subset_1000.csv", "_pfdelta"),
}
CKPT_ROOT = ROOT / "checkpoints/genco-pf-datakit/case14_ieee"
GRAPHKIT = ROOT / "venv/Scripts/gridfm_graphkit.exe"

# Unknowns the model has to predict, per bus type (see results/data_format.md §2).
# Every other (bus type, variable) pair is copied from the input and has zero error.
TARGETS = {"PQ": ["Vm", "Va"], "PV": ["Va", "Qg"], "REF": ["Vm", "Qg", "Pg"]}
UNITS = {"Vm": "p.u.", "Va": "rad", "Pg": "MW", "Qg": "MVAr"}


def make_config(size, n, subset, suffix):
    """Checkpoint YAML -> v1 inference config for the subset (written to configs/ for the record)."""
    from gridfm_graphkit.io.config_version import resolve_config

    ckpt_yaml = CKPT_ROOT / size / "seed0" / f"case14_ieee_{size}_seed0.yaml"
    cfg = resolve_config(yaml.safe_load(open(ckpt_yaml)), config_path=str(ckpt_yaml), conversion_mode="auto_inline")

    splits = subset / ("splits" if n == 1000 else f"splits_first{n}")
    if n != 1000:  # first n scenarios of the same seeded subset
        splits.mkdir(exist_ok=True)
        torch.save(torch.arange(n), splits / "test.pt")
        torch.save(torch.tensor([], dtype=torch.long), splits / "train.pt")
        torch.save(torch.tensor([], dtype=torch.long), splits / "val.pt")

    cfg["data"].update(
        split_from_existing_files=str(splits.relative_to(ROOT)).replace("\\", "/"),
        split_by_load_scenario_idx=False,  # graphkit allows only one split mode
        workers=0,
    )
    # "auto" makes the CLI pick a DDP strategy on non-MPS machines; force plain CPU
    cfg["training"].update(accelerator="cpu", devices=1, strategy="auto")
    out = ROOT / "configs" / f"case14_{size}_infer{suffix}.yaml"
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    return out


def run_evaluate(config, size, n, subset, suffix):
    ckpt = CKPT_ROOT / size / "seed0"
    run_name = f"{size}_n{n}{suffix}"
    cmd = [
        str(GRAPHKIT), "evaluate",
        "--config", str(config.relative_to(ROOT)),
        "--model_path", str((ckpt / "best_model_state_dict.pt").relative_to(ROOT)),
        "--normalizer_stats", str((ckpt / "normalizer_stats.pt").relative_to(ROOT)),
        "--data_path", str(subset.relative_to(ROOT)),
        "--log_dir", "mlruns",
        "--exp_name", "phase4_inference",
        "--run_name", run_name,
        "--num_workers", "0",
        "--save_output",
    ]
    print(">>", " ".join(cmd))
    # graphkit logs to a local ./mlruns folder; MLflow >= 3.x refuses file stores unless opted in
    env = dict(os.environ, MLFLOW_ALLOW_FILE_STORE="true", MLFLOW_DISABLE_AGENT_HINT="1", PYTHONIOENCODING="utf-8")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)
    # newest run folder with this run name that holds the saved predictions
    runs = [Path(p).parents[1] for p in glob.glob(str(ROOT / "mlruns/*/*/tags/mlflow.runName"))
            if Path(p).read_text().strip() == run_name]
    runs = [r / "artifacts" for r in runs if (r / "artifacts/test/case14_ieee_predictions.parquet").exists()]
    return max(runs, key=os.path.getmtime)


def per_bus_errors(bus):
    """Long table: one row per (scenario, bus, variable) for the unknowns only."""
    bus["bus_type"] = np.select([bus.REF == 1, bus.PV == 1], ["REF", "PV"], "PQ")
    rows = []
    for bt, variables in TARGETS.items():
        b = bus[bus.bus_type == bt]
        for v in variables:
            rows.append(pd.DataFrame({
                "scenario": b.scenario, "bus": b.bus, "bus_type": bt, "variable": v,
                "err": b[f"{v}_pred"] - b[f"{v}_target"],
            }))
    long = pd.concat(rows, ignore_index=True)
    long["abs_err"] = long.err.abs()
    return long


def metrics(group):
    return pd.Series({
        "n": len(group),
        "MAE": group.abs_err.mean(),
        "RMSE": np.sqrt((group.err ** 2).mean()),
        "max_abs": group.abs_err.max(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="base", choices=["base", "small", "tiny"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--dataset", default="datakit", choices=list(DATASETS))
    parser.add_argument("--artifacts", type=Path, help="skip inference, reuse an MLflow artifacts dir")
    args = parser.parse_args()
    assert args.n <= 1000, "course requirement: at most 1000 samples for inference"
    sys.stdout.reconfigure(encoding="utf-8")
    subset_dir, subset_csv, ds_suffix = DATASETS[args.dataset]

    art = args.artifacts or run_evaluate(make_config(args.size, args.n, subset_dir, ds_suffix),
                                         args.size, args.n, subset_dir, ds_suffix)
    print(f"\nartifacts: {art}")
    bus = pd.read_parquet(art / "test/case14_ieee_predictions.parquet")
    branch = pd.read_parquet(art / "test/case14_ieee_branch_predictions.parquet")
    subset = pd.read_csv(subset_csv).set_index("subset_id")
    assert bus.scenario.nunique() == args.n, f"expected {args.n} scenarios, got {bus.scenario.nunique()}"

    long = per_bus_errors(bus)
    long["outage_type"] = long.scenario.map(subset.outage_type)

    by_type = long.groupby(["variable", "bus_type"]).apply(metrics)
    by_outage = long.groupby(["variable", "bus_type", "outage_type"]).apply(metrics)
    by_var = long.groupby("variable").apply(metrics)

    # per-sample table: RMSE and max-abs per variable/bus-type pair + residuals
    per = long.pivot_table(index="scenario", columns=["variable", "bus_type"], values="err",
                           aggfunc=lambda e: np.sqrt((e ** 2).mean()))
    per.columns = [f"rmse_{v}_{bt}" for v, bt in per.columns]
    mx = long.pivot_table(index="scenario", columns="variable", values="abs_err", aggfunc="max")
    mx.columns = [f"maxabs_{v}" for v in mx.columns]
    per = per.join(mx)
    # overall score: mean over variables of (sample RMSE / subset RMSE), so 1.0 = typical sample
    rmse_var = long.groupby(["scenario", "variable"]).err.apply(lambda e: np.sqrt((e ** 2).mean())).unstack()
    per["overall_score"] = (rmse_var / by_var.RMSE).mean(axis=1)
    # same score but on a fixed scale across datasets: relative to the datakit base-model RMSE
    ref = ROOT / "results/inference_by_variable.csv"
    ref_rmse = pd.read_csv(ref, index_col="variable").RMSE if ref.exists() else by_var.RMSE
    per["overall_score_vs_datakit"] = (rmse_var / ref_rmse).mean(axis=1)
    res = bus.groupby("scenario").agg(pbe_mean=("PBE", "mean"), pbe_max=("PBE", "max"),
                                      res_P_mean=("active res. (MW)", "mean"),
                                      res_Q_mean=("reactive res. (MVar)", "mean"))
    per = per.join(res)
    id_cols = [c for c in ["topology", "orig_scenario", "outage_type", "out_branches", "out_gens"] if c in subset]
    per = subset[id_cols].join(per, how="inner")
    per.index.name = "sample_id"
    suffix = ds_suffix + ("" if args.size == "base" and args.n == 1000 else f"_{args.size}_n{args.n}")
    per.to_csv(ROOT / f"results/inference_per_sample{suffix}.csv", float_format="%.6g")

    # graphkit's own RMSE table for this run vs the model card (same code, same format)
    ours = pd.read_csv(art / "test/case14_ieee_RMSE.csv").set_index("Metric")
    card = pd.read_csv(next((ROOT / f"checkpoints/genco-pf-datakit/mlflow/eval/case14_ieee/{args.size}/seed0").glob(
        "*/artifacts/test/case14_ieee_RMSE.csv"))).set_index("Metric")
    compare = pd.concat({"this run": ours, "model card": card, "ratio": ours / card}, axis=1)
    residual_csv = pd.read_csv(art / "test/case14_ieee_metrics.csv")

    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n=== error per variable x bus type (unknowns only) ===\n" + by_type.to_string())
    print("\n=== by outage type ===\n" + by_outage.to_string())
    print("\n=== graphkit RMSE table vs model card ===\n" + compare.T.to_string())
    print("\n=== graphkit residual metrics (0.9.0.3 definition) ===\n" + residual_csv.to_string(index=False))
    print("\n=== per-sample residuals (PBE, MVA) ===\n" + per[["pbe_mean", "pbe_max"]].describe().to_string())
    print("\n=== overall score by outage type ===\n" + per.groupby("outage_type").overall_score.describe().to_string())
    print(f"\nbranch prediction columns: {list(branch.columns)}")

    tables = {"by_bus_type": by_type, "by_outage_type": by_outage, "by_variable": by_var,
              "rmse_vs_card": compare.T, "residuals": residual_csv.set_index("Metric")}
    for name, t in tables.items():
        t.to_csv(ROOT / f"results/inference_{name}{suffix}.csv", float_format="%.6g")
    print(f"\nwrote results/inference_per_sample{suffix}.csv and results/inference_<table>{suffix}.csv")


if __name__ == "__main__":
    main()
