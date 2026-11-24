"""Part 2 (GENCO side): CPU inference time of the frozen replay checkpoint on the three 1,000-sample test subsets.

Inference only, no training. Runs in the existing venv (torch + graphkit), but writes nothing into the repo's data/,
configs/ or mlruns/: each test set's raw parquet files and split files are COPIED to a scratch folder (passed as
--work), where graphkit builds its processed graph cache; the config is resolved in memory from the checkpoint yaml
(same settings as finetune_pipeline.make_config: split files, workers 0, CPU) and saved only in the scratch folder.

Timed per batch (torch.no_grad, model.eval(), after 5 warm-up batches; data loading is not timed):
  forward       model.model(batch)                         = the GNN itself
  predict_step  model.predict_step(batch, i)               = forward + de-normalisation + physics decoder
                                                             (branch flows, injections, residuals) = what `predict` runs
for batch size 1 and batch size 64 (the checkpoint's training/predict batch size, used in all inference runs).
Threads: torch default (printed). pandapower NR runs single-threaded, so a 1-thread GENCO run is also reported.

Check:  predictions of the timed model vs the saved replay predictions (data/predictions/stage3/replay/<set>).
Output: results/extra/genco_timing.csv
Usage:  venv/Scripts/python.exe -B scripts/extra/genco_timing.py --work <scratch folder>
"""

import argparse
import copy
import os
import platform
import shutil
import time
from pathlib import Path

import common  # noqa: F401
import numpy as np
import pandas as pd
import torch
import yaml

from common import EXTRA, ROOT, SEED, TESTS
from finetune_pipeline import CKPT_ROOT, model_path

SIZE = "base"
BATCHES = [1, 64]


def prepare(name, work):
    src = TESTS[name][0].parent.parent  # data/<set>
    dst = Path(work) / name
    if not (dst / "case14_ieee/raw").exists():
        shutil.copytree(src / "case14_ieee/raw", dst / "case14_ieee/raw")
        shutil.copytree(src / "splits", dst / "splits")
    return dst


def config(splits, batch_size, work):
    from gridfm_graphkit.io.config_version import resolve_config
    from gridfm_graphkit.io.param_handler import NestedNamespace

    ckpt_yaml = CKPT_ROOT / SIZE / "seed0" / f"case14_ieee_{SIZE}_seed0.yaml"
    cfg = resolve_config(yaml.safe_load(open(ckpt_yaml)), config_path=str(ckpt_yaml), conversion_mode="auto_inline")
    cfg["data"].update(split_from_existing_files=str(splits), split_by_load_scenario_idx=False, workers=0)
    cfg["training"].update(accelerator="cpu", devices=1, strategy="auto", batch_size=batch_size)
    yaml.safe_dump(cfg, open(Path(work) / f"genco_timing_bs{batch_size}.yaml", "w"), sort_keys=False)
    ns = NestedNamespace(**cfg)
    ns.get_embeddings = False
    return ns


def load_model(cfg, data_dir):
    from gridfm_graphkit.cli import _normalize_loaded_state_dict_keys
    from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
    from gridfm_graphkit.io.param_handler import get_task

    dm = LitGridHeteroDataModule(cfg, str(data_dir), normalizer_stats_path=str(CKPT_ROOT / SIZE / "seed0/normalizer_stats.pt"))
    dm.setup("predict")
    model = get_task(cfg, dm.data_normalizers)
    sd = torch.load(model_path("replay", SIZE), map_location="cpu")
    model.load_state_dict(_normalize_loaded_state_dict_keys(sd))
    model.eval()
    return dm, model


def time_set(name, bs, work, warmup=5):
    d = prepare(name, work)
    cfg = config(d / "splits", bs, work)
    dm, model = load_model(cfg, d)
    loader = dm.predict_dataloader()[0]
    t_fwd, t_pred, n_samples, outs = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            b2 = copy.deepcopy(batch)
            t0 = time.perf_counter()
            model.model(batch)
            t1 = time.perf_counter()
            out = model.predict_step(b2, i)
            t2 = time.perf_counter()
            outs.append(pd.DataFrame(out["bus"])[["scenario", "bus", "Vm_pred", "Va_pred"]])
            if i >= warmup:
                t_fwd.append(t1 - t0)
                t_pred.append(t2 - t1)
                n_samples.append(int(batch["bus"].batch.max()) + 1 if bs > 1 else 1)
    n = np.array(n_samples)
    # the timed model must reproduce the frozen replay predictions saved by the pipeline
    mine = pd.concat(outs).set_index(["scenario", "bus"]).sort_index()
    ref = pd.read_parquet(ROOT / "data/predictions/stage3/replay" / name / "case14_ieee_predictions.parquet")
    ref = ref.set_index(["scenario", "bus"]).sort_index().loc[mine.index]
    dvm = float(np.abs(mine.Vm_pred.values - ref.Vm_pred.values).max())
    return {"max |Vm_pred - saved replay Vm_pred|": dvm, "dataset": name, "batch size": bs, "threads": torch.get_num_threads(), "batches timed": len(n),
            "samples timed": int(n.sum()),
            "forward ms/sample": 1e3 * np.sum(t_fwd) / n.sum(),
            "predict_step ms/sample": 1e3 * np.sum(t_pred) / n.sum(),
            "predict_step ms/batch median": 1e3 * np.median(t_pred),
            "throughput samples/s (predict_step)": n.sum() / np.sum(t_pred)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="scratch folder outside the repo")
    args = p.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"CPU: {platform.processor()}, {os.cpu_count()} logical cores, torch {torch.__version__}")
    rows = []
    for threads in [torch.get_num_threads(), 1]:
        torch.set_num_threads(threads)
        for name in TESTS:
            for bs in BATCHES:
                r = time_set(name, bs, args.work)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(EXTRA / "genco_timing.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
