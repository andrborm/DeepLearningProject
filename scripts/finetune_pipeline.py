"""Stage 2 fine-tuning on PFΔ case14 N-0/N-1 (Task 2.2-2.3). N-2 is never read.

Data: data/pfdelta_ft (scripts/make_pfdelta_finetune_data.py): 15,000 TRAIN-position mining samples
(7,500 N-0 + 7,500 N-1) and 1,000 PFΔ VAL-position samples (500 + 500) for early stopping.

Stages (run in order):
  splits                       chunked mining splits (1000 each) + 5,000-sample timing split
  time   --size S              fine-tune 1 epoch on 5,000 random mining samples, report CPU time
  mine   --size S              base-model error on every mining sample (15 evaluate runs of 1000)
  select --k K                 hard = top K per topology by base error; random = K per topology, seeded
  finetune --set hard|random   fine-tune from the checkpoint, PFΔ val for early stopping
  finetune --set replay        same, on 5,000 datakit train + 5,000 random PFΔ (data/stage3), mixed val
  eval   --model NAME          evaluate on datakit test1000 and PFΔ test1000 (forgetting / shift)
  compare                      one table: original vs fine-tuned

graphkit's `finetune` CLI has no --normalizer_stats flag and would refit baseMVA on the PFΔ train set
(95th percentile of the loads), which changes the input scaling the checkpoint was trained with. This
script calls graphkit's `main_cli` directly so the checkpoint's normalizer_stats.pt is used everywhere.
"""

import argparse
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import torch
import yaml

from phase5_analysis import confusion, labels_from_predictions, long_errors, robust, true_labels_from_raw
from run_inference import per_bus_errors

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "data/pfdelta_ft"
S3 = ROOT / "data/stage3"  # scripts/make_stage3_data.py
RESULTS = ROOT / "results"
CKPT_ROOT = ROOT / "checkpoints/genco-pf-datakit/case14_ieee"
MAPPING = RESULTS / "pfdelta_ft_mapping.csv"
EXP = "stage2_finetune"
EVAL_SETS = {  # name -> (data folder, subset csv)
    "datakit": (ROOT / "data/case14_test1000", RESULTS / "test_subset_1000.csv"),
    "pfdelta": (ROOT / "data/pfdelta_test1000", RESULTS / "pfdelta_test_subset_1000.csv"),
}


def rel(p):
    """Repo-relative path with forward slashes, as graphkit configs expect."""
    return str(Path(p).relative_to(ROOT)).replace("\\", "/")


def write_split(name, train=(), val=(), test=()):
    """Write graphkit split files (id tensors) to data/pfdelta_ft/splits_<name>."""
    d = FT / f"splits_{name}"
    d.mkdir(parents=True, exist_ok=True)
    for k, ids in [("train", train), ("val", val), ("test", test)]:
        torch.save(torch.tensor(np.sort(np.asarray(ids, dtype=np.int64)), dtype=torch.long), d / f"{k}.pt")
    return d


def make_config(size, splits, name, lr=None, epochs=None, patience=None):
    """Checkpoint YAML -> CPU config with our split files (optional lr / epochs / patience), saved in configs/."""
    from gridfm_graphkit.io.config_version import resolve_config

    ckpt_yaml = CKPT_ROOT / size / "seed0" / f"case14_ieee_{size}_seed0.yaml"
    cfg = resolve_config(yaml.safe_load(open(ckpt_yaml)), config_path=str(ckpt_yaml), conversion_mode="auto_inline")
    cfg["data"].update(split_from_existing_files=rel(splits), split_by_load_scenario_idx=False, workers=0)
    cfg["training"].update(accelerator="cpu", devices=1, strategy="auto")
    if lr is not None:
        cfg["optimizer"]["learning_rate"] = lr
        cfg["optimizer"]["scheduler_params"].update(patience=2, factor=0.5)
    if epochs is not None:
        cfg["training"]["epochs"] = epochs
    if patience is not None:
        cfg["callbacks"].update(patience=patience, tol=0)
    out = ROOT / "configs" / f"{name}.yaml"
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    return out


def run(command, config, model_path, size, data_path, run_name, save_output=False):
    """Call graphkit's CLI in-process with the checkpoint normalizer; return wall time (s)."""
    from gridfm_graphkit.cli import main_cli

    args = Namespace(command=command, config=rel(config), model_path=str(model_path),
                     normalizer_stats=str(CKPT_ROOT / size / "seed0" / "normalizer_stats.pt"),
                     data_path=rel(data_path), log_dir="mlruns", exp_name=EXP, run_name=run_name,
                     num_workers=0, save_output=save_output, report_performance=True,
                     config_conversion="no", plugins=[])
    patch_sanity_check_save()
    t0 = time.perf_counter()
    main_cli(args)
    return time.perf_counter() - t0


def patch_sanity_check_save():
    """graphkit 0.9.0.3 SaveBestModelStateDict only skips the sanity-check pass when the metric is missing.
    If it is present, the 2-batch sanity loss becomes 'best' and can block every later save. Skip it always."""
    from gridfm_graphkit.training.callbacks import SaveBestModelStateDict

    orig = SaveBestModelStateDict.on_validation_end
    if getattr(orig, "_patched", False):
        return

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        return orig(self, trainer, pl_module)

    on_validation_end._patched = True
    SaveBestModelStateDict.on_validation_end = on_validation_end


def find_run_dir(run_name, needs):
    """Newest MLflow run folder with this run name that contains `needs`."""
    runs = [p.parents[1] for p in (ROOT / "mlruns").glob("*/*/tags/mlflow.runName")
            if p.read_text().strip() == run_name and (p.parents[1] / needs).exists()]
    return max(runs, key=lambda r: r.stat().st_mtime)


def base_model(size):
    """Weights of the original checkpoint."""
    return CKPT_ROOT / size / "seed0" / "best_model_state_dict.pt"


def sample_scores(bus):
    """Per-sample overall score = mean over Vm/Va/Pg/Qg of sample RMSE / datakit base RMSE (1.0 = datakit level)."""
    long = per_bus_errors(bus.copy())
    ref = pd.read_csv(RESULTS / "inference_by_variable.csv", index_col="variable").RMSE
    rmse_var = long.groupby(["scenario", "variable"]).err.apply(lambda e: np.sqrt((e ** 2).mean())).unstack()
    return (rmse_var / ref).mean(axis=1).rename("score")


# --------------------------------------------------------------------------- stages
def stage_splits(args):
    """Chunked mining splits (1,000 each) and the 5,000-sample timing split."""
    m = pd.read_csv(MAPPING)
    mine = m[m.role == "mine"].ft_id.values
    val = m[m.role == "val"].ft_id.values
    for i in range(0, len(mine), 1000):
        write_split(f"mine{i // 1000:02d}", test=mine[i:i + 1000])
    rng = np.random.default_rng(0)
    write_split("time", train=rng.choice(mine, 5000, replace=False), val=val, test=val[:100])
    print(f"{len(range(0, len(mine), 1000))} mining chunks, timing split 5000 train / {len(val)} val")


def stage_time(args):
    """Time one fine-tuning epoch on 5,000 samples."""
    cfg = make_config(args.size, FT / "splits_time", f"ft_time_{args.size}", lr=args.lr, epochs=1, patience=5)
    wall = run("finetune", cfg, base_model(args.size), args.size, FT, f"time_{args.size}")
    print(f"\n[timing] {args.size}: 1 epoch on 5000 samples (+ 1000 val, + dataset setup): {wall:.1f} s wall")


def stage_mine(args):
    """Score every mining sample with the base model (error score and residual)."""
    m = pd.read_csv(MAPPING).set_index("ft_id")
    chunks = sorted(FT.glob("splits_mine*"))
    cfg = None
    scores = []
    for ch in chunks:
        cfg = make_config(args.size, ch, f"ft_mine_{args.size}")
        name = f"mine_{args.size}_{ch.name[-2:]}"
        wall = run("evaluate", cfg, base_model(args.size), args.size, FT, name, save_output=True)
        bus = pd.read_parquet(find_run_dir(name, "artifacts/test/case14_ieee_predictions.parquet")
                              / "artifacts/test/case14_ieee_predictions.parquet")
        assert bus.scenario.nunique() <= 1000
        assert m.loc[bus.scenario.unique(), "role"].eq("mine").all()
        s = sample_scores(bus)
        pbe = bus.groupby("scenario").PBE.mean().rename("pbe_mean")
        scores.append(pd.concat([s, pbe], axis=1))
        print(f"[mine] {ch.name}: {bus.scenario.nunique()} samples, {wall:.0f} s")
    out = pd.concat(scores).rename_axis("ft_id").join(m[["topology", "outage_type", "pfdelta_position"]])
    out.to_csv(RESULTS / "pfdelta_ft_mining_scores.csv", float_format="%.6g")
    print(out.groupby("topology").score.describe().to_string())


def stage_select(args):
    """Hard set = top k per topology by base error; random control = k per topology."""
    s = pd.read_csv(RESULTS / "pfdelta_ft_mining_scores.csv", index_col="ft_id")
    val = pd.read_csv(MAPPING).query("role == 'val'").ft_id.values
    rng = np.random.default_rng(1)
    hard, rand = [], []
    for topo, g in s.groupby("topology"):
        hard.append(g.score.nlargest(args.k).index.values)
        rand.append(rng.choice(g.index.values, args.k, replace=False))
        print(f"{topo}: hard threshold score >= {g.score.nlargest(args.k).min():.1f} "
              f"(pool median {g.score.median():.1f})")
    hard, rand = np.concatenate(hard), np.concatenate(rand)
    write_split("hard", train=hard, val=val, test=val[:100])
    write_split("random", train=rand, val=val, test=val[:100])
    sel = s.assign(hard=s.index.isin(hard), random=s.index.isin(rand))
    sel.to_csv(RESULTS / "pfdelta_ft_selection.csv", float_format="%.6g")
    print(f"hard {len(hard)}, random {len(rand)}, overlap {len(set(hard) & set(rand))}")
    print(sel.groupby(["topology", "hard"]).score.median().to_string())
    print(sel[sel.hard].groupby(["topology", "outage_type"]).size().to_string())


def replay_splits():
    """Replay set (data/stage3): 5,000 datakit TRAIN + the 5,000 random PFΔ train samples.
    Validation: 250 N-0 + 250 N-1 PFΔ val + 500 datakit val (1,000 total), so early stopping sees both."""
    m = pd.read_csv(RESULTS / "stage3_mapping.csv")
    rng = np.random.default_rng(2)
    train = m[m.role.isin(["dk_train", "pf_train"])].s3_id.values
    pf_val = np.concatenate([rng.choice(g.s3_id.values, 250, replace=False)
                             for _, g in m[m.role == "pf_val"].groupby("topology")])
    dk_val = rng.choice(m[m.role == "dk_val"].s3_id.values, 500, replace=False)
    val = np.concatenate([pf_val, dk_val])
    d = S3 / "splits_replay"
    d.mkdir(parents=True, exist_ok=True)
    for k, ids in [("train", train), ("val", val), ("test", val[:100])]:
        torch.save(torch.tensor(np.sort(ids), dtype=torch.long), d / f"{k}.pt")
    return d


def stage_finetune(args):
    """Fine-tune the checkpoint on the hard, random or replay set."""
    name = f"ft_{args.set}_{args.size}"
    data, splits = (S3, replay_splits()) if args.set == "replay" else (FT, FT / f"splits_{args.set}")
    cfg = make_config(args.size, splits, name, lr=args.lr, epochs=args.epochs, patience=args.patience)
    wall = run("finetune", cfg, base_model(args.size), args.size, data, name)
    d = find_run_dir(name, "artifacts/model/best_model_state_dict.pt")
    print(f"\n[finetune] {name}: {wall / 60:.1f} min wall, best model {d / 'artifacts/model/best_model_state_dict.pt'}")


def model_path(name, size):
    """Weights file of a model name (original / hard / random / replay)."""
    if name == "original":
        return base_model(size)
    # replay: graphkit saved its "best" model in the 2-batch sanity check (datakit-only batches, tiny loss),
    # i.e. the unchanged original weights. Use the epoch-40 weights extracted from checkpoints/last.ckpt.
    f = MODEL_FILE.get(name, "best_model_state_dict.pt")
    return find_run_dir(f"ft_{name}_{size}", f"artifacts/model/{f}") / "artifacts/model" / f


MODEL_FILE = {"replay": "last_epoch_state_dict.pt"}


def stage_eval(args):
    """Evaluate one model on the datakit and PFΔ test subsets (saves predictions)."""
    for ds, (data, subset_csv) in EVAL_SETS.items():
        cfg = make_config(args.size, data / "splits", f"eval_{ds}_{args.size}")
        name = f"eval_{args.model}_{args.size}_{ds}"
        run("evaluate", cfg, model_path(args.model, args.size), args.size, data, name, save_output=True)


def metrics_for(model, size, ds):
    """Errors, predict-then-check confusion and scores of one model on one test subset."""
    data, subset_csv = EVAL_SETS[ds]
    art = find_run_dir(f"eval_{model}_{size}_{ds}", "artifacts/test/case14_ieee_predictions.parquet") / "artifacts/test"
    bus = pd.read_parquet(art / "case14_ieee_predictions.parquet")
    branch = pd.read_parquet(art / "case14_ieee_branch_predictions.parquet")
    subset = pd.read_csv(subset_csv).set_index("subset_id")
    assert bus.scenario.nunique() == 1000
    assert "topology" not in subset or not subset.topology.eq("N-2").any()
    long = long_errors(bus, subset.outage_type)
    err = long.groupby(["variable", "bus_type"]).apply(robust)[["RMSE", "median", "p95", "p99"]]
    truth = true_labels_from_raw(data / "case14_ieee/raw")
    pred = labels_from_predictions(bus, branch, "pred", slack_setpoint=True)
    ptc = {("all", k): confusion(truth[k].values, pred.loc[truth.index, k].values) for k in ["y_voltage", "y_overload"]}
    if "topology" in subset:
        for t in ["N-0", "N-1"]:
            idx = truth.index[subset.topology.reindex(truth.index).eq(t).values]
            for k in ["y_voltage", "y_overload"]:
                ptc[(t, k)] = confusion(truth.loc[idx, k].values, pred.loc[idx, k].values)
    score = sample_scores(bus)
    topo = subset.topology if "topology" in subset else None
    return err, ptc, score, topo


def stage_compare(args):
    """One comparison table over models: errors, predict-then-check, scores."""
    rows_err, rows_ptc, rows_score = [], [], []
    for model in args.models:
        for ds in EVAL_SETS:
            try:
                err, ptc, score, topo = metrics_for(model, args.size, ds)
            except ValueError:
                print(f"skip {model}/{ds}: no evaluate run")
                continue
            rows_err.append(err.assign(model=model, dataset=ds).reset_index())
            for (scope, k), c in ptc.items():
                rows_ptc.append({"model": model, "dataset": ds, "scope": scope, "label": k, **c})
            rows_score.append({"model": model, "dataset": ds, "score median": score.median(),
                               "score p95": score.quantile(.95), "score p99": score.quantile(.99),
                               **({f"score median {t}": score[topo == t].median() for t in ["N-0", "N-1"]}
                                  if topo is not None else {})})
    err = pd.concat(rows_err)
    ptc, score = pd.DataFrame(rows_ptc), pd.DataFrame(rows_score)
    err.to_csv(RESULTS / f"finetune_errors_{args.size}.csv", index=False, float_format="%.6g")
    ptc.to_csv(RESULTS / f"finetune_predict_then_check_{args.size}.csv", index=False, float_format="%.4f")
    score.to_csv(RESULTS / f"finetune_scores_{args.size}.csv", index=False, float_format="%.4g")
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    pd.set_option("display.width", 250)
    wide = err.pivot_table(index=["dataset", "variable", "bus_type"], columns="model",
                           values=["RMSE", "median", "p95", "p99"])
    wide = wide.reindex(columns=args.models, level=1)
    print("=== errors (unknowns only) ===\n" + wide.to_string())
    print("\n=== predict-then-check (slack Vm = known setpoint) ===\n"
          + ptc.set_index(["dataset", "scope", "label", "model"]).sort_index().to_string())
    print("\n=== per-sample overall score (1.0 = datakit base RMSE level) ===\n" + score.to_string(index=False))


def main():
    """CLI: run one stage."""
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["splits", "time", "mine", "select", "finetune", "eval", "compare"])
    p.add_argument("--size", default="base", choices=["base", "small", "tiny"])
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--k", type=int, default=2500)
    p.add_argument("--set", default="hard", choices=["hard", "random", "replay"])
    p.add_argument("--model", default="original")
    p.add_argument("--models", nargs="+", default=["original", "hard", "random"])
    args = p.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
