"""Sample a seeded subset of case14 test scenarios into a small graphkit-ready folder.

Reads the checkpoint's own train/val/test split (scenario_splits.json), samples N test
scenarios, renumbers them 0..N-1 and writes:

  <out>/case14_ieee/raw/{bus,gen,branch}_data.parquet   flat parquet, graphkit input
  <out>/splits/{train,val,test}.pt                      all N scenarios in "test"
  results/test_subset_<N>.csv                           subset_id -> original scenario + outage info

Use the splits with graphkit via `data.split_from_existing_files: <out>/splits`.

Usage:
  python scripts/make_test_subset.py --n 1000 --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/case14_ieee/raw"
SPLITS_JSON = next(
    (ROOT / "checkpoints/genco-pf-datakit/mlflow/eval/case14_ieee/base/seed0").glob(
        "*/artifacts/stats/case14_ieee_scenario_splits.json",
    ),
)


def read_table(name, scenarios):
    """Read one datakit table for the given scenario ids (Hive-partitioned parquet)."""
    table = ds.dataset(RAW / f"{name}.parquet", format="parquet", partitioning="hive")
    df = table.to_table(filter=ds.field("scenario").isin(scenarios)).to_pandas()
    return df.drop(columns=["scenario_partition"], errors="ignore")


def outage_info(gen, branch):
    """Describe the contingency of each scenario from br_status / in_service."""
    br_out = branch[branch.br_status == 0]
    gen_out = gen[gen.in_service == 0]
    n_br = br_out.groupby("scenario").size()
    n_gen = gen_out.groupby("scenario").size()

    info = pd.DataFrame(index=sorted(gen.scenario.unique()))
    info["n_branch_out"] = n_br.reindex(info.index, fill_value=0)
    info["n_gen_out"] = n_gen.reindex(info.index, fill_value=0)
    k = info.n_branch_out + info.n_gen_out
    info["contingency"] = np.where(k == 0, "N-0", "N-" + k.astype(str))
    info["outage_type"] = np.select(
        [k == 0, info.n_gen_out == 0, info.n_branch_out == 0],
        ["none", "branch", "gen"],
        default="mixed",
    )
    info["out_branches"] = (
        br_out.groupby("scenario")
        .apply(lambda d: ";".join(f"{i:.0f}({f}-{t})" for i, f, t in zip(d.idx, d.from_bus, d.to_bus)))
        .reindex(info.index, fill_value="")
    )
    info["out_gens"] = (
        gen_out.groupby("scenario")
        .apply(lambda d: ";".join(f"{i:.0f}(bus{b})" for i, b in zip(d.idx, d.bus)))
        .reindex(info.index, fill_value="")
    )
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    assert args.n <= 1000, "course requirement: at most 1000 samples for inference"
    out = args.out or ROOT / f"data/case14_test{args.n}"

    test_ids = json.load(open(SPLITS_JSON))["test"]
    rng = np.random.default_rng(args.seed)
    picked = np.sort(rng.choice(test_ids, size=args.n, replace=False))
    new_id = {int(s): i for i, s in enumerate(picked)}

    tables = {name: read_table(name, picked.tolist()) for name in ["bus_data", "gen_data", "branch_data"]}
    info = outage_info(tables["gen_data"], tables["branch_data"])

    raw_out = out / "case14_ieee/raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        sort_cols = ["scenario", "bus"] if name == "bus_data" else ["scenario", "idx"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df["scenario"] = df["scenario"].map(new_id)
        df.to_parquet(raw_out / f"{name}.parquet", index=False)

    splits = out / "splits"
    splits.mkdir(exist_ok=True)
    torch.save(torch.arange(args.n), splits / "test.pt")
    torch.save(torch.tensor([], dtype=torch.long), splits / "train.pt")
    torch.save(torch.tensor([], dtype=torch.long), splits / "val.pt")

    mapping = info.rename_axis("orig_scenario").reset_index()
    mapping.insert(0, "subset_id", mapping.orig_scenario.map(new_id))
    load_idx = tables["bus_data"].groupby("scenario").load_scenario_idx.first()
    mapping.insert(2, "load_scenario_idx", mapping.orig_scenario.map(load_idx))
    mapping.to_csv(ROOT / f"results/test_subset_{args.n}.csv", index=False)

    print(f"Wrote {args.n} scenarios (seed {args.seed}) from {len(test_ids)} test ids to {out}")
    print(mapping.contingency.value_counts().to_string())
    print(mapping.outage_type.value_counts().to_string())


if __name__ == "__main__":
    main()
