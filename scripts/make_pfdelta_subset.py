"""Sample a seeded subset of PFΔ case14 TEST samples (N-0 and N-1 only) into a graphkit-ready folder.

Prerequisite: converted PFΔ feasible data (scripts/vendor/batch_convert_pfdelta.py) in
data/pfdelta_case14/{n,n-1}/raw, and the PFΔ shuffle files in data/pfdelta_orig/shuffle_files.

PFΔ's fixed test split = the last 2,000 positions of each topology's shuffled list
(core/datasets/pfdelta_dataset.py: shuffled file k = sample_{shuffle_map[k] + 1}.json).
The converter maps sample_N.json -> scenario N-1, so the converted scenario id is shuffle_map[k].

N-2 is never read here: it stays untouched until the final evaluation.

Writes:
  <out>/case14_ieee/raw/{bus,gen,branch}_data.parquet   renumbered 0..n-1 (N-0 first, then N-1)
  <out>/splits/{train,val,test}.pt                      all samples in "test"
  results/pfdelta_test_subset_<n>.csv                   subset_id -> topology, PFΔ file, outage info

Usage:
  python scripts/make_pfdelta_subset.py --per-topology 500 --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from make_test_subset import outage_info

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "data/pfdelta_case14"
SHUFFLE = ROOT / "data/pfdelta_orig/shuffle_files"
TOPOLOGIES = {"n": "N-0", "n-1": "N-1"}  # N-2 deliberately excluded
PFDELTA_TEST_SIZE = 2000


def test_scenarios(topo):
    """Converted scenario ids of PFΔ's feasible test split for one topology."""
    shuffle = {int(k): int(v) for k, v in json.load(open(SHUFFLE / topo / "raw_shuffle.json")).items()}
    n = len(shuffle)
    return np.array([shuffle[k] for k in range(n - PFDELTA_TEST_SIZE, n)]), n


def read(topo, name, scenarios):
    """Read one converted PFΔ table for the given scenario ids."""
    table = ds.dataset(CONVERTED / topo / "raw" / f"{name}.parquet", format="parquet", partitioning="hive")
    df = table.to_table(filter=ds.field("scenario").isin(scenarios.tolist())).to_pandas()
    return df.drop(columns=["scenario_partition"], errors="ignore")


def main():
    """Sample each topology's test split, renumber (N-0 first), write tables, splits and mapping."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-topology", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    n_total = args.per_topology * len(TOPOLOGIES)
    assert n_total <= 1000, "course requirement: at most 1000 samples for inference"
    out = ROOT / f"data/pfdelta_test{n_total}"

    rng = np.random.default_rng(args.seed)
    tables = {name: [] for name in ["bus_data", "gen_data", "branch_data"]}
    mapping = []
    next_id = 0
    # per topology: seeded sample of PFΔ's test split, contiguous new ids
    for topo, label in TOPOLOGIES.items():
        ids, n_files = test_scenarios(topo)
        assert len(set(ids)) == PFDELTA_TEST_SIZE and ids.max() < n_files
        picked = np.sort(rng.choice(ids, size=args.per_topology, replace=False))
        new_id = {int(s): next_id + i for i, s in enumerate(picked)}
        next_id += len(picked)
        dfs = {name: read(topo, name, picked) for name in tables}
        info = outage_info(dfs["gen_data"], dfs["branch_data"])
        for name, df in dfs.items():
            sort_cols = ["scenario", "bus"] if name == "bus_data" else ["scenario", "idx"]
            df = df.sort_values(sort_cols).reset_index(drop=True)
            df["scenario"] = df["scenario"].map(new_id)
            tables[name].append(df)
        m = info.rename_axis("orig_scenario").reset_index()
        m.insert(0, "subset_id", m.orig_scenario.map(new_id))
        m.insert(1, "topology", label)
        m.insert(2, "pfdelta_file", [f"case14/{topo}/raw/sample_{s + 1}.json" for s in m.orig_scenario])
        mapping.append(m)
        print(f"{label}: {n_files} feasible files, test split {len(ids)}, sampled {len(picked)}")

    # write the combined tables
    raw_out = out / "case14_ieee/raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_parquet(raw_out / f"{name}.parquet", index=False)

    # graphkit split files: every sample goes to 'test'
    splits = out / "splits"
    splits.mkdir(exist_ok=True)
    torch.save(torch.arange(n_total), splits / "test.pt")
    torch.save(torch.tensor([], dtype=torch.long), splits / "train.pt")
    torch.save(torch.tensor([], dtype=torch.long), splits / "val.pt")

    mapping = pd.concat(mapping, ignore_index=True).sort_values("subset_id")
    mapping.to_csv(ROOT / f"results/pfdelta_test_subset_{n_total}.csv", index=False)
    print(f"\nwrote {n_total} samples to {out}")
    print(mapping.groupby(["topology", "outage_type"]).size().to_string())


if __name__ == "__main__":
    main()
