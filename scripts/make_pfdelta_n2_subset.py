"""Final-test subset: 1,000 seeded samples from PFΔ case14 N-2 TEST split (feasible only).

Prerequisite: converted N-2 data (scripts/vendor/batch_convert_pfdelta.py) in data/pfdelta_case14/n-2/raw.
Same split rule as make_pfdelta_subset.py: PFΔ test = last 2,000 positions of raw_shuffle.json.

Writes:
  data/pfdelta_n2_test1000/case14_ieee/raw/{bus,gen,branch}_data.parquet   renumbered 0..999
  data/pfdelta_n2_test1000/splits/{train,val,test}.pt                    all samples in "test"
  results/pfdelta_n2_test_subset_1000.csv

Usage:
  python scripts/make_pfdelta_n2_subset.py --n 1000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from make_pfdelta_subset import PFDELTA_TEST_SIZE, read, test_scenarios
from make_test_subset import outage_info

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/pfdelta_n2_test1000"


def main():
    """Sample the N-2 test split, renumber 0..n-1, write tables, splits and mapping."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert args.n <= 1000, "course requirement: at most 1000 samples for inference"

    # PFΔ N-2 test split (last 2,000 shuffle positions), seeded sample
    ids, n_files = test_scenarios("n-2")
    assert len(set(ids)) == PFDELTA_TEST_SIZE and ids.max() < n_files
    picked = np.sort(np.random.default_rng(args.seed).choice(ids, size=args.n, replace=False))
    new_id = {int(s): i for i, s in enumerate(picked)}
    names = ["bus_data", "gen_data", "branch_data"]
    dfs = {name: read("n-2", name, picked) for name in names}
    info = outage_info(dfs["gen_data"], dfs["branch_data"])
    assert (info.contingency == "N-2").all(), info.contingency.value_counts()

    # write the tables with the new ids
    raw_out = OUT / "case14_ieee/raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        sort_cols = ["scenario", "bus"] if name == "bus_data" else ["scenario", "idx"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df["scenario"] = df["scenario"].map(new_id)
        df.to_parquet(raw_out / f"{name}.parquet", index=False)

    # graphkit split files: every sample goes to 'test'
    splits = OUT / "splits"
    splits.mkdir(exist_ok=True)
    torch.save(torch.arange(args.n), splits / "test.pt")
    torch.save(torch.tensor([], dtype=torch.long), splits / "train.pt")
    torch.save(torch.tensor([], dtype=torch.long), splits / "val.pt")

    # mapping: subset_id -> PFΔ file and outage description
    m = info.rename_axis("orig_scenario").reset_index()
    m.insert(0, "subset_id", m.orig_scenario.map(new_id))
    m.insert(1, "topology", "N-2")
    m.insert(2, "pfdelta_file", [f"case14/n-2/raw/sample_{s + 1}.json" for s in m.orig_scenario])
    m = m.sort_values("subset_id")
    m.to_csv(ROOT / f"results/pfdelta_n2_test_subset_{args.n}.csv", index=False)
    print(f"N-2: {n_files} feasible files, test split {len(ids)}, sampled {args.n} -> {OUT}")
    print(m.outage_type.value_counts().to_string())


if __name__ == "__main__":
    main()
