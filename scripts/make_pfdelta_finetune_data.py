"""Build the PFΔ case14 fine-tuning folder (N-0 and N-1 TRAIN / VAL positions only).

PFΔ split per topology (shuffle_files/<topo>/raw_shuffle.json, n = #feasible files):
  test  = last 2,000 positions                       -> never read here
  pool  = positions 0 .. n-2000
  val   = last 10 % of the pool (PFΔ convention 0.9·pool .. pool)
  train = first 90 % of the pool
N-2 is never read.

Writes data/pfdelta_ft/case14_ieee/raw/{bus,gen,branch}_data.parquet with contiguous ids:
  0 .. 2*M-1         mining pool: M seeded TRAIN-position samples per topology (N-0 first)
  2*M .. 2*M+2*V-1   validation: V seeded VAL-position samples per topology
and results/pfdelta_ft_mapping.csv (ft_id -> role, topology, PFΔ position, file, outage info).
Split folders are made later by finetune_pipeline.py.

Usage:
  python scripts/make_pfdelta_finetune_data.py --mine-per-topology 7500 --val-per-topology 500
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_pfdelta_subset import PFDELTA_TEST_SIZE, SHUFFLE, TOPOLOGIES, read
from make_test_subset import outage_info

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/pfdelta_ft"


def positions(topo):
    """Shuffle map plus the TRAIN and VAL position ranges of one topology (test excluded)."""
    shuffle = {int(k): int(v) for k, v in json.load(open(SHUFFLE / topo / "raw_shuffle.json")).items()}
    n = len(shuffle)
    pool = n - PFDELTA_TEST_SIZE
    val_start = int(0.9 * pool)
    return shuffle, np.arange(0, val_start), np.arange(val_start, pool)


def main():
    """Draw mining and validation samples per topology; write data/pfdelta_ft and its mapping."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mine-per-topology", type=int, default=7500)
    parser.add_argument("--val-per-topology", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # 1. seeded positions: 'mine' from TRAIN positions, 'val' from VAL positions
    plan = []  # (role, topo, positions)
    for role, size in [("mine", args.mine_per_topology), ("val", args.val_per_topology)]:
        for topo in TOPOLOGIES:
            shuffle, train_pos, val_pos = positions(topo)
            src = train_pos if role == "mine" else val_pos
            picked = np.sort(rng.choice(src, size=size, replace=False))
            plan.append((role, topo, shuffle, picked))
            print(f"{role} {TOPOLOGIES[topo]}: {len(src)} positions available, picked {size}")

    # 2. read the picked samples and renumber them contiguously (ft_id)
    tables = {name: [] for name in ["bus_data", "gen_data", "branch_data"]}
    mapping, next_id = [], 0
    for role, topo, shuffle, pos in plan:
        scen = np.array([shuffle[int(k)] for k in pos])
        order = np.argsort(scen)
        scen, pos = scen[order], pos[order]
        new_id = {int(s): next_id + i for i, s in enumerate(scen)}
        next_id += len(scen)
        dfs = {name: read(topo, name, scen) for name in tables}
        assert dfs["bus_data"].scenario.nunique() == len(scen)
        info = outage_info(dfs["gen_data"], dfs["branch_data"])
        for name, df in dfs.items():
            sort_cols = ["scenario", "bus"] if name == "bus_data" else ["scenario", "idx"]
            df = df.sort_values(sort_cols).reset_index(drop=True)
            df["scenario"] = df["scenario"].map(new_id)
            tables[name].append(df)
        m = info.reindex(scen).rename_axis("orig_scenario").reset_index()
        m.insert(0, "ft_id", m.orig_scenario.map(new_id))
        m.insert(1, "role", role)
        m.insert(2, "topology", TOPOLOGIES[topo])
        m.insert(3, "pfdelta_position", pos)
        m.insert(4, "pfdelta_file", [f"case14/{topo}/raw/sample_{s + 1}.json" for s in scen])
        mapping.append(m)

    # 3. write tables and mapping
    raw_out = OUT / "case14_ieee/raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_parquet(raw_out / f"{name}.parquet", index=False)
    mapping = pd.concat(mapping, ignore_index=True).sort_values("ft_id")
    assert (mapping.ft_id.values == np.arange(len(mapping))).all()
    # no overlap with the PFΔ test positions used by data/pfdelta_test1000
    test = pd.read_csv(ROOT / "results/pfdelta_test_subset_1000.csv")
    assert not set(zip(mapping.topology, mapping.orig_scenario)) & set(zip(test.topology, test.orig_scenario))
    mapping.to_csv(ROOT / "results/pfdelta_ft_mapping.csv", index=False)
    print(f"\nwrote {len(mapping)} samples to {OUT}")
    print(mapping.groupby(["role", "topology", "outage_type"]).size().to_string())


if __name__ == "__main__":
    main()
