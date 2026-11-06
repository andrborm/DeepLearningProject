"""Build data/stage3: one graphkit folder for replay fine-tuning and the Task 3 detectors. N-2 is never read.

Blocks (contiguous ids, in this order; mapping in results/stage3_mapping.csv):
  dk_train   5,000  datakit, checkpoint TRAIN split          -> replay fine-tuning
  dk_val     1,000  datakit, checkpoint VAL split            -> replay early stopping (500), detector tuning
  dk_probe   2,000  datakit, checkpoint TRAIN split, disjoint from dk_train -> probe training
  pf_train   5,000  PFΔ train positions = the random fine-tuning set of data/pfdelta_ft
  pf_val     1,000  PFΔ val positions (same as data/pfdelta_ft val) -> early stopping, detector tuning
  pf_probe   2,000  PFΔ train positions from the mining pool, not in pf_train (1,000 N-0 + 1,000 N-1)
None of these overlap the datakit test1000 or PFΔ test1000 subsets (asserted).

Columns: the PFΔ/datakit common set (datakit-only DC baseline + load_scenario_idx dropped), datakit dtypes.

Usage:
  python scripts/make_stage3_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from make_test_subset import SPLITS_JSON, outage_info, read_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/stage3"
PF_FT = ROOT / "data/pfdelta_ft/case14_ieee/raw"
NAMES = ["bus_data", "gen_data", "branch_data"]
SORT = {"bus_data": ["scenario", "bus"], "gen_data": ["scenario", "idx"], "branch_data": ["scenario", "idx"]}


def main():
    rng = np.random.default_rng(7)
    splits = json.load(open(SPLITS_JSON))
    dk_test = set(pd.read_csv(ROOT / "results/test_subset_1000.csv").orig_scenario)
    train_ids = rng.permutation(splits["train"])
    blocks = [  # (role, source, original ids)
        ("dk_train", "datakit", np.sort(train_ids[:5000])),
        ("dk_val", "datakit", np.sort(rng.choice(splits["val"], 1000, replace=False))),
        ("dk_probe", "datakit", np.sort(train_ids[5000:7000])),
    ]
    ftmap = pd.read_csv(ROOT / "results/pfdelta_ft_mapping.csv").set_index("ft_id")
    sel = pd.read_csv(ROOT / "results/pfdelta_ft_selection.csv", index_col="ft_id")
    pf_train = sel.index[sel.random].values
    pf_val = ftmap.index[ftmap.role == "val"].values
    rest = sel[~sel.random]
    pf_probe = np.concatenate([rng.choice(g.index.values, 1000, replace=False) for _, g in rest.groupby("topology")])
    blocks += [("pf_train", "pfdelta", np.sort(pf_train)), ("pf_val", "pfdelta", np.sort(pf_val)),
               ("pf_probe", "pfdelta", np.sort(pf_probe))]
    for role, src, ids in blocks:
        if src == "datakit":
            assert not set(ids.tolist()) & dk_test

    dk_all = np.concatenate([b[2] for b in blocks if b[1] == "datakit"])
    assert len(set(dk_all.tolist())) == len(dk_all)
    dk = {n: read_table(n, dk_all.tolist()) for n in NAMES}
    pf_all = np.concatenate([b[2] for b in blocks if b[1] == "pfdelta"])
    pfs = {n: ds.dataset(PF_FT / f"{n}.parquet").to_table(filter=ds.field("scenario").isin(pf_all.tolist())).to_pandas()
           for n in NAMES}

    tables = {n: [] for n in NAMES}
    mapping, next_id = [], 0
    for role, src, ids in blocks:
        new_id = {int(s): next_id + i for i, s in enumerate(ids)}
        next_id += len(ids)
        dfs = {}
        for n in NAMES:
            ref = dk[n]
            df = (dk[n] if src == "datakit" else pfs[n])
            df = df[df.scenario.isin(ids)].copy()
            common = [c for c in pfs[n].columns if c in ref.columns]
            df = df[common].astype({c: ref[c].dtype for c in common})
            dfs[n] = df
        info = outage_info(dfs["gen_data"], dfs["branch_data"])
        for n, df in dfs.items():
            df = df.sort_values(SORT[n]).reset_index(drop=True)
            df["scenario"] = df["scenario"].map(new_id)
            tables[n].append(df)
        m = info.reindex(ids).rename_axis("orig_id").reset_index()
        m.insert(0, "s3_id", m.orig_id.map(new_id))
        m.insert(1, "role", role)
        m.insert(2, "source", src)
        if src == "pfdelta":
            m["topology"] = ftmap.loc[ids, "topology"].values
            m["pfdelta_file"] = ftmap.loc[ids, "pfdelta_file"].values
        else:
            m["topology"] = "datakit N-1"
        mapping.append(m)
        print(f"{role}: {len(ids)}")

    raw = OUT / "case14_ieee/raw"
    raw.mkdir(parents=True, exist_ok=True)
    for n, parts in tables.items():
        df = pd.concat(parts, ignore_index=True)
        assert df.scenario.nunique() == next_id
        df.to_parquet(raw / f"{n}.parquet", index=False)
    mapping = pd.concat(mapping, ignore_index=True)
    assert (mapping.s3_id.values == np.arange(len(mapping))).all()
    pf_test = pd.read_csv(ROOT / "results/pfdelta_test_subset_1000.csv")
    pfm = mapping[mapping.source == "pfdelta"]
    assert not set(pfm.pfdelta_file) & set(pf_test.pfdelta_file)
    assert not mapping.topology.eq("N-2").any()
    mapping.to_csv(ROOT / "results/stage3_mapping.csv", index=False)
    print(mapping.groupby(["role", "topology"]).size().to_string())


if __name__ == "__main__":
    main()
