"""Shared helpers for the additive report analyses in scripts/extra/ (frozen pipeline is only read, never changed).

Paths, the three test sets, raw-input features and the palette. Existing functions are imported read-only.
"""

import sys

sys.dont_write_bytecode = True  # never write __pycache__ into scripts/ when importing the frozen modules

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS = ROOT / "results"
EXTRA = RESULTS / "extra"
FIGS = EXTRA / "figures"
S3_RAW = ROOT / "data/stage3/case14_ieee/raw"
LABELS = ["y_voltage", "y_overload"]
SEED = 0

# test name -> (raw folder, subset csv); same three sets as scripts/final_n2_test.py
TESTS = {
    "test_datakit": (ROOT / "data/case14_test1000/case14_ieee/raw", RESULTS / "test_subset_1000.csv"),
    "test_pfdelta": (ROOT / "data/pfdelta_test1000/case14_ieee/raw", RESULTS / "pfdelta_test_subset_1000.csv"),
    "test_n2": (ROOT / "data/pfdelta_n2_test1000/case14_ieee/raw", RESULTS / "pfdelta_n2_test_subset_1000.csv"),
}
TEST_LABEL = {"test_datakit": "datakit (N-1)", "test_pfdelta": "PFΔ N-0/N-1", "test_n2": "PFΔ N-2"}


def read_raw(raw_dir, ids=None):
    """Raw bus / gen / branch tables of a folder, optionally only some scenario ids."""
    out = {}
    for n in ["bus_data", "gen_data", "branch_data"]:
        df = pd.read_parquet(Path(raw_dir) / f"{n}.parquet")
        if ids is not None:
            df = df[df.scenario.isin(ids)]
        out[n] = df
    return out


def raw_features(raw_dir, ids=None):
    """Raw model inputs per scenario, no GENCO output.

    bus Pd, Qd (14 + 14); non-slack generator Pg setpoints (4); generator Vm setpoints = bus Vm of in-service
    generators (5, 0 if the generator is out); total Pd, total Qd; outage indicators for 20 branches and 5 generators.
    The slack generator's Pg is a power-flow *result* and is left out.
    """
    t = read_raw(raw_dir, ids)
    bus, gen, br = t["bus_data"], t["gen_data"].copy(), t["branch_data"].copy()
    bus = bus.astype({"bus": int})
    gen = gen.astype({"idx": int, "bus": int})
    br = br.astype({"idx": int})
    # one row per scenario, one column per bus / generator / branch
    pd_ = bus.pivot(index="scenario", columns="bus", values="Pd").add_prefix("Pd_b")
    qd_ = bus.pivot(index="scenario", columns="bus", values="Qd").add_prefix("Qd_b")
    vm_bus = bus.set_index(["scenario", "bus"]).Vm
    g = gen.copy()
    g["vm_set"] = vm_bus.reindex(pd.MultiIndex.from_arrays([g.scenario, g.bus])).values * g.in_service
    pg = g[g.is_slack_gen == 0].pivot(index="scenario", columns="idx", values="p_mw").add_prefix("Pg_g")
    vm = g.pivot(index="scenario", columns="idx", values="vm_set").add_prefix("Vmset_g")
    gout = (1 - g.pivot(index="scenario", columns="idx", values="in_service")).add_prefix("gen_out_")
    bout = (1 - br.pivot(index="scenario", columns="idx", values="br_status")).add_prefix("br_out_")
    tot = pd.DataFrame({"total_Pd": bus.groupby("scenario").Pd.sum(), "total_Qd": bus.groupby("scenario").Qd.sum()})
    X = pd.concat([pd_, qd_, pg, vm, tot, gout, bout], axis=1).sort_index()
    assert not X.isna().any().any()
    return X.astype(np.float64)


# palette and rcParams of scripts/phase5_analysis.py (importing it applies the same rcParams)
def palette():
    import phase5_analysis as p5
    return dict(BLUE=p5.BLUE, ORANGE=p5.ORANGE, AQUA=p5.AQUA, GRAY=p5.GRAY, INK=p5.INK, INK_2=p5.INK_2,
                GRID=p5.GRID, SURFACE=p5.SURFACE)
