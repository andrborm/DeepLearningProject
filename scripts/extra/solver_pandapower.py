"""Part 2: classical AC power flow (pandapower Newton-Raphson) on the three 1,000-sample test subsets.

Runs in the SEPARATE venv D:/school/DeepLearning/venv_extra (pandapower 3.1.2 + numba; see
scripts/extra/requirements_extra.txt). Self-contained: it does not import the frozen scripts (they need torch/graphkit).

Per sample a PYPOWER case is built from the sample's raw data (case14 base MVA 100):
  bus     type from the PQ/PV/REF flags, Pd/Qd, Gs/Bs (p.u. in the data -> MW/MVAr at V = 1), Vm/Va limits
  gen     p_mw setpoint, Vm setpoint = the bus Vm of the generator's bus (PV/REF), in_service -> status
  branch  r, x, b, rate_a, tap (0 = plain line when tap = 1 and shift = 0), shift, br_status
and converted with pandapower.converter.from_ppc. Only `pp.runpp` is timed (construction is not).
Settings: algorithm="nr", init="flat", tolerance_mva=1e-8 (default), max_iteration=50, enforce_q_lims=False
(PFΔ's solved samples exceed generator Q limits, so the ground truth does not enforce them).

Stages:
  sanity   first 20 samples of each test set: max |Vm - Vm_true|, max |Va - Va_true| (deg)  -> stop on mismatch
  time     all 3 x 1,000 samples, numba on (after warm-up) and numba off; per-sample wall time, convergence,
           labels recomputed from the NR solution vs the dataset labels
           -> results/extra/solver_nr_per_sample.csv, results/extra/solver_nr_summary.csv

Usage (from the repo root):
  ../venv_extra/Scripts/python.exe -B scripts/extra/solver_pandapower.py sanity
  ../venv_extra/Scripts/python.exe -B scripts/extra/solver_pandapower.py time
"""

import argparse
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.converter import from_ppc
import logging

logging.getLogger("pandapower").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
EXTRA = ROOT / "results/extra"
TESTS = {
    "test_datakit": ROOT / "data/case14_test1000/case14_ieee/raw",
    "test_pfdelta": ROOT / "data/pfdelta_test1000/case14_ieee/raw",
    "test_n2": ROOT / "data/pfdelta_n2_test1000/case14_ieee/raw",
}
BASE_MVA = 100.0
TOL = 1e-6


def load(raw):
    t = {n: pd.read_parquet(raw / f"{n}.parquet") for n in ["bus_data", "gen_data", "branch_data"]}
    return {n: dict(tuple(df.sort_values([c for c in ["scenario", "bus", "idx"] if c in df]).groupby("scenario")))
            for n, df in t.items()}


def ppc_for(bus, gen, br):
    btype = np.select([bus.REF == 1, bus.PV == 1], [3, 2], 1)
    b = np.zeros((len(bus), 13))
    b[:, 0] = bus.bus.values + 1
    b[:, 1] = btype
    b[:, 2], b[:, 3] = bus.Pd.values, bus.Qd.values
    b[:, 4], b[:, 5] = bus.GS.values * BASE_MVA, bus.BS.values * BASE_MVA
    b[:, 6] = 1
    b[:, 7] = bus.Vm.values * (btype > 1) + 1.0 * (btype == 1)  # PQ start values are ignored (flat start)
    b[:, 8] = 0.0
    b[:, 9] = 1.0  # vn_kv = 1 in pglib case14 (all buses equal)
    b[:, 10] = 1
    b[:, 11], b[:, 12] = bus.max_vm_pu.values, bus.min_vm_pu.values
    vm_bus = dict(zip(bus.bus.astype(int), bus.Vm))
    g = np.zeros((len(gen), 10))
    g[:, 0] = gen.bus.values + 1
    g[:, 1] = gen.p_mw.values
    g[:, 3], g[:, 4] = gen.max_q_mvar.values, gen.min_q_mvar.values
    g[:, 5] = [vm_bus[int(x)] for x in gen.bus]
    g[:, 6] = BASE_MVA
    g[:, 7] = gen.in_service.values
    g[:, 8], g[:, 9] = gen.max_p_mw.values, gen.min_p_mw.values
    r = np.zeros((len(br), 13))
    r[:, 0], r[:, 1] = br.from_bus.values + 1, br.to_bus.values + 1
    r[:, 2], r[:, 3], r[:, 4] = br.r.values, br.x.values, br.b.values
    r[:, 5:8] = br.rate_a.values[:, None]
    plain = (np.abs(br.tap.values - 1) < 1e-12) & (np.abs(br["shift"].values) < 1e-12)
    r[:, 8] = np.where(plain, 0.0, br.tap.values)
    r[:, 9] = br["shift"].values
    r[:, 10] = br.br_status.values
    r[:, 11], r[:, 12] = -360, 360
    return {"version": "2", "baseMVA": BASE_MVA, "bus": b, "gen": g, "branch": r}


def build(t, s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = from_ppc(ppc_for(t["bus_data"][s], t["gen_data"][s], t["branch_data"][s]), f_hz=60, validate_conversion=False)
    # from_ppc (pandapower 3.1.2) drops the branch status for transformers: an outaged transformer stays in service.
    # Set in_service for every branch element explicitly from br_status (ppc branch order = data branch order).
    lk = net._from_ppc_lookups["branch"]
    status = t["branch_data"][s].br_status.values.astype(bool)
    for (tab, i), on in zip(zip(lk["element_type"], lk["element"].astype(int)), status):
        net[tab].at[i, "in_service"] = bool(on)
    return net


def solve(net, numba):
    t0 = time.perf_counter()
    try:
        pp.runpp(net, algorithm="nr", init="flat", max_iteration=50, enforce_q_lims=False, numba=numba,
                 calculate_voltage_angles=True)
        ok = bool(net.converged)
    except pp.LoadflowNotConverged:
        ok = False
    return time.perf_counter() - t0, ok


def compare(net, bus):
    vm = net.res_bus.vm_pu.values
    va = net.res_bus.va_degree.values
    return np.abs(vm - bus.Vm.values).max(), np.abs(va - bus.Va.values).max()


def labels_from_net(net, bus, br):
    vm = net.res_bus.vm_pu.values
    yv = bool(((vm < bus.min_vm_pu.values - TOL) | (vm > bus.max_vm_pu.values + TOL)).any())
    # max end apparent power per in-service branch (ppc order) vs rate_a, same rule as true_labels_from_raw
    rate = br.rate_a.values[br.br_status.values == 1]
    loading = np.array(_ordered_flows(net))[br.br_status.values == 1] / rate
    return yv, bool((loading > 1 + TOL).any())


def _ordered_flows(net):
    """Max end apparent power per ppc branch, in ppc branch order (from_ppc stores the ppc index in
    net._from_ppc_lookups['branch'] as (element type, index))."""
    lk = net._from_ppc_lookups["branch"]
    out = []
    for tab, i in zip(lk["element_type"], lk["element"].astype(int)):
        if tab == "line":
            r = net.res_line.loc[i]
            out.append(max(np.hypot(r.p_from_mw, r.q_from_mvar), np.hypot(r.p_to_mw, r.q_to_mvar)))
        else:
            r = net.res_trafo.loc[i]
            out.append(max(np.hypot(r.p_hv_mw, r.q_hv_mvar), np.hypot(r.p_lv_mw, r.q_lv_mvar)))
    return np.nan_to_num(np.array(out, float))


def true_labels(bus, br):
    yv = bool(((bus.Vm < bus.min_vm_pu - TOL) | (bus.Vm > bus.max_vm_pu + TOL)).any())
    b = br[br.br_status == 1]
    lo = np.maximum(np.hypot(b.pf, b.qf), np.hypot(b.pt, b.qt)) / b.rate_a
    return yv, bool((lo > 1 + TOL).any())


def stage_sanity(args):
    rows = []
    for name, raw in TESTS.items():
        t = load(raw)
        for s in sorted(t["bus_data"])[:20]:
            net = build(t, s)
            _, ok = solve(net, numba=False)
            dvm, dva = compare(net, t["bus_data"][s]) if ok else (np.nan, np.nan)
            rows.append({"dataset": name, "scenario": s, "converged": ok, "max |dVm| (pu)": dvm, "max |dVa| (deg)": dva})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.groupby("dataset")[["converged", "max |dVm| (pu)", "max |dVa| (deg)"]].agg(
        {"converged": "sum", "max |dVm| (pu)": "max", "max |dVa| (deg)": "max"}).to_string())
    worst = df.sort_values("max |dVm| (pu)", ascending=False).head(5)
    print("\nworst 5:\n" + worst.to_string(index=False))
    EXTRA.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXTRA / "solver_nr_sanity20.csv", index=False, float_format="%.3g")
    ok = df.converged.all() and df["max |dVm| (pu)"].max() < 1e-4 and df["max |dVa| (deg)"].max() < 1e-2
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return ok


def stage_time(args):
    print(f"CPU: {platform.processor()}, {os.cpu_count()} logical cores, pandapower {pp.__version__}, single process")
    rows = []
    for name, raw in TESTS.items():
        t = load(raw)
        scen = sorted(t["bus_data"])
        warm = build(t, scen[0])
        solve(warm, numba=True)  # numba JIT compile, not timed
        for s in scen:
            bus, br = t["bus_data"][s], t["branch_data"][s]
            row = {"dataset": name, "scenario": s}
            for numba in [True, False]:
                net = build(t, s)
                dt, ok = solve(net, numba=numba)
                tag = "numba" if numba else "no_numba"
                row[f"time_s_{tag}"] = dt
                row[f"converged_{tag}"] = ok
            if row["converged_numba"]:
                row["max_dVm"], row["max_dVa_deg"] = compare(net, bus)
                row["iterations"] = int(net._ppc["iterations"]) if "iterations" in net._ppc else np.nan
                yv, yo = labels_from_net(net, bus, br)
                tv, to = true_labels(bus, br)
                row.update(y_voltage_nr=yv, y_overload_nr=yo, y_voltage_true=tv, y_overload_true=to)
            rows.append(row)
        print(f"{name}: done")
    df = pd.DataFrame(rows)
    df.to_csv(EXTRA / "solver_nr_per_sample.csv", index=False, float_format="%.6g")
    summ = []
    for name, g in df.groupby("dataset", sort=False):
        c = g[g.converged_numba]
        summ.append({"dataset": name, "n": len(g),
                     "NR failures (numba)": int((~g.converged_numba).sum()),
                     "NR failures (no numba)": int((~g.converged_no_numba).sum()),
                     "NR ms/sample median (numba)": 1e3 * g.time_s_numba.median(),
                     "NR ms/sample mean (numba)": 1e3 * g.time_s_numba.mean(),
                     "NR ms/sample median (no numba)": 1e3 * g.time_s_no_numba.median(),
                     "NR ms/sample mean (no numba)": 1e3 * g.time_s_no_numba.mean(),
                     "max |dVm| pu": c.max_dVm.max(), "max |dVa| deg": c.max_dVa_deg.max(),
                     "label agreement voltage": (c.y_voltage_nr == c.y_voltage_true).mean(),
                     "label agreement overload": (c.y_overload_nr == c.y_overload_true).mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(EXTRA / "solver_nr_summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summ.T.to_string())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sanity", "time"])
    args = p.parse_args()
    if args.stage == "sanity":
        sys.exit(0 if stage_sanity(args) else 1)
    stage_time(args)


if __name__ == "__main__":
    main()
