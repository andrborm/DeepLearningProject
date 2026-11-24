"""Explore the case14 data formats used in this project.

Parts (run all, or pick with --parts):
  A  datakit case14 (pf_small_case14_ieee): sizes, columns, bus-type encoding, contingency mix
  B  graphkit PowerFlow masking: which inputs are hidden per bus type, what the model predicts
  C  per-variable summary statistics + limit violations already present in the data
  D  branch flow limits (rate_a) for later overload labels
  E  original PFΔ case14 archive: samples per category, voltage violations / overloads (feasible)

Outputs: printed report, results/case14_variable_stats.csv, results/case14_violation_counts.csv,
results/pfdelta_case14_counts.csv. The written summary is results/data_format.md.

Usage:
  python scripts/explore_data.py            # all parts
  python scripts/explore_data.py --parts AB
"""

import argparse
import json
import math
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/case14_ieee/raw"
PFDELTA_TAR = ROOT / "data/pfdelta_orig/case14.tar.gz"  # read in place; no need to extract
RESULTS = ROOT / "results"
SPLITS_JSON = next(
    (ROOT / "checkpoints/genco-pf-datakit/mlflow/eval/case14_ieee/base/seed0").glob(
        "*/artifacts/stats/case14_ieee_scenario_splits.json",
    ),
)
VM_MIN, VM_MAX = 0.94, 1.06
TOL = 1e-6  # numerical tolerance for limit checks

COLUMN_MEANING = {
    "bus": {
        "Pd, Qd": "active / reactive load at the bus (MW, MVAr)",
        "Pg, Qg": "active / reactive generation summed over the bus's generators (MW, MVAr)",
        "Vm, Va": "voltage magnitude (p.u.) and angle (degrees)",
        "PQ, PV, REF": "one-hot bus type (load bus, generator bus, slack/reference bus)",
        "vn_kv": "nominal voltage (kV; 1.0 for pglib case14)",
        "min_vm_pu, max_vm_pu": "voltage magnitude limits (p.u.)",
        "GS, BS": "shunt conductance / susceptance",
        "Va_dc, Pg_dc": "DC power flow baseline results",
        "load_scenario_idx": "id of the load profile the scenario was derived from",
    },
    "gen": {
        "p_mw, q_mvar": "generator active / reactive output",
        "min/max_p_mw, min/max_q_mvar": "generator limits",
        "cp0/cp1/cp2": "cost coefficients (used by OPF, masked in PF)",
        "in_service": "1 = online, 0 = outaged in this scenario",
        "is_slack_gen": "generator at the reference bus",
    },
    "branch": {
        "from_bus, to_bus": "endpoint bus indices (0-based)",
        "pf, qf, pt, qt": "active / reactive flow at from / to end (MW, MVAr)",
        "r, x, b": "series resistance, reactance, total line charging",
        "Yff/Yft/Ytf/Ytt _r/_i": "branch admittance matrix entries (real / imag)",
        "tap, shift": "transformer ratio and phase shift",
        "ang_min, ang_max": "angle-difference limits (deg)",
        "rate_a": "thermal rating (long-term), basis for overload labels",
        "br_status": "1 = in service, 0 = outaged in this scenario",
    },
}


def read(name, columns=None):
    """Read (some columns of) one Hive-partitioned datakit table."""
    table = ds.dataset(RAW / f"{name}.parquet", format="parquet", partitioning="hive")
    return table.to_table(columns=columns).to_pandas()


def section(title):
    """Print a section header."""
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


# --------------------------------------------------------------------------- A
def part_a():
    """Sizes, columns, bus-type encoding and contingency mix of datakit case14."""
    section("A. datakit case14 (gridfm/pf_small_case14_ieee): format")
    bus = read("bus_data", ["scenario", "bus", "PQ", "PV", "REF", "load_scenario_idx"])
    gen = read("gen_data", ["scenario", "idx", "bus", "in_service", "is_slack_gen"])
    branch = read("branch_data", ["scenario", "idx", "from_bus", "to_bus", "br_status", "tap"])

    n_scen = bus.scenario.nunique()
    print(f"scenarios: {n_scen:,}  (load scenarios: {bus.load_scenario_idx.nunique():,})")
    print(f"buses/scenario: {len(bus) // n_scen}, gens/scenario: {len(gen) // n_scen}, "
          f"branches/scenario: {len(branch) // n_scen} "
          f"(of which transformers, tap != 1: {(branch[branch.scenario == 0].tap != 1).sum()})")

    for name in ["bus_data", "gen_data", "branch_data"]:
        cols = ds.dataset(RAW / f"{name}.parquet", partitioning="hive").schema.names
        print(f"{name} columns: {cols}")
    for table, meanings in COLUMN_MEANING.items():
        print(f"\n{table} column meaning:")
        for col, meaning in meanings.items():
            print(f"  {col:32s} {meaning}")

    print("\nBus type encoding: one-hot columns PQ / PV / REF (exactly one is 1 per row):",
          bool(((bus.PQ + bus.PV + bus.REF) == 1).all()))
    bus["type"] = np.select([bus.REF == 1, bus.PV == 1], ["REF", "PV"], "PQ")
    types = bus.groupby("bus").type.value_counts().unstack(fill_value=0)
    print("bus type per bus index across all scenarios:\n" + types.to_string())

    gen_out = gen[gen.in_service == 0]
    print("\ngenerators (scenario 0):\n" + gen[gen.scenario == 0][["idx", "bus", "is_slack_gen"]].to_string(index=False))
    print("outaged generator -> count:", gen_out.bus.value_counts().sort_index().to_dict(), "(by bus)")
    # Does a generator outage turn its PV bus into a PQ bus?
    g = gen_out[["scenario", "bus"]].merge(bus[["scenario", "bus", "type"]], on=["scenario", "bus"])
    print("bus type of the bus whose generator is out:", g.type.value_counts().to_dict())

    n_br = (branch.br_status == 0).groupby(branch.scenario).sum()
    n_gen = (gen.in_service == 0).groupby(gen.scenario).sum()
    k = n_br + n_gen
    mix = pd.DataFrame({"k": k, "type": np.select([k == 0, n_gen == 0], ["none", "branch"], "gen")})
    print("\ncontingency mix, all scenarios:")
    print("  " + ("N-" + mix.k.astype(str)).value_counts().to_string().replace("\n", "\n  "))
    print("  " + mix.type.value_counts().to_string().replace("\n", "\n  "))
    test_ids = json.load(open(SPLITS_JSON))["test"]
    print(f"contingency mix, checkpoint test split ({len(test_ids):,}):",
          mix.loc[test_ids].type.value_counts().to_dict())
    print("outaged branch -> count:", branch[branch.br_status == 0].idx.astype(int).value_counts().sort_index().to_dict())


# --------------------------------------------------------------------------- B
def part_b():
    """Which inputs graphkit's PowerFlow mask hides per bus type, on one real scenario."""
    section("B. graphkit PowerFlow masking (AddPFHeteroMask) on one real scenario")
    from gridfm_graphkit.datasets.graph_builder import BUS_FEATURES, GEN_FEATURES, FORWARD_BRANCH_FEATURES, build_hetero_data
    from gridfm_graphkit.datasets.masking import AddPFHeteroMask
    from gridfm_graphkit.datasets.transforms import RemoveInactiveBranches, RemoveInactiveGenerators

    flt = ds.field("scenario") == 0
    load = lambda n: ds.dataset(RAW / f"{n}.parquet", partitioning="hive").to_table(filter=flt).to_pandas()
    bus, gen, branch = load("bus_data"), load("gen_data"), load("branch_data")
    # same Q-limit aggregation graphkit does before building graphs
    agg = gen.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
    bus = bus.merge(agg, on=["scenario", "bus"], how="left").fillna(0).sort_values("bus")

    data = build_hetero_data(bus, gen.sort_values("idx"), branch.sort_values("idx"), 0)
    for t in (RemoveInactiveBranches(), RemoveInactiveGenerators(), AddPFHeteroMask()):
        data = t(data)
    m = data.mask_dict
    gen_feats = [f for f in GEN_FEATURES if f != "in_service"]
    br_feats = [f for f in FORWARD_BRANCH_FEATURES if f != "br_status"]

    print("Masked (= hidden from the model, set to mask_value 0.0) bus features per bus type:")
    for bt in ["PQ", "PV", "REF"]:
        row = m["bus"][m[bt]][0]
        hidden = [f for f, h in zip(BUS_FEATURES, row.tolist()) if h]
        print(f"  {bt:3s}: {hidden}")
    print("Always-visible bus inputs: Pd, Qd, bus-type one-hot, GS, BS; plus Vm at PV/REF, Va at REF, Qg at PQ (=0)")
    ref_bus = int(m["REF"].nonzero()[0])
    gen_bus = data["gen", "connected_to", "bus"].edge_index[1]
    for label, sel in [("gen at REF bus", gen_bus == ref_bus), ("other gens", gen_bus != ref_bus)]:
        row = m["gen"][sel][0]
        print(f"  {label:14s} hidden: {[f for f, h in zip(gen_feats, row.tolist()) if h]}")
    print(f"  branches hidden: {[f for f, h in zip(br_feats, m['branch'][0].tolist()) if h]}")
    print("Model heads predict Vm, Va per bus and Pg per generator; bus Pg/Qg are then derived from the")
    print("power-flow equations. Unknowns scored per bus type = the masked subset of [Vm, Va, Pg, Qg]:")
    for bt in ["PQ", "PV", "REF"]:
        row = m["bus"][m[bt]][0]
        unknown = [f for f in ["Vm", "Va", "Qg"] if row[BUS_FEATURES.index(f)]]
        if bt == "REF":
            unknown.append("Pg (slack gen)")
        print(f"  {bt:3s} -> {unknown}")


# --------------------------------------------------------------------------- C
def describe(df, cols, table):
    """min / max / mean / std of some columns, tagged with the table name."""
    out = df[cols].agg(["min", "max", "mean", "std"]).T
    out.insert(0, "table", table)
    return out


def violation_counts(bus, gen, branch, scenario_subset=None):
    """Per-scenario flags for limit violations present in the solved power flow."""
    if scenario_subset is not None:
        bus, gen, branch = (d[d.scenario.isin(scenario_subset)] for d in (bus, gen, branch))
    vm_fixed = (bus.Vm < VM_MIN - TOL) | (bus.Vm > VM_MAX + TOL)
    vm_own = (bus.Vm < bus.min_vm_pu - TOL) | (bus.Vm > bus.max_vm_pu + TOL)
    g = gen[gen.in_service == 1]
    qg = (g.q_mvar < g.min_q_mvar - TOL) | (g.q_mvar > g.max_q_mvar + TOL)
    pg = (g.p_mw < g.min_p_mw - TOL) | (g.p_mw > g.max_p_mw + TOL)
    b = branch[branch.br_status == 1]
    loading = np.maximum(np.hypot(b.pf, b.qf), np.hypot(b.pt, b.qt)) / b.rate_a
    flags = pd.DataFrame({
        "vm_outside_0.94_1.06": vm_fixed.groupby(bus.scenario).any(),
        "vm_outside_own_limits": vm_own.groupby(bus.scenario).any(),
        "vm_low": (bus.Vm < VM_MIN - TOL).groupby(bus.scenario).any(),
        "vm_high": (bus.Vm > VM_MAX + TOL).groupby(bus.scenario).any(),
        "line_overload": (loading > 1 + TOL).groupby(b.scenario).any(),
        "qg_limit": qg.groupby(g.scenario).any(),
        "pg_limit": pg.groupby(g.scenario).any(),
    }).fillna(False)
    flags["any_voltage_or_overload"] = flags["vm_outside_0.94_1.06"] | flags["line_overload"]
    return flags, loading


def part_c():
    """Variable statistics and limit-violation counts (all, test split, test subset)."""
    section("C. per-variable statistics and limit violations (all 219,606 scenarios)")
    bus = read("bus_data", ["scenario", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF", "min_vm_pu", "max_vm_pu"])
    gen = read("gen_data", ["scenario", "p_mw", "q_mvar", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar", "in_service", "is_slack_gen"])
    branch = read("branch_data", ["scenario", "idx", "pf", "qf", "pt", "qt", "rate_a", "br_status"])

    stats = pd.concat([
        describe(bus, ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"], "bus"),
        describe(gen[gen.in_service == 1], ["p_mw", "q_mvar"], "gen (in service)"),
        describe(branch[branch.br_status == 1], ["pf", "qf", "pt", "qt", "rate_a"], "branch (in service)"),
    ])
    bus["type"] = np.select([bus.REF == 1, bus.PV == 1], ["REF", "PV"], "PQ")
    for bt in ["PQ", "PV", "REF"]:
        s = describe(bus[bus.type == bt], ["Vm", "Va", "Pg", "Qg"], f"bus {bt}")
        s.index = [f"{i} ({bt})" for i in s.index]
        stats = pd.concat([stats, s])
    stats.to_csv(RESULTS / "case14_variable_stats.csv", float_format="%.6g")
    print(stats.to_string(float_format=lambda v: f"{v:.4g}"))

    flags, loading = violation_counts(bus, gen, branch)
    test_ids = json.load(open(SPLITS_JSON))["test"]
    subset = pd.read_csv(RESULTS / "test_subset_1000.csv") if (RESULTS / "test_subset_1000.csv").exists() else None
    rows = {"all": flags, "test split": flags.loc[test_ids]}
    if subset is not None:
        rows["test subset 1000"] = flags.loc[subset.orig_scenario]
    counts = pd.DataFrame({k: v.sum() for k, v in rows.items()})
    counts.loc["n_scenarios"] = [len(v) for v in rows.values()]
    counts.to_csv(RESULTS / "case14_violation_counts.csv")
    print("\nscenarios with at least one violation:\n" + counts.to_string())
    print(f"\nbranch loading |S|/rate_a (in-service): mean {loading.mean():.3f}, "
          f"p99 {loading.quantile(0.99):.3f}, max {loading.max():.3f}")
    print("Vm bounds in data: min_vm_pu", bus.min_vm_pu.unique(), "max_vm_pu", bus.max_vm_pu.unique())


# --------------------------------------------------------------------------- D
def part_d():
    """rate_a per branch: present, positive and constant across scenarios?"""
    section("D. branch flow limits (rate_a)")
    branch = read("branch_data", ["scenario", "idx", "from_bus", "to_bus", "rate_a", "tap", "br_status"])
    per_branch = branch.groupby("idx").rate_a.agg(["min", "max", "nunique"])
    base = branch[branch.scenario == 0].set_index("idx")[["from_bus", "to_bus", "tap"]]
    print(base.join(per_branch).to_string())
    print(f"rate_a present: {branch.rate_a.notna().all()}, all > 0: {(branch.rate_a > 0).all()}, "
          f"constant per branch across scenarios: {(per_branch['nunique'] == 1).all()}")
    print("units: same as pf/qf (MW/MVA); overload = max(|pf+j qf|, |pt+j qt|) / rate_a > 1")


# --------------------------------------------------------------------------- E
def pfdelta_sample(raw_json):
    """Outage counts and violation flags for one feasible PFΔ JSON (schema {network, solution})."""
    d = json.loads(raw_json)
    net, sol = d["network"], d["solution"]["solution"]
    vm = [sol["bus"][k]["vm"] for k in net["bus"]]
    vmin = [net["bus"][k]["vmin"] for k in net["bus"]]
    vmax = [net["bus"][k]["vmax"] for k in net["bus"]]
    loading = []
    for k, br in net["branch"].items():
        if br["br_status"] == 1:
            f = sol["branch"][k]
            s = max(math.hypot(f["pf"], f["qf"]), math.hypot(f["pt"], f["qt"]))
            loading.append(s / br["rate_a"])
    return {
        "n_branch_out": sum(b["br_status"] == 0 for b in net["branch"].values()),
        "n_gen_out": sum(g["gen_status"] == 0 for g in net["gen"].values()),
        "vm_low": any(v < lo - TOL for v, lo in zip(vm, vmin)),
        "vm_high": any(v > hi + TOL for v, hi in zip(vm, vmax)),
        "line_overload": any(x > 1 + TOL for x in loading),
        "min_vm": min(vm),
        "max_loading": max(loading),
        "status": d["solution"]["termination_status"],
    }


def pfdelta_feasible_json(tar_path, counts):
    """Stream the archive once; count every JSON per category, yield (topology, name, bytes) for feasible ones."""
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")  # case14/<topo>/<category>[/<split>]/<file>.json
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            topo, category = parts[1], "/".join(parts[2:-1])
            counts[(topo, category)] = counts.get((topo, category), 0) + 1
            if category == "raw":
                yield topo, parts[-1], tar.extractfile(member).read()


def part_e():
    """Category counts and violation rates of the original PFΔ case14 archive."""
    section("E. original PFΔ case14 archive (read directly from data/pfdelta_orig/case14.tar.gz)")
    if not PFDELTA_TAR.exists():
        print(f"missing {PFDELTA_TAR}: hf download pfdelta/pfdelta case14.tar.gz --repo-type dataset")
        return
    counts, meta = {}, []

    def raw_bytes():
        """Feasible JSON bytes; remembers (topology, file) in the same order."""
        for topo, name, raw_json in pfdelta_feasible_json(PFDELTA_TAR, counts):
            meta.append((topo, name))
            yield raw_json

    with Pool() as pool:
        flags = list(pool.imap(pfdelta_sample, raw_bytes(), chunksize=256))
    feas = pd.DataFrame(flags)
    feas.insert(0, "topology", [t for t, _ in meta])
    feas.insert(1, "file", [n for _, n in meta])

    counts = pd.Series(counts).rename_axis(["topology", "category"]).unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts.loc["total"] = counts.sum()
    print("samples per category (raw = feasible power flow solutions):\n" + counts.to_string())

    feas["k"] = feas.n_branch_out + feas.n_gen_out
    feas["vm_violation"] = feas.vm_low | feas.vm_high
    feas["both"] = feas.vm_violation & feas.line_overload
    feas["normal"] = ~(feas.vm_violation | feas.line_overload)
    summary = feas.groupby("topology").agg(
        samples=("status", "size"),
        solved_ok=("status", lambda s: (s == "LOCALLY_SOLVED").sum()),
        outages_min=("k", "min"),
        outages_max=("k", "max"),
        vm_violation=("vm_violation", "sum"),
        vm_low=("vm_low", "sum"),
        vm_high=("vm_high", "sum"),
        line_overload=("line_overload", "sum"),
        both=("both", "sum"),
        normal=("normal", "sum"),
    )
    summary.loc["total"] = summary.sum()
    summary.loc["total", ["outages_min", "outages_max"]] = [feas.k.min(), feas.k.max()]
    counts.to_csv(RESULTS / "pfdelta_case14_counts.csv")
    summary.to_csv(RESULTS / "pfdelta_case14_feasible_violations.csv")
    print("\nfeasible samples: violation counts (sample-level, any bus / any branch):\n" + summary.to_string())
    print("\nfeasible samples by outage type:\n" + feas.groupby(["topology", "n_branch_out", "n_gen_out"]).size().to_string())
    print(f"\nfeasible: min Vm {feas.min_vm.min():.4f}, max branch loading {feas.max_loading.max():.2f}")
    print("statuses:", feas.status.value_counts().to_dict())


def main():
    """Run the selected parts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", default="ABCDE")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    for p in args.parts.upper():
        {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e}[p]()


if __name__ == "__main__":
    main()
