"""First distribution-shift test: base model on 1000 PFΔ case14 N-0/N-1 TEST samples.

Needs: `run_inference.py --dataset pfdelta` (evaluate run "base_n1000_pfdelta") and the datakit results from
Phases 4-5 for comparison. N-2 is not read anywhere.

  1. conversion sanity checks (flows vs admittances, labels vs original JSON, input ranges)
  2. errors: RMSE / median / p95 / p99 per bus type, PFΔ vs datakit, per topology
  3. predict-then-check (slack Vm = known setpoint): precision / recall, overall and per topology
  4. reconstruction error and power-balance residual on samples where predict-then-check is wrong vs right

Usage:
  python scripts/pfdelta_shift_eval.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from explore_data import pfdelta_sample
from phase5_analysis import (confusion, find_run, labels_from_predictions, long_errors, robust,
                             true_labels_from_raw)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = ROOT / "data/pfdelta_test1000/case14_ieee/raw"
DK_RAW = ROOT / "data/case14_test1000/case14_ieee/raw"
FMT = lambda v: f"{v:.3g}"


def section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def auc(score, positive):
    """P(score of a positive > score of a negative), ties count half (Mann-Whitney U / n1 n0)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    art = find_run("base_n1000_pfdelta") / "artifacts/test"
    bus = pd.read_parquet(art / "case14_ieee_predictions.parquet")
    branch = pd.read_parquet(art / "case14_ieee_branch_predictions.parquet")
    per = pd.read_csv(RESULTS / "inference_per_sample_pfdelta.csv", index_col="sample_id")
    mapping = pd.read_csv(RESULTS / "pfdelta_test_subset_1000.csv", index_col="subset_id")
    assert not mapping.topology.eq("N-2").any(), "N-2 must stay untouched"
    print(f"evaluate run: {art.parents[1].name}; samples: {per.topology.value_counts().to_dict()}")

    # ---------------------------------------------------------------- 1
    section("1. Conversion sanity checks")
    raw_br = pd.read_parquet(RAW / "branch_data.parquet").query("br_status == 1")
    ends = pd.concat([
        raw_br.assign(fb=raw_br.from_bus, tb=raw_br.to_bus, P=raw_br.pf, Q=raw_br.qf),
        raw_br.assign(fb=raw_br.to_bus, tb=raw_br.from_bus, P=raw_br.pt, Q=raw_br.qt),
    ])[["scenario", "fb", "tb", "P", "Q"]].rename(columns={"fb": "from_bus", "tb": "to_bus"})
    j = branch.merge(ends, on=["scenario", "from_bus", "to_bus"])
    print(f"directed edges matched {len(j)} / {len(branch)}; max |Pft_target - P_raw| = "
          f"{(j.Pft_target - j.P).abs().max():.3g} MW, max |Qft_target - Q_raw| = {(j.Qft_target - j.Q).abs().max():.3g} MVAr")
    print("   (graphkit recomputes true flows from the true Vm/Va through the admittances: small = consistent conversion)")

    truth = true_labels_from_raw(RAW)
    json_flags = pd.DataFrame([pfdelta_sample(open(ROOT / "data/pfdelta_orig" / f, "rb").read())
                               for f in mapping.pfdelta_file], index=mapping.index)
    agree_v = ((json_flags.vm_low | json_flags.vm_high) == truth.y_voltage).mean()
    agree_o = (json_flags.line_overload == truth.y_overload).mean()
    print(f"labels from converted parquet vs original JSON: y_voltage agree {agree_v:.1%}, y_overload agree {agree_o:.1%}")

    def load_stats(raw_dir):
        b = pd.read_parquet(raw_dir / "bus_data.parquet")
        g = b.groupby("scenario")
        return pd.DataFrame({"total_Pd": g.Pd.sum(), "total_Qd": g.Qd.sum(), "min_Vm": g.Vm.min(), "max_Vm": g.Vm.max()})
    dk, pf = load_stats(DK_RAW), load_stats(RAW)
    rng = pd.concat({"datakit test": dk.describe(percentiles=[.05, .5, .95]).T,
                     "PFΔ test": pf.describe(percentiles=[.05, .5, .95]).T})[["min", "5%", "50%", "95%", "max"]]
    print("\ninput / state ranges:\n" + rng.to_string(float_format=FMT))
    pf = pf.join(mapping.topology)
    out_of_range = (pf.total_Pd > dk.total_Pd.max()) | (pf.total_Pd < dk.total_Pd.min())
    print(f"PFΔ samples with total load outside the datakit test range: {out_of_range.sum()} "
          f"({pf[out_of_range].topology.value_counts().to_dict()})")

    # ---------------------------------------------------------------- 2
    section("2. Errors: PFΔ vs datakit (base model, unknowns only)")
    long = long_errors(bus, mapping.outage_type)
    long["topology"] = long.scenario.map(mapping.topology)
    pf_rob = long.groupby(["variable", "bus_type"]).apply(robust)
    dk_rob = pd.read_csv(RESULTS / "inference_percentiles_by_bus_type.csv", index_col=["variable", "bus_type"])
    cols = ["RMSE", "median", "p95", "p99"]
    side = pd.concat({"datakit": dk_rob[cols], "PFΔ": pf_rob[cols], "PFΔ / datakit": pf_rob[cols] / dk_rob[cols]}, axis=1)
    print(side.to_string(float_format=FMT))
    by_topo = long.groupby(["variable", "bus_type", "topology"]).apply(robust)[cols]
    print("\nPFΔ by topology:\n" + by_topo.to_string(float_format=FMT))
    side.to_csv(RESULTS / "pfdelta_vs_datakit_errors.csv", float_format="%.6g")
    by_topo.to_csv(RESULTS / "pfdelta_errors_by_topology.csv", float_format="%.6g")
    s = per.overall_score_vs_datakit
    print(f"\noverall score (1.0 = datakit RMSE level): median {s.median():.1f}, p95 {s.quantile(.95):.1f}, "
          f"max {s.max():.1f} | by topology: {per.groupby('topology').overall_score_vs_datakit.median().round(1).to_dict()}")

    # ---------------------------------------------------------------- 3
    section("3. Predict-then-check (slack Vm treated as known setpoint)")
    pred = labels_from_predictions(bus, branch, "pred", slack_setpoint=True)
    tgt = labels_from_predictions(bus, branch, "target")
    print(f"graphkit target labels vs raw labels agree: "
          f"{(tgt.y_voltage == truth.y_voltage).mean():.1%} voltage, {(tgt.y_overload == truth.y_overload).mean():.1%} overload")
    y = truth.join(pred, rsuffix="_pred").join(mapping[["topology", "outage_type"]])
    rows = {}
    for scope, g in [("all", y)] + [(t, g) for t, g in y.groupby("topology")]:
        for k in ["y_voltage", "y_overload"]:
            rows[(scope, k)] = confusion(g[k].values, g[f"{k}_pred"].values)
        rows[(scope, "any violation")] = confusion((g.y_voltage | g.y_overload).values,
                                                   (g.y_voltage_pred | g.y_overload_pred).values)
    cm = pd.DataFrame(rows).T
    cm["base rate"] = (cm.TP + cm.FN) / (cm.TP + cm.FN + cm.FP + cm.TN)
    print(cm.to_string(float_format=FMT))
    cm.to_csv(RESULTS / "pfdelta_predict_then_check.csv", float_format="%.4f")
    dk_cm = pd.read_csv(RESULTS / "predict_then_check_confusion.csv", index_col=0)
    print("\ndatakit reference (slack known): voltage P/R "
          f"{dk_cm.loc['y_voltage (slack setpoint known)', 'precision']:.3f}/{dk_cm.loc['y_voltage (slack setpoint known)', 'recall']:.3f}, "
          f"overload P/R {dk_cm.loc['y_overload', 'precision']:.3f}/{dk_cm.loc['y_overload', 'recall']:.3f}")

    # ---------------------------------------------------------------- 4
    section("4. Reconstruction error and residual on wrong vs right predict-then-check labels")
    y = y.join(per[["overall_score_vs_datakit", "pbe_mean", "pbe_max"]])
    y["wrong_voltage"] = y.y_voltage != y.y_voltage_pred
    y["wrong_overload"] = y.y_overload != y.y_overload_pred
    y["wrong_any"] = y.wrong_voltage | y.wrong_overload
    y["case_voltage"] = np.select([y.y_voltage & y.y_voltage_pred, ~y.y_voltage & y.y_voltage_pred,
                                   y.y_voltage & ~y.y_voltage_pred], ["TP", "FP", "FN"], "TN")
    y["case_overload"] = np.select([y.y_overload & y.y_overload_pred, ~y.y_overload & y.y_overload_pred,
                                    y.y_overload & ~y.y_overload_pred], ["TP", "FP", "FN"], "TN")
    scores = ["overall_score_vs_datakit", "pbe_mean", "pbe_max"]
    for label in ["voltage", "overload"]:
        tab = y.groupby(f"case_{label}")[scores].agg(["count", "median"])
        tab = tab.loc[:, [(scores[0], "count")] + [(s, "median") for s in scores]]
        tab.columns = ["n"] + [f"median {s}" for s in scores]
        print(f"\n{label}: medians per confusion cell\n" + tab.to_string(float_format=FMT))
    rows = {}
    for scope, g in [("all", y)] + [(t, g) for t, g in y.groupby("topology")]:
        for wrong in ["wrong_voltage", "wrong_overload", "wrong_any"]:
            rows[(scope, wrong)] = {"n wrong": int(g[wrong].sum()), "n right": int((~g[wrong]).sum()),
                                    **{f"AUC {s}": auc(g[s], g[wrong]) for s in scores}}
    sep = pd.DataFrame(rows).T
    print("\nhow well does each score flag a wrong label? (AUC: 0.5 = no signal, 1.0 = perfect)\n"
          + sep.to_string(float_format=FMT))
    print(f"\nSpearman(overall score, pbe_mean) = {y.overall_score_vs_datakit.corr(y.pbe_mean, method='spearman'):.3f}")
    sep.to_csv(RESULTS / "pfdelta_wrong_vs_right_auc.csv", float_format="%.4f")
    y.to_csv(RESULTS / "pfdelta_labels_true_vs_pred.csv", float_format="%.6g")


if __name__ == "__main__":
    main()
