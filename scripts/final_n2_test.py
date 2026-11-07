"""Final N-2 test: the frozen Task 3 pipeline (stage3_detectors.py, results/stage3_summary.md) on PFΔ case14 N-2.

Run once. Nothing here is tuned: the gate thresholds are recomputed with the frozen rule on the same val
data (and asserted equal to the frozen values), and the B probes are loaded from data/stage3_probes.

Test sets (each one inference run of 1,000 samples):
  test_n2        data/pfdelta_n2_test1000 (scripts/make_pfdelta_n2_subset.py)
  test_pfdelta   data/pfdelta_test1000 (N-0/N-1), test_datakit   data/case14_test1000   (re-reported for one table)

Load level: a sample is "within" datakit's training load range if its total Pd is <= the largest total Pd of
the checkpoint's TRAIN split (321.8 MW), else "above". (The replay model has also seen PFΔ N-0/N-1 loads.)

Stages:
  extract --model M     graphkit predict --get_embeddings on N-2 -> data/predictions/stage3/<M>/test_n2/
  evaluate              both models -> results/final_n2_*.csv

Usage:
  python scripts/final_n2_test.py extract --model replay
  python scripts/final_n2_test.py extract --model original
  python scripts/final_n2_test.py evaluate
"""

import argparse
import json
import os
import sys
import warnings
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from finetune_pipeline import CKPT_ROOT, EVAL_SETS, make_config, model_path
from make_test_subset import SPLITS_JSON
from phase5_analysis import confusion, long_errors, robust
from stage3_detectors import (LABELS, PRED, PROBES, SIZE, blocks, choose_gate, features, gated, load_block,
                              safe_auc, wrong)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N2 = ROOT / "data/pfdelta_n2_test1000"
TESTS = {"test_n2": (N2, RESULTS / "pfdelta_n2_test_subset_1000.csv"),
         "test_pfdelta": EVAL_SETS["pfdelta"], "test_datakit": EVAL_SETS["datakit"]}
FROZEN_GATE = {"original": 1.27, "replay": np.inf}  # results/stage3_summary.md
MODELS = ["original", "replay"]
PROBE_KINDS = ["logreg", "mlp"]


def stage_extract(args):
    from gridfm_graphkit.cli import main_cli

    out = PRED / args.model / "test_n2"
    assert not (out / "case14_ieee_bus_embeddings.parquet").exists(), "final test runs once"
    cfg = make_config(SIZE, N2 / "splits", f"final_n2_predict_{SIZE}")
    main_cli(Namespace(command="predict", config=str(cfg.relative_to(ROOT)), model_path=str(model_path(args.model, SIZE)),
                       normalizer_stats=str(CKPT_ROOT / SIZE / "seed0/normalizer_stats.pt"),
                       data_path=str(N2.relative_to(ROOT)), log_dir="mlruns", exp_name="final_n2",
                       run_name=f"{args.model}_test_n2", num_workers=0, get_embeddings=True,
                       output_path=str(out), config_conversion="no", plugins=[]))


def datakit_train_max_load():
    split = json.load(open(SPLITS_JSON))["train"]
    t = ds.dataset(ROOT / "data/case14_ieee/raw/bus_data.parquet", format="parquet", partitioning="hive")
    tot = t.to_table(columns=["scenario", "Pd"]).to_pandas().groupby("scenario").Pd.sum()
    return float(tot.loc[split].max())


def load_test(model, name):
    data, subset_csv = TESTS[name]
    t, f = load_block(model, name, data)
    sub = pd.read_csv(subset_csv).set_index("subset_id")
    bus = pd.read_parquet(PRED / model / name / "case14_ieee_predictions.parquet")
    raw_bus = pd.read_parquet(data / "case14_ieee/raw/bus_data.parquet")
    t["total_load"] = raw_bus.groupby("scenario").Pd.sum()
    t["outage_type"] = sub.outage_type.reindex(t.index)
    t["topology"] = sub.topology.reindex(t.index) if "topology" in sub else "datakit"
    return t, f, bus


def scopes(t, max_load):
    out = {"all": t.index}
    out["load within"] = t.index[t.total_load <= max_load]
    out["load above"] = t.index[t.total_load > max_load]
    for k in ["N-0", "N-1", "N-2"]:
        if (t.topology == k).any():
            out[k] = t.index[t.topology == k]
    if (t.topology == "N-2").all():
        for k in ["branch", "gen", "mixed"]:
            out[f"N-2 {k}"] = t.index[t.outage_type == k]
    return out


def stage_evaluate(args):
    warnings.filterwarnings("ignore")
    max_load = datakit_train_max_load()
    print(f"datakit TRAIN max total load: {max_load:.1f} MW")
    det, trust, sweep, errs, scores = [], [], [], [], []
    for model in MODELS:
        val = pd.concat([load_block(model, n, d)[0] for n, (d, _) in blocks().items() if n.startswith(("pf_val", "dk_val"))])
        thr = choose_gate(val)
        assert (np.isinf(thr) and np.isinf(FROZEN_GATE[model])) or abs(thr - FROZEN_GATE[model]) < 0.01, thr
        qthr = {q: (np.inf if q == 0 else np.quantile(val.pbe_mean, 1 - q)) for q in [0, .01, .02, .05, .10, .20]}
        probes = {(p, lab): joblib.load(PROBES / f"{model}_{p}_{lab}.joblib") for p in PROBE_KINDS for lab in LABELS}
        print(f"[{model}] gate t = {thr:.4g} (frozen {FROZEN_GATE[model]})")
        for name in TESTS:
            t, f, bus = load_test(model, name)
            for (p, lab), pr in probes.items():
                t[f"{lab}_{p}_prob"] = pr["clf"].predict_proba(features(f, pr["features"]))[:, 1]
                t[f"{lab}_{p}"] = t[f"{lab}_{p}_prob"] >= pr["threshold"]
            t.to_csv(RESULTS / f"final_n2_samples_{model}_{name}.csv", float_format="%.6g")
            long = long_errors(bus, t.outage_type)
            for scope, idx in scopes(t, max_load).items():
                s = t.loc[idx]
                if len(s) == 0:
                    continue
                e = long[long.scenario.isin(idx)].groupby(["variable", "bus_type"]).apply(robust)
                errs.append(e[["RMSE", "median", "p95", "p99"]].assign(model=model, dataset=name, scope=scope).reset_index())
                scores.append({"model": model, "dataset": name, "scope": scope, "n": len(s),
                               "score median": s.score.median(), "score p95": s.score.quantile(.95),
                               "score p99": s.score.quantile(.99), "pbe median": s.pbe_mean.median(),
                               "voltage rate": s.y_voltage.mean(), "overload rate": s.y_overload.mean()})
                rho = spearmanr(s.pbe_mean, s.score).statistic if len(s) > 2 else np.nan
                for lab in ["y_voltage", "y_overload", "any"]:
                    w = wrong(s, lab).values
                    trust.append({"model": model, "dataset": name, "scope": scope, "n": len(s), "wrong label": lab,
                                  "n wrong": int(w.sum()), "spearman(pbe, error)": rho,
                                  "AUROC": safe_auc(roc_auc_score, w, s.pbe_mean),
                                  "PR-AUC": safe_auc(average_precision_score, w, s.pbe_mean), "PR-AUC chance": w.mean()})
                base = {"model": model, "dataset": name, "scope": scope, "n": len(s)}
                for lab in LABELS:
                    y = s[lab].values
                    det.append({**base, "label": lab, "method": "baseline PtC",
                                **confusion(y, s[f"{lab}_ptc"].values), "PR-AUC": np.nan, "solver share": 0.0})
                    share, c, cd = gated(s, thr, lab)
                    det.append({**base, "label": lab, "method": "A gated PtC (solver)", **c, "PR-AUC": np.nan,
                                "solver share": share})
                    det.append({**base, "label": lab, "method": "A gated PtC (drop)", **cd, "PR-AUC": np.nan,
                                "solver share": 0.0})
                    for p in PROBE_KINDS:
                        det.append({**base, "label": lab, "method": f"B {p}", **confusion(y, s[f"{lab}_{p}"].values),
                                    "PR-AUC": safe_auc(average_precision_score, y, s[f"{lab}_{p}_prob"]),
                                    "solver share": 0.0})
                    for q, th in qthr.items():
                        share, c, cd = gated(s, th, lab)
                        sweep.append({**base, "label": lab, "val share target": q, "threshold": th, "solver share": share,
                                      **{f"solver {k}": c[k] for k in ["precision", "recall", "FP", "FN"]},
                                      **{f"drop {k}": cd[k] for k in ["precision", "recall", "FP", "FN"]}})
    det, trust, sweep = pd.DataFrame(det), pd.DataFrame(trust), pd.DataFrame(sweep)
    errs, scores = pd.concat(errs), pd.DataFrame(scores)
    det.to_csv(RESULTS / "final_n2_detectors.csv", index=False, float_format="%.4g")
    trust.to_csv(RESULTS / "final_n2_A_trust.csv", index=False, float_format="%.4g")
    sweep.to_csv(RESULTS / "final_n2_A_gate_sweep.csv", index=False, float_format="%.4g")
    errs.to_csv(RESULTS / "final_n2_errors.csv", index=False, float_format="%.6g")
    scores.to_csv(RESULTS / "final_n2_scores.csv", index=False, float_format="%.4g")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    main_scopes = ["all", "load within", "load above"]
    e = errs[errs.scope.isin(main_scopes)].pivot_table(index=["dataset", "scope", "variable", "bus_type"],
                                                       columns="model", values=["RMSE", "median", "p95", "p99"])
    print("=== errors ===\n" + e.to_string())
    print("\n=== scores ===\n" + scores.to_string(index=False))
    print("\n=== A trust ===\n" + trust[trust.scope.isin(main_scopes)].to_string(index=False))
    cols = ["dataset", "scope", "label", "method", "model", "precision", "recall", "F1", "PR-AUC", "FP", "FN", "solver share"]
    print("\n=== detectors ===\n" + det[cols].sort_values(cols[:5]).to_string(index=False))
    print("\n=== gate sweep (N-2) ===\n" + sweep[(sweep.dataset == "test_n2") & (sweep.scope == "all")].to_string(index=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["extract", "evaluate"])
    p.add_argument("--model", default="replay", choices=MODELS)
    args = p.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
