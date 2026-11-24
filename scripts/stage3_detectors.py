"""Task 3 detectors on PFΔ N-0/N-1 and datakit (no N-2). Multi-label: y_voltage, y_overload; slack Vm = known setpoint.

  Baseline  predict-then-check (PtC): labels from the model's predicted state.
  A         trust signal: per-sample power-balance residual (mean PBE, MVA, available without ground truth).
            - Spearman with true error, AUROC / PR-AUC for flagging wrong PtC labels
            - gated PtC: samples with residual > t go to a classical solver (exact labels), the rest keep PtC.
              t is chosen on validation; "drop" variant: high-residual alarms are simply removed
  B         frozen-embedding probes (logistic regression, small MLP) per label

Data (≤ 1000 samples per inference run):
  train  data/stage3 pf_probe (2,000 PFΔ train positions) [+ dk_probe (2,000 datakit train) with --with-datakit]
  val    data/stage3 pf_val (1,000) [+ dk_val (1,000)]   all tuning: hyperparameters, decision thresholds, gate t
  test   data/pfdelta_test1000, data/case14_test1000

Stages:
  extract --model M      graphkit predict --get_embeddings on every block -> data/predictions/stage3/<M>/<block>/
  analyze --model M      A and B for one model -> results/stage3_<M>_*.csv
  compare                one table across models and methods -> results/stage3_comparison.csv

Usage:
  python scripts/stage3_detectors.py extract --model replay
  python scripts/stage3_detectors.py analyze --model replay --with-datakit
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
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from finetune_pipeline import CKPT_ROOT, EVAL_SETS, S3, make_config, model_path, sample_scores
from phase5_analysis import confusion, labels_from_predictions, true_labels_from_raw

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRED = ROOT / "data/predictions/stage3"
PROBES = ROOT / "data/stage3_probes"  # fitted probes + thresholds (gitignored)
LABELS = ["y_voltage", "y_overload"]
SIZE = "base"


def blocks():
    """name -> (data folder, ids or None for 'all test ids', source tag)."""
    m = pd.read_csv(RESULTS / "stage3_mapping.csv")
    out = {}
    for role in ["pf_probe", "dk_probe", "pf_val", "dk_val"]:
        ids = m[m.role == role].s3_id.values
        for i in range(0, len(ids), 1000):
            out[f"{role}{i // 1000}"] = (S3, ids[i:i + 1000])
    out["test_pfdelta"] = (EVAL_SETS["pfdelta"][0], None)
    out["test_datakit"] = (EVAL_SETS["datakit"][0], None)
    return out


def stage_extract(args):
    """graphkit predict --get_embeddings on every block (blocks already done are skipped)."""
    from gridfm_graphkit.cli import main_cli

    for name, (data, ids) in blocks().items():
        out = PRED / args.model / name
        if (out / "case14_ieee_bus_embeddings.parquet").exists():
            continue
        if ids is None:
            splits = data / "splits"
        else:
            assert len(ids) <= 1000
            splits = S3 / f"splits_pred_{name}"
            splits.mkdir(parents=True, exist_ok=True)
            for k, v in [("train", []), ("val", []), ("test", ids)]:
                torch.save(torch.tensor(np.sort(np.asarray(v, dtype=np.int64)), dtype=torch.long), splits / f"{k}.pt")
        cfg = make_config(SIZE, splits, f"stage3_predict_{SIZE}")
        main_cli(Namespace(command="predict", config=str(cfg.relative_to(ROOT)), model_path=str(model_path(args.model, SIZE)),
                           normalizer_stats=str(CKPT_ROOT / SIZE / "seed0/normalizer_stats.pt"),
                           data_path=str(data.relative_to(ROOT)), log_dir="mlruns", exp_name="stage3_predict",
                           run_name=f"{args.model}_{name}", num_workers=0, get_embeddings=True,
                           output_path=str(out), config_conversion="no", plugins=[]))


# --------------------------------------------------------------------------- per-sample table
def load_block(model, name, data):
    """Per-sample table (truth, PtC labels, error score, residual) and (n, 14, 384) embeddings of one block."""
    d = PRED / model / name
    bus = pd.read_parquet(d / "case14_ieee_predictions.parquet")
    branch = pd.read_parquet(d / "case14_ieee_branch_predictions.parquet")
    emb = pd.read_parquet(d / "case14_ieee_bus_embeddings.parquet").sort_values(["scenario", "bus"])
    truth = true_labels_from_raw(data / "case14_ieee/raw")
    ptc = labels_from_predictions(bus, branch, "pred", slack_setpoint=True)
    scen = np.sort(bus.scenario.unique())
    df = pd.DataFrame(index=pd.Index(scen, name="scenario"))
    df = df.join(truth[LABELS]).join(ptc.add_suffix("_ptc"))
    df["score"] = sample_scores(bus)
    df["pbe_mean"] = bus.groupby("scenario").PBE.mean()
    df["block"] = name
    e = emb.drop(columns=["scenario", "bus"]).to_numpy(np.float32)
    nb = emb.groupby("scenario").size()
    assert nb.nunique() == 1 and (np.sort(emb.scenario.unique()) == scen).all()
    feats = e.reshape(len(scen), nb.iloc[0], -1)
    return df, feats


def features(f, kind):
    """Probe input: 'pooled' = mean/max/min over buses (1,152-d), otherwise flat 14 x 384 (5,376-d)."""
    if kind == "pooled":
        return np.concatenate([f.mean(1), f.max(1), f.min(1)], axis=1)
    return f.reshape(len(f), -1)  # case14: fixed bus order, so per-bus identity is kept


def load_all(model):
    """All blocks of one model, concatenated per group (pf_probe, dk_val, test_pfdelta, ...)."""
    tabs, feats = {}, {}
    for name, (data, _) in blocks().items():
        t, f = load_block(model, name, data)
        group = name.rstrip("0123456789")
        tabs.setdefault(group, []).append(t)
        feats.setdefault(group, []).append(f)
    return ({g: pd.concat(v) for g, v in tabs.items()}, {g: np.concatenate(v) for g, v in feats.items()})


# --------------------------------------------------------------------------- A
def wrong(t, label):
    """True where the PtC label differs from the truth (one label or 'any')."""
    if label == "any":
        return (t.y_voltage != t.y_voltage_ptc) | (t.y_overload != t.y_overload_ptc)
    return t[label] != t[f"{label}_ptc"]


def safe_auc(fn, y, s):
    """AUC-type metric, NaN when only one class is present."""
    return fn(y, s) if 0 < y.sum() < len(y) else float("nan")


def gated(t, thr, label):
    """PtC, but samples with residual > thr get the exact label (solver); 'drop' just removes their alarms."""
    sent = t.pbe_mean > thr
    solver = np.where(sent, t[label], t[f"{label}_ptc"])
    drop = np.where(sent, False, t[f"{label}_ptc"])
    return sent.mean(), confusion(t[label].values, solver.astype(bool)), confusion(t[label].values, drop.astype(bool))


def choose_gate(val, target=0.98):
    """Smallest solver share whose gated (solver) PtC reaches precision and recall >= target on both labels."""
    cands = np.r_[np.inf, np.sort(val.pbe_mean.values)[::-1]]
    for thr in cands:
        ok = True
        for lab in LABELS:
            _, c, _ = gated(val, thr, lab)
            ok &= (c["precision"] >= target or np.isnan(c["precision"])) and c["recall"] >= target
        if ok:
            return thr
    return cands[-1]


def approach_a(tabs, val_groups, model):
    """A: residual as trust signal; gate chosen on val; sweep over val-quantile thresholds."""
    val = pd.concat([tabs[g] for g in val_groups])
    rows, sweep = [], []
    thr = choose_gate(val)
    share_val = (val.pbe_mean > thr).mean()
    print(f"[A] gate chosen on val ({'+'.join(val_groups)}): residual > {thr:.4g} MVA -> solver "
          f"({share_val:.1%} of val samples)")
    for ds in ["test_pfdelta", "test_datakit", *val_groups]:
        t = tabs[ds]
        rho = spearmanr(t.pbe_mean, t.score).statistic
        for lab in ["y_voltage", "y_overload", "any"]:
            w = wrong(t, lab).values
            rows.append({"model": model, "dataset": ds, "wrong label": lab, "n wrong": int(w.sum()),
                         "spearman(pbe, error)": rho,
                         "AUROC": safe_auc(roc_auc_score, w, t.pbe_mean), "PR-AUC": safe_auc(average_precision_score, w, t.pbe_mean),
                         "PR-AUC chance": w.mean()})
        for q in [0, .01, .02, .05, .10, .20]:  # thresholds from val quantiles
            th = np.inf if q == 0 else np.quantile(val.pbe_mean, 1 - q)
            for lab in LABELS:
                share, c, cd = gated(t, th, lab)
                sweep.append({"model": model, "dataset": ds, "label": lab, "val share target": q, "threshold": th,
                              "solver share": share, **{f"solver {k}": c[k] for k in ["precision", "recall", "FP", "FN"]},
                              **{f"drop {k}": cd[k] for k in ["precision", "recall", "FP", "FN"]}})
    a = pd.DataFrame(rows)
    sw = pd.DataFrame(sweep)
    gate = []
    for ds in ["test_pfdelta", "test_datakit"]:
        for lab in LABELS:
            share, c, cd = gated(tabs[ds], thr, lab)
            gate.append({"model": model, "dataset": ds, "label": lab, "threshold": thr, "solver share": share,
                         **{f"solver {k}": c[k] for k in ["precision", "recall", "F1", "TP", "FP", "FN", "TN"]},
                         **{f"drop {k}": cd[k] for k in ["precision", "recall", "F1", "TP", "FP", "FN", "TN"]}})
    return a, sw, pd.DataFrame(gate)


# --------------------------------------------------------------------------- B
def best_threshold(y, p):
    """Decision threshold with the highest F1 (candidates: every score and 0.5)."""
    cand = np.unique(np.r_[p, 0.5])
    f1 = [confusion(y, p >= c)["F1"] for c in cand]
    f1 = np.nan_to_num(np.array(f1, float), nan=-1)
    return cand[int(np.argmax(f1))]


def approach_b(tabs, feats, train_groups, val_groups, model):
    """B: tune and fit one probe per label on frozen embeddings, save it, score the test sets."""
    Xtr_raw = np.concatenate([feats[g] for g in train_groups])
    Xva_raw = np.concatenate([feats[g] for g in val_groups])
    ttr = pd.concat([tabs[g] for g in train_groups])
    tva = pd.concat([tabs[g] for g in val_groups])
    grids = {
        "logreg": [dict(C=c) for c in [1e-3, 1e-2, 1e-1, 1]],
        "mlp": [dict(alpha=a, hidden=h) for a in [1e-4, 1e-2, 1] for h in [64, 256]],
    }
    rows, chosen = [], []
    # grid search: feature type x hyperparameters x label, scored by val PR-AUC
    for kind in ["pooled", "flat"]:
        Xtr, Xva = features(Xtr_raw, kind), features(Xva_raw, kind)
        for probe, grid in grids.items():
            for hp in grid:
                for lab in LABELS:
                    clf = make_clf(probe, hp)
                    clf.fit(Xtr, ttr[lab].values)
                    p = clf.predict_proba(Xva)[:, 1]
                    rows.append({"probe": probe, "features": kind, "hp": json.dumps(hp), "label": lab,
                                 "val PR-AUC": average_precision_score(tva[lab], p)})
    tune = pd.DataFrame(rows)
    tune.insert(0, "model", model)
    # refit the best setting per probe and label, pick the threshold on val, freeze to disk
    results, fits = [], []
    for probe in grids:
        for lab in LABELS:
            best = tune[(tune.probe == probe) & (tune.label == lab)].sort_values("val PR-AUC").iloc[-1]
            hp = json.loads(best.hp)
            Xtr, Xva = features(Xtr_raw, best.features), features(Xva_raw, best.features)
            clf = make_clf(probe, hp).fit(Xtr, ttr[lab].values)
            thr = best_threshold(tva[lab].values, clf.predict_proba(Xva)[:, 1])
            PROBES.mkdir(parents=True, exist_ok=True)  # frozen for the N-2 test
            joblib.dump({"clf": clf, "features": best.features, "threshold": thr, "hp": hp},
                        PROBES / f"{model}_{probe}_{lab}.joblib")
            chosen.append({"model": model, "probe": probe, "label": lab, "features": best.features, "hp": best.hp,
                           "val PR-AUC": best["val PR-AUC"], "decision threshold": thr})
            for ds in ["test_pfdelta", "test_datakit"]:
                p = clf.predict_proba(features(feats[ds], best.features))[:, 1]
                y = tabs[ds][lab].values
                c = confusion(y, p >= thr)
                results.append({"model": model, "method": f"B {probe}", "dataset": ds, "label": lab, **c,
                                "PR-AUC": safe_auc(average_precision_score, y, p)})
    return tune, pd.DataFrame(chosen), pd.DataFrame(results)


def make_clf(probe, hp):
    """Standardised logistic regression or one-hidden-layer MLP probe."""
    if probe == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], max_iter=3000, class_weight="balanced"))
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(hp["hidden"],), alpha=hp["alpha"],
                                                         max_iter=300, early_stopping=True, random_state=0))


def stage_analyze(args):
    """A and B for one model; writes results/stage3_<model>_*.csv."""
    warnings.filterwarnings("ignore")
    tabs, feats = load_all(args.model)
    train_groups = ["pf_probe"] + (["dk_probe"] if args.with_datakit else [])
    val_groups = ["pf_val"] + (["dk_val"] if args.with_datakit else [])
    for g, t in tabs.items():
        print(f"{g}: n={len(t)}, voltage {t.y_voltage.mean():.1%}, overload {t.y_overload.mean():.1%}, "
              f"median score {t.score.median():.3g}, median pbe {t.pbe_mean.median():.3g}")
    a, sweep, gate = approach_a(tabs, val_groups, args.model)
    tune, chosen, b = approach_b(tabs, feats, train_groups, val_groups, args.model)
    ptc = []
    for ds in ["test_pfdelta", "test_datakit"]:
        t = tabs[ds]
        for lab in LABELS:
            ptc.append({"model": args.model, "method": "baseline PtC", "dataset": ds, "label": lab,
                        **confusion(t[lab].values, t[f"{lab}_ptc"].values), "PR-AUC": float("nan")})
            g = gate[(gate.dataset == ds) & (gate.label == lab)].iloc[0]
            for v in ["solver", "drop"]:
                ptc.append({"model": args.model, "method": f"A gated PtC ({v})", "dataset": ds, "label": lab,
                            **{k: g[f"{v} {k}"] for k in ["TP", "FP", "FN", "TN", "precision", "recall", "F1"]},
                            "PR-AUC": float("nan"), "solver share": g["solver share"] if v == "solver" else 0.0})
    table = pd.concat([pd.DataFrame(ptc), b], ignore_index=True)
    sfx = args.model
    a.to_csv(RESULTS / f"stage3_{sfx}_A_trust.csv", index=False, float_format="%.4g")
    sweep.to_csv(RESULTS / f"stage3_{sfx}_A_gate_sweep.csv", index=False, float_format="%.4g")
    tune.to_csv(RESULTS / f"stage3_{sfx}_B_tuning.csv", index=False, float_format="%.4g")
    chosen.to_csv(RESULTS / f"stage3_{sfx}_B_chosen.csv", index=False, float_format="%.4g")
    table.to_csv(RESULTS / f"stage3_{sfx}_detectors.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.3g}")
    print("\n=== A: residual as trust signal ===\n" + a.to_string(index=False))
    print("\n=== A: gate sweep (thresholds = val quantiles) ===\n" + sweep[sweep.dataset.str.startswith("test")].to_string(index=False))
    print("\n=== B: chosen probes ===\n" + chosen.to_string(index=False))
    print("\n=== detectors on test subsets ===\n" + table.to_string(index=False))


def stage_compare(args):
    """Merge the per-model detector tables into stage3_comparison.csv."""
    t = pd.concat([pd.read_csv(RESULTS / f"stage3_{m}_detectors.csv") for m in args.models])
    t["FN"] = t["FN"].astype(int)
    t["FP"] = t["FP"].astype(int)
    cols = ["dataset", "label", "method", "model", "precision", "recall", "F1", "PR-AUC", "FP", "FN", "solver share"]
    t = t[cols].sort_values(["dataset", "label", "method", "model"])
    t.to_csv(RESULTS / "stage3_comparison.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:.3g}")
    print(t.to_string(index=False))


def main():
    """CLI: run one stage."""
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["extract", "analyze", "compare"])
    p.add_argument("--model", default="original")
    p.add_argument("--with-datakit", action="store_true")
    p.add_argument("--models", nargs="+", default=["original", "replay"])
    args = p.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
