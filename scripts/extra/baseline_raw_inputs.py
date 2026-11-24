"""Part 1: non-GridFM baseline that predicts y_voltage / y_overload directly from the raw model inputs.

Features (scripts/extra/common.py raw_features): per-bus Pd/Qd, non-slack generator Pg, generator Vm setpoints,
total load, branch and generator outage indicators. No GENCO embeddings or predictions.

Samples: exactly the frozen probes' split (scripts/stage3_detectors.py with --with-datakit; results/stage3_mapping.csv):
  train = pf_probe + dk_probe (4,000), val = pf_val + dk_val (2,000). N-2 is only used for the final evaluation.
Selection rule = the probes' rule: hyperparameters by val PR-AUC, decision threshold maximizing val F1
(stage3_detectors.best_threshold, imported unchanged).
Models: logistic regression (same C grid and pipeline as the logreg probe) and HistGradientBoostingClassifier.

Outputs: results/extra/baseline_raw_inputs.csv (test metrics), baseline_raw_inputs_tuning.csv,
         baseline_raw_inputs_comparison.csv (new baseline next to the existing numbers read from
         results/final_n2_detectors.csv, scope "all"), baseline_raw_inputs_log.txt via tee.

Usage: venv/Scripts/python.exe scripts/extra/baseline_raw_inputs.py
"""

import json
import warnings

import common  # noqa: F401  (sets sys.dont_write_bytecode and sys.path first)
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import EXTRA, LABELS, RESULTS, S3_RAW, SEED, TESTS, raw_features
from phase5_analysis import confusion, true_labels_from_raw
from stage3_detectors import best_threshold, safe_auc

GRIDS = {
    "logreg": [dict(C=c) for c in [1e-3, 1e-2, 1e-1, 1]],
    "hgb": [dict(learning_rate=lr, max_leaf_nodes=m) for lr in [0.05, 0.1] for m in [15, 31]],
}


def make_model(kind, hp):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], max_iter=5000, class_weight="balanced"))
    return HistGradientBoostingClassifier(max_iter=300, early_stopping=False, random_state=SEED, **hp)


def split(roles):
    m = pd.read_csv(RESULTS / "stage3_mapping.csv")
    ids = np.sort(m[m.role.isin(roles)].s3_id.values)
    X = raw_features(S3_RAW, ids)
    y = true_labels_from_raw(S3_RAW).loc[X.index, LABELS]
    return X, y


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)
    EXTRA.mkdir(parents=True, exist_ok=True)
    Xtr, ytr = split(["pf_probe", "dk_probe"])
    Xva, yva = split(["pf_val", "dk_val"])
    print(f"train {len(Xtr)} (pf_probe + dk_probe), val {len(Xva)} (pf_val + dk_val), {Xtr.shape[1]} raw features")
    print(f"train rates: voltage {ytr.y_voltage.mean():.3f}, overload {ytr.y_overload.mean():.3f}")
    tests = {}
    for name, (raw, _) in TESTS.items():
        X = raw_features(raw)
        tests[name] = (X[Xtr.columns], true_labels_from_raw(raw).loc[X.index, LABELS])

    tune, rows, chosen = [], [], []
    for kind, grid in GRIDS.items():
        for lab in LABELS:
            best = None
            for hp in grid:
                p = make_model(kind, hp).fit(Xtr, ytr[lab]).predict_proba(Xva)[:, 1]
                ap = average_precision_score(yva[lab], p)
                tune.append({"model": kind, "label": lab, "hp": json.dumps(hp), "val PR-AUC": ap})
                if best is None or ap > best[0]:
                    best = (ap, hp)
            clf = make_model(kind, best[1]).fit(Xtr, ytr[lab])
            thr = best_threshold(yva[lab].values, clf.predict_proba(Xva)[:, 1])
            chosen.append({"model": kind, "label": lab, "hp": json.dumps(best[1]), "val PR-AUC": best[0],
                           "decision threshold": thr})
            for name, (X, y) in tests.items():
                p = clf.predict_proba(X)[:, 1]
                c = confusion(y[lab].values, p >= thr)
                rows.append({"dataset": name, "label": lab, "method": f"raw-input {kind}", **c,
                             "false alarms": c["FP"], "misses": c["FN"],
                             "PR-AUC": safe_auc(average_precision_score, y[lab].values, p)})
    res = pd.DataFrame(rows)
    res.to_csv(EXTRA / "baseline_raw_inputs.csv", index=False, float_format="%.4g")
    pd.DataFrame(tune).to_csv(EXTRA / "baseline_raw_inputs_tuning.csv", index=False, float_format="%.4g")
    pd.DataFrame(chosen).to_csv(EXTRA / "baseline_raw_inputs_chosen.csv", index=False, float_format="%.4g")

    # existing numbers, read (not recomputed) from the frozen final test output
    ex = pd.read_csv(RESULTS / "final_n2_detectors.csv")
    ex = ex[(ex.scope == "all") & ex.method.isin(["baseline PtC", "A gated PtC (solver)", "B logreg", "B mlp"])]
    names = {"baseline PtC": "limit check (PtC)", "A gated PtC (solver)": "residual routing",
             "B logreg": "probe logreg", "B mlp": "probe MLP"}
    ex = ex.assign(method=ex.method.map(names) + " [" + ex.model + "]")
    new = res.assign(model="none", **{"solver share": 0.0})
    keep = ["dataset", "label", "method", "precision", "recall", "F1", "FP", "FN", "PR-AUC", "solver share"]
    comp = pd.concat([ex[keep], new[keep]], ignore_index=True)
    order = {d: i for i, d in enumerate(["test_datakit", "test_pfdelta", "test_n2"])}
    comp = comp.sort_values(["dataset", "label", "method"], key=lambda s: s.map(order) if s.name == "dataset" else s)
    comp.to_csv(EXTRA / "baseline_raw_inputs_comparison.csv", index=False, float_format="%.4g")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n=== chosen (val) ===\n" + pd.DataFrame(chosen).to_string(index=False))
    print("\n=== comparison (test) ===\n" + comp.to_string(index=False))


if __name__ == "__main__":
    main()
