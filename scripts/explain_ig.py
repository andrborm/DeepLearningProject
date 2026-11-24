"""Explainability for the replay-model embedding probes (approach B): Integrated Gradients, no retraining.

The frozen probes (data/stage3_probes/replay_<probe>_<label>.joblib, sklearn) are re-implemented in torch, and their
logit is checked against sklearn. Input = the 14 x 384 frozen bus embeddings of one sample.
IG baseline = per-bus mean embedding of the probe-TRAIN samples (pf_probe + dk_probe) that are negative for that
label ("typical normal grid"). 64-step trapezoid path; completeness (sum of attributions = logit(x) - logit(baseline))
is checked per sample.

Attribution of an element:
  bus     sum of the IG values over its 384 embedding dims (signed; positive = pushes towards the alarm)
  branch  bus attribution of its two end buses, summed (the probes see only bus embeddings)

Localization on correct alarms (true positive of the probe):
  voltage   top-3 buses by attribution contain a bus with a true Vm violation
  overload  top-3 in-service branches by attribution contain a truly overloaded branch
References with the same top-3 rule: chance (random top-3), per-bus residual |PBE| (branch = sum of its end buses),
and the predicted-state ranking (predicted voltage violation / thermal excess, i.e. what PtC would point at).

Stages:
  localize   -> results/explain_ig_localization.csv, results/explain_ig_samples.csv
  examples   -> results/figures/fig_explain_grid_<k>.png (case14 grid with residual + attributions)

Usage:
  python scripts/explain_ig.py localize
  python scripts/explain_ig.py examples
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from finetune_pipeline import EVAL_SETS
from stage3_detectors import PRED, PROBES, blocks

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
MODEL = "replay"
TOL = 1e-6
STEPS = 64
DATASETS = {"test_pfdelta": EVAL_SETS["pfdelta"][0], "test_n2": ROOT / "data/pfdelta_n2_test1000",
            "test_datakit": EVAL_SETS["datakit"][0]}
LABELS = ["y_voltage", "y_overload"]
PROBE_KINDS = ["logreg", "mlp"]


# --------------------------------------------------------------------------- probe in torch
# frozen sklearn probe (scaler + logreg / MLP) rebuilt in torch, so gradients w.r.t. embeddings exist
class TorchProbe:
    def __init__(self, path):
        d = joblib.load(path)
        self.kind, self.threshold = d["features"], d["threshold"]
        scaler, clf = d["clf"][0], d["clf"][-1]
        self.mu = torch.tensor(scaler.mean_, dtype=torch.float64)
        self.sd = torch.tensor(scaler.scale_, dtype=torch.float64)
        if hasattr(clf, "coefs_"):
            self.layers = [(torch.tensor(W, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
                           for W, b in zip(clf.coefs_, clf.intercepts_)]
        else:
            self.layers = [(torch.tensor(clf.coef_.T, dtype=torch.float64), torch.tensor(clf.intercept_, dtype=torch.float64))]
        self.sk = d["clf"]

    def features(self, E):  # E: (..., 14, 384)
        if self.kind == "pooled":
            return torch.cat([E.mean(-2), E.amax(-2), E.amin(-2)], dim=-1)
        return E.flatten(-2)

    def logit(self, E):
        """Probe logit (before the sigmoid) for embeddings E of shape (..., 14, 384)."""
        h = (self.features(E) - self.mu) / self.sd
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h[..., 0]


def integrated_gradients(probe, E, base):
    """E, base: (14, 384) -> IG (14, 384), with trapezoid rule over STEPS segments."""
    alphas = torch.linspace(0, 1, STEPS + 1, dtype=torch.float64)[:, None, None]
    path = (base + alphas * (E - base)).requires_grad_(True)
    probe.logit(path).sum().backward()
    g = path.grad
    avg = (g[:-1] + g[1:]).sum(0) / (2 * STEPS)
    return (E - base) * avg


# --------------------------------------------------------------------------- data
def embeddings(name):
    """Scenario ids and (n, 14, 384) replay bus embeddings of one block."""
    e = pd.read_parquet(PRED / MODEL / name / "case14_ieee_bus_embeddings.parquet").sort_values(["scenario", "bus"])
    scen = np.sort(e.scenario.unique())
    return scen, e.drop(columns=["scenario", "bus"]).to_numpy(np.float64).reshape(len(scen), 14, -1)


def baseline_embeddings(label):
    """Per-bus mean embedding of label-negative probe-TRAIN samples."""
    from phase5_analysis import true_labels_from_raw

    from stage3_detectors import S3
    truth = true_labels_from_raw(S3 / "case14_ieee/raw")
    parts = []
    for name in blocks():
        if name.startswith(("pf_probe", "dk_probe")):
            scen, E = embeddings(name)
            parts.append(E[~truth.loc[scen, label].values])
    return np.concatenate(parts).mean(0)


def element_truth(raw):
    """Per-bus voltage-violation and per-branch overload flags from the solved power flow."""
    bus = pd.read_parquet(raw / "bus_data.parquet")
    br = pd.read_parquet(raw / "branch_data.parquet")
    bus["v_violation"] = np.maximum(bus.min_vm_pu - bus.Vm, bus.Vm - bus.max_vm_pu)
    bus["violating"] = bus.v_violation > TOL
    br["loading"] = np.maximum(np.hypot(br.pf, br.qf), np.hypot(br.pt, br.qt)) / br.rate_a
    br["overloaded"] = (br.br_status == 1) & (br.loading > 1 + TOL)
    return bus, br


def topk_hit(score, true_mask, k=3):
    """Does the top k of `score` contain a true element?"""
    order = np.argsort(-score, kind="stable")[:k]
    return bool(true_mask[order].any())


def chance_hit(n_true, n_total, k=3):
    """Probability that a random top k holds at least one of n_true true elements."""
    from math import comb
    return 1 - comb(n_total - n_true, k) / comb(n_total, k)


# --------------------------------------------------------------------------- localize
def stage_localize(args):
    """IG on every correct probe alarm; top-3 localization vs residual, PtC and chance."""
    probes = {(p, lab): TorchProbe(PROBES / f"{MODEL}_{p}_{lab}.joblib") for p in PROBE_KINDS for lab in LABELS}
    bases = {lab: torch.tensor(baseline_embeddings(lab)) for lab in LABELS}
    rows, summary = [], []
    for ds, data in DATASETS.items():
        scen, E = embeddings(ds)
        samples = pd.read_csv(RESULTS / f"final_n2_samples_{MODEL}_{ds}.csv", index_col=0).loc[scen]
        tb, tbr = element_truth(data / "case14_ieee/raw")
        pb = pd.read_parquet(PRED / MODEL / ds / "case14_ieee_predictions.parquet").sort_values(["scenario", "bus"])
        pbr = pd.read_parquet(PRED / MODEL / ds / "case14_ieee_branch_predictions.parquet")
        pb["v_viol_pred"] = np.maximum(pb.Vm_min - pb.Vm_pred, pb.Vm_pred - pb.Vm_max)
        pb.loc[pb.REF == 1, "v_viol_pred"] = -np.inf  # slack Vm is a known setpoint
        pbr["pair"] = [tuple(sorted(x)) for x in zip(pbr.from_bus.astype(int), pbr.to_bus.astype(int))]
        texc = pbr.groupby(["scenario", "pair"]).thermal_excess.max()
        tb_g, tbr_g, pb_g = tb.groupby("scenario"), tbr.groupby("scenario"), pb.groupby("scenario")
        for (p, lab), probe in probes.items():
            # sklearn agreement check on the whole dataset
            with torch.no_grad():
                lt = probe.logit(torch.tensor(E)).numpy()
            pk = probe.sk.predict_proba(np.asarray(probe.features(torch.tensor(E)).numpy()))[:, 1]
            assert np.allclose(1 / (1 + np.exp(-lt)), pk, atol=1e-6), (p, lab)
            # correct alarms only: the probe fires and the truth is positive
            tp = samples.index[samples[lab] & samples[f"{lab}_{p}"]]
            for s in tp:
                i = int(np.searchsorted(scen, s))
                x = torch.tensor(E[i])
                ig = integrated_gradients(probe, x, bases[lab])
                with torch.no_grad():
                    gap = float(probe.logit(x) - probe.logit(bases[lab]))
                bus_attr = ig.sum(1).numpy()
                b, br, pbs = tb_g.get_group(s).sort_values("bus"), tbr_g.get_group(s).sort_values("idx"), pb_g.get_group(s)
                pbe = pbs.PBE.abs().to_numpy()
                row = {"dataset": ds, "probe": p, "label": lab, "scenario": s, "logit gap": gap,
                       "completeness err": abs(bus_attr.sum() - gap),
                       "top3 buses": ",".join(map(str, np.argsort(-bus_attr, kind="stable")[:3]))}
                # voltage: rank buses; overload: rank in-service branches by the sum of their end buses
                if lab == "y_voltage":
                    true = b.violating.to_numpy()
                    row.update({"n true": int(true.sum()), "n candidates": 14,
                                "IG hit": topk_hit(bus_attr, true), "residual hit": topk_hit(pbe, true),
                                "PtC hit": topk_hit(pbs.v_viol_pred.to_numpy(), true),
                                "chance": chance_hit(int(true.sum()), 14)})
                else:
                    live = br[br.br_status == 1]
                    fb, tb_ = live.from_bus.astype(int).to_numpy(), live.to_bus.astype(int).to_numpy()
                    true = live.overloaded.to_numpy()
                    pred = np.array([texc.get((s, tuple(sorted((f, t)))), -np.inf) for f, t in zip(fb, tb_)])
                    row.update({"n true": int(true.sum()), "n candidates": len(live),
                                "IG hit": topk_hit(bus_attr[fb] + bus_attr[tb_], true),
                                "residual hit": topk_hit(pbe[fb] + pbe[tb_], true),
                                "PtC hit": topk_hit(pred, true),
                                "chance": chance_hit(int(true.sum()), len(live)),
                                "IG bus hit (endpoint)": topk_hit(bus_attr, ends := np.isin(np.arange(14), np.r_[fb[true], tb_[true]])),
                                "residual bus hit (endpoint)": topk_hit(pbe, ends),
                                "chance (endpoint)": chance_hit(int(ends.sum()), 14)})
                rows.append(row)
            print(f"{ds} {p} {lab}: {len(tp)} correct alarms")
    df = pd.DataFrame(rows)
    df["rel completeness err"] = df["completeness err"] / df["logit gap"].abs().clip(lower=1)
    assert (df["rel completeness err"] < 0.02).all(), df["rel completeness err"].max()  # max/min pooling + ReLU: piecewise path
    df.to_csv(RESULTS / "explain_ig_samples.csv", index=False, float_format="%.5g")
    agg = df.groupby(["dataset", "label", "probe"]).agg(
        n=("IG hit", "size"), IG=("IG hit", "mean"), residual=("residual hit", "mean"), PtC=("PtC hit", "mean"),
        chance=("chance", "mean"), mean_true=("n true", "mean"),
        IG_endpoint_bus=("IG bus hit (endpoint)", "mean"), residual_endpoint_bus=("residual bus hit (endpoint)", "mean"),
        chance_endpoint_bus=("chance (endpoint)", "mean")).reset_index()
    agg.to_csv(RESULTS / "explain_ig_localization.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"completeness: max error {df['completeness err'].max():.2e} logit, max relative {df['rel completeness err'].max():.2%}")


# --------------------------------------------------------------------------- figures
# IEEE 14-bus one-line layout (bus index 0..13 = bus 1..14), hand-placed
POS = np.array([[0.0, 3.0], [0.6, 0.6], [3.0, 0.0], [3.2, 1.6], [1.3, 2.0], [1.2, 3.8], [4.6, 2.4], [5.6, 2.4],
                [4.2, 3.4], [3.2, 4.0], [2.2, 4.4], [0.4, 5.2], [1.6, 5.6], [3.4, 5.4]])
GEN_BUSES = {0, 1, 2, 5, 7}


def draw_grid(ax, bus_val, br, title, cmap, vlim, bus_true=None, label_fmt=None, colorbar_label=""):
    """Draw case14: buses coloured by bus_val; outaged, overloaded and normal branches styled apart."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize

    norm = TwoSlopeNorm(0, -vlim, vlim) if cmap == "RdBu_r" else Normalize(0, vlim)
    for _, r in br.iterrows():
        f, t = int(r.from_bus), int(r.to_bus)
        xs, ys = POS[[f, t], 0], POS[[f, t], 1]
        if r.br_status == 0:
            ax.plot(xs, ys, ls=(0, (3, 3)), color="#9a9a9a", lw=1.4, zorder=1)
            ax.text(xs.mean(), ys.mean(), "✕", ha="center", va="center", color="#555", fontsize=10, zorder=2)
        elif r.overloaded:
            ax.plot(xs, ys, color="#d62728", lw=4.5, zorder=1, solid_capstyle="round")
            ax.text(xs.mean(), ys.mean(), f"{r.loading:.0%}", fontsize=7, color="#d62728", ha="center", va="bottom",
                    zorder=3, bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
        else:
            ax.plot(xs, ys, color="#606060", lw=1.0 + 2.0 * min(r.loading, 1.0), alpha=0.75, zorder=1)
    sc = ax.scatter(POS[:, 0], POS[:, 1], c=bus_val, cmap=cmap, norm=norm, s=330, zorder=4,
                    edgecolors=["#d62728" if (bus_true is not None and bus_true[i]) else "#333" for i in range(14)],
                    linewidths=[3.0 if (bus_true is not None and bus_true[i]) else 0.8 for i in range(14)],
                    marker="o")
    for i in range(14):
        ax.text(POS[i, 0], POS[i, 1], str(i + 1) + ("G" if i in GEN_BUSES else ""), ha="center", va="center",
                fontsize=7, zorder=5, color="black")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.6, 6.2)
    ax.set_ylim(-0.6, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label(colorbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=7)


def pick_examples():
    """(name, dataset, scenario, label to explain). Chosen by rule, not by looking at the explanations."""
    s2 = pd.read_csv(RESULTS / f"final_n2_samples_{MODEL}_test_n2.csv", index_col=0)
    ig = pd.read_csv(RESULTS / "explain_ig_samples.csv")
    ex = []
    normal = s2[~s2.y_voltage & ~s2.y_overload & ~s2.y_voltage_logreg & ~s2.y_overload_logreg]
    ex.append(("normal", "test_n2", int(normal.pbe_mean.sub(normal.pbe_mean.median()).abs().idxmin()), "y_voltage"))
    v = ig[(ig.dataset == "test_n2") & (ig.label == "y_voltage") & (ig.probe == "logreg")]
    v = v[s2.loc[v.scenario, "y_overload"].values == False]
    ex.append(("voltage violation", "test_n2", int(v.scenario.iloc[len(v) // 2]), "y_voltage"))
    o = ig[(ig.dataset == "test_n2") & (ig.label == "y_overload") & (ig.probe == "logreg")]
    ex.append(("overload", "test_n2", int(o.scenario.iloc[len(o) // 2]), "y_overload"))
    ex.append(("datakit sample 438", "test_datakit", 438, "y_overload"))
    return ex


def stage_examples(args):
    """Grid figure: residual and IG attribution for four rule-picked example scenarios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    bases = {lab: torch.tensor(baseline_embeddings(lab)) for lab in LABELS}
    ex = pick_examples()
    fig, axes = plt.subplots(len(ex), 2, figsize=(9.5, 4.3 * len(ex)))
    rows = []
    for k, (name, ds, s, lab) in enumerate(ex):
        probe = TorchProbe(PROBES / f"{MODEL}_logreg_{lab}.joblib")
        scen, E = embeddings(ds)
        i = int(np.searchsorted(scen, s))
        x = torch.tensor(E[i])
        attr = integrated_gradients(probe, x, bases[lab]).sum(1).numpy()
        with torch.no_grad():
            prob = float(torch.sigmoid(probe.logit(x)))
        tb, tbr = element_truth(DATASETS[ds] / "case14_ieee/raw")
        b, br = tb[tb.scenario == s].sort_values("bus"), tbr[tbr.scenario == s].sort_values("idx")
        pb = pd.read_parquet(PRED / MODEL / ds / "case14_ieee_predictions.parquet")
        pbe = pb[pb.scenario == s].sort_values("bus").PBE.abs().to_numpy()
        smp = pd.read_csv(RESULTS / f"final_n2_samples_{MODEL}_{ds}.csv", index_col=0).loc[s]
        gen = pd.read_parquet(DATASETS[ds] / "case14_ieee/raw/gen_data.parquet")
        gen_out = gen[(gen.scenario == s) & (gen.in_service == 0)].bus.astype(int) + 1
        outs = ", ".join([f"{int(f) + 1}–{int(t) + 1}" for f, t in br[br.br_status == 0][["from_bus", "to_bus"]].values]
                         + [f"gen @ bus {g}" for g in gen_out])
        head = (f"{name} ({ds.replace('test_', '')} #{s}; out: {outs or 'none'})\n"
                f"truth V={bool(smp.y_voltage)} O={bool(smp.y_overload)}, PtC V={bool(smp.y_voltage_ptc)} "
                f"O={bool(smp.y_overload_ptc)}")
        vmax = max(pbe.max(), 1e-3)
        draw_grid(axes[k, 0], pbe, br, head + f"\nper-bus residual |PBE| (mean {pbe.mean():.3g} MVA)", "viridis", vmax,
                  b.violating.to_numpy(), colorbar_label="|PBE| (MVA)")
        lim = max(np.abs(attr).max(), 1e-6)
        top = np.argsort(-attr)[:3] + 1
        draw_grid(axes[k, 1], attr, br, f"IG attribution, {lab.replace('y_', '')} probe (logreg)\n"
                  f"p(alarm) = {prob:.2f}, top-3 buses {', '.join(map(str, top))}", "RdBu_r", lim,
                  b.violating.to_numpy(), colorbar_label="attribution (logit)")
        rows.append({"example": name, "dataset": ds, "scenario": s, "label": lab, "p_alarm": prob,
                     "top3 buses (1-based)": " ".join(map(str, top)),
                     "violating buses (1-based)": " ".join(map(str, np.flatnonzero(b.violating) + 1)),
                     "overloaded branches": " ".join(f"{int(f) + 1}-{int(t) + 1}" for f, t in
                                                     br[br.overloaded][["from_bus", "to_bus"]].values)})
    fig.text(0.5, 0.005, "red ring = bus with a true voltage violation · thick red line = overloaded branch (true loading) · "
             "dashed ✕ = outaged branch · G = generator bus", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.012, 1, 1))
    fig.savefig(FIG / "fig_explain_grid_examples.png", dpi=150)
    pd.DataFrame(rows).to_csv(RESULTS / "explain_ig_examples.csv", index=False, float_format="%.4g")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    """CLI: run one stage."""
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["localize", "examples"])
    args = p.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
