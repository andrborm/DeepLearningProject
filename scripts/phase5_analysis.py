"""Phase 5 groundwork on the base model's 1000 test predictions.

Needs the output of graphkit `predict --get_embeddings` in data/predictions/base_n1000
(command in README / SETUP_LOG) and results/inference_per_sample.csv from run_inference.py.

  1. embeddings: bus embeddings -> .npy with sample ids, mean-pooled graph embeddings, PCA
  2. predict-then-check: y_voltage / y_overload from predicted state vs true labels
     (confusion matrix, precision, recall) + check that thermal_excess maps onto y_overload
  3. robust error summary: median / p95 / p99 next to RMSE, per bus type and outage type
  4. top-5 % hard cases
  5. three presentation figures in results/figures/

Usage:
  python scripts/phase5_analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data/predictions/base_n1000"
SUBSET_RAW = ROOT / "data/case14_test1000/case14_ieee/raw"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
EMB_OUT = ROOT / "data/embeddings"
TOL = 1e-6

TARGETS = {"PQ": ["Vm", "Va"], "PV": ["Va", "Qg"], "REF": ["Vm", "Qg", "Pg"]}

# Reference palette (dataviz skill, light mode); validated all-pairs for the 3 categorical slots
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#b8b7b2"  # recessive "normal" class, not a series hue
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})


# --------------------------------------------------------------------------- labels
def true_labels_from_raw():
    """Ground-truth labels from the raw solved power flow (both branch ends, in-service only)."""
    bus = pd.read_parquet(SUBSET_RAW / "bus_data.parquet")
    br = pd.read_parquet(SUBSET_RAW / "branch_data.parquet")
    br = br[br.br_status == 1]
    loading = np.maximum(np.hypot(br.pf, br.qf), np.hypot(br.pt, br.qt)) / br.rate_a
    return pd.DataFrame({
        "y_voltage": ((bus.Vm < bus.min_vm_pu - TOL) | (bus.Vm > bus.max_vm_pu + TOL)).groupby(bus.scenario).any(),
        "y_overload": (loading > 1 + TOL).groupby(br.scenario).any(),
        "max_loading": loading.groupby(br.scenario).max(),
        "min_vm": bus.Vm.groupby(bus.scenario).min(),
        "max_vm": bus.Vm.groupby(bus.scenario).max(),
    })


def labels_from_predictions(bus, branch, which, vm_tol=TOL, slack_setpoint=False):
    """y_voltage / y_overload from graphkit's predict output. which = 'pred' or 'target'.

    vm_tol:         a voltage violation needs Vm beyond the limit by more than vm_tol
    slack_setpoint: use the true slack Vm (an operator setpoint, known in practice) instead of the
                    model's prediction; graphkit masks it, and it sits exactly on the 1.06 limit
    """
    vm = bus["Vm_pred" if which == "pred" else "Vm_target"]
    if slack_setpoint:
        vm = vm.where(bus.REF == 0, bus.Vm_target)
    excess = branch["thermal_excess" if which == "pred" else "thermal_excess_target"]
    return pd.DataFrame({
        "y_voltage": ((vm < bus.Vm_min - vm_tol) | (vm > bus.Vm_max + vm_tol)).groupby(bus.scenario).any(),
        "y_overload": (excess > TOL).groupby(branch.scenario).any(),
    })


def confusion(true, pred):
    tp = int((true & pred).sum()); fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum()); tn = int((~true & ~pred).sum())
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "F1": 2 * tp / (2 * tp + fp + fn) if tp else float("nan"),
    }


# --------------------------------------------------------------------------- errors
def long_errors(bus, outage):
    bus = bus.assign(bus_type=np.select([bus.REF == 1, bus.PV == 1], ["REF", "PV"], "PQ"))
    rows = []
    for bt, variables in TARGETS.items():
        b = bus[bus.bus_type == bt]
        for v in variables:
            rows.append(pd.DataFrame({"scenario": b.scenario, "bus": b.bus, "bus_type": bt, "variable": v,
                                      "abs_err": (b[f"{v}_pred"] - b[f"{v}_target"]).abs()}))
    long = pd.concat(rows, ignore_index=True)
    long["outage_type"] = long.scenario.map(outage)
    return long


def robust(g):
    e = g.abs_err
    return pd.Series({"n": len(e), "median": e.median(), "p95": e.quantile(0.95), "p99": e.quantile(0.99),
                      "max": e.max(), "MAE": e.mean(), "RMSE": np.sqrt((e ** 2).mean())})


# --------------------------------------------------------------------------- PCA
def pca(x, k=2):
    xc = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    explained = s ** 2 / (s ** 2).sum()
    return xc @ vt[:k].T, explained


# --------------------------------------------------------------------------- figures
def fig_error_distribution(per, path):
    s = per.overall_score
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bins = np.logspace(np.log10(s.min() * 0.9), np.log10(s.max() * 1.1), 45)
    ax.hist(s, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xscale("log")
    top = ax.get_ylim()[1]
    for (q, label), y in zip([(0.5, "median"), (0.95, "95th percentile"), (0.99, "99th percentile")],
                             [0.97, 0.80, 0.63]):
        v = s.quantile(q)
        ax.axvline(v, color=INK_2, linestyle=(0, (4, 3)), linewidth=1.2)
        left = label == "median"  # tallest bars sit just right of the median line
        ax.text(v / 1.06 if left else v * 1.06, top * y, f"{label} = {v:.2f}", va="top",
                ha="right" if left else "left", fontsize=11, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    x438 = s.loc[438]
    ax.annotate(f"sample 438\nbranch 0 (bus 0–1) out\nscore {x438:.1f}",
                xy=(x438, 1), xytext=(x438 / 3.2, ax.get_ylim()[1] * 0.45),
                fontsize=11, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2))
    ax.set_xlabel("Overall error score per sample (log scale; 1.0 = subset-wide RMSE level)")
    ax.set_ylabel("Number of samples")
    ax.set_title("GENCO base on 1,000 case14 N-1 test samples: errors are heavy-tailed", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_error_by_outage(long, path):
    panels = [("Vm", "PQ", "Vm at PQ buses (p.u.)"), ("Va", "PQ", "Va at PQ buses (rad)"),
              ("Qg", "PV", "Qg at PV buses (MVAr)"), ("Pg", "REF", "Pg at slack bus (MW)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.6))
    n_zero = 0
    for ax, (v, bt, title) in zip(axes, panels):
        g = long[(long.variable == v) & (long.bus_type == bt)]
        n_zero += int((g.abs_err == 0).sum())
        # exact-zero errors (float32 ties) cannot sit on a log axis; they are counted in the caption
        data = [g[(g.outage_type == t) & (g.abs_err > 0)].abs_err for t in ["branch", "gen"]]
        bp = ax.boxplot(data, whis=(5, 95), widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_2,
                                        markeredgecolor="none", alpha=0.35),
                        medianprops=dict(color=INK, linewidth=2))
        for patch, color in zip(bp["boxes"], [BLUE, ORANGE]):
            patch.set(facecolor=color, edgecolor=SURFACE, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xticks([1, 2], [f"branch\n(n={data[0].size:,})", f"generator\n(n={data[1].size:,})"])
        ax.set_title(title, fontsize=13, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Absolute error (log scale)")
    fig.suptitle("Absolute error by outage type (box = 25–75 %, whiskers = 5–95 %, dots beyond)",
                 x=0.01, ha="left", fontsize=15)
    fig.text(0.01, 0.01, f"GENCO base, 1,000 case14 N-1 test samples; one point per bus. "
             f"{n_zero} exact-zero errors omitted from the log axis.", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_pca(z, explained, per, labels, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    xlab = f"PC1 ({explained[0]:.0%} var.)"
    ylab = f"PC2 ({explained[1]:.0%} var.)"

    # (a) overall error, sequential blue on log scale
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    order = np.argsort(per.overall_score.values)
    sc = axes[0].scatter(z[order, 0], z[order, 1], c=per.overall_score.values[order], cmap=cmap,
                         norm=LogNorm(), s=22, edgecolors=SURFACE, linewidths=0.4)
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)
    cb.set_label("overall error score (log)")
    i438 = per.index.get_loc(438)
    axes[0].annotate("438", xy=z[i438], xytext=(8, 8), textcoords="offset points", fontsize=11)
    axes[0].set_title("(a) overall error", loc="left")

    # (b) outage type
    for t, color, marker in [("branch", BLUE, "o"), ("gen", ORANGE, "s")]:
        m = (per.outage_type == t).values
        axes[1].scatter(z[m, 0], z[m, 1], s=22, color=color, marker=marker, edgecolors=SURFACE,
                        linewidths=0.4, label=f"{'generator' if t == 'gen' else t} outage (n={m.sum()})")
    axes[1].legend(loc="best", frameon=False)
    axes[1].set_title("(b) outage type", loc="left")

    # (c) true violation labels (multi-label -> 4 display groups)
    cls = np.select([labels.y_voltage & labels.y_overload, labels.y_voltage, labels.y_overload],
                    ["both", "voltage", "overload"], "normal")
    styles = [("normal", GRAY, "o", 14), ("voltage", BLUE, "^", 46),
              ("overload", ORANGE, "s", 40), ("both", AQUA, "D", 46)]
    for name, color, marker, size in styles:
        m = cls == name
        if m.any():
            axes[2].scatter(z[m, 0], z[m, 1], s=size, color=color, marker=marker, edgecolors=SURFACE,
                            linewidths=0.5, label=f"{name} (n={m.sum()})", zorder=3 if name != "normal" else 2)
    axes[2].legend(loc="best", frameon=False)
    axes[2].set_title("(c) true violation (voltage / overload)", loc="left")

    for ax in axes:
        ax.set_xlabel(xlab)
    axes[0].set_ylabel(ylab)
    fig.suptitle("PCA of mean-pooled bus embeddings (384-d), base model, 1,000 case14 test samples",
                 x=0.01, ha="left", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    FIGS.mkdir(parents=True, exist_ok=True)
    EMB_OUT.mkdir(parents=True, exist_ok=True)

    bus = pd.read_parquet(PRED / "case14_ieee_predictions.parquet")
    branch = pd.read_parquet(PRED / "case14_ieee_branch_predictions.parquet")
    emb = pd.read_parquet(PRED / "case14_ieee_bus_embeddings.parquet")
    per = pd.read_csv(RESULTS / "inference_per_sample.csv", index_col="sample_id")

    # predict run must reproduce the evaluate run used for inference_per_sample.csv
    ev = find_run("base_n1000") / "artifacts/test/case14_ieee_predictions.parquet"
    ev_bus = pd.read_parquet(ev)
    diff = np.abs(ev_bus[["Vm_pred", "Va_pred", "Pg_pred", "Qg_pred"]].values
                  - bus[["Vm_pred", "Va_pred", "Pg_pred", "Qg_pred"]].values).max()
    print(f"predict vs evaluate predictions, max abs difference: {diff:.3g}  (run {ev.parents[2].name})")

    # ---- 1. embeddings
    section("1. Embeddings")
    emb = emb.sort_values(["scenario", "bus"])
    cols = [c for c in emb.columns if c.startswith("emb_")]
    node = emb[cols].to_numpy(np.float32).reshape(emb.scenario.nunique(), -1, len(cols))
    ids = np.sort(emb.scenario.unique())
    graph = node.mean(axis=1)
    np.save(EMB_OUT / "base_bus_embeddings.npy", node)
    np.save(EMB_OUT / "base_graph_embeddings_meanpool.npy", graph)
    np.save(EMB_OUT / "sample_ids.npy", ids)
    print(f"bus embeddings {node.shape} (samples, buses, dim); graph (mean-pooled) {graph.shape}")
    print(f"embedding = final-layer h_bus fed into the Vm/Va head; dim = hidden_size 48 x 8 heads = {len(cols)}")
    print(f"value range {node.min():.2f}..{node.max():.2f}, mean {node.mean():.3f}, std {node.std():.3f}")
    per_dim_std = graph.std(axis=0)
    print(f"graph-level dims with std < 1e-6 (dead): {(per_dim_std < 1e-6).sum()} / {graph.shape[1]}")
    z, explained = pca(graph)
    print("PCA explained variance, first 5 PCs:", np.round(explained[:5], 3).tolist(),
          f"| PCs for 90 %: {int(np.searchsorted(np.cumsum(explained), 0.9)) + 1}")

    # ---- 2. predict-then-check
    section("2. Predict-then-check: labels from predicted state vs truth")
    raw = true_labels_from_raw()
    tgt = labels_from_predictions(bus, branch, "target")
    prd = labels_from_predictions(bus, branch, "pred")
    agree = {k: int((raw[k] == tgt[k]).sum()) for k in ["y_voltage", "y_overload"]}
    print(f"graphkit target labels vs raw-data labels, agreement: {agree} of {len(raw)}")
    mism = raw.index[raw.y_overload != tgt.y_overload]
    if len(mism):
        print("  overload mismatches (raw max_loading):", raw.loc[mism, "max_loading"].round(6).to_dict())
    # thermal_excess vs raw loading, continuous check on every directed edge
    br_raw = pd.read_parquet(SUBSET_RAW / "branch_data.parquet").query("br_status == 1")
    s_raw = pd.concat([
        pd.DataFrame({"scenario": br_raw.scenario, "from_bus": br_raw.from_bus, "to_bus": br_raw.to_bus,
                      "S_raw": np.hypot(br_raw.pf, br_raw.qf), "rate_a_raw": br_raw.rate_a}),
        pd.DataFrame({"scenario": br_raw.scenario, "from_bus": br_raw.to_bus, "to_bus": br_raw.from_bus,
                      "S_raw": np.hypot(br_raw.pt, br_raw.qt), "rate_a_raw": br_raw.rate_a}),
    ])
    j = branch.merge(s_raw, on=["scenario", "from_bus", "to_bus"], how="inner")
    s_tgt = np.hypot(j.Pft_target, j.Qft_target)
    print(f"directed edges matched: {len(j)} of {len(branch)} | max |S_target - S_raw| = "
          f"{np.abs(s_tgt - j.S_raw).max():.3g} MVA | max |rate_a - rate_a_raw| = {np.abs(j.rate_a - j.rate_a_raw).max():.3g}")
    print(f"max |thermal_excess_target - relu(S_raw - rate_a)| = "
          f"{np.abs(j.thermal_excess_target - np.maximum(j.S_raw - j.rate_a_raw, 0)).max():.3g} MVA")

    y = raw.join(prd, rsuffix="_pred")
    rows = {}
    for k in ["y_voltage", "y_overload"]:
        rows[k] = confusion(y[k].values, y[f"{k}_pred"].values)
    normal_true = ~(y.y_voltage | y.y_overload)
    normal_pred = ~(y.y_voltage_pred | y.y_overload_pred)
    rows["any violation"] = confusion(~normal_true.values, ~normal_pred.values)
    cm = pd.DataFrame(rows).T
    print("(a) naive: predicted Vm (incl. predicted slack Vm) vs limits, tolerance 1e-6")
    print(cm.to_string(float_format=lambda v: f"{v:.3f}"))

    # All naive voltage false positives sit at the slack bus: true slack Vm = 1.06 = upper limit in
    # ~99 % of scenarios, and graphkit masks slack Vm, so any positive prediction error flags "over-voltage".
    fp = y.index[~y.y_voltage & y.y_voltage_pred]
    vm_viol = bus[((bus.Vm_pred < bus.Vm_min - TOL) | (bus.Vm_pred > bus.Vm_max + TOL)) & bus.scenario.isin(fp)]
    print(f"\nvoltage false positives: {len(fp)}; bus of the predicted violation: {vm_viol.bus.value_counts().to_dict()}; "
          f"true slack Vm on the 1.06 limit in {(np.abs(bus[bus.REF == 1].Vm_target - 1.06) < 1e-6).mean():.1%} of samples")
    variants = {"(a) naive, tol 1e-6": cm.loc["y_voltage"]}
    for tol in [1e-4, 5e-4, 1e-3, 2e-3]:
        p = labels_from_predictions(bus, branch, "pred", vm_tol=tol)
        variants[f"(b) tolerance {tol:g} p.u."] = pd.Series(confusion(y.y_voltage.values, p.y_voltage.values))
    p = labels_from_predictions(bus, branch, "pred", slack_setpoint=True)
    variants["(c) slack setpoint known, tol 1e-6"] = pd.Series(confusion(y.y_voltage.values, p.y_voltage.values))
    y["y_voltage_pred_slack_known"] = p.y_voltage
    vm_table = pd.DataFrame(variants).T
    print("\ny_voltage under different checking rules:\n" + vm_table.to_string(float_format=lambda v: f"{v:.3f}"))
    vm_table.to_csv(RESULTS / "predict_then_check_voltage_variants.csv", float_format="%.4f")
    rows["y_voltage (slack setpoint known)"] = confusion(y.y_voltage.values, p.y_voltage.values)
    both_true = y.y_voltage | y.y_overload
    rows["any violation (slack setpoint known)"] = confusion(both_true.values, (p.y_voltage | y.y_overload_pred).values)
    cm = pd.DataFrame(rows).T
    # which samples are misclassified, and how close to the limit are they?
    y["vm_margin"] = np.minimum(y.min_vm - 0.94, 1.06 - y.max_vm)  # < 0 = violation
    y["overload_margin"] = 1 - y.max_loading                       # < 0 = overload
    wrong = y[(y.y_voltage != y.y_voltage_pred_slack_known) | (y.y_overload != y.y_overload_pred)]
    print(f"\nmisclassified with rule (c): {len(wrong)}")
    if len(wrong):
        print(wrong[["y_voltage", "y_voltage_pred_slack_known", "vm_margin", "min_vm", "max_vm", "y_overload",
                     "y_overload_pred", "overload_margin"]].to_string(float_format=lambda v: f"{v:.4g}"))
    cm.to_csv(RESULTS / "predict_then_check_confusion.csv", float_format="%.4f")
    y.to_csv(RESULTS / "labels_true_vs_pred_base_n1000.csv", float_format="%.6g")

    # ---- 3. robust error summary
    section("3. Median / p95 / p99 next to RMSE")
    outage = per.outage_type
    long = long_errors(bus, outage)
    by_type = long.groupby(["variable", "bus_type"]).apply(robust)
    by_outage = long.groupby(["variable", "bus_type", "outage_type"]).apply(robust)
    print(by_type.to_string(float_format=lambda v: f"{v:.3g}"))
    print("\n" + by_outage.to_string(float_format=lambda v: f"{v:.3g}"))
    by_type.to_csv(RESULTS / "inference_percentiles_by_bus_type.csv", float_format="%.6g")
    by_outage.to_csv(RESULTS / "inference_percentiles_by_outage_type.csv", float_format="%.6g")
    s = per.overall_score
    print(f"\noverall score: median {s.median():.3f}, p95 {s.quantile(.95):.3f}, p99 {s.quantile(.99):.3f}, max {s.max():.2f}")

    # ---- 4. hard cases
    section("4. Top 5 % hard cases")
    top = per.nlargest(int(0.05 * len(per)), "overall_score").join(y[["y_voltage", "y_overload"]])
    top.to_csv(RESULTS / "hard_cases_top5pct.csv", float_format="%.6g")
    print(f"{len(top)} samples; outage type {top.outage_type.value_counts().to_dict()} "
          f"(subset: {per.outage_type.value_counts(normalize=True).round(3).to_dict()})")
    print("outaged element:", top.out_branches.fillna(top.out_gens).value_counts().head(6).to_dict())
    print(f"with a true violation: {int((top.y_voltage | top.y_overload).sum())} of {len(top)} "
          f"(subset rate {(y.y_voltage | y.y_overload).mean():.3f})")
    # where on the grid do the largest errors sit?
    worst_bus = long[long.scenario.isin(top.index)].sort_values("abs_err").groupby(["scenario", "variable"]).tail(1)
    print("bus holding the max error of each variable (top-5 % samples):")
    print(worst_bus.groupby("variable").bus.value_counts().groupby(level=0).head(3).to_string())
    corr = {k: round(per.overall_score.corr(y[k].astype(float), method="spearman"), 3)
            for k in ["y_voltage", "y_overload"]}
    print("Spearman(overall score, true label):", corr)

    # ---- 5. figures
    fig_error_distribution(per, FIGS / "fig1_error_distribution.png")
    fig_error_by_outage(long, FIGS / "fig2_error_by_outage_type.png")
    fig_pca(z, explained, per, y.loc[per.index], FIGS / "fig3_embeddings_pca.png")
    print(f"\nfigures written to {FIGS}")


def section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def find_run(run_name):
    """Newest MLflow file-store run with the given run name that holds evaluate outputs."""
    runs = [p.parents[1] for p in (ROOT / "mlruns").glob("*/*/tags/mlflow.runName")
            if p.read_text().strip() == run_name
            and (p.parents[1] / "artifacts/test/case14_ieee_predictions.parquet").exists()]
    return max(runs, key=lambda r: r.stat().st_mtime)


if __name__ == "__main__":
    main()
