"""Presentation figures (results/figures/), from the committed CSVs of the final N-2 test and the IG analysis.

  fig_summary_detectors.png     F1 (and FP / FN) of baseline / A / B, both models, on datakit, PFΔ N-0/N-1, N-2
  fig_residual_vs_error.png     per-sample residual vs true error, original vs replay, all three test sets
  fig_explain_localization.png  IG top-3 localization vs residual and chance, PFΔ N-0/N-1 vs N-2
  fig_explain_grid_example.png  one N-2 overload scenario: residual and IG attribution on the case14 grid

Palette and rcParams come from phase5_analysis.py (dataviz reference palette, validated). Static PNGs for slides.

Usage:
  python scripts/make_presentation_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from phase5_analysis import AQUA, BLUE, GRAY, GRID, INK, INK_2, ORANGE, SEQ_BLUE, SURFACE

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
SETS = [("test_datakit", "datakit"), ("test_pfdelta", "PFΔ N-0/N-1"), ("test_n2", "PFΔ N-2 (final)")]
SET_COLOR = {"test_datakit": BLUE, "test_pfdelta": ORANGE, "test_n2": AQUA}
METHODS = [("baseline PtC", "Predict-then-check"), ("A gated PtC (solver)", "A: residual gate + solver"),
           ("B logreg", "B: probe (logreg)"), ("B mlp", "B: probe (MLP)")]


def fig_summary():
    """Heatmap of F1 (with FP / FN) per detector, model, test set and label."""
    d = pd.read_csv(RESULTS / "final_n2_detectors.csv")
    d = d[d.scope == "all"]
    rows = [(m, mt, ml) for m in ["replay", "original"] for mt, ml in METHODS]
    cols = [(ds, lab) for ds, _ in SETS for lab in ["y_voltage", "y_overload"]]
    F = np.full((len(rows), len(cols)), np.nan)
    txt = [[""] * len(cols) for _ in rows]
    for i, (m, mt, _) in enumerate(rows):
        for j, (ds, lab) in enumerate(cols):
            r = d[(d.model == m) & (d.method == mt) & (d.dataset == ds) & (d.label == lab)].iloc[0]
            F[i, j] = r.F1
            extra = f"\n{int(r.FP)} FP / {int(r.FN)} FN"
            if mt.startswith("A") and r["solver share"] > 0:
                extra += f"\n{r['solver share']:.0%} to solver"
            txt[i][j] = f"{r.F1:.3f}{extra}"
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + SEQ_BLUE[:6])
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.grid(False)
    vmin = 0.6
    ax.imshow(np.clip(F, vmin, 1), cmap=cmap, vmin=vmin, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            dark = F[i, j] > 0.93
            best = F[i, j] >= np.nanmax(F[:, j]) - 1e-9
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=9.5, linespacing=1.25,
                    color="white" if dark else INK, fontweight="bold" if best else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([("voltage" if lab == "y_voltage" else "overload") for _, lab in cols])
    for k, (_, name) in enumerate(SETS):
        ax.text(2 * k + 0.5, -0.95, name, ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([ml for _, _, ml in rows])
    for k in [1.5, 3.5]:
        ax.axvline(k, color=SURFACE, lw=6)
    ax.axhline(3.5, color=SURFACE, lw=6)
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y, name in [(1.5, "replay model (final)"), (5.5, "original model")]:
        ax.text(-0.235, y, name, ha="center", va="center", fontsize=12, fontweight="bold", rotation=90, transform=tr)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("F1 of each detector on the three 1,000-sample test sets (bold = best in column)",
                 x=0.01, ha="left", fontsize=14)
    fig.text(0.01, 0.01, "Cells show F1, false positives / false negatives. A gate t was chosen on validation "
             "(replay: t = ∞, nothing sent; original: 1.27 MVA). Probes frozen before N-2. Colour scale clipped at F1 0.6.",
             fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.25, right=0.99, top=0.84, bottom=0.07)
    fig.savefig(FIG / "fig_summary_detectors.png", dpi=200)
    plt.close(fig)


def fig_residual():
    """Per-sample residual vs true error for both models; wrong PtC labels circled."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, model in zip(axes, ["original", "replay"]):
        for ds, name in SETS:
            s = pd.read_csv(RESULTS / f"final_n2_samples_{model}_{ds}.csv", index_col=0)
            rho = spearmanr(s.pbe_mean, s.score).statistic
            wrong = (s.y_voltage != s.y_voltage_ptc) | (s.y_overload != s.y_overload_ptc)
            ax.scatter(s.pbe_mean[~wrong], s.score[~wrong], s=10, color=SET_COLOR[ds], alpha=0.45, lw=0,
                       label=f"{name}  (ρ = {rho:.2f})")
            ax.scatter(s.pbe_mean[wrong], s.score[wrong], s=34, facecolors="none", edgecolors=SET_COLOR[ds], lw=1.3)
        ax.scatter([], [], s=34, facecolors="none", edgecolors=INK_2, lw=1.3, label="wrong predict-then-check label")
        if model == "original":
            ax.axvline(1.27, color=INK_2, ls=(0, (4, 3)), lw=1.2)
            ax.text(1.35, 0.03, "gate t = 1.27 MVA", fontsize=9, color=INK_2, rotation=90, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{model} model" + (" (final)" if model == "replay" else ""), loc="left")
        ax.set_xlabel("power-balance residual, mean |PBE| per sample (MVA)")
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    axes[0].set_ylabel("true error (score; 1 = datakit level)")
    fig.suptitle("The residual tracks the error when the model is out of its depth (Spearman ρ per test set)",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "fig_residual_vs_error.png", dpi=200)
    plt.close(fig)


def fig_localization():
    """Bars: IG vs residual vs chance localization of correct alarms."""
    a = pd.read_csv(RESULTS / "explain_ig_localization.csv")
    a = a[(a.probe == "logreg") & a.dataset.isin(["test_pfdelta", "test_n2"])]
    groups = [("y_voltage", "voltage: top-3 buses\ncontain a violating bus", "IG", "residual", "chance"),
              ("y_overload", "overload: top-3 buses contain an\nendpoint of an overloaded line",
               "IG_endpoint_bus", "residual_endpoint_bus", "chance_endpoint_bus"),
              ("y_overload", "overload: top-3 branches\ncontain an overloaded line", "IG", "residual", "chance")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    w = 0.26
    for ax, (lab, title, ig, res, ch) in zip(axes, groups):
        for k, (ds, name) in enumerate([("test_pfdelta", "PFΔ N-0/N-1"), ("test_n2", "N-2")]):
            r = a[(a.dataset == ds) & (a.label == lab)].iloc[0]
            for o, (col, color, lbl) in enumerate([(ig, BLUE, "Integrated Gradients (probe)"),
                                                   (res, ORANGE, "per-bus residual |PBE|"),
                                                   (ch, GRAY, "random top-3 (chance)")]):
                x = k + (o - 1) * w
                ax.bar(x, r[col], w - 0.03, color=color, label=lbl if k == 0 else None)
                ax.text(x, r[col] + 0.015, f"{r[col]:.0%}", ha="center", fontsize=9, color=INK)
            ax.text(k, -0.13, f"n = {int(r.n)}", ha="center", fontsize=8.5, color=INK_2,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["PFΔ N-0/N-1", "N-2 (unseen)"])
        ax.set_title(title, fontsize=11.5, loc="left")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of correct alarms localized")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle("Where does the replay probe look? IG localization on correct alarms (logreg probe)",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_explain_localization.png", dpi=200)
    plt.close(fig)


def fig_grid_example():
    """Residual and IG attribution on the grid for the N-2 overload example."""
    import torch
    import explain_ig as X

    ex = pd.read_csv(RESULTS / "explain_ig_examples.csv")
    r = ex[ex.example == "overload"].iloc[0]
    ds, s, lab = r.dataset, int(r.scenario), r.label
    probe = X.TorchProbe(X.PROBES / f"{X.MODEL}_logreg_{lab}.joblib")
    scen, E = X.embeddings(ds)
    attr = X.integrated_gradients(probe, torch.tensor(E[int(np.searchsorted(scen, s))]),
                                  torch.tensor(X.baseline_embeddings(lab))).sum(1).numpy()
    tb, tbr = X.element_truth(X.DATASETS[ds] / "case14_ieee/raw")
    b, br = tb[tb.scenario == s].sort_values("bus"), tbr[tbr.scenario == s].sort_values("idx")
    pb = pd.read_parquet(X.PRED / X.MODEL / ds / "case14_ieee_predictions.parquet")
    pbe = pb[pb.scenario == s].sort_values("bus").PBE.abs().to_numpy()
    plt.rcParams.update({"axes.grid": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
    X.draw_grid(axes[0], pbe, br, "per-bus power-balance residual |PBE|", "viridis", pbe.max(),
                b.violating.to_numpy(), colorbar_label="|PBE| (MVA)")
    X.draw_grid(axes[1], attr, br, f"IG attribution of the overload alarm (p = {r.p_alarm:.2f})\n"
                f"top-3 buses: {r['top3 buses (1-based)'].replace(' ', ', ')}", "RdBu_r", np.abs(attr).max(),
                b.violating.to_numpy(), colorbar_label="attribution (logit)")
    outs = ", ".join(f"{int(f) + 1}–{int(t) + 1}" for f, t in br[br.br_status == 0][["from_bus", "to_bus"]].values)
    fig.suptitle(f"N-2 scenario #{s}: lines {outs} out → line 1–5 overloaded ({br[br.overloaded].loading.max():.0%})\n"
                 f"The overload probe's attribution concentrates on buses {r['top3 buses (1-based)'].replace(' ', ', ')} around the overloaded line's end bus 5",
                 x=0.01, ha="left", fontsize=12.5)
    fig.text(0.01, 0.015, "red ring = true voltage violation · thick red line = overloaded branch · dashed ✕ = outaged "
             "branch · G = generator bus · line width ∝ loading", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(FIG / "fig_explain_grid_example.png", dpi=200)
    plt.close(fig)


def main():
    """Write all four presentation figures."""
    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(exist_ok=True)
    fig_summary()
    fig_residual()
    fig_localization()
    fig_grid_example()
    print("wrote", *sorted(p.name for p in FIG.glob("fig_*")), sep="\n  ")


if __name__ == "__main__":
    main()
