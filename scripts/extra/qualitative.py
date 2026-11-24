"""Part 4: qualitative before/after fine-tuning: predicted vs true PQ-bus Vm on the PFΔ N-0/N-1 test subset.

Uses the SAVED predictions of the frozen pipeline (no inference): data/predictions/stage3/<model>/test_pfdelta/
case14_ieee_predictions.parquet (graphkit predict, stage3_detectors.py extract). Models: original checkpoint and
replay (epoch 40). PQ buses = rows with PQ == 1 (generator-outage buses that became PQ are included).

Output: results/extra/figures/fig_finetune_qualitative.png, results/extra/qualitative_vm_stats.csv
Usage:  venv/Scripts/python.exe -B scripts/extra/qualitative.py
"""

import common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import EXTRA, FIGS, ROOT, palette

PRED = ROOT / "data/predictions/stage3"
LO, HI = 0.94, 1.06


def main():
    c = palette()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharex=True, sharey=True)
    data = {m: pd.read_parquet(PRED / m / "test_pfdelta/case14_ieee_predictions.parquet") for m in ["original", "replay"]}
    pq = {m: d[d.PQ == 1] for m, d in data.items()}
    allv = np.concatenate([np.r_[d.Vm_target, d.Vm_pred] for d in pq.values()])
    lim = (min(allv.min(), 0.70) - 0.01, max(allv.max(), 1.10) + 0.01)
    rows = []
    for ax, (m, title, col) in zip(axes, [("original", "(a) original checkpoint", c["ORANGE"]),
                                          ("replay", "(b) replay fine-tuned (final model)", c["BLUE"])]):
        d = pq[m]
        t, p = d.Vm_target.values, d.Vm_pred.values
        err = p - t
        side_t = np.select([t < LO, t > HI], [-1, 1], 0)
        side_p = np.select([p < LO, p > HI], [-1, 1], 0)
        flip = (side_t != side_p).mean()
        rmse = np.sqrt((err ** 2).mean())
        rows.append({"model": m, "n PQ bus values": len(d), "RMSE (pu)": rmse, "median |err| (pu)": np.median(np.abs(err)),
                     "max |err| (pu)": np.abs(err).max(), "share on wrong side of a limit": flip})
        ax.axvspan(lim[0], LO, color=c["GRID"], alpha=0.6, lw=0)
        ax.axhspan(lim[0], LO, color=c["GRID"], alpha=0.6, lw=0)
        ax.axvspan(HI, lim[1], color=c["GRID"], alpha=0.6, lw=0)
        ax.axhspan(HI, lim[1], color=c["GRID"], alpha=0.6, lw=0)
        ax.scatter(t, p, s=7, color=col, alpha=0.35, edgecolors="none", rasterized=True)
        ax.plot(lim, lim, color=c["INK"], lw=1.2, label="y = x")
        for v in (LO, HI):
            ax.axvline(v, color=c["INK_2"], lw=1, ls=(0, (4, 3)))
            ax.axhline(v, color=c["INK_2"], lw=1, ls=(0, (4, 3)))
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left")
        ax.set_xlabel("True Vm (p.u.)")
        ax.text(0.03, 0.97, f"RMSE {rmse:.2g} p.u.\n{flip:.1%} of PQ-bus values on the\nwrong side of 0.94 / 1.06",
                transform=ax.transAxes, va="top", fontsize=11, color=c["INK"],
                bbox=dict(facecolor=c["SURFACE"], edgecolor="none", pad=2))
    axes[0].set_ylabel("Predicted Vm (p.u.)")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("PQ-bus voltage magnitude on the PFΔ N-0/N-1 test subset (1,000 samples), before vs after fine-tuning",
                 x=0.01, ha="left", fontsize=14)
    fig.text(0.01, 0.01, f"One point per PQ bus and sample (n = {len(pq['replay']):,}). Dashed lines: voltage limits "
             "0.94 / 1.06 p.u.; shaded: outside the limits. Saved graphkit predictions, no new inference.",
             fontsize=9.5, color=c["INK_2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_finetune_qualitative.png", dpi=200)
    plt.close(fig)
    st = pd.DataFrame(rows)
    st.to_csv(EXTRA / "qualitative_vm_stats.csv", index=False, float_format="%.5g")
    print(st.to_string(index=False))


if __name__ == "__main__":
    main()
