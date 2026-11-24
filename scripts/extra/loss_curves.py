"""Part 3: fine-tuning loss curves from the existing MLflow logs (read-only).

Runs (experiment stage2_finetune, mlruns/657056988967417188): ft_hard_base, ft_random_base, ft_replay_base.
  validation loss: logged once per epoch (metric "Validation loss"; 30 / 30 / 40 values)
  training loss:   graphkit's Trainer uses log_every_n_steps=1000, so "Training Loss" exists only every 1,000
                   optimizer steps (2 / 2 / 6 values), i.e. NOT per epoch. Those points are plotted at
                   epoch = step / batches per epoch (79 for hard/random, 157 for replay).
Used weights: hard / random = best_model_state_dict.pt, written at the minimum-val-loss epoch (= last epoch, checked);
replay = last epoch (epoch 40) because of the best-checkpoint bug (finetune_summary.md).
Pretraining panel: the checkpoint's metrics.csv only holds final test metrics (PBE, residuals), no loss per epoch,
so no second panel is drawn.

Outputs: results/extra/figures/fig_finetune_loss_curves.png, results/extra/finetune_loss_curves.csv
Usage:   venv/Scripts/python.exe -B scripts/extra/loss_curves.py
"""

import common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import EXTRA, FIGS, ROOT, palette

EXP = ROOT / "mlruns/657056988967417188"
RUNS = {  # name -> (run id, batches per epoch, epoch whose weights were used: "best" or "last")
    "hard": ("95c890fec4fc44c7a418062e00f14fd5", 79, "best"),
    "random": ("1f1199cb93004031970696db1fa7dac3", 79, "best"),
    "replay": ("f061c2d3f30c40169fef7f4bc0845665", 157, "last"),
}
CARD_METRICS = ROOT / "checkpoints/genco-pf-datakit/case14_ieee/base/seed0/metrics.csv"


def metric(run, name):
    """One MLflow metric file as a table (timestamp, value, step)."""
    m = pd.read_csv(EXP / run / "metrics" / name, sep=" ", header=None, names=["ts", "value", "step"])
    return m.sort_values(["step", "ts"]).reset_index(drop=True)


def main():
    """Plot validation (per epoch) and train (per 1,000 steps) loss of the three runs."""
    c = palette()
    colors = {"hard": c["BLUE"], "random": c["ORANGE"], "replay": c["AQUA"]}
    labels = {"hard": "hard cases", "random": "random control", "replay": "replay (final model)"}
    rows = []
    fig, ax = plt.subplots(figsize=(11, 6.4))
    for name, (run, bpe, used) in RUNS.items():
        assert (EXP / run / "tags/mlflow.runName").read_text().strip() == f"ft_{name}_base"
        val = metric(run, "Validation loss")
        val["epoch"] = np.arange(1, len(val) + 1)
        tr = metric(run, "Training Loss")
        tr["epoch"] = (tr.step + 1) / bpe
        used_ep = int(val.epoch.iloc[-1]) if used == "last" else int(val.epoch[val.value.idxmin()])
        if used == "best":
            assert used_ep == val.epoch.iloc[-1], f"{name}: best epoch is not the last one"
        for _, r in val.iterrows():
            rows.append({"run": name, "kind": "validation", "epoch": r.epoch, "step": r.step, "loss": r.value})
        for _, r in tr.iterrows():
            rows.append({"run": name, "kind": "train (every 1000 steps)", "epoch": r.epoch, "step": r.step,
                         "loss": r.value})
        col = colors[name]
        ax.plot(val.epoch, val.value, color=col, lw=2.2, label=f"{labels[name]}: val")
        ax.plot(tr.epoch, tr.value, color=col, lw=1.3, alpha=0.45, ls="--", marker="o", ms=5,
                label=f"{labels[name]}: train")
        v = val.value[val.epoch == used_ep].iloc[0]
        ax.scatter([used_ep], [v], s=130, marker="*", color=col, edgecolor=c["INK"], lw=0.8, zorder=5)
        print(f"{name}: {len(val)} epochs, val {val.value.iloc[0]:.3g} -> {val.value.iloc[-1]:.3g}, "
              f"min {val.value.min():.3g} at epoch {int(val.epoch[val.value.idxmin()])}, used epoch {used_ep}; "
              f"train points at epochs {np.round(tr.epoch.values, 1).tolist()}")
    ax.scatter([], [], s=130, marker="*", color="white", edgecolor=c["INK"], label="weights used (★)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("GENCO base fine-tuning: loss per epoch (0.1 physics + 0.9 masked bus MSE)", loc="left")
    ax.legend(frameon=False, fontsize=10.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), columnspacing=1.2)
    fig.text(0.01, 0.01, "Hard / random: 5,000 PFΔ N-0/N-1 samples, 30-epoch cap, PFΔ val; best = last epoch (30). "
             "Replay: + 5,000 datakit, 40-epoch cap,\nmixed val; used the last epoch (40), not the val minimum (39), "
             "because of the best-checkpoint bug. Train loss: single-step values, logged only\nevery 1,000 optimizer "
             "steps by graphkit (not per epoch). The runs use different validation sets, so levels are not directly "
             "comparable.", fontsize=9.5, color=c["INK_2"])
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_finetune_loss_curves.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(EXTRA / "finetune_loss_curves.csv", index=False, float_format="%.6g")
    card = pd.read_csv(CARD_METRICS)
    print(f"checkpoint metrics.csv: {len(card)} rows, metrics {card.Metric.tolist()} -> no per-epoch loss, no 2nd panel")


if __name__ == "__main__":
    main()
