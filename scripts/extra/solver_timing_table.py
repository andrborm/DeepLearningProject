"""Part 2 table: combine NR (solver_nr_summary.csv) and GENCO (genco_timing.csv) timings -> results/extra/solver_timing.csv.

Both measured on the same CPU, one after the other (never in parallel). NR: pandapower runpp, single thread, per sample
(construction of the pandapower net not timed). GENCO: replay checkpoint, predict_step (forward + physics decoder),
data loading not timed. Speed-up = NR median ms/sample (numba) / GENCO ms/sample.

Usage: venv/Scripts/python.exe -B scripts/extra/solver_timing_table.py
"""

import common  # noqa: F401
import pandas as pd

from common import EXTRA, TEST_LABEL


def main():
    nr = pd.read_csv(EXTRA / "solver_nr_summary.csv").set_index("dataset")
    g = pd.read_csv(EXTRA / "genco_timing.csv")
    rows = []
    for ds in ["test_datakit", "test_pfdelta", "test_n2"]:
        r = nr.loc[ds]
        base = {"dataset": ds, "test set": TEST_LABEL[ds], "n": int(r.n)}
        rows.append({**base, "method": "pandapower NR (numba)", "threads": 1, "batch size": 1,
                     "ms/sample": r["NR ms/sample median (numba)"], "ms/sample mean": r["NR ms/sample mean (numba)"],
                     "NR failures": int(r["NR failures (numba)"]), "max |dVm| vs truth (pu)": r["max |dVm| pu"]})
        rows.append({**base, "method": "pandapower NR (no numba)", "threads": 1, "batch size": 1,
                     "ms/sample": r["NR ms/sample median (no numba)"], "ms/sample mean": r["NR ms/sample mean (no numba)"],
                     "NR failures": int(r["NR failures (no numba)"])})
        for _, x in g[g.dataset == ds].iterrows():
            rows.append({**base, "method": "GENCO replay (predict_step)", "threads": int(x.threads),
                         "batch size": int(x["batch size"]), "ms/sample": x["predict_step ms/sample"],
                         "forward-only ms/sample": x["forward ms/sample"],
                         "throughput samples/s": x["throughput samples/s (predict_step)"]})
    t = pd.DataFrame(rows)
    ref = t[t.method == "pandapower NR (numba)"].set_index("dataset")["ms/sample"]
    t["speed-up vs NR (numba)"] = ref.reindex(t.dataset).values / t["ms/sample"]
    t.to_csv(EXTRA / "solver_timing.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(t.drop(columns=["test set"]).to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
