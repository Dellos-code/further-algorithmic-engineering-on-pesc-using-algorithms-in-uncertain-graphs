"""Confirmation runs: the frozen configurations W1 and W3 and the comparators on
the held-out panels, with Spearman rho, Kendall tau-b, top-10% overlap and
NDCG@10% against the evaluation ground truth yA (oracles are fitted on yB).

CTRL_lin0.9 (the dual-operator configuration of the paper: Phase B on P with
alpha 0.9, 5 steps, gamma 1, beta 0.5) belonged to the declared comparator family
but is absent from the stored confirmation files; it is evaluated here as well
and reported separately.

For every instance the reference implementation (w1.algorithm.w1_scores) is
checked against the search-engine evaluation of the frozen configuration.

    python scripts/run_confirm.py [panel ...]      (default: all confirmation panels)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from _common import CONFIRMATION, COMPARATORS, DATA, ORIG, out_dir, winners, write_json
from w1.algorithm import W1, W3, w1_scores
from w1.hybrid import CTRL_LIN09, baseline_scores, evaluate
from w1.instances import load_instance, panel_paths
from w1.metrics import full_metrics, ranks_of

REF = {"W1": W1, "W3": W3}


def run(panel: str) -> dict:
    frozen = winners()
    rows, ref_gap = [], 0.0
    t0 = time.time()
    for path in panel_paths(DATA, panel):
        inst, yA, yB, _ = load_instance(path)
        rA = ranks_of(yA)
        cache = {}
        for name in ("W1", "W3"):
            s = evaluate(inst, yA, yB, frozen[name], cache, return_vector=True)
            ref = w1_scores(inst, REF[name])
            ref_gap = max(ref_gap, float(np.max(np.abs(ref - s))))
            rows.append({"id": int(path.stem), "method": name, **full_metrics(s, yA, rA)})
        base = baseline_scores(inst, yB, names=COMPARATORS)
        for name in COMPARATORS:
            rows.append({"id": int(path.stem), "method": name, **full_metrics(base[name], yA, rA)})
        s = evaluate(inst, yA, yB, CTRL_LIN09, cache, return_vector=True)
        rows.append({"id": int(path.stem), "method": "CTRL_lin0.9", **full_metrics(s, yA, rA)})
    df = pd.DataFrame(rows)
    out = out_dir("confirm")
    df.to_csv(out / f"confirm_{panel}.csv.gz", index=False, compression="gzip")

    # comparison with the stored confirmation file of the source study
    orig = pd.read_csv(ORIG / "confirm" / f"confirm_{panel}.csv.gz")
    merged = orig.merge(df, on=["id", "method"], suffixes=("_orig", "_new"))
    cmp = {m: float((merged[f"{m}_orig"] - merged[f"{m}_new"]).abs().max())
           for m in ("rho", "tau", "topk", "ndcg")}
    rep = dict(panel=panel, instances=int(df.id.nunique()), rows=len(df),
               rows_compared_with_source=len(merged), rows_in_source=len(orig),
               max_abs_diff_vs_source=cmp, reference_vs_engine_max_score_gap=ref_gap,
               seconds=time.time() - t0,
               means={k: float(v) for k, v in df.groupby("method").rho.mean().items()})
    write_json(out / f"confirm_{panel}_check.json", rep)
    print(panel, {k: rep[k] for k in ("instances", "rows_compared_with_source",
                                      "max_abs_diff_vs_source", "reference_vs_engine_max_score_gap",
                                      "seconds")}, flush=True)
    return rep


if __name__ == "__main__":
    for p in (sys.argv[1:] or CONFIRMATION):
        run(p)
