"""External hold-out: the 100-instance hold-out benchmark of the earlier
escape-probability study (vendored unmodified in external/escape_probability,
source commit c5377f4).  Instances and their ground truth (3,000 Monte-Carlo
worlds) are rebuilt with that study's own code; rho is that study's metric
(scipy Spearman against the ground truth).

Metric note.  W1, W3 and CTRL_lin0.9 are scored with this project's Spearman
(w1.metrics.spearman, ranks merged within a relative tolerance of 1e-12), as in
the source study; rank-fused scores contain exact ties that floating-point
rounding can split, which scipy's Spearman would rank apart.  Their scipy values
are reported as well (columns *_scipy); the two differ by less than 4e-4 per
instance.  IRIE and the escape-scaled rule are computed with the vendored code and
scipy Spearman, like every stored method of that benchmark.

W1, W3, the paper's CTRL_lin0.9, IRIE and the calibrated escape-scaled rule are
evaluated here; the other methods of that benchmark (including Monte Carlo with
10/50/200 worlds, PMIA and the scalar oracle) are taken from its stored results.

    python scripts/run_external_holdout.py
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd

from _common import EXTERNAL, ORIG, out_dir, write_json
from w1.algorithm import W1, W3, w1_scores
from w1.hybrid import CTRL_LIN09, evaluate
from w1.instances import Instance
from w1.metrics import ranks_of, spearman

spec = importlib.util.spec_from_file_location("escape_lib", EXTERNAL / "escape_lib.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def main() -> dict:
    H = json.loads((EXTERNAL / "holdout.json").read_text())
    coef = tuple(H["coef"])
    rows = []
    for rec in H["records"]:
        old = E.make_instance(rec["topology"], 100, rec["lo"], rec["hi"], rec["seed"])
        gt = E.ground_truth(old, 3000, seed=5000 + rec["seed"])
        inst = Instance(f"old_{rec['topology']}_{rec['lo']}_{rec['seed']}", old.n,
                        np.stack([old.ei, old.ej], 1).astype(np.int64), old.p, old.x, {})
        rg = ranks_of(gt)
        scores = {"W1": w1_scores(inst, W1), "W3": w1_scores(inst, W3),
                  "CTRL_lin0_9": evaluate(inst, gt, None, CTRL_LIN09, return_vector=True)}
        row = dict(topology=rec["topology"], lo=rec["lo"], hi=rec["hi"], seed=rec["seed"])
        row.update({k: spearman(v, rg) for k, v in scores.items()})
        row.update({f"{k}_scipy": E.rho(v, gt) for k, v in scores.items()})
        row.update(IRIE_exact=E.rho(E.m_irie(old, 0.7), gt),
                   rule_EW_recomputed=E.rho(E.m_escape_pull(old, E.alpha_from_rule(old, coef)), gt))
        row.update({f"stored:{k}": v for k, v in rec["rho"].items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    out = out_dir("external")
    df.to_csv(out / "external_holdout.csv", index=False)
    src = pd.DataFrame(json.loads((ORIG / "external_holdout.json").read_text()))
    m = src.merge(df, on=["topology", "lo", "seed"], suffixes=("_src", ""))
    rep = dict(instances=len(df),
               ground_truth_reproduced=bool(np.allclose(df.rule_EW_recomputed, df["stored:THIS WORK"],
                                                        atol=1e-12, rtol=0)),
               max_abs_diff_vs_source={k: float((m[f"{k}_src"] - m[k]).abs().max())
                                       for k in ("W1", "W3", "IRIE_exact")},
               means={c.replace("stored:", ""): float(df[c].mean())
                      for c in df.columns if c not in ("topology", "lo", "hi", "seed")})
    write_json(out / "external_holdout_check.json", rep)
    print(json.dumps({k: rep[k] for k in ("instances", "ground_truth_reproduced", "max_abs_diff_vs_source")}))
    return rep


if __name__ == "__main__":
    main()
