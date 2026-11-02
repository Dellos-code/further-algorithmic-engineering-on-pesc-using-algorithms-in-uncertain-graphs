"""Evaluate the frozen rule on available held-out synthetic panels."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion.gfamilies import evaluate_rule
from fusion.members import member_scores, rank_fusion
from w1.instances import load_instance, panel_paths
from w1.metrics import ranks_of, spearman
from w1.stats import holm, paired, bootstrap_ci


def main():
    frozen = json.loads((ROOT / "results" / "fusion" / "frozen_fusion_v2.json").read_text())
    panels = ("test2", "scale2", "prior2", "real2", "big")
    rows = []
    for panel in panels:
        for path in panel_paths(ROOT / "data", panel):
            inst, y_a, y_b, _ = load_instance(path)
            w, m = member_scores(inst)
            phi = __import__("fusion.covariates", fromlist=["covariates"]).covariates(inst)[frozen["covariate"]]
            c = float(evaluate_rule(frozen["rule"], np.array([phi]))[0])
            c0 = frozen["constant_c0"]
            target_a = ranks_of(y_a)
            rows.append({"panel": panel, "id": path.stem,
                         "F_v2": spearman(rank_fusion(w, m, c), target_a),
                         "F_c0": spearman(rank_fusion(w, m, c0), target_a),
                         "W1": spearman(w, target_a),
                         "MP2": spearman(m, target_a)})
    out = ROOT / "results" / "fusion"
    out.mkdir(exist_ok=True, parents=True)
    with (out / "confirm_v2.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    stats_rows = []
    for panel in panels:
        subset = [r for r in rows if r["panel"] == panel]
        family = {k: paired([r["F_v2"] for r in subset], [r[k] for r in subset])
                  for k in ("W1", "F_c0", "MP2")}
        adjusted = holm({k: v["p"] for k, v in family.items()})
        for method, result in family.items():
            mean, lo, hi = bootstrap_ci(
                [r["F_v2"] for r in subset], [r[method] for r in subset],
                ("fusion-v2", panel), resamples=20000)
            stats_rows.append({"panel": panel, "contrast": f"F_v2-{method}",
                               **result, "holm_p": adjusted[method],
                               "ci_lo": lo, "ci_hi": hi})
    with (out / "confirm_v2_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_rows[0])
        writer.writeheader()
        writer.writerows(stats_rows)
    print(json.dumps({"rows": len(rows), "stats": len(stats_rows)}, indent=2))


if __name__ == "__main__":
    main()
