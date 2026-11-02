"""Fit the second-round fusion rule on development data only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion.covariates import covariates
from fusion.gfamilies import fit_candidates
from fusion.members import member_scores, rank_fusion
from w1.instances import load_instance, panel_paths
from w1.metrics import ranks_of, spearman


def main():
    data = ROOT / "data"
    rows = []
    for path in panel_paths(data, "dev"):
        inst, y_a, y_b, _ = load_instance(path)
        w, m = member_scores(inst)
        c = covariates(inst)
        grid = np.linspace(0, 1, 41)
        curve = np.array([spearman(rank_fusion(w, m, value), ranks_of(y_b)) for value in grid])
        rows.append({"id": path.stem, "covariates": c, "w": w, "m": m,
                     "curve": curve, "y_a": y_a, "y_b": y_b})
    names = list(rows[0]["covariates"])
    best = None
    for name in names:
        x = np.array([r["covariates"][name] for r in rows])
        curves = np.array([r["curve"] for r in rows])
        rule = fit_candidates(x, curves, np.linspace(0, 1, 41))
        c = np.array([rule["params"][0]] * len(rows)) if rule["family"] == "constant" else None
        if c is None:
            from fusion.gfamilies import evaluate_rule
            c = evaluate_rule(rule, x)
        score = float(np.mean([
            spearman(rank_fusion(r["w"], r["m"], ci), ranks_of(r["y_a"]))
            for r, ci in zip(rows, c)]))
        candidate = {"covariate": name, "rule": rule, "cv_score_dev": score}
        if best is None or score > best["cv_score_dev"]:
            best = candidate
    out = ROOT / "results" / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    best.update({
        "members": {"w": "w1.algorithm.W1", "m": "w1.diffusion.knz_mp(iters=2)"},
        "constant_c0": float(np.argmax(np.mean([r["curve"] for r in rows], axis=0)) / 40),
        "hypotheses": ["H1", "H2", "H3"],
        "selection_data": "dev only; yB objective",
        "utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    })
    (out / "frozen_fusion_v2.json").write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
