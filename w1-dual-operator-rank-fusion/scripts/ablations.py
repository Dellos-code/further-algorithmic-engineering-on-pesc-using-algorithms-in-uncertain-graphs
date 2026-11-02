"""EXPLORATORY (not preregistered): sensitivity analyses of W1.

1. bridge_normalised  Phase-B start (x + gamma z)/(1 + gamma) instead of x + gamma z.
2. phaseA_unnormalised  Phase A without the per-step normalisation (the lazy affine
   iteration with the same stopping rule; for alpha_A < 1 it converges to the
   resolvent solution (1 - a)(I - a EW)^-1 x, for alpha_A = 1 its mass decays).
3. K sweep  number of Phase-B steps, K in {1, 2, 3, 4, 5, 6, 8, 12, 20}, all other
   parameters frozen; reported per panel (graphs of 100 to 10,000 vertices).
4. gamma sweep on the development panel only: gamma = 16 is the largest value of
   the search grid, so larger values were never examined.

These analyses use the confirmation panels after the fact (except item 4); they
describe the frozen algorithm and are not used to change it.

    python scripts/ablations.py
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from _common import DATA, out_dir
from w1.algorithm import W1, excess_embedding, phase_a, phase_b
from w1.instances import load_instance, panel_paths
from w1.metrics import ranks_of, spearman

PANELS = ("test2", "scale2", "prior2", "real2", "big")
KS = (1, 2, 3, 4, 5, 6, 8, 12, 20)
GAMMAS = (4.0, 16.0, 32.0, 64.0, 128.0, 1024.0)


def fused(a, b, beta=W1.beta):
    w = np.array([1.0 - beta, beta])
    return np.column_stack([ranks_of(a), ranks_of(b)]) @ (w / w.sum())


def phase_a_unnormalised(EW, x, alpha, tol=W1.tol, maxit=W1.max_iter):
    y = x.copy()
    for it in range(1, maxit + 1):
        yn = 0.5 * (y + (1.0 - alpha) * x + alpha * (EW @ y))
        d = np.abs(yn - y).sum()
        y = yn
        if d < tol:
            break
    return y, it


def main():
    rows = []
    for panel in PANELS + ("dev",):
        for path in panel_paths(DATA, panel):
            inst, yA, _, _ = load_instance(path)
            rA = ranks_of(yA)
            ops, x = inst.ops(), inst.x
            a, alpha_A, it = phase_a(ops["EW"], x, ops["Pbar"], W1)
            row = dict(panel=panel, id=int(path.stem), n=inst.n, alpha_A=alpha_A,
                       family=inst.meta.get("family"), regime=inst.meta.get("prange"))
            if panel != "dev":
                b0 = excess_embedding(a, x, W1.gamma)
                row["W1"] = spearman(fused(a, phase_b(ops["EP"], x, b0, W1.alpha_B, W1.K)), rA)
                row["phaseA_only"] = spearman(a, rA)
                b0n = b0 / (1.0 + W1.gamma)
                row["bridge_normalised"] = spearman(fused(a, phase_b(ops["EP"], x, b0n, W1.alpha_B, W1.K)), rA)
                u, itu = phase_a_unnormalised(ops["EW"], x, alpha_A)
                row["phaseA_unnormalised_only"] = spearman(u, rA)
                row["phaseA_unnormalised_mass"] = float(u.sum())
                bu = phase_b(ops["EP"], x, excess_embedding(u, x, W1.gamma), W1.alpha_B, W1.K)
                row["phaseA_unnormalised"] = spearman(fused(u, bu), rA)
                for K in KS:
                    row[f"K{K}"] = spearman(fused(a, phase_b(ops["EP"], x, b0, W1.alpha_B, K)), rA)
            else:
                for g in GAMMAS:
                    prm = replace(W1, gamma=g)
                    b = phase_b(ops["EP"], x, excess_embedding(a, x, g), prm.alpha_B, prm.K)
                    row[f"gamma{g:g}"] = spearman(fused(a, b), rA)
            rows.append(row)
        print(panel, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir("exploratory") / "ablations.csv", index=False)
    return df


if __name__ == "__main__":
    main()
