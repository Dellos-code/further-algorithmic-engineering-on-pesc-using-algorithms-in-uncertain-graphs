"""EXPLORATORY (not preregistered): accuracy and single-thread wall time of W1
against Monte-Carlo estimation with k sampled worlds, on identical instances.

W1 is timed end to end with the reference implementation (operator construction
included); Monte Carlo is the batched estimator of w1.instances.mc_reach.  Each
measurement is the minimum over 3 repetitions (1 on graphs with more than 1,000
vertices).  Timings depend on the machine; run nothing else concurrently.

    python scripts/bench_cost.py [panel ...]     (default: test2 scale2 big)
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from _common import DATA, out_dir
from w1.algorithm import W1, w1_scores
from w1.instances import Instance, load_instance, mc_reach, panel_paths, stable_seed
from w1.metrics import ranks_of, spearman

KS = (50, 100, 200, 500)


def tmin(f, reps):
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t0)
    return out, 1000 * best


def run(panel: str) -> pd.DataFrame:
    rows = []
    for path in panel_paths(DATA, panel):
        inst, yA, yB, _ = load_instance(path)
        rA = ranks_of(yA)
        reps = 3 if inst.n <= 1000 else 1

        def w1_fresh():
            fresh = Instance(inst.name, inst.n, inst.edges, inst.p, inst.x, inst.meta)
            return w1_scores(fresh, W1, return_parts=True)

        (s, parts), ms = tmin(w1_fresh, reps)
        row = dict(panel=panel, id=int(path.stem), n=inst.n, m=inst.m,
                   family=inst.meta.get("family"), regime=inst.meta.get("prange"),
                   W1_rho=spearman(s, rA), W1_ms=ms, phaseA_iterations=parts["iterations"],
                   alpha_A=parts["alpha_A"], ceiling_rho_yB=spearman(yB, rA))
        for k in KS:
            y, ms = tmin(lambda: mc_reach(inst, k, stable_seed(inst.name, "bench", k)), reps)
            row[f"MC{k}_rho"], row[f"MC{k}_ms"] = spearman(y, rA), ms
        rows.append(row)
        print(panel, path.stem, f"W1 {row['W1_rho']:.3f}/{row['W1_ms']:.1f}ms",
              f"MC200 {row['MC200_rho']:.3f}/{row['MC200_ms']:.1f}ms", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir("exploratory") / f"cost_{panel}.csv", index=False)
    return df


if __name__ == "__main__":
    for p in (sys.argv[1:] or ("test2", "scale2", "big")):
        run(p)
