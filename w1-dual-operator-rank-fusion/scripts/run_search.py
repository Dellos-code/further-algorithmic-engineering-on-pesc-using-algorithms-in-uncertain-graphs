"""Re-evaluate search-stage configurations on the development panel and compare
with the stored values.

Default (about 20 minutes on one core): every configuration of stages s0, s4 and
the comparator stage, the 30 best stage-s3 configurations (W1 and W3 among them),
100 further random stage-s3 and 200 random stage-s2 configurations, on all 180
development instances.  ``--full`` re-evaluates the complete stages s2 and s3
(about 20,000 configurations; many hours on one core).

    python scripts/run_search.py [--full]
"""
from __future__ import annotations

import argparse
import gzip
import json
import random

import numpy as np

from _common import DATA, ORIG, out_dir, write_json
from w1.hybrid import evaluate
from w1.instances import load_instance, panel_paths


def stage(name):
    with np.load(ORIG / "search" / f"{name}_dev.npz") as d:
        return list(d["config_id"]), d["rho"], d["instance"]


def main(full: bool) -> dict:
    cfgs = {}
    with gzip.open(ORIG / "configs.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            cfgs[row["config_id"]] = row["config"]
    rng = random.Random(20261101)
    plan = {}
    for name in ("s0", "baselines", "s4"):
        ids, rho, _ = stage(name)
        plan[name] = (ids, rho, list(range(len(ids))))
    ids3, rho3, _ = stage("s3")
    order = list(np.argsort(-rho3.mean(1), kind="stable"))
    pick3 = order if full else order[:30] + rng.sample(order[30:], 100)
    plan["s3"] = (ids3, rho3, pick3)
    ids2, rho2, _ = stage("s2")
    pick2 = list(range(len(ids2))) if full else rng.sample(range(len(ids2)), 200)
    plan["s2"] = (ids2, rho2, pick2)

    paths = panel_paths(DATA, "dev")
    report = {}
    worst = 0.0
    for name, (ids, rho, pick) in plan.items():
        diffs = []
        for j, path in enumerate(paths):
            inst, yA, yB, _ = load_instance(path)
            cache = {}
            for k in pick:
                v = evaluate(inst, yA, yB, cfgs[ids[k]], cache)
                diffs.append(abs(v - rho[k, j]))
        report[name] = dict(configurations=len(pick), instances=len(paths),
                            evaluations=len(diffs), max_abs_diff=float(max(diffs)),
                            identical=int(sum(d == 0.0 for d in diffs)))
        worst = max(worst, float(max(diffs)))
        print(name, report[name], flush=True)
    report["max_abs_diff_overall"] = worst
    write_json(out_dir("selection") / ("search_full.json" if full else "search_spotcheck.json"), report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    main(ap.parse_args().full)
