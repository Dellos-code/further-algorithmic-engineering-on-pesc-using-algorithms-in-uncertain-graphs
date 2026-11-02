"""Verify the frozen selection against the stored development search.

1. SHA-256 of evidence/original/selected_mixing.json equals the recorded digest.
2. Stage s2: the stored 50 best configurations are re-derived from the matrix.
3. Stage s3: W1 is the configuration with the highest mean development rho; the
   frozen development means of W1 and W3 are recovered exactly.
4. Plateau: how many configurations lie within 0.001 / 0.002 of the best, and
   what they have in common.
5. Instance-level bootstrap top-1 frequencies of W1 and W3 among all stage-s3
   configurations (the source study did not commit the code of its bootstrap;
   the frequencies here use this script's own seed).
6. Stage s4 (alternating schedules) against W1.

    python scripts/verify_freeze.py
"""
from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pandas as pd

from _common import ORIG, frozen, out_dir, write_json
from w1.hybrid import config_id
from w1.stats import bootstrap_top1


def load_stage(stage: str):
    with np.load(ORIG / "search" / f"{stage}_dev.npz") as d:
        return list(d["config_id"]), d["rho"], d["ms_est"]


def main() -> dict:
    out = out_dir("selection")
    raw = (ORIG / "selected_mixing.json").read_bytes()
    recorded = (ORIG / "selected_mixing.sha256").read_text().split()[0]
    digest = hashlib.sha256(raw).hexdigest()
    sel = frozen()
    cfgs = {}
    with gzip.open(ORIG / "configs.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            cfgs[row["config_id"]] = row["config"]
    rep = dict(sha256=digest, sha256_matches=digest == recorded,
               frozen_utc=sel["created_utc"], n_configs_registered=len(cfgs))

    # stage s2
    ids2, rho2, _ = load_stage("s2")
    m2 = rho2.mean(1)
    order2 = np.argsort(-m2, kind="stable")
    stored50 = pd.read_csv(ORIG / "stage_tables" / "s2_dev_top50.csv")
    rep["s2"] = dict(configurations=len(ids2), instances=rho2.shape[1], best=float(m2[order2[0]]),
                     top50_rederived=[ids2[k] for k in order2[:50]] == list(stored50.config_id),
                     top50_means_equal=bool(np.allclose(m2[order2[:50]], stored50.rho, atol=1e-15, rtol=0)))

    # stage s3
    ids3, rho3, ms3 = load_stage("s3")
    m3 = rho3.mean(1)
    order3 = np.argsort(-m3, kind="stable")
    best = float(m3[order3[0]])
    w = {x["name"]: x for x in sel["winners"]}
    pos = {c: k for k, c in enumerate(ids3)}
    # The freeze file labels the configurations "W1"/"W3"; the search registered them under
    # the stage label "S1".  The label is not used in evaluation and is excluded here.
    for name in ("W1", "W3"):
        searched = cfgs[w[name]["config_id"]]
        assert config_id(searched) == w[name]["config_id"]
        assert {k: v for k, v in searched.items() if k != "name"} == \
               {k: v for k, v in w[name]["config"].items() if k != "name"}
    rank_of = {c: int(np.where(order3 == pos[c])[0][0]) + 1 for c in (w["W1"]["config_id"], w["W3"]["config_id"])}
    rep["s3"] = dict(configurations=len(ids3), rows_per_instance_in_source=15750, best=best,
                     W1_is_top1=ids3[order3[0]] == w["W1"]["config_id"],
                     W1_rank=rank_of[w["W1"]["config_id"]], W3_rank=rank_of[w["W3"]["config_id"]],
                     W1_dev_mean=float(m3[pos[w["W1"]["config_id"]]]),
                     W3_dev_mean=float(m3[pos[w["W3"]["config_id"]]]),
                     W1_dev_mean_matches_frozen=float(m3[pos[w["W1"]["config_id"]]]) == w["W1"]["dev_rho"],
                     W3_dev_mean_matches_frozen=float(m3[pos[w["W3"]["config_id"]]]) == w["W3"]["dev_rho"])

    # plateau and the declared tie-breaks
    plateau = {}
    for tol in (0.001, 0.002, 0.005):
        idx = [k for k in order3 if m3[k] >= best - tol]
        sub = [cfgs[ids3[k]] for k in idx]
        plateau[str(tol)] = dict(
            count=len(idx),
            operators=sorted({c.get("operator") for c in sub}),
            steps=sorted({c.get("steps") for c in sub}),
            dampings=sorted({str(c.get("damping")) for c in sub}),
            gammas=sorted({c.get("gamma") for c in sub}),
            betas=sorted({c.get("beta") for c in sub}),
        )
    near = [k for k in order3 if m3[k] >= best - 0.001]
    cost = {ids3[k]: float(np.median(ms3[k])) for k in near}
    rep["plateau"] = plateau
    rep["tiebreak_within_0.001"] = dict(
        candidates=len(near),
        lowest_median_cost_config=min(cost, key=cost.get),
        lowest_median_cost_is_W1=min(cost, key=cost.get) == w["W1"]["config_id"],
        W1_median_cost_ms=cost[w["W1"]["config_id"]],
        min_median_cost_ms=min(cost.values()))
    top = []
    for r, k in enumerate(order3[:20], 1):
        c = cfgs[ids3[k]]
        top.append(dict(rank=r, config_id=ids3[k], dev_mean=float(m3[k]), operator=c["operator"],
                        damping=c["damping"], steps=c["steps"], gamma=c["gamma"], beta=c["beta"]))
    pd.DataFrame(top).to_csv(out / "s3_top20.csv", index=False)

    # bootstrap top-1 frequencies among all stage-s3 configurations
    freq = bootstrap_top1(rho3, 2000, ("w1-project", "bootstrap-top1", "s3"))
    rep["bootstrap_top1"] = dict(resamples=2000,
                                 W1=float(freq[pos[w["W1"]["config_id"]]]),
                                 W3=float(freq[pos[w["W3"]["config_id"]]]),
                                 W1_source=w["W1"]["bootstrap_top1"], W3_source=w["W3"]["bootstrap_top1"],
                                 configurations_ever_top1=int((freq > 0).sum()))

    # stage s4 and the development comparators
    ids4, rho4, _ = load_stage("s4")
    rep["s4"] = {cfgs[c]["name"]: float(v) for c, v in zip(ids4, rho4.mean(1))}
    idsb, rhob, _ = load_stage("baselines")
    rep["dev_baselines"] = {cfgs[c]["name"]: float(v) for c, v in zip(idsb, rhob.mean(1))}
    ids0, rho0, _ = load_stage("s0")
    rep["dev_s0"] = {cfgs[c]["name"]: float(v) for c, v in zip(ids0, rho0.mean(1))}
    write_json(out / "freeze_verification.json", rep)
    print(json.dumps({k: rep[k] for k in ("sha256_matches", "s2", "s3", "bootstrap_top1", "s4")}, indent=1))
    return rep


if __name__ == "__main__":
    main()
