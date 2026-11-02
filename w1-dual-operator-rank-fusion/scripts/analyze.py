"""Statistics of the confirmation runs.

Confirmatory family (as frozen): for every panel, the 12 contrasts {W1, W3} x
{IRIE_exact, rule_EW, rule_WT, MP2, fixed085, oracle:scalar}; two-sided Wilcoxon
signed-rank tests on per-instance Spearman rho (zero differences dropped), Holm
adjustment within the panel, and 20,000-resample percentile bootstrap intervals
of the mean difference.  The results are compared with the statistics files of
the source study.

Added after the fact (reported separately, not part of the frozen family):
W1 vs CTRL_lin0.9 on every panel, Holm-adjusted across panels.

Exploratory breakdowns: means by edge-probability regime and graph family.

    python scripts/analyze.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import CONFIRMATION, COMPARATORS, DATA, ORIG, REPRO, out_dir, write_json
from w1.stats import bootstrap_ci, friedman, holm, paired

METRICS = ("rho", "tau", "topk", "ndcg")


def load(panel: str) -> pd.DataFrame:
    return pd.read_csv(REPRO / "confirm" / f"confirm_{panel}.csv.gz")


def specs(panel: str) -> pd.DataFrame:
    s = pd.DataFrame(json.loads((DATA / panel / "specs.json").read_text()))
    s["id"] = np.arange(len(s))
    s["family"] = s["fam"] if "fam" in s else s["g"]
    s["regime"] = s["pr"] if "pr" in s else s["scheme"]
    return s[["id", "family", "regime", "prior"]]


def contrasts(panel: str, piv: pd.DataFrame) -> pd.DataFrame:
    rows, pv = [], {}
    for w in ("W1", "W3"):
        for c in COMPARATORS:
            name = f"{w}-vs-{c}"
            r = paired(piv[w], piv[c])
            est, lo, hi = bootstrap_ci(piv[w].to_numpy(), piv[c].to_numpy(), ("ci", panel, name))
            rows.append(dict(contrast=name, **r, ci_lo=lo, ci_hi=hi))
            pv[name] = r["p"]
    adj = holm(pv)
    for r in rows:
        r["p_holm"] = adj[r["contrast"]]
    return pd.DataFrame(rows)


def main() -> dict:
    out = out_dir("stats")
    rep, means, ctrl_rows, ctrl_p = {}, [], [], {}
    for panel in CONFIRMATION:
        df = load(panel)
        piv = df.pivot_table(index="id", columns="method", values="rho")
        for m in METRICS:
            t = df.groupby("method")[m].mean()
            means.append({"panel": panel, "metric": m, "n": int(df.id.nunique()), **t.to_dict()})
        con = contrasts(panel, piv)
        con.to_csv(out / f"contrasts_{panel}.csv", index=False)
        stat, p = friedman([piv[c].to_numpy() for c in ("W1",) + COMPARATORS])
        entry = dict(friedman_stat=stat, friedman_p=p)
        src = ORIG / "confirm" / f"confirm_{panel}_stats.csv"
        if src.exists():
            o = pd.read_csv(src).set_index("contrast")
            n = con.set_index("contrast").loc[o.index]
            entry["matches_source"] = dict(
                mean_diff=float((o.mean_diff - n.mean_diff).abs().max()),
                p=float((o.p - n.p).abs().max()), p_holm=float((o.p_holm - n.p_holm).abs().max()),
                wins=bool((o.w == n.wins).all()), losses_labelled_t=bool((o.t == n.losses).all()),
                ties_labelled_l=bool((o.l == n.ties).all()))
            fr = dict(line.split("=") for line in
                      (ORIG / "confirm" / f"confirm_{panel}_stats.friedman.txt").read_text().split())
            entry["matches_source"]["friedman_stat"] = abs(float(fr["stat"]) - stat)
        rep[panel] = entry
        r = paired(piv["W1"], piv["CTRL_lin0.9"])
        est, lo, hi = bootstrap_ci(piv["W1"].to_numpy(), piv["CTRL_lin0.9"].to_numpy(),
                                   ("ci", panel, "W1-vs-CTRL"))
        ctrl_rows.append(dict(panel=panel, **r, ci_lo=lo, ci_hi=hi))
        ctrl_p[panel] = r["p"]
        # exploratory breakdowns
        sp = specs(panel)
        piv2 = piv.reset_index().merge(sp, on="id")
        for key in ("regime", "family"):
            g = piv2.groupby(key)[["W1", "IRIE_exact", "oracle:scalar", "rule_EW", "CTRL_lin0.9", "MP2"]]
            tab = g.mean()
            tab.insert(0, "n", g.size())
            tab.to_csv(out / f"by_{key}_{panel}.csv")
    adj = holm(ctrl_p)
    for r in ctrl_rows:
        r["p_holm_across_panels"] = adj[r["panel"]]
    pd.DataFrame(ctrl_rows).to_csv(out / "posthoc_W1_vs_CTRL.csv", index=False)
    pd.DataFrame(means).to_csv(out / "means.csv", index=False)
    write_json(out / "summary.json", rep)
    for k, v in rep.items():
        print(k, v.get("matches_source"))
    return rep


if __name__ == "__main__":
    main()
