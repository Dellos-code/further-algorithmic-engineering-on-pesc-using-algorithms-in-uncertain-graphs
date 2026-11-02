"""Write every number and table of the evidence report, and the result tables of
the README files, from evidence/original and evidence/reproduced.

Outputs
    docs/evidence/generated/values.tex     \\V{key} values (one definition per number)
    docs/evidence/generated/tables_en.tex  table bodies, English labels
    docs/evidence/generated/tables_gr.tex  table bodies, Greek labels
    README.md, README_GR.md                blocks between RESULTS/REPRO markers

    python scripts/make_doc_assets.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from _common import CONFIRMATION, COMPARATORS, DATA, ORIG, REPRO, ROOT, frozen
from w1.instances import load_instance, panel_paths
from w1.metrics import ranks_of, spearman

GEN = ROOT / "docs" / "evidence" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
V: dict[str, str] = {}

KEY = {"W1": "W1", "W3": "W3", "CTRL_lin0.9": "CTRL", "IRIE_exact": "IRIE", "rule_EW": "ruleEW",
       "rule_WT": "ruleWT", "MP2": "MP2", "fixed085": "fixed", "oracle:scalar": "oracle"}
LABEL = {
    "en": {"W1": "W1", "W3": "W3", "CTRL_lin0.9": "CTRL", "IRIE_exact": "IRIE",
           "rule_EW": "EW rule", "rule_WT": "WT rule", "MP2": "MP2",
           "fixed085": r"$\alpha{=}0.85$", "oracle:scalar": "oracle"},
    "gr": {"W1": "W1", "W3": "W3", "CTRL_lin0.9": "CTRL", "IRIE_exact": "IRIE",
           "rule_EW": "κανόνας EW", "rule_WT": "κανόνας WT", "MP2": "MP2",
           "fixed085": r"$\alpha{=}0.85$", "oracle:scalar": "μαντείο"},
}
PANEL_ORDER = ("test2", "scale2", "prior2", "real2", "big", "test", "prior")
METHODS = ("W1", "W3", "CTRL_lin0.9", "oracle:scalar", "IRIE_exact", "rule_EW", "rule_WT",
           "MP2", "fixed085")


def f(x, d=4):
    return f"{x:.{d}f}"


def sgn(x, d=4):
    return f"{x:+.{d}f}".replace("-", r"\ensuremath{-}").replace("+", r"\ensuremath{+}")


def pv(p):
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return rf"\ensuremath{{{m}\cdot10^{{{int(e)}}}}}"


def sci(x):
    """x as a power of ten with one significant digit, e.g. 1e-16 for 1.1e-16."""
    m, e = f"{x:.0e}".split("e")
    return rf"\ensuremath{{{'' if m == '1' else m + r'\cdot'}10^{{{int(e)}}}}}"


def put(key, value):
    if not re.fullmatch(r"[A-Za-z0-9./-]+", key):
        raise ValueError(key)
    V[key] = value


def jload(path):
    return json.loads(path.read_text())


# ----------------------------------------------------------------- panels
def panels():
    info = {}
    for panel in ("dev",) + PANEL_ORDER:
        paths = panel_paths(DATA, panel)
        ns, ceil = [], []
        for p in paths:
            inst, yA, yB, _ = load_instance(p)
            ns.append(inst.n)
            ceil.append(spearman(yB, ranks_of(yA)))
        info[panel] = dict(count=len(paths), nmin=min(ns), nmax=max(ns), ceiling=float(np.mean(ceil)))
        put(f"{panel}/count", str(len(paths)))
        put(f"{panel}/ceiling", f(np.mean(ceil), 3))
        put(f"{panel}/n", str(min(ns)) if min(ns) == max(ns) else f"{min(ns)}--{max(ns)}")
    put("total/instances", f"{sum(v['count'] for v in info.values()):,}".replace(",", "\\,"))
    return info


PANEL_ROLE = {
    "en": {"dev": "selection", "test2": "primary confirmation", "scale2": "primary confirmation",
           "prior2": "secondary confirmation", "real2": "secondary confirmation",
           "big": "secondary confirmation", "test": "earlier held-out", "prior": "earlier held-out"},
    "gr": {"dev": "επιλογή", "test2": "κύρια επιβεβαίωση", "scale2": "κύρια επιβεβαίωση",
           "prior2": "δευτερεύουσα επιβεβαίωση", "real2": "δευτερεύουσα επιβεβαίωση",
           "big": "δευτερεύουσα επιβεβαίωση", "test": "προηγούμενο σύνολο ελέγχου",
           "prior": "προηγούμενο σύνολο ελέγχου"},
}
PANEL_CONTENT = {
    "en": {"dev": "9 families, R1--R5, uniform/Dirichlet(1)",
           "test2": "9 families, R1--R5, uniform/Dirichlet(1)",
           "scale2": "9 families, R1--R5, uniform/Dirichlet(1)",
           "prior2": "5 families, R2--R4, 10 priors",
           "real2": "3 classic networks, 4 probability schemes",
           "big": "5 families, R2 and R4, uniform",
           "test": "9 families, R1--R5, uniform/Dirichlet(1)",
           "prior": "5 families, R2--R4, 10 priors"},
    "gr": {"dev": "9 οικογένειες, R1--R5, ομοιόμορφη/Dirichlet(1)",
           "test2": "9 οικογένειες, R1--R5, ομοιόμορφη/Dirichlet(1)",
           "scale2": "9 οικογένειες, R1--R5, ομοιόμορφη/Dirichlet(1)",
           "prior2": "5 οικογένειες, R2--R4, 10 αρχικές κατανομές",
           "real2": "3 κλασικά δίκτυα, 4 σχήματα πιθανοτήτων",
           "big": "5 οικογένειες, R2 και R4, ομοιόμορφη",
           "test": "9 οικογένειες, R1--R5, ομοιόμορφη/Dirichlet(1)",
           "prior": "5 οικογένειες, R2--R4, 10 αρχικές κατανομές"},
}


def table_panels(info, lang):
    head = {"en": r"Panel & Role & Instances & Vertices & Content & Ceiling \\",
            "gr": r"Σύνολο & Ρόλος & Στιγμιότυπα & Κορυφές & Περιεχόμενο & Όριο θορύβου \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    for p in ("dev",) + PANEL_ORDER:
        i = info[p]
        n = str(i["nmin"]) if i["nmin"] == i["nmax"] else f"{i['nmin']}--{i['nmax']}"
        rows.append(rf"\texttt{{{p}}} & {PANEL_ROLE[lang][p]} & {i['count']} & {n} & "
                    rf"{PANEL_CONTENT[lang][p]} & {f(i['ceiling'], 3)} \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


# --------------------------------------------------------------- selection
def selection():
    r = jload(REPRO / "selection" / "freeze_verification.json")
    sel = frozen()
    put("sha", r["sha256"][:16])
    put("freeze/utc", sel["created_utc"][:19].replace("T", " "))
    put("s2/configs", f"{r['s2']['configurations']:,}".replace(",", "\\,"))
    put("s2/best", f(r["s2"]["best"]))
    put("s3/configs", f"{r['s3']['configurations']:,}".replace(",", "\\,"))
    put("s3/rows", "15\\,750")
    put("dev/W1", f(r["s3"]["W1_dev_mean"]))
    put("dev/W3", f(r["s3"]["W3_dev_mean"]))
    put("dev/S4best", f(max(r["s4"].values())))
    put("dev/S4alt", f(min(r["s4"].values())))
    for k, v in r["dev_baselines"].items():
        if k in KEY:
            put(f"dev/{KEY[k]}", f(v))
    put("dev/CTRL", f(r["dev_s0"]["CTRL_lin0.9"]))
    put("dev/MPinf", f(r["dev_baselines"]["MPinf"]))
    put("dev/TotalRank", f(r["dev_baselines"]["TotalRank"]))
    for tol in ("0.001", "0.002", "0.005"):
        put(f"plateau/{tol}", str(r["plateau"][tol]["count"]))
    b = r["bootstrap_top1"]
    put("boot/W1", f(b["W1"], 3)); put("boot/W3", f(b["W3"], 3))
    put("boot/W1src", f(b["W1_source"], 3)); put("boot/W3src", f(b["W3_source"], 3))
    put("boot/ever", str(b["configurations_ever_top1"]))
    t = r["tiebreak_within_0.001"]
    put("tie/W1cost", f(t["W1_median_cost_ms"], 3)); put("tie/mincost", f(t["min_median_cost_ms"], 3))
    top = pd.read_csv(REPRO / "selection" / "s3_top20.csv")
    alt = top[top.config_id == t["lowest_median_cost_config"]].iloc[0]
    put("tie/altrank", str(int(alt["rank"]))); put("tie/altdev", f(alt.dev_mean))
    put("tie/altop", alt.operator); put("tie/altdamp", f"{float(alt.damping):g}")
    put("tie/altbeta", f"{alt.beta:g}")
    # damping laws and structure of the search space
    import gzip
    cfgs = {}
    with gzip.open(ORIG / "configs.jsonl.gz", "rt") as fh:
        for line in fh:
            row = json.loads(line)
            cfgs[row["config_id"]] = row["config"]
    with np.load(ORIG / "search" / "s3_dev.npz") as d:
        ids, m3 = list(d["config_id"]), d["rho"].mean(1)
    def best_of(pred):
        k = [i for i, c in enumerate(ids) if pred(cfgs[c])]
        return float(max(m3[k])), len(k)
    g4const = best_of(lambda c: c.get("gamma") == 4.0 and isinstance(c.get("damping"), (int, float)))
    put("law/const4", f(g4const[0]))
    for law, key in (("D2", "D2"), ("D3", "D3"), ("D4", "D4")):
        v, n = best_of(lambda c, law=law: str(c.get("damping")).startswith(law + ":"))
        put(f"law/{key}", f(v)); put(f"law/{key}n", str(n))
    with np.load(ORIG / "search" / "s2_dev.npz") as d:
        ids2, m2 = list(d["config_id"]), d["rho"].mean(1)
    by_steps, by_op = {}, {}
    for i, c in enumerate(ids2):
        cf = cfgs[c]
        by_steps[cf["steps"]] = max(by_steps.get(cf["steps"], -1), m2[i])
        by_op[cf["operator"]] = max(by_op.get(cf["operator"], -1), m2[i])
    for k, v in by_steps.items():
        put(f"s2steps/{k}", f(v))
    for k, v in by_op.items():
        put(f"s2op/{k}", f(v))
    return r, top, by_steps, by_op


def table_top(top, lang):
    head = {"en": r"Rank & Operator & $\alpha_B$ & $K$ & $\gamma$ & $\beta$ & Development $\bar\rho$ \\",
            "gr": r"Θέση & Τελεστής & $\alpha_B$ & $K$ & $\gamma$ & $\beta$ & $\bar\rho$ ανάπτυξης \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    for _, t in top.head(10).iterrows():
        name = {1: " (W1)", 2: " (W3)"}.get(int(t["rank"]), "")
        rows.append(rf"{int(t['rank'])}{name} & {t.operator} & {float(t.damping):g} & {int(t.steps)} & "
                    rf"{t.gamma:g} & {t.beta:g} & {f(t.dev_mean)} \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def table_steps(by_steps, by_op, lang):
    ks = sorted(by_steps)
    head = {"en": r"$K$", "gr": r"$K$"}[lang]
    rows = [r"\toprule", head + " & " + " & ".join(str(k) for k in ks) + r" \\", r"\midrule",
            {"en": r"best $\bar\rho$", "gr": r"καλύτερο $\bar\rho$"}[lang] + " & "
            + " & ".join(f(by_steps[k], 3) for k in ks) + r" \\", r"\bottomrule"]
    return "\n".join(rows)


# ----------------------------------------------------------- confirmation
def confirmation():
    means = {}
    for panel in PANEL_ORDER:
        df = pd.read_csv(REPRO / "confirm" / f"confirm_{panel}.csv.gz")
        g = df.groupby("method")[["rho", "tau", "topk", "ndcg"]].mean()
        means[panel] = g
        for m in METHODS:
            for metric in ("rho", "tau", "topk", "ndcg"):
                put(f"{panel}/{KEY[m]}/{metric}", f(g.loc[m, metric]))
        put(f"{panel}/W1-oracle", sgn(g.loc["W1", "rho"] - g.loc["oracle:scalar", "rho"]))
        con = pd.read_csv(REPRO / "stats" / f"contrasts_{panel}.csv").set_index("contrast")
        for w in ("W1", "W3"):
            for c in COMPARATORS:
                r = con.loc[f"{w}-vs-{c}"]
                k = f"{panel}/{w}vs{KEY[c]}"
                put(k + "/d", sgn(r.mean_diff)); put(k + "/lo", sgn(r.ci_lo)); put(k + "/hi", sgn(r.ci_hi))
                put(k + "/wins", str(int(r.wins))); put(k + "/losses", str(int(r.losses)))
                put(k + "/p", pv(r.p)); put(k + "/ph", pv(r.p_holm))
    ctrl = pd.read_csv(REPRO / "stats" / "posthoc_W1_vs_CTRL.csv").set_index("panel")
    for panel in PANEL_ORDER:
        r = ctrl.loc[panel]
        k = f"{panel}/W1vsCTRL"
        put(k + "/d", sgn(r.mean_diff)); put(k + "/wins", str(int(r.wins)))
        put(k + "/losses", str(int(r.losses))); put(k + "/ph", pv(r.p_holm_across_panels))
    return means, ctrl


def table_means(means, lang, metric="rho"):
    cols = PANEL_ORDER
    head = ({"en": "Method", "gr": "Μέθοδος"}[lang] + " & "
            + " & ".join(rf"\texttt{{{p}}}" for p in cols) + r" \\")
    rows = [r"\toprule", head, r"\midrule"]
    for m in METHODS:
        vals = []
        for p in cols:
            col = means[p][metric]
            best = col.drop("oracle:scalar").max()
            v = f(col[m])
            vals.append(rf"\textbf{{{v}}}" if m != "oracle:scalar" and col[m] == best else v)
        rows.append(LABEL[lang][m] + " & " + " & ".join(vals) + r" \\")
        if m == "CTRL_lin0.9":
            rows.append(r"\midrule")
        if m == "oracle:scalar":
            rows.append(r"\midrule")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def table_contrasts(lang, w="W1"):
    head = {"en": r"Panel & Comparator & $\Delta\bar\rho$ & 95\% CI & W/L & $p$ & $p_{\mathrm{Holm}}$ \\",
            "gr": r"Σύνολο & Μέθοδος & $\Delta\bar\rho$ & Δ.Ε. 95\% & Ν/Η & $p$ & $p_{\mathrm{Holm}}$ \\"}[lang]
    rows = [r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    for panel in PANEL_ORDER:
        con = pd.read_csv(REPRO / "stats" / f"contrasts_{panel}.csv").set_index("contrast")
        for j, c in enumerate(COMPARATORS):
            r = con.loc[f"{w}-vs-{c}"]
            pn = rf"\texttt{{{panel}}}" if j == 0 else ""
            star = r"$^{*}$" if r.p_holm < 0.05 else ""
            rows.append(rf"{pn} & {LABEL[lang][c]} & {sgn(r.mean_diff)}{star} & "
                        rf"[{sgn(r.ci_lo)}, {sgn(r.ci_hi)}] & {int(r.wins)}/{int(r.losses)} & "
                        rf"{pv(r.p)} & {pv(r.p_holm)} \\")
        if panel != PANEL_ORDER[-1]:
            rows.append(r"\midrule")
    return "\n".join(rows)


def table_ctrl(ctrl, means, lang):
    head = {"en": r"Panel & W1 & CTRL & $\Delta\bar\rho$ & W/L & $p_{\mathrm{Holm}}$ \\",
            "gr": r"Σύνολο & W1 & CTRL & $\Delta\bar\rho$ & Ν/Η & $p_{\mathrm{Holm}}$ \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    for panel in PANEL_ORDER:
        r = ctrl.loc[panel]
        rows.append(rf"\texttt{{{panel}}} & {f(means[panel].loc['W1', 'rho'])} & "
                    rf"{f(means[panel].loc['CTRL_lin0.9', 'rho'])} & {sgn(r.mean_diff)} & "
                    rf"{int(r.wins)}/{int(r.losses)} & {pv(r.p_holm_across_panels)} \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def table_size(means, info, lang):
    head = {"en": r"Panel & Vertices & W1 & oracle & W1 $-$ oracle & IRIE & W1 $-$ IRIE \\",
            "gr": r"Σύνολο & Κορυφές & W1 & μαντείο & W1 $-$ μαντείο & IRIE & W1 $-$ IRIE \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    order = sorted(PANEL_ORDER, key=lambda p: (info[p]["nmax"], info[p]["nmin"]))
    for p in order:
        g = means[p]["rho"]
        i = info[p]
        n = str(i["nmin"]) if i["nmin"] == i["nmax"] else f"{i['nmin']}--{i['nmax']}"
        rows.append(rf"\texttt{{{p}}} & {n} & {f(g['W1'])} & {f(g['oracle:scalar'])} & "
                    rf"{sgn(g['W1'] - g['oracle:scalar'])} & {f(g['IRIE_exact'])} & "
                    rf"{sgn(g['W1'] - g['IRIE_exact'])} \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def table_regime(lang):
    head = {"en": r"Panel & Regime & $n$ & W1 & oracle & IRIE & MP2 & EW rule \\",
            "gr": r"Σύνολο & Καθεστώς & $n$ & W1 & μαντείο & IRIE & MP2 & κανόνας EW \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    for panel in ("test2", "scale2", "big"):
        t = pd.read_csv(REPRO / "stats" / f"by_regime_{panel}.csv")
        for j, (_, r) in enumerate(t.iterrows()):
            pn = rf"\texttt{{{panel}}}" if j == 0 else ""
            for key, col in (("W1", "W1"), ("oracle", "oracle:scalar"), ("IRIE", "IRIE_exact"),
                             ("MP2", "MP2"), ("ruleEW", "rule_EW")):
                put(f"reg/{panel}/{r.regime}/{key}", f(r[col], 3))
            rows.append(rf"{pn} & {r.regime} & {int(r.n)} & {f(r.W1, 3)} & {f(r['oracle:scalar'], 3)} & "
                        rf"{f(r.IRIE_exact, 3)} & {f(r.MP2, 3)} & {f(r.rule_EW, 3)} \\")
        if panel != "big":
            rows.append(r"\midrule")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


# ---------------------------------------------------------- external hold-out
EXT_LABEL = {
    "en": {"W1": "W1 (this work)", "W3": "W3 (this work)", "CTRL_lin0_9": "CTRL (paper configuration)",
           "MC 200 samples": "Monte Carlo, 200 worlds", "MC 50 samples": "Monte Carlo, 50 worlds",
           "MC 10 samples": "Monte Carlo, 10 worlds", "RIS 50 samples": "RIS, 50 samples",
           "IRIE (ICDM12)": "IRIE", "IRIE row-normalised": "IRIE, row-normalised",
           "MIA/PMIA (KDD10)": "PMIA", "oracle a": "oracle (best constant damping)",
           "THIS WORK": "escape-scaled pull, calibrated rule",
           "PPR push rule-a": "PPR push, calibrated rule", "PPR push a=.85": r"PPR push, $\alpha=0.85$",
           "PPR pull rule-a": "PPR pull, calibrated rule", "PPR pull a=.85": r"PPR pull, $\alpha=0.85$",
           "pull x P_esc post hoc": r"PPR pull $\times P_{\mathrm{esc}}$",
           "escape-scaled a=1": r"escape-scaled pull, $\alpha=1$",
           "P_esc alone": r"$P_{\mathrm{esc}}$", "sum_j p_ij": r"$\sum_j p_{ij}$",
           "degree discount (KDD09)": "degree discount", "x only": "prior only"},
    "gr": {"W1": "W1 (παρούσα εργασία)", "W3": "W3 (παρούσα εργασία)",
           "CTRL_lin0_9": "CTRL (διαμόρφωση του άρθρου)",
           "MC 200 samples": "Monte Carlo, 200 κόσμοι", "MC 50 samples": "Monte Carlo, 50 κόσμοι",
           "MC 10 samples": "Monte Carlo, 10 κόσμοι", "RIS 50 samples": "RIS, 50 δείγματα",
           "IRIE (ICDM12)": "IRIE", "IRIE row-normalised": "IRIE, κανονικοποίηση γραμμών",
           "MIA/PMIA (KDD10)": "PMIA", "oracle a": "μαντείο (βέλτιστη σταθερή απόσβεση)",
           "THIS WORK": "έλξη με κλιμάκωση διαφυγής, βαθμονομημένος κανόνας",
           "PPR push rule-a": "PPR ώθησης, βαθμονομημένος κανόνας",
           "PPR push a=.85": r"PPR ώθησης, $\alpha=0.85$",
           "PPR pull rule-a": "PPR έλξης, βαθμονομημένος κανόνας",
           "PPR pull a=.85": r"PPR έλξης, $\alpha=0.85$",
           "pull x P_esc post hoc": r"PPR έλξης $\times P_{\mathrm{esc}}$",
           "escape-scaled a=1": r"έλξη με κλιμάκωση διαφυγής, $\alpha=1$",
           "P_esc alone": r"$P_{\mathrm{esc}}$", "sum_j p_ij": r"$\sum_j p_{ij}$",
           "degree discount (KDD09)": "έκπτωση βαθμού (degree discount)",
           "x only": "μόνο η αρχική κατανομή"},
}


def external():
    rep = jload(REPRO / "external" / "external_holdout_check.json")
    df = pd.read_csv(REPRO / "external" / "external_holdout.csv")
    cols = {"W1": df.W1.mean(), "W3": df.W3.mean(), "CTRL_lin0_9": df.CTRL_lin0_9.mean()}
    for c in df.columns:
        if c.startswith("stored:"):
            cols[c[7:]] = df[c].mean()
    ranking = sorted(cols.items(), key=lambda kv: -kv[1])
    put("ext/n", str(rep["instances"]))
    for k, v in (("W1", "W1"), ("W3", "W3"), ("CTRL", "CTRL_lin0_9"), ("MC200", "MC 200 samples"),
                 ("MC50", "MC 50 samples"), ("IRIE", "IRIE (ICDM12)"), ("PMIA", "MIA/PMIA (KDD10)"),
                 ("oracle", "oracle a"), ("rule", "THIS WORK"), ("push", "PPR push rule-a")):
        put(f"ext/{k}", f(cols[v]))
    put("ext/W1rank", str([k for k, _ in ranking].index("W1") + 1))
    put("ext/methods", str(len(ranking)))
    put("ext/scipymax", sci((df.W1 - df.W1_scipy).abs().max()))
    by = df.groupby("topology")[["W1", "stored:MC 200 samples", "stored:IRIE (ICDM12)", "stored:THIS WORK"]].mean()
    for t, r in by.iterrows():
        put(f"ext/{t}/W1", f(r.W1, 3)); put(f"ext/{t}/MC200", f(r["stored:MC 200 samples"], 3))
        put(f"ext/{t}/IRIE", f(r["stored:IRIE (ICDM12)"], 3)); put(f"ext/{t}/rule", f(r["stored:THIS WORK"], 3))
    return ranking, by


def table_external(ranking, lang):
    head = {"en": r"\# & Method & $\bar\rho$ \\", "gr": r"\# & Μέθοδος & $\bar\rho$ \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    for i, (k, v) in enumerate(ranking, 1):
        lab = EXT_LABEL[lang].get(k, k)
        if k in ("W1", "W3", "CTRL_lin0_9"):
            lab = rf"\textbf{{{lab}}}" if k == "W1" else lab
        rows.append(rf"{i} & {lab} & {f(v)} \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def table_external_topo(by, lang):
    names = {"en": {"er": "Erdős--Rényi", "ba": "Barabási--Albert", "grid": "grid", "ws": "Watts--Strogatz"},
             "gr": {"er": "Erdős--Rényi", "ba": "Barabási--Albert", "grid": "πλέγμα", "ws": "Watts--Strogatz"}}[lang]
    head = {"en": r"Topology & W1 & MC-200 & IRIE & escape-scaled rule \\",
            "gr": r"Τοπολογία & W1 & MC-200 & IRIE & κανόνας διαφυγής \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    for t in ("er", "ba", "ws", "grid"):
        r = by.loc[t]
        rows.append(rf"{names[t]} & {f(r.W1, 3)} & {f(r['stored:MC 200 samples'], 3)} & "
                    rf"{f(r['stored:IRIE (ICDM12)'], 3)} & {f(r['stored:THIS WORK'], 3)} \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


# ------------------------------------------------------------- exploratory
def exploratory():
    out = {}
    cost = pd.concat([pd.read_csv(REPRO / "exploratory" / f"cost_{p}.csv") for p in ("test2", "scale2", "big")])
    out["cost"] = cost
    for p, g in cost.groupby("panel"):
        put(f"cost/{p}/W1rho", f(g.W1_rho.mean(), 3)); put(f"cost/{p}/W1ms", f(g.W1_ms.median(), 1))
        put(f"cost/{p}/itA", str(int(g.phaseA_iterations.median())))
        for k in (50, 100, 200, 500):
            put(f"cost/{p}/MC{k}rho", f(g[f"MC{k}_rho"].mean(), 3))
            put(f"cost/{p}/MC{k}ms", f(g[f"MC{k}_ms"].median(), 1))
        from w1.stats import paired
        for k in (200, 500):
            r = paired(g.W1_rho, g[f"MC{k}_rho"])
            put(f"cost/{p}/vsMC{k}/d", sgn(r["mean_diff"], 3))
            put(f"cost/{p}/vsMC{k}/wl", f"{r['wins']}/{r['losses']}")
            put(f"cost/{p}/vsMC{k}/p", pv(r["p"]))
        put(f"cost/{p}/ratio200", f(g.MC200_ms.median() / g.W1_ms.median(), 1))
        put(f"cost/{p}/ratio500", f(g.MC500_ms.median() / g.W1_ms.median(), 1))
    big = cost[cost.panel != "test2"]
    for reg, g in big.groupby("regime"):
        put(f"cost/large/{reg}/W1", f(g.W1_rho.mean(), 3)); put(f"cost/large/{reg}/MC200", f(g.MC200_rho.mean(), 3))
        put(f"cost/large/{reg}/MC500", f(g.MC500_rho.mean(), 3))
    put("cost/maxit", str(int(cost.phaseA_iterations.max())))
    put("cost/nonconv", str(int((cost.phaseA_iterations >= 20000).sum())))
    put("cost/total", str(len(cost)))
    ab = pd.read_csv(REPRO / "exploratory" / "ablations.csv")
    out["ab"] = ab
    conf = ab[ab.panel != "dev"]
    for p, g in conf.groupby("panel"):
        for c in ("W1", "phaseA_only", "bridge_normalised", "phaseA_unnormalised", "phaseA_unnormalised_only"):
            put(f"ab/{p}/{c.replace('_', '')}", f(g[c].mean()))
        put(f"ab/{p}/alpha1", str(int((g.alpha_A >= 1.0).sum())))
    put("ab/alpha1all", str(int((conf.alpha_A >= 1.0).sum())))
    put("ab/instances", str(len(conf)))
    dev = ab[ab.panel == "dev"]
    for c in [c for c in ab.columns if c.startswith("gamma")]:
        put(f"ab/dev/{c}", f(dev[c].mean()))
    for p, g in conf.groupby("panel"):
        for c in [c for c in ab.columns if re.fullmatch(r"K\d+", c)]:
            put(f"ab/{p}/{c}", f(g[c].mean()))
    cell = conf.groupby(["panel", "family", "regime"])[[c for c in ab.columns if re.fullmatch(r"K\d+", c)]].mean()
    for (p, fam, reg), r in cell.iterrows():
        for c, v in r.items():
            put(f"ab/{p}/{fam}/{reg}/{c}", f(v, 3))
    reg = conf.groupby(["panel", "regime"])[["W1", "phaseA_unnormalised", "alpha_A"]].mean()
    for (p, rg), r in reg.iterrows():
        put(f"ab/{p}/{rg}/W1", f(r.W1, 3)); put(f"ab/{p}/{rg}/unnorm", f(r.phaseA_unnormalised, 3))
        put(f"ab/{p}/{rg}/alpha", f(r.alpha_A, 3))
    regk = conf.groupby(["panel", "regime"])[[c for c in ab.columns if re.fullmatch(r"K\d+", c)]].mean()
    for (p, rg), r in regk.iterrows():
        for c, v in r.items():
            put(f"ab/{p}/{rg}/{c}", f(v, 3))
    return out


def table_cost(cost, lang):
    head = {"en": r"Panel & Vertices & W1 & MC-50 & MC-100 & MC-200 & MC-500 \\",
            "gr": r"Σύνολο & Κορυφές & W1 & MC-50 & MC-100 & MC-200 & MC-500 \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    for p in ("test2", "scale2", "big"):
        g = cost[cost.panel == p]
        n = str(g.n.min()) if g.n.min() == g.n.max() else f"{g.n.min()}--{g.n.max()}"
        cells = [rf"{f(g.W1_rho.mean(), 3)} / {f(g.W1_ms.median(), 1)}"]
        for k in (50, 100, 200, 500):
            cells.append(rf"{f(g[f'MC{k}_rho'].mean(), 3)} / {f(g[f'MC{k}_ms'].median(), 1)}")
        rows.append(rf"\texttt{{{p}}} & {n} & " + " & ".join(cells) + r" \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def table_cost_regime(cost, lang):
    head = {"en": r"Edge probabilities & Instances & W1 & MC-200 & MC-500 & W1 ms & MC-200 ms \\",
            "gr": r"Πιθανότητες ακμών & Στιγμιότυπα & W1 & MC-200 & MC-500 & W1 ms & MC-200 ms \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    big = cost[cost.panel != "test2"]
    from w1.instances import PROB_RANGES
    for reg, g in big.groupby("regime"):
        lo, hi = PROB_RANGES[reg]
        rows.append(rf"{reg} [{lo:g}, {hi:g}] & {len(g)} & {f(g.W1_rho.mean(), 3)} & "
                    rf"{f(g.MC200_rho.mean(), 3)} & {f(g.MC500_rho.mean(), 3)} & "
                    rf"{f(g.W1_ms.median(), 1)} & {f(g.MC200_ms.median(), 1)} \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def table_ablation(ab, lang):
    head = {"en": r"Panel & W1 & bridge $/(1{+}\gamma)$ & no Phase-A normalisation & Phase A only \\",
            "gr": r"Σύνολο & W1 & γέφυρα $/(1{+}\gamma)$ & χωρίς κανονικοποίηση Φάσης Α & μόνο Φάση Α \\"}[lang]
    rows = [r"\toprule", head, r"\midrule"]
    conf = ab[ab.panel != "dev"]
    for p in ("test2", "scale2", "prior2", "real2", "big"):
        g = conf[conf.panel == p]
        rows.append(rf"\texttt{{{p}}} & {f(g.W1.mean())} & {f(g.bridge_normalised.mean())} & "
                    rf"{f(g.phaseA_unnormalised.mean())} & {f(g.phaseA_only.mean())} \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def table_ksweep(ab, lang):
    ks = [c for c in ab.columns if re.fullmatch(r"K\d+", c)]
    head = {"en": "Panel", "gr": "Σύνολο"}[lang] + " & " + " & ".join(f"$K{{=}}{c[1:]}$" for c in ks) + r" \\"
    rows = [r"\toprule", head, r"\midrule"]
    conf = ab[ab.panel != "dev"]
    for p in ("test2", "scale2", "prior2", "real2", "big"):
        g = conf[conf.panel == p][ks].mean()
        best = g.max()
        cells = [rf"\textbf{{{f(v, 3)}}}" if v == best else f(v, 3) for v in g]
        rows.append(rf"\texttt{{{p}}} & " + " & ".join(cells) + r" \\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


# ---------------------------------------------------------- reproduction
def reproduction():
    reg = jload(REPRO / "data" / "regeneration.json")
    put("repro/instances", str(sum(v["instances"] for v in reg.values())))
    put("repro/regenerated", str(sum(v["regenerated_identical"] for v in reg.values())))
    put("repro/gt", str(sum(v["ground_truth_identical"] for v in reg.values())))
    sc = jload(REPRO / "selection" / "search_spotcheck.json")
    ev = sum(v["evaluations"] for k, v in sc.items() if isinstance(v, dict))
    cf = sum(v["configurations"] for k, v in sc.items() if isinstance(v, dict))
    put("repro/searchevals", f"{ev:,}".replace(",", "\\,")); put("repro/searchcfgs", str(cf))
    put("repro/searchmax", sci(sc["max_abs_diff_overall"]))
    worst, rows = 0.0, 0
    for p in PANEL_ORDER:
        c = jload(REPRO / "confirm" / f"confirm_{p}_check.json")
        worst = max(worst, max(c["max_abs_diff_vs_source"].values()))
        rows += c["rows_compared_with_source"]
    put("repro/confirmrows", f"{rows:,}".replace(",", "\\,"))
    put("repro/confirmmax", sci(worst))
    st = jload(REPRO / "stats" / "summary.json")
    smax = max(max(v["matches_source"]["mean_diff"], v["matches_source"]["p"], v["matches_source"]["p_holm"])
               for v in st.values() if "matches_source" in v)
    put("repro/statsmax", sci(smax) if smax > 0 else "0")
    return reg, sc


# ----------------------------------------------------------------- README
def readme_blocks(means):
    cols = ("test2", "scale2", "prior2", "real2", "big")
    for lang, path in (("en", ROOT / "README.md"), ("gr", ROOT / "README_GR.md")):
        lab = {"en": "method", "gr": "μέθοδος"}[lang]
        lines = [f"| {lab} | " + " | ".join(f"`{c}`" for c in cols) + " |",
                 "|---|" + "---:|" * len(cols)]
        for m in ("W1", "W3", "CTRL_lin0.9", "IRIE_exact", "rule_EW", "rule_WT", "MP2", "oracle:scalar"):
            name = {"CTRL_lin0.9": "CTRL_lin0.9 (paper)" if lang == "en" else "CTRL_lin0.9 (άρθρο)",
                    "oracle:scalar": "oracle:scalar" + (" (upper bound)" if lang == "en" else " (άνω φράγμα)")}.get(m, m)
            vals = []
            for c in cols:
                v = means[c].loc[m, "rho"]
                best = means[c]["rho"].drop("oracle:scalar").max()
                vals.append(f"**{v:.4f}**" if (m != "oracle:scalar" and v == best) else f"{v:.4f}")
            lines.append(f"| {name} | " + " | ".join(vals) + " |")
        n = {c: len(panel_paths(DATA, c)) for c in cols}
        note = {"en": "Instances: " + ", ".join(f"`{c}` {n[c]}" for c in cols)
                + ". Paired Wilcoxon tests with Holm correction, bootstrap intervals and the external hold-out are in the evidence report.",
                "gr": "Στιγμιότυπα: " + ", ".join(f"`{c}` {n[c]}" for c in cols)
                + ". Οι ζευγαρωτοί έλεγχοι Wilcoxon με διόρθωση Holm, τα διαστήματα bootstrap και το εξωτερικό σύνολο ελέγχου βρίσκονται στην αναφορά τεκμηρίωσης."}[lang]
        block = "\n".join(lines) + "\n\n" + note
        txt = path.read_text()
        txt = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->",
                     lambda _: f"<!-- RESULTS:BEGIN -->\n{block}\n<!-- RESULTS:END -->", txt, flags=re.S)
        repro = {"en": [
            f"* all {V['repro/instances']} instances regenerate bit for bit from their specifications ({V['repro/gt']} ground truths recomputed and identical);",
            "* the SHA-256 digest of the frozen selection matches, and W1 is the configuration with the highest mean development ρ among the stored search results;",
            f"* {V['repro/searchevals'].replace(chr(92) + ',', ',')} re-evaluations of {V['repro/searchcfgs']} search configurations on the development panel agree with the stored values to within 1e-16;",
            f"* the {V['repro/confirmrows'].replace(chr(92) + ',', ',')} stored confirmation rows (4 metrics each) and their statistics are reproduced to within 1e-16;",
            "* the external hold-out values of W1 and W3 are reproduced exactly."],
            "gr": [
            f"* και τα {V['repro/instances']} στιγμιότυπα αναπαράγονται bit προς bit από τις προδιαγραφές τους ({V['repro/gt']} τιμές αναφοράς υπολογίστηκαν ξανά και είναι ταυτόσημες)·",
            "* το αποτύπωμα SHA-256 της κλειδωμένης επιλογής συμφωνεί, και ο W1 είναι η διαμόρφωση με τον μεγαλύτερο μέσο ρ ανάπτυξης στα αποθηκευμένα αποτελέσματα της αναζήτησης·",
            f"* {V['repro/searchevals'].replace(chr(92) + ',', '.')} νέες αξιολογήσεις {V['repro/searchcfgs']} διαμορφώσεων της αναζήτησης στο σύνολο ανάπτυξης συμφωνούν με τις αποθηκευμένες τιμές με απόκλιση έως 1e-16·",
            f"* οι {V['repro/confirmrows'].replace(chr(92) + ',', '.')} αποθηκευμένες γραμμές επιβεβαίωσης (4 μέτρα η καθεμία) και τα στατιστικά τους αναπαράγονται με απόκλιση έως 1e-16·",
            "* οι τιμές των W1 και W3 στο εξωτερικό σύνολο ελέγχου αναπαράγονται ακριβώς."]}[lang]
        txt = re.sub(r"<!-- REPRO:BEGIN -->.*?<!-- REPRO:END -->",
                     lambda _: "<!-- REPRO:BEGIN -->\n" + "\n".join(repro) + "\n<!-- REPRO:END -->", txt, flags=re.S)
        path.write_text(txt)


def main():
    info = panels()
    sel, top, by_steps, by_op = selection()
    means, ctrl = confirmation()
    ranking, by = external()
    ex = exploratory()
    reproduction()
    tables = {}
    for lang in ("en", "gr"):
        tables[lang] = {
            "TabPanels": table_panels(info, lang),
            "TabTop": table_top(top, lang),
            "TabSteps": table_steps(by_steps, by_op, lang),
            "TabMeans": table_means(means, lang),
            "TabMeansTau": table_means(means, lang, "tau"),
            "TabMeansTopk": table_means(means, lang, "topk"),
            "TabMeansNdcg": table_means(means, lang, "ndcg"),
            "TabContrastsWone": table_contrasts(lang, "W1"),
            "TabContrastsWthree": table_contrasts(lang, "W3"),
            "TabCtrl": table_ctrl(ctrl, means, lang),
            "TabSize": table_size(means, info, lang),
            "TabRegime": table_regime(lang),
            "TabExternal": table_external(ranking, lang),
            "TabExternalTopo": table_external_topo(by, lang),
            "TabCost": table_cost(ex["cost"], lang),
            "TabCostRegime": table_cost_regime(ex["cost"], lang),
            "TabAblation": table_ablation(ex["ab"], lang),
            "TabKsweep": table_ksweep(ex["ab"], lang),
        }
    with open(GEN / "values.tex", "w") as fh:
        fh.write("% generated by scripts/make_doc_assets.py -- do not edit\n")
        for k in sorted(V):
            fh.write(f"\\expandafter\\def\\csname v@{k}\\endcsname{{{V[k]}}}\n")
    for lang in ("en", "gr"):
        with open(GEN / f"tables_{lang}.tex", "w") as fh:
            fh.write("% generated by scripts/make_doc_assets.py -- do not edit\n")
            for name, body in tables[lang].items():
                fh.write(f"\\newcommand{{\\{name}}}{{%\n{body}\n}}\n")
    readme_blocks(means)
    print(f"{len(V)} values, tables and README blocks written")


if __name__ == "__main__":
    main()
