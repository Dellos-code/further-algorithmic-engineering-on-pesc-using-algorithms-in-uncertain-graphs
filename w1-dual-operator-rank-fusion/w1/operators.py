"""Escape probability, propagation operators and the calibrated damping rules."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

P_CLIP = 1.0 - 1e-12          # p used inside logarithms is clipped below 1

# Calibrated scalar damping rules of the earlier escape-probability study:
# alpha = clip(c0 + c1 * mean(P_esc), 0.4, 1.0)
RULE_EW = (0.502, 0.558)      # escape-scaled pull operator EW
RULE_WT = (0.241, 0.859)      # push personalised PageRank operator W^T
RULE_CLIP = (0.4, 1.0)


def escape_probability(n: int, E: np.ndarray, p: np.ndarray) -> np.ndarray:
    """P_esc[i] = 1 - prod_{j~i} (1 - p_ij): probability that i has at least one live edge."""
    pc = np.minimum(p, P_CLIP)
    l1m = np.log1p(-pc)
    logq = np.bincount(E[:, 0], weights=l1m, minlength=n) + np.bincount(E[:, 1], weights=l1m, minlength=n)
    return -np.expm1(logq)


def build_operators(inst) -> dict:
    """Operators used by W1 and by the comparators.

    W   row-stochastic, weights w_ij = -ln(1 - p_ij)
    EW  diag(P_esc) W            (escape-scaled pull; Phase A of W1)
    WT  W^T                      (push personalised PageRank)
    P   raw probability adjacency
    EP  diag(P_esc) P            (Phase B of W1)
    """
    n, E, p = inst.n, inst.edges, inst.p
    i, j = E[:, 0], E[:, 1]
    pc = np.minimum(p, P_CLIP)
    lw = -np.log1p(-pc)
    rows, cols = np.r_[i, j], np.r_[j, i]
    Alw = sp.csr_matrix((np.r_[lw, lw], (rows, cols)), shape=(n, n))
    P = sp.csr_matrix((np.r_[p, p], (rows, cols)), shape=(n, n))
    dw = np.asarray(Alw.sum(1)).ravel()
    inv = np.divide(1.0, dw, out=np.zeros(n), where=dw > 0)
    W = (sp.diags(inv) @ Alw).tocsr()
    Pesc = escape_probability(n, E, p)
    return dict(
        Pesc=Pesc,
        Pbar=float(Pesc.mean()),
        deg=np.bincount(E.ravel(), minlength=n),
        deff=np.asarray(P.sum(1)).ravel(),
        W=W,
        EW=(sp.diags(Pesc) @ W).tocsr(),
        WT=W.T.tocsr(),
        P=P,
        EP=(sp.diags(Pesc) @ P).tocsr(),
    )


def search_operators(inst) -> dict:
    """``build_operators`` plus the two extra Phase-B operators of the search space."""
    ops = dict(inst.ops())
    P = ops["P"].tocsr()
    plog = P.copy()
    plog.data = -np.log1p(-np.minimum(plog.data, 1.0 - 1e-12))
    ops["Plog"] = plog
    rs = np.asarray(P.sum(1)).ravel()
    ops["Prow"] = (sp.diags(np.divide(1.0, rs, out=np.zeros(inst.n), where=rs > 0)) @ P).tocsr()
    return ops


def rule_alpha(Pbar: float, family: str = "EW") -> float:
    c0, c1 = RULE_EW if family == "EW" else RULE_WT
    return float(np.clip(c0 + c1 * Pbar, *RULE_CLIP))
