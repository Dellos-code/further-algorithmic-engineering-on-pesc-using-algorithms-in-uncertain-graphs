"""Ground-truth-free graph covariates used by fusion selection."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from w1.diffusion import knz_mp


def _rho_p(inst):
    n = inst.n
    e = inst.edges
    p = inst.p
    mat = coo_matrix((np.r_[p, p], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                     shape=(n, n)).tocsr()
    if n < 3:
        return float(mat.toarray().max(initial=0.0))
    return float(max(eigsh(mat, k=1, which="LM", return_eigenvectors=False)[0], 0.0))


def rho_nb(inst, iterations: int = 200) -> float:
    """Power-iterate the p-weighted non-backtracking operator."""
    e, p = inst.edges, np.minimum(inst.p, 1.0 - 1e-15)
    m = len(e)
    if m == 0:
        return 0.0
    src = np.r_[e[:, 0], e[:, 1]]
    dst = np.r_[e[:, 1], e[:, 0]]
    rev = np.r_[np.arange(m) + m, np.arange(m)]
    directed_p = np.r_[p, p]
    v = np.ones(2 * m) / np.sqrt(2 * m)
    value = 0.0
    for _ in range(iterations):
        sums = np.bincount(dst, weights=v, minlength=inst.n)
        nxt = directed_p * (sums[src] - v[rev])
        norm = np.linalg.norm(nxt)
        if norm == 0:
            return 0.0
        v = nxt / norm
        value = norm
    return float(value)


def covariates(inst) -> dict[str, float]:
    """Compute all specified covariates from an instance alone."""
    p = np.asarray(inst.p, float)
    e = inst.edges
    degree = np.zeros(inst.n, float)
    np.add.at(degree, e[:, 0], p)
    np.add.at(degree, e[:, 1], p)
    mu, _ = knz_mp(inst, iters=None)
    pbar = float(p.mean()) if len(p) else 0.0
    dbar = float(degree.mean())
    mean_k = dbar
    second = float(np.mean(degree * degree))
    kappa = pbar * (second - mean_k) / mean_k if mean_k > 0 else 0.0
    log_no_escape = np.bincount(
        np.r_[e[:, 0], e[:, 1]],
        weights=np.log1p(-np.minimum(np.r_[p, p], 1.0 - 1e-15)),
        minlength=inst.n)
    pesc = 1.0 - np.exp(log_no_escape)
    return {
        "pbar_E": pbar,
        "dbar": dbar,
        "Pesc_bar": float(pesc.mean()) if len(pesc) else 0.0,
        "rho_P": _rho_p(inst),
        "rho_NB": rho_nb(inst),
        "S_MP": float(np.dot(inst.x, mu)),
        "kappa_E": float(kappa),
    }
