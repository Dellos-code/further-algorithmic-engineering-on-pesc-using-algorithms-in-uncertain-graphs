"""Damped diffusions and the propagation-based comparators."""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.special import expit, logit

from .instances import stable_seed
from .operators import P_CLIP

ALPHA_GRID = np.round(np.arange(1, 41) * 0.025, 4)          # 0.025, 0.050, ..., 1.000


def diffuse(T: sp.csr_matrix, x: np.ndarray, alpha, y0: np.ndarray | None = None,
            tol: float = 1e-12, maxit: int = 20000, lazy: bool = True):
    """Normalised damped diffusion  y <- N((1-a) x + a T y), scalar or per-node a.

    The fixed point is the Perron vector of B = ((1-a) x) 1^T + diag(a) T.  The
    lazy form iterates (I+B)/2, which has the same Perron vector and converges on
    bipartite graphs, where the plain power method oscillates.
    Returns (y, iterations)."""
    a = np.broadcast_to(np.asarray(alpha, float), x.shape)
    u = (1.0 - a) * x
    y = (x if y0 is None else np.asarray(y0, float)).copy()
    y /= y.sum()
    it = 0
    for it in range(1, maxit + 1):
        By = u + a * (T @ y)
        yn = 0.5 * (y + By) if lazy else By
        s = yn.sum()
        if not s > 0:
            return x.copy(), it
        yn /= s
        d = np.abs(yn - y).sum()
        y = yn
        if d < tol:
            break
    return y, it


def diffuse_batch(T: sp.csr_matrix, x: np.ndarray, A: np.ndarray, Y0: np.ndarray | None = None,
                  tol: float = 1e-12, maxit: int = 20000):
    """Batched normalised diffusion; A is (K,) scalars or (n, K) per-node dampings."""
    n = len(x)
    if A.ndim == 1:
        A = np.broadcast_to(A[None, :], (n, len(A)))
    U = (1.0 - A) * x[:, None]
    Y = (np.repeat(x[:, None], A.shape[1], 1) if Y0 is None else np.array(Y0, float)).copy()
    Y /= Y.sum(0, keepdims=True)
    it = 0
    for it in range(1, maxit + 1):
        Yn = 0.5 * (Y + U + A * (T @ Y))
        Yn /= Yn.sum(0, keepdims=True)
        d = np.abs(Yn - Y).sum(0).max()
        Y = Yn
        if d < tol:
            break
    return Y, it


def scalar_sweep(T, x, grid=ALPHA_GRID, tol=1e-12, maxit=20000):
    """Fixed points of the normalised diffusion for every scalar damping of the grid."""
    return diffuse_batch(T, x, np.asarray(grid, float), tol=tol, maxit=maxit)


def anchored_alpha(q: np.ndarray, alpha0: float, eta: float, amax: float = 1.0 - 1e-6,
                   amin: float = 1e-6) -> np.ndarray:
    """Anchored proportional law AP(eta): odds_i = exp(c + eta (log q_i - mean log q)),
    with c chosen so that mean_i alpha_i = alpha0 (safeguarded Newton)."""
    n = len(q)
    if eta == 0.0:
        return np.full(n, float(alpha0))
    a0 = float(np.clip(alpha0, amin, amax))
    lq = np.log(np.maximum(q, 1e-300))
    l = eta * (lq - lq.mean())
    lo, hi = logit(amin) - np.abs(l).max() - 1.0, logit(amax) + np.abs(l).max() + 1.0
    c = float(logit(a0))
    for _ in range(60):
        s = expit(c + l)
        f = s.mean() - a0
        if abs(f) < 1e-13:
            break
        if f < 0:
            lo = c
        else:
            hi = c
        d = float((s * (1.0 - s)).mean())
        cn = c - f / d if d > 1e-300 else 0.5 * (lo + hi)
        c = cn if lo < cn < hi else 0.5 * (lo + hi)
    return np.clip(expit(c + l), amin, amax)


@lru_cache(maxsize=2048)
def _spectral_radius_cached(key: bytes, shape: tuple, data: tuple, indices: tuple,
                            indptr: tuple, seed: int) -> float:
    T = sp.csr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=shape)
    if T.shape[0] == 1:
        return float(abs(T[0, 0]))
    try:
        # NOTE: eigsh assumes a symmetric matrix.  The source study applied it to
        # every operator, including the non-symmetric EW, EP, W and Prow; the
        # value only enters the "spectral:c" damping family of the search space,
        # which no frozen configuration uses.  Kept unchanged for reproducibility.
        return float(abs(eigsh(T, k=1, which="LM", tol=1e-10, maxiter=5000,
                               v0=np.ones(T.shape[0]))[0][0]))
    except Exception:
        v = np.random.default_rng(seed).random(T.shape[0])
        v /= np.linalg.norm(v)
        r = 0.0
        for _ in range(250):
            w = T @ v
            nw = np.linalg.norm(w)
            if nw == 0:
                return 0.0
            v, r = w / nw, nw
        return float(r)


def spectral_radius(T: sp.spmatrix, seed: int = 0) -> float:
    A = T.tocsr()
    return _spectral_radius_cached(
        hashlib.sha1(A.data.tobytes() + A.indices.tobytes() + A.indptr.tobytes()).digest(),
        A.shape, tuple(A.data), tuple(A.indices), tuple(A.indptr), int(seed))


# ----------------------------------------------------------------- comparators
def knz_mp(inst, iters: int | None = None, tol: float = 1e-12, maxit: int = 2000):
    """Karrer-Newman-Zdeborova message passing in membership variables.
    iters=None runs to convergence; iters=0 returns P_esc exactly."""
    n, E = inst.n, inst.edges
    m = len(E)
    pc = np.minimum(inst.p, P_CLIP)
    src = np.r_[E[:, 0], E[:, 1]]
    dst = np.r_[E[:, 1], E[:, 0]]
    pe = np.r_[pc, pc]
    rev = np.r_[np.arange(m) + m, np.arange(m)]
    nu = np.ones(2 * m)
    K = maxit if iters is None else iters
    k = 0
    for k in range(K):
        lg = np.log1p(-pe * nu)
        Lsum = np.bincount(dst, weights=lg, minlength=n)
        nu_new = -np.expm1(Lsum[src] - lg[rev])
        d = np.abs(nu_new - nu).max()
        nu = nu_new
        if iters is None and d < tol:
            break
    mu = -np.expm1(np.bincount(dst, weights=np.log1p(-pe * nu), minlength=n))
    return mu, k


def irie_exact(inst) -> np.ndarray:
    """IRIE influence ranking (Jung, Heo and Chen, ICDM 2012) on the raw probability
    matrix: y = x + a P y with a = min(0.7, 0.95 / rho(P))."""
    T = inst.ops()["P"]
    rng = np.random.default_rng(stable_seed(inst.name, "irie", "rho"))
    v = rng.random(inst.n)
    v /= np.linalg.norm(v)
    rho = 0.0
    for _ in range(60):
        w = T @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            rho = 0.0
            break
        v, rho = w / nw, float(nw)
    a = min(0.7, 0.95 / rho) if rho > 0 else 0.0
    y = inst.x.copy()
    for _ in range(400):
        yn = inst.x + a * (T @ y)
        if np.max(np.abs(yn - y)) < 1e-12 * max(1.0, np.max(np.abs(y))):
            y = yn
            break
        y = yn
    return y


def irie_wc(inst) -> np.ndarray:
    T = inst.ops()["W"]
    y = inst.x.copy()
    for _ in range(400):
        yn = inst.x + 0.7 * (T @ y)
        if np.max(np.abs(yn - y)) < 1e-12 * max(1.0, np.max(np.abs(y))):
            y = yn
            break
        y = yn
    return y


def heat_kernel(T: sp.csr_matrix, x: np.ndarray, t: float, tol: float = 1e-12) -> np.ndarray:
    y = x.copy()
    term = x.copy()
    for k in range(1, 10000):
        term = (t / k) * (T @ term)
        y += term
        if np.max(np.abs(term)) <= tol * max(1.0, np.max(np.abs(y))):
            break
    return y


def total_rank(T: sp.csr_matrix, x: np.ndarray, nodes: int = 16) -> np.ndarray:
    q, w = np.polynomial.legendre.leggauss(nodes)
    out = np.zeros_like(x)
    for z, wt in zip(q, w):
        y, _ = diffuse(T, x, (z + 1.0) / 2.0, tol=1e-10, maxit=5000)
        out += 0.5 * wt * y
    return out


def rapr(T: sp.csr_matrix, x: np.ndarray, a: float, b: float, nodes: int = 16) -> np.ndarray:
    q, w = np.polynomial.legendre.leggauss(nodes)
    out = np.zeros_like(x)
    for z, wt in zip(q, w):
        alpha = (z + 1.0) / 2.0
        density = alpha ** (a - 1.0) * (1 - alpha) ** (b - 1.0)
        y, _ = diffuse(T, x, alpha, tol=1e-10, maxit=5000)
        out += 0.5 * wt * density * y
    return out
