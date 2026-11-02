"""Reference implementation of W1, line by line as in the specification
(docs/specification/W1_specification_en.pdf, Algorithm 1).

    a = Phase A   normalised lazy diffusion on EW with the calibrated scalar
                  damping alpha_A = clip(0.502 + 0.558 mean(P_esc), 0.4, 1),
                  run to the fixed point
    b = Phase B   K = 3 raw steps  b <- (1 - 0.7) x + 0.7 EP b, started from the
                  excess-scale embedding  b0 = x + 16 z  of Phase A
    s = 0.3 r(a) + 0.7 r(b)      (rank fusion; larger s = higher rank)

The search engine in ``w1.hybrid`` evaluates the same configuration through a
general code path; tests/test_algorithm.py checks that both agree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .diffusion import diffuse
from .instances import Instance
from .metrics import ranks_of
from .operators import RULE_CLIP, RULE_EW


@dataclass(frozen=True)
class W1Params:
    c0: float = RULE_EW[0]          # calibrated rule of the earlier study
    c1: float = RULE_EW[1]
    alpha_min: float = RULE_CLIP[0]
    alpha_max: float = RULE_CLIP[1]
    tol: float = 1e-12              # Phase-A convergence (L1 norm)
    max_iter: int = 20000           # Phase-A iteration cap
    gamma: float = 16.0             # mean excess ratio of the Phase-B start
    alpha_B: float = 0.7            # Phase-B damping
    K: int = 3                      # Phase-B steps
    beta: float = 0.7               # weight of Phase B in the rank fusion

    def as_dict(self) -> dict:
        return asdict(self)


W1 = W1Params()
W3 = W1Params(beta=0.75)            # the second frozen configuration


def phase_a(EW, x: np.ndarray, Pbar: float, prm: W1Params = W1):
    """Lines 8-16: returns (a, alpha_A, iterations)."""
    alpha_A = float(np.clip(prm.c0 + prm.c1 * Pbar, prm.alpha_min, prm.alpha_max))
    a, it = diffuse(EW, x, alpha_A, y0=x / x.sum(), tol=prm.tol, maxit=prm.max_iter)
    return a, alpha_A, it


def excess_embedding(a: np.ndarray, x: np.ndarray, gamma: float) -> np.ndarray:
    """Lines 17-18: z = a / mean(a/x), b0 = x + gamma z, so that mean((b0 - x)/x) = gamma."""
    z = np.maximum(a, 1e-300)
    z = z / np.mean(z / x)
    return x + gamma * z


def phase_b(EP, x: np.ndarray, b0: np.ndarray, alpha_B: float, K: int) -> np.ndarray:
    """Lines 19-20: K raw (unnormalised) affine steps."""
    b = b0.copy()
    for _ in range(K):
        b = (1.0 - alpha_B) * x + alpha_B * (EP @ b)
    return b


def w1_scores(inst: Instance, prm: W1Params = W1, return_parts: bool = False):
    """Scores s of W1 for an instance; rank vertices by decreasing s."""
    ops = inst.ops()
    x = inst.x
    a, alpha_A, it = phase_a(ops["EW"], x, ops["Pbar"], prm)
    b = phase_b(ops["EP"], x, excess_embedding(a, x, prm.gamma), prm.alpha_B, prm.K)
    # s = (1 - beta) r(a) + beta r(b), with the weights normalised exactly as in the
    # search engine (w1.hybrid.fuse), so that tied scores stay bit-identical
    weights = np.array([1.0 - prm.beta, prm.beta])
    s = np.column_stack([ranks_of(a), ranks_of(b)]) @ (weights / weights.sum())
    if return_parts:
        return s, dict(a=a, b=b, alpha_A=alpha_A, iterations=it)
    return s


def w1_from_arrays(edges, p, x=None, prm: W1Params = W1, n: int | None = None) -> np.ndarray:
    """Convenience entry point: edges (m, 2) of an undirected graph, edge probabilities
    p (m,), optional prior x (uniform if omitted).  Returns the W1 scores."""
    E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    p = np.asarray(p, float)
    if len(E) != len(p):
        raise ValueError("edges and p must have the same length")
    if np.any(E[:, 0] == E[:, 1]):
        raise ValueError("self-loops are not allowed")
    key = np.sort(E, axis=1)
    if len(np.unique(key, axis=0)) != len(key):
        raise ValueError("duplicate edges are not allowed")
    order = np.lexsort((key[:, 1], key[:, 0]))
    E, p = key[order], p[order]
    n = int(n if n is not None else (E.max() + 1 if len(E) else 0))
    x = np.full(n, 1.0 / n) if x is None else np.asarray(x, float) / np.sum(x)
    return w1_scores(Instance("user", n, E, p, x), prm)
