"""Rank-based evaluation metrics."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import kendalltau, rankdata

RANK_REL = 1e-12              # scores equal to within this relative tolerance are ties


def ranks_of(y: np.ndarray, rel: float = RANK_REL) -> np.ndarray:
    """Ascending average ranks after merging scores equal to within a relative tolerance."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        s = np.nanmax(np.abs(y))
        if not np.isfinite(s) or s == 0:
            return np.full(len(y), (len(y) + 1) / 2.0)
        return rankdata(np.round(y / s / rel))
    s = np.nanmax(np.abs(y), axis=0)
    s[~np.isfinite(s) | (s == 0)] = 1.0
    return rankdata(np.round(y / s[None, :] / rel), axis=0)


def corr_ranks(R: np.ndarray, rt: np.ndarray) -> np.ndarray | float:
    """Pearson correlation of rank vector(s) R, shape (n,) or (n, K), with target ranks rt.
    A constant prediction has correlation 0."""
    rt = rt - rt.mean()
    nt = np.sqrt((rt * rt).sum())
    if R.ndim == 1:
        r = R - R.mean()
        nr = np.sqrt((r * r).sum())
        return 0.0 if nr == 0 or nt == 0 else float((r @ rt) / (nr * nt))
    Rc = R - R.mean(0, keepdims=True)
    nr = np.sqrt((Rc * Rc).sum(0))
    out = np.zeros(R.shape[1])
    ok = (nr > 0) & (nt > 0)
    out[ok] = (Rc[:, ok].T @ rt) / (nr[ok] * nt)
    return out


def spearman(y: np.ndarray, target_ranks: np.ndarray) -> float:
    """Spearman correlation of scores y with precomputed target ranks."""
    return corr_ranks(ranks_of(y), target_ranks)


def full_metrics(y: np.ndarray, ystar: np.ndarray, target_ranks: np.ndarray,
                 kfrac: float = 0.1) -> dict:
    """Spearman rho, Kendall tau-b, top-10% overlap and NDCG@10%."""
    n = len(y)
    yr = np.round(y / (np.max(np.abs(y)) or 1.0) / RANK_REL)
    rho = corr_ranks(rankdata(yr), target_ranks)
    tau = 0.0 if np.ptp(yr) == 0 else float(kendalltau(yr, ystar).statistic)
    k = max(1, int(math.ceil(kfrac * n)))
    o_pred = np.lexsort((np.arange(n), -yr))[:k]
    o_true = np.lexsort((np.arange(n), -ystar))[:k]
    topk = len(np.intersect1d(o_pred, o_true)) / k
    g = (ystar - ystar.min()) / (np.ptp(ystar) or 1.0)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((g[o_pred] * disc).sum())
    idcg = float((np.sort(g)[::-1][:k] * disc).sum())
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return dict(rho=rho, tau=tau, topk=topk, ndcg=ndcg)
