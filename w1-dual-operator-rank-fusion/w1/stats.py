"""Paired statistics used by the evidence scripts."""
from __future__ import annotations

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from .instances import stable_seed


def holm(pvalues: dict) -> dict:
    """Holm step-down adjustment of a family of named p-values."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * float(p)))
        adj[name] = running
    return adj


def paired(a, b) -> dict:
    """Two-sided Wilcoxon signed-rank test of a - b; exact zero differences are dropped.

    wins = #(a > b), losses = #(a < b), ties = #(a == b).  (The statistics files of the
    source study labelled losses 't' and ties 'l'; see evidence/original/PROVENANCE.md.)"""
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    return {"mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()), "ties": int((d == 0).sum()), "p": p}


def bootstrap_ci(a, b, seed_parts, resamples: int = 20000, level: float = 0.95):
    """Percentile bootstrap interval of the mean paired difference a - b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(stable_seed(*seed_parts))
    n = len(a)
    draws = rng.integers(0, n, size=(resamples, n))
    means = (a[draws] - b[draws]).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float((a - b).mean()), float(lo), float(hi)


def bootstrap_top1(matrix: np.ndarray, resamples: int, seed_parts, chunk: int = 50) -> np.ndarray:
    """Frequency with which each row (configuration) has the highest mean over
    columns (instances) in instance-level bootstrap resamples."""
    rng = np.random.default_rng(stable_seed(*seed_parts))
    n_cfg, n_inst = matrix.shape
    counts = np.zeros(n_cfg)
    for start in range(0, resamples, chunk):
        b = min(chunk, resamples - start)
        idx = rng.integers(0, n_inst, size=(b, n_inst))
        w = np.zeros((b, n_inst))
        np.add.at(w, (np.repeat(np.arange(b), n_inst), idx.ravel()), 1.0)
        means = matrix @ w.T              # (n_cfg, b)
        counts += np.bincount(np.argmax(means, axis=0), minlength=n_cfg)
    return counts / resamples


def friedman(columns: list) -> tuple[float, float]:
    stat, p = friedmanchisquare(*columns)
    return float(stat), float(p)
