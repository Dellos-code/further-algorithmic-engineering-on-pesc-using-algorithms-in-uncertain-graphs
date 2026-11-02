"""Small deterministic candidate family implementations."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def clipped_linear(x, params):
    a, b = params
    z = (x - np.mean(x)) / (np.std(x) or 1.0)
    return np.clip(a * z + b, 0.0, 1.0)


def logistic(x, params):
    a, tau = params
    z = (x - tau) / (np.std(x) or 1.0)
    return expit(a * z)


def fit_constant(x, curves, grid):
    del x
    mean_curve = np.mean(curves, axis=0)
    return {"family": "constant", "params": [float(grid[int(np.argmax(mean_curve))])]}


def fit_candidates(x, curves, grid):
    """Fit constant, hard gate, clipped linear and logistic on pooled curves."""
    x = np.asarray(x, float)
    curves = np.asarray(curves, float)
    mean_curve = lambda c: float(np.mean([
        np.interp(ci, grid, row) for ci, row in zip(c, curves)
    ]))
    candidates = [fit_constant(x, curves, grid)]
    thresholds = np.unique(np.quantile(x, np.linspace(0.1, 0.9, 9)))
    for tau in thresholds:
        candidates.append({"family": "hard_gate", "params": [float(tau)]})
    for family, fn, starts in (
        ("clipped_linear", clipped_linear, [(-1, .5), (0, .5), (1, .5)]),
        ("logistic", logistic, [(1, np.median(x)), (3, np.median(x))]),
    ):
        for start in starts:
            result = minimize(lambda q: -mean_curve(fn(x, q)), start, method="Nelder-Mead")
            candidates.append({"family": family, "params": result.x.tolist()})
    def values(candidate):
        f, par = candidate["family"], candidate["params"]
        if f == "constant":
            c = np.full(len(x), par[0])
        elif f == "hard_gate":
            c = (x > par[0]).astype(float)
        elif f == "clipped_linear":
            c = clipped_linear(x, par)
        else:
            c = logistic(x, par)
        return mean_curve(c)
    return max(candidates, key=values)


def evaluate_rule(rule, x):
    f, p = rule["family"], rule["params"]
    if f == "constant":
        return np.full(len(x), np.clip(p[0], 0, 1))
    if f == "hard_gate":
        return (np.asarray(x) > p[0]).astype(float)
    if f == "clipped_linear":
        return clipped_linear(np.asarray(x), p)
    if f == "logistic":
        return logistic(np.asarray(x), p)
    raise ValueError(f"unknown fusion family: {f}")
