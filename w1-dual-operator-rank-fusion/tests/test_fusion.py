"""Focused tests for the additive fusion implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.covariates import covariates, rho_nb
from fusion.members import rank_fusion
from w1.instances import Instance
from w1.metrics import ranks_of


def graph(edges, p, x=None):
    n = max(max(e) for e in edges) + 1
    return Instance("toy", n, np.asarray(edges, dtype=np.int64),
                    np.asarray(p, float),
                    np.ones(n) / n if x is None else np.asarray(x, float))


def test_endpoints_are_rank_members():
    w = np.array([3.0, 1.0, 2.0])
    m = np.array([1.0, 3.0, 2.0])
    assert np.array_equal(rank_fusion(w, m, 0), ranks_of(w))
    assert np.array_equal(rank_fusion(w, m, 1), ranks_of(m))


def test_pbar_and_expected_degree():
    c = covariates(graph([(0, 1), (1, 2)], [0.2, 0.4]))
    assert np.isclose(c["pbar_E"], 0.3)
    assert np.isclose(c["dbar"], 0.4)


def test_nonbacktracking_tree_is_zero():
    assert rho_nb(graph([(0, 1), (1, 2), (2, 3)], [0.5, 0.5, 0.5])) < 1e-12


def test_nonbacktracking_regular_cycle():
    # A cycle is 2-regular, so rho_NB=p for uniform p.
    value = rho_nb(graph([(0, 1), (1, 2), (2, 3), (3, 0)], [0.25] * 4))
    assert np.isclose(value, 0.25, atol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
