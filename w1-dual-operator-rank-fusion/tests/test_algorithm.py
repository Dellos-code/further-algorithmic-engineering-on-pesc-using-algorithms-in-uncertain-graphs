"""Unit tests of W1 and of the identities stated in the specification.

Run with ``python tests/test_algorithm.py`` (or pytest, if installed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from w1.algorithm import W1, W1Params, excess_embedding, phase_a, phase_b, w1_from_arrays, w1_scores  # noqa: E402
from w1.hybrid import config_id, evaluate, fuse, phase_B_trajectory  # noqa: E402
from w1.instances import load_instance, make_synthetic  # noqa: E402
from w1.metrics import ranks_of  # noqa: E402
from w1.operators import escape_probability  # noqa: E402

DEV0 = ROOT / "data" / "dev" / "0000.npz"
W1_CFG = {"beta": 0.7, "damping": 0.7, "gamma": 16.0, "kind": "hybrid", "name": "S1",
          "operator": "EP", "steps": 3}


def test_escape_probability_identity():
    """P_esc,i = 1 - prod (1 - p_ij) = 1 - exp(-d_i) with d_i the weighted degree of W."""
    inst = make_synthetic("er", 60, "R3", "dir1", 0, panel="unit")
    pe = escape_probability(inst.n, inst.edges, inst.p)
    direct = np.ones(inst.n)
    for (i, j), p in zip(inst.edges, inst.p):
        direct[i] *= 1 - p
        direct[j] *= 1 - p
    assert np.allclose(pe, 1 - direct, atol=1e-14)
    w = -np.log1p(-inst.p)
    d = np.bincount(inst.edges[:, 0], w, inst.n) + np.bincount(inst.edges[:, 1], w, inst.n)
    assert np.allclose(pe, 1 - np.exp(-d), atol=1e-14)


def test_phase_a_is_perron_vector():
    """The normalised lazy fixed point is the Perron vector of B = (1-a) x 1^T + a EW."""
    inst = make_synthetic("ws", 50, "R2", "dir1", 0, panel="unit")
    ops = inst.ops()
    a, alpha, _ = phase_a(ops["EW"], inst.x, ops["Pbar"])
    B = (1 - alpha) * np.outer(inst.x, np.ones(inst.n)) + alpha * ops["EW"].toarray()
    vals, vecs = np.linalg.eig(B)
    v = np.real(vecs[:, np.argmax(np.real(vals))])
    v = v / v.sum()
    assert np.allclose(a, v, atol=1e-9)
    lam = np.max(np.real(vals))
    # effective damping alpha / lambda: a is proportional to (I - (alpha/lam) EW)^-1 x
    r = np.linalg.solve(np.eye(inst.n) - (alpha / lam) * ops["EW"].toarray(), inst.x)
    assert np.allclose(a, r / r.sum(), atol=1e-9)


def test_excess_embedding_mean_ratio():
    rng = np.random.default_rng(0)
    x = rng.random(40)
    x /= x.sum()
    a = rng.random(40)
    a /= a.sum()
    b0 = excess_embedding(a, x, 16.0)
    assert abs(np.mean((b0 - x) / x) - 16.0) < 1e-12


def test_phase_b_affine_and_scale_dependent():
    """Phase B is affine: scaling its start changes the ranking (why b0 is not divided by 1+gamma)."""
    T = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    x = np.array([0.6, 0.4])
    b_full = phase_b(T, x, np.array([2.0, 1.0]), 0.7, 1)
    b_small = phase_b(T, x, np.array([2.0, 1.0]) / 17, 0.7, 1)
    assert np.argmax(b_full) != np.argmax(b_small)


def test_raw_phase_b_matches_scaled_engine():
    inst, *_ = load_instance(DEV0)
    ops = inst.ops()
    a, _, _ = phase_a(ops["EW"], inst.x, ops["Pbar"])
    b0 = excess_embedding(a, inst.x, 16.0)
    direct = phase_b(ops["EP"], inst.x, b0, 0.7, 3)
    traj, _ = phase_B_trajectory(ops["EP"], inst.x, ops, 0.7, b0, (3,), "none")
    assert np.allclose(direct, traj[3], rtol=1e-12, atol=0)


def test_reference_equals_engine_bitwise():
    inst, yA, yB, _ = load_instance(DEV0)
    assert np.array_equal(w1_scores(inst), evaluate(inst, yA, yB, W1_CFG, return_vector=True))


def test_fusion_endpoints():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 1.0, 2.0])
    assert np.array_equal(ranks_of(fuse([a, b], beta=0)), ranks_of(a))
    assert np.array_equal(ranks_of(fuse([a, b], beta=1)), ranks_of(b))


def test_config_id_of_frozen_w1():
    assert config_id(W1_CFG) == "7580cfc826f3f26c8b587f9ea90bcab2d2a81d27"


def test_array_entry_point_is_order_invariant():
    inst, *_ = load_instance(DEV0)
    s = w1_scores(inst)
    perm = np.random.default_rng(1).permutation(inst.m)
    E = inst.edges[perm][:, ::-1]           # shuffled and reversed endpoints
    s2 = w1_from_arrays(E, inst.p[perm], inst.x, n=inst.n)
    assert np.array_equal(ranks_of(s), ranks_of(s2))


def test_params_defaults_are_frozen_values():
    assert W1 == W1Params(c0=0.502, c1=0.558, alpha_min=0.4, alpha_max=1.0, tol=1e-12,
                          max_iter=20000, gamma=16.0, alpha_B=0.7, K=3, beta=0.7)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
