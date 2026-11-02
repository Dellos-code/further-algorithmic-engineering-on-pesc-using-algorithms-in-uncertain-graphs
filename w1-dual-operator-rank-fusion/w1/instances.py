"""Uncertain-graph instances: generation, storage and Monte-Carlo ground truth.

An instance is an undirected graph G=(V,E) whose edges exist independently
with probabilities p, together with a prior x > 0 that sums to one.  The
target of every method in this project is

    y*_i = E[ sum_{v in C(i)} x_v ],

the expected prior mass of the connected component of i in a sampled world.

Every random choice is derived from explicit seeds (``stable_seed``), so the
panels shipped in ``data/`` can be regenerated bit for bit with the pinned
library versions (see ``requirements.txt``).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

M_GT = 2000  # Monte-Carlo worlds per ground-truth sample


def stable_seed(*parts) -> int:
    """Deterministic 63-bit seed from arbitrary parts."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


@dataclass
class Instance:
    name: str
    n: int
    edges: np.ndarray            # (m, 2) int64, i < j, unique
    p: np.ndarray                # (m,) edge probabilities in (0, 1]
    x: np.ndarray                # (n,) prior, > 0, sums to 1
    meta: dict = field(default_factory=dict)
    _ops: dict | None = field(default=None, repr=False)

    @property
    def m(self) -> int:
        return int(len(self.p))

    def ops(self) -> dict:
        if self._ops is None:
            from .operators import build_operators
            self._ops = build_operators(self)
        return self._ops


def canon_edges(E) -> np.ndarray:
    E = np.asarray(E, dtype=np.int64).reshape(-1, 2)
    E = E[E[:, 0] != E[:, 1]]
    E = np.sort(E, axis=1)
    if len(E):
        E = np.unique(E, axis=0)
    return E


def _nx_edges(G):
    import networkx as nx
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return canon_edges(np.array(list(G.edges()), dtype=np.int64).reshape(-1, 2)), G.number_of_nodes()


# ----------------------------------------------------------------- topologies
def topology(kind: str, n: int, seed: int, **kw):
    """Return (edges, n_actual, info) for a synthetic topology family."""
    import networkx as nx
    s = int(seed % (2 ** 31 - 1))
    rng = np.random.default_rng(seed)
    info = {}
    if kind == "er":
        d = kw.get("deg", 5.0)
        G = nx.gnm_random_graph(n, int(round(n * d / 2)), seed=s)
    elif kind == "ba":
        G = nx.barabasi_albert_graph(n, kw.get("m", 2), seed=s)
    elif kind == "ws":
        G = nx.connected_watts_strogatz_graph(n, kw.get("k", 6), kw.get("beta", 0.1), seed=s)
    elif kind == "grid":
        side = int(round(math.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
        info["shape"] = (side, side)
    elif kind == "rect":
        l1 = int(round(math.sqrt(n / 4.0)))
        l2 = int(round(n / l1))
        G = nx.grid_2d_graph(l1, l2)
        info["shape"] = (l1, l2)
    elif kind == "sbm":
        blocks = kw.get("blocks", 4)
        sizes = [n // blocks] * blocks
        sizes[-1] += n - sum(sizes)
        din, dout = kw.get("din", 5.0), kw.get("dout", 1.0)
        bs = sizes[0]
        pin, pout = din / (bs - 1), dout / (n - bs)
        probs = [[pin if a == b else pout for b in range(blocks)] for a in range(blocks)]
        G = nx.stochastic_block_model(sizes, probs, seed=s)
    elif kind == "cp":                       # core-periphery
        nc = max(4, int(round(0.15 * n)))
        E = [(i, j) for i in range(nc) for j in range(i + 1, nc) if rng.random() < 0.5]
        for v in range(nc, n):
            k = 1 + int(rng.random() < 0.5)
            for c in rng.choice(nc, size=k, replace=False):
                E.append((v, int(c)))
        npp = int(round(0.3 * (n - nc)))
        a = rng.integers(nc, n, size=npp)
        b = rng.integers(nc, n, size=npp)
        E += list(zip(a.tolist(), b.tolist()))
        info["core"] = nc
        return canon_edges(E), n, info
    elif kind == "tree":
        G = nx.random_labeled_tree(n, seed=s)
    elif kind == "barbell":
        m1 = kw.get("m1", int(round(0.3 * n)))
        m2 = n - 2 * m1
        G = nx.barbell_graph(m1, m2)
        info["m1"], info["m2"] = m1, m2
    else:
        raise ValueError(kind)
    E, nn = _nx_edges(G)
    return E, nn, info


PROB_RANGES = {
    "R1": (0.01, 0.06),
    "R2": (0.06, 0.18),
    "R3": (0.18, 0.40),
    "R4": (0.40, 0.70),
    "R5": (0.70, 0.95),
}


def make_prior(kind: str, n: int, seed: int, floor_rel: float = 1e-12) -> np.ndarray:
    """uniform; dirT (Dirichlet, concentration T); lognS (log-normal, sigma S);
    seedK_E (K seeds sharing mass 1-E, mass E spread uniformly)."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        x = np.ones(n)
    elif kind.startswith("dir"):
        x = rng.standard_gamma(float(kind[3:]), size=n)
    elif kind.startswith("logn"):
        x = np.exp(float(kind[4:]) * rng.standard_normal(n))
    elif kind.startswith("seed"):
        k_str, e_str = kind[4:].split("_")
        k, eps = int(k_str), float(e_str)
        x = np.full(n, eps / n)
        seeds = rng.choice(n, size=k, replace=False)
        x[seeds] += (1.0 - eps) / k
    else:
        raise ValueError(kind)
    x = np.asarray(x, float)
    x = np.maximum(x, floor_rel * x.max())
    return x / x.sum()


def prior_stats(x: np.ndarray) -> dict:
    n = len(x)
    H = -np.sum(x * np.log(x))
    return dict(kappa=float(x.max() / x.min()), Hnorm=float(H / math.log(n)),
                log10_kappa=float(np.log10(x.max() / x.min())))


def assign_probs(m: int, rng_seed: int, lo: float, hi: float) -> np.ndarray:
    return np.random.default_rng(rng_seed).uniform(lo, hi, size=m)


def make_synthetic(kind: str, n: int, prange: str, prior: str, rep: int,
                   panel: str = "", **kw) -> Instance:
    name = f"{panel}:{kind}:n{n}:{prange}:{prior}:r{rep}"
    sd = stable_seed(panel, kind, n, prange, rep)          # topology/probs do not depend on prior
    E, nn, info = topology(kind, n, sd, **kw)
    lo, hi = PROB_RANGES[prange]
    p = assign_probs(len(E), stable_seed(sd, "p"), lo, hi)
    x = make_prior(prior, nn, stable_seed(sd, "x", prior))
    meta = dict(panel=panel, family=kind, n=nn, prange=prange, prior=prior, rep=rep, **info)
    return Instance(name, nn, E, p, x, meta)


def real_probs(E: np.ndarray, n: int, scheme: str, seed: int) -> np.ndarray:
    m = len(E)
    rng = np.random.default_rng(seed)
    if scheme.startswith("const"):
        return np.full(m, float(scheme[5:]))
    if scheme.startswith("unif"):
        lo, hi = map(float, scheme[4:].split("-"))
        return rng.uniform(lo, hi, size=m)
    if scheme == "wc":                       # weighted cascade, symmetrised
        deg = np.bincount(E.ravel(), minlength=n)
        return 1.0 / np.maximum(deg[E[:, 0]], deg[E[:, 1]])
    raise ValueError(scheme)


def make_instance(spec: dict) -> Instance:
    """Build the instance described by one entry of ``data/<panel>/specs.json``."""
    if spec["kind"] == "syn":
        return make_synthetic(spec["fam"], spec["n"], spec["pr"], spec["prior"], spec["rep"],
                              panel=spec["panel"], **spec.get("kw", {}))
    if spec["kind"] == "classic":
        import networkx as nx
        G = {"karate": nx.karate_club_graph, "lesmis": nx.les_miserables_graph,
             "davis": nx.davis_southern_women_graph}[spec["g"]]()
        E, n = _nx_edges(G)
    else:
        raise ValueError("snowball-sampled real networks need the raw graphs, which are not shipped")
    sd = stable_seed(spec["kind"], spec["g"], spec.get("rep", 0), spec["scheme"])
    p = real_probs(E, n, spec["scheme"], sd)
    x = make_prior(spec["prior"], n, stable_seed(sd, "x", spec["prior"]))
    name = f"real:{spec['g']}:{spec['scheme']}:{spec['prior']}:r{spec.get('rep', 0)}"
    meta = dict(panel="real", family=spec["g"], n=n, prange=spec["scheme"], prior=spec["prior"],
                rep=spec.get("rep", 0), genre="classic")
    return Instance(name, n, E, p, x, meta)


# --------------------------------------------------------------- ground truth
def mc_reach(inst: Instance, M: int, seed: int, max_nodes: int = 400_000,
             max_edges: int = 4_000_000, return_second_moment: bool = False):
    """Monte-Carlo estimate of y*_i over M sampled worlds.

    Worlds are processed in batches stacked as one block-diagonal graph, so a
    single connected_components call labels all worlds of a batch."""
    rng = np.random.default_rng(seed)
    n, m, x = inst.n, inst.m, inst.x
    if m == 0:
        return (x.copy(), np.zeros(n)) if return_second_moment else x.copy()
    B = int(max(1, min(M, max_nodes // n, max_edges // m)))
    tot = np.zeros(n)
    tot2 = np.zeros(n)
    i0, j0 = inst.edges[:, 0], inst.edges[:, 1]
    xt = np.tile(x, B)
    done = 0
    while done < M:
        b = min(B, M - done)
        keep = rng.random((b, m)) < inst.p
        bi, ei = np.nonzero(keep)
        off = bi * n
        r, c = i0[ei] + off, j0[ei] + off
        N = b * n
        A = sp.csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(N, N))
        nc, lab = connected_components(A, directed=False)
        mass = np.bincount(lab, weights=xt[:N], minlength=nc)
        v = mass[lab].reshape(b, n)
        tot += v.sum(0)
        if return_second_moment:
            tot2 += (v * v).sum(0)
        done += b
    if return_second_moment:
        return tot / M, tot2 / M
    return tot / M


def ground_truth(inst: Instance) -> dict:
    """The two independent ground-truth samples stored with every instance:
    yA (evaluation) and yB (oracle fitting), each over M_GT worlds."""
    yA, yA2 = mc_reach(inst, M_GT, stable_seed(inst.name, "A"), return_second_moment=True)
    yB = mc_reach(inst, M_GT, stable_seed(inst.name, "B"))
    return dict(yA=yA, yA2=yA2, yB=yB)


# --------------------------------------------------------------------- panels
FAMILIES = ["er", "ba", "ws", "grid", "rect", "sbm", "cp", "tree", "barbell"]
RANGES = ["R1", "R2", "R3", "R4", "R5"]
PRIORS10 = ["uniform", "dir10", "dir3", "dir1", "dir0.3", "logn0.5", "logn1", "logn2",
            "seed5_0.1", "seed1_0.1"]
PRIOR_FAMILIES = (("grid", 144), ("er", 150), ("ba", 150), ("ws", 150), ("sbm", 150))


def spec_list(panel: str) -> list[dict]:
    """Specifications of the panels used in this project (identical to the source study)."""
    if panel in ("dev", "test"):
        reps = (0, 1) if panel == "dev" else (2, 3)
        return [dict(kind="syn", fam=fam, n=100, pr=pr, prior=prior, rep=r, panel="main")
                for fam in FAMILIES for pr in RANGES for prior in ("uniform", "dir1") for r in reps]
    if panel == "test2":
        return [dict(kind="syn", fam=fam, n=100, pr=pr, prior=prior, rep=rep, panel="main")
                for fam in FAMILIES for pr in RANGES for prior in ("uniform", "dir1")
                for rep in (4, 5)]
    if panel in ("prior", "prior2"):
        rep = 0 if panel == "prior" else 1
        return [dict(kind="syn", fam=fam, n=n, pr=pr, prior=prior, rep=rep, panel="prior")
                for fam, n in PRIOR_FAMILIES for pr in ("R2", "R3", "R4") for prior in PRIORS10]
    if panel == "scale2":
        return [dict(kind="syn", fam=fam, n=900, pr=pr, prior=prior, rep=1, panel="scale")
                for fam in FAMILIES for pr in RANGES for prior in ("uniform", "dir1")]
    if panel == "big":
        return [dict(kind="syn", fam=fam, n=n, pr=pr, prior="uniform", rep=0, panel="big")
                for fam in ("er", "ba", "grid", "ws", "sbm")
                for n in (2000, 5000, 10000) for pr in ("R2", "R4")]
    if panel == "real2":
        # classic networks only: the snowball-sampled citation and co-purchase
        # graphs of the original specification were unavailable to the source study
        return [dict(kind="classic", g=g, scheme=sc, prior=prior, rep=1)
                for g in ("karate", "lesmis", "davis")
                for sc in ("unif0.05-0.2", "unif0.2-0.5", "unif0.5-0.9", "wc")
                for prior in ("uniform", "dir1")]
    raise ValueError(panel)


def load_instance(path):
    """Load ``data/<panel>/NNNN.npz``: returns (instance, yA, yB, yA2)."""
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        inst = Instance(meta["name"], len(d["x"]), d["edges"], d["p"], d["x"], meta)
        return inst, d["yA"], d["yB"], d["yA2"]


def panel_paths(data_dir: Path, panel: str) -> list[Path]:
    return sorted((Path(data_dir) / panel).glob("[0-9][0-9][0-9][0-9].npz"))
