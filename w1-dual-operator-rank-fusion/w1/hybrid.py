"""The dual-operator search engine: configuration space, identifiers and evaluation.

This module reproduces, function for function, the engine with which the
source study searched the dual-operator family on the development panel and
evaluated the frozen configurations.  A configuration is a JSON-serialisable
dict; its identifier is the SHA-1 digest of its canonical JSON form, so the
identifiers in ``evidence/original`` can be recomputed from the dicts.

Configuration kinds
-------------------
hybrid    Phase A (EW, calibrated rule, fixed point) -> excess embedding with
          ``gamma`` -> ``steps`` raw Phase-B steps on ``operator`` with damping
          law ``damping`` -> rank fusion with weight ``beta`` on Phase B.
A         a single normalised diffusion (``operator``, ``damping``, ``steps``).
schedule  alternating blocks (stage S4).
baseline  a named comparator (see ``baseline_scores``).
irie      IRIE on the raw probability matrix.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .diffusion import (anchored_alpha, diffuse, heat_kernel, irie_exact, irie_wc, knz_mp,
                        rapr, scalar_sweep, spectral_radius, total_rank)
from .instances import mc_reach, stable_seed
from .metrics import corr_ranks, ranks_of, spearman
from .operators import rule_alpha, search_operators


# ------------------------------------------------------------- identifiers
@dataclass(frozen=True)
class Block:
    operator: str
    damping: str | float
    steps: int | str
    normalisation: str = "none"
    start: str = "previous"
    name: str = "snapshot"
    gamma: float = 1.0
    eta: float = 1.0


@dataclass(frozen=True)
class Schedule:
    blocks: tuple
    snapshots: tuple = ()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (Block, Schedule)):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (tuple, list)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def canonical_json(cfg: Any) -> str:
    return json.dumps(_jsonable(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_id(cfg: Any) -> str:
    return hashlib.sha1(canonical_json(cfg).encode("utf-8")).hexdigest()


# ------------------------------------------------------------- damping laws
def _normalise(y: np.ndarray) -> np.ndarray:
    s = float(np.sum(y))
    if not np.isfinite(s) or s <= 0:
        raise FloatingPointError("non-positive or non-finite propagation mass")
    return y / s


def damping_law(x, ops, law, y=None, T=None, eta=1.0):
    """Scalar or per-node damping.  Laws: a number; 'constant:c'; 'rule_EW'; 'rule_WT';
    'spectral:c' (c / rho(T)); 'D2:g' (per node g S/(x + g S), S = T y);
    'D3:g' (mean of D2); 'D4:eta' (anchored proportional law around rule_EW)."""
    if isinstance(law, (float, int)):
        return float(law)
    if law == "rule_EW":
        return rule_alpha(ops["Pbar"], "EW")
    if law == "rule_WT":
        return rule_alpha(ops["Pbar"], "WT")
    if law.startswith("constant:"):
        return float(law.split(":", 1)[1])
    if law.startswith("spectral:"):
        return float(law.split(":", 1)[1]) / spectral_radius(T, 0)
    if y is None or T is None:
        raise ValueError(f"{law} requires the current vector and operator")
    q = np.divide(y, np.maximum(x, 1e-300))
    if law.startswith("D2:"):
        g = float(law.split(":", 1)[1])
        S = T @ y
        return np.clip(g * S / (x + g * S), 0.0, 1.0 - 1e-12)
    if law.startswith("D3:"):
        g = float(law.split(":", 1)[1])
        S = T @ y
        return float(np.mean(g * S / (x + g * S)))
    if law.startswith("D4:"):
        return anchored_alpha(q, rule_alpha(ops["Pbar"], "EW"), eta)
    raise ValueError(f"unknown damping law: {law}")


def phase_A(T, x, ops, operator="EW", damping="rule_EW", steps="FP", y0=None, eta=1.0):
    """Lazy normalised diffusion; steps='FP' runs to the fixed point."""
    y = _normalise(x if y0 is None else np.asarray(y0, float))
    if steps == "FP":
        return diffuse(T, x, damping_law(x, ops, damping), y0=y)
    for _ in range(int(steps)):
        a = damping_law(x, ops, damping, y, T, eta)
        z = (1.0 - a) * x + np.asarray(a) * (T @ y)
        y = _normalise(0.5 * (y + z))
    return y, int(steps)


# --------------------------------------------------- raw (unnormalised) Phase B
@dataclass
class ScaledVector:
    """A non-negative vector represented as exp(log_scale) * unit (overflow-safe)."""
    unit: np.ndarray
    log_scale: float = 0.0

    def finite(self) -> np.ndarray:
        if self.log_scale > math.log(np.finfo(float).max):
            return np.full_like(self.unit, np.inf)
        return np.exp(self.log_scale) * self.unit


def _raw_step_scaled(state: ScaledVector, x, T, alpha) -> ScaledVector:
    a = np.broadcast_to(np.asarray(alpha, float), x.shape)
    b = np.maximum((1.0 - a) * x, 0.0)
    v = np.maximum(a * (T @ state.unit), 0.0)
    ls = float(state.log_scale)
    if not np.isfinite(ls):
        raise FloatingPointError("invalid log scale")
    if ls > 0:
        b_scaled = b * math.exp(-ls) if ls < 700 else np.zeros_like(b)
        q = b_scaled + v
        scale = ls + math.log(max(float(np.max(q)), np.finfo(float).tiny))
    else:
        q = b + math.exp(ls) * v
        scale = math.log(max(float(np.max(q)), np.finfo(float).tiny))
    unit = q / max(float(np.max(q)), np.finfo(float).tiny)
    return ScaledVector(unit, scale)


def phase_B_trajectory(T, x, ops, damping, start, steps, normalisation="none", eta=1.0):
    """Run Phase B and return {k: vector} for every requested step count k."""
    requested = tuple(steps)
    finite_steps = [int(k) for k in requested if k != "FP"]
    max_steps = max(finite_steps, default=0)
    raw_start = np.asarray(start, float)
    if normalisation == "L1":
        y = _normalise(raw_start)
        state = ScaledVector(y.copy(), 0.0)
    elif normalisation == "none":
        scale0 = float(np.max(np.abs(raw_start)))
        if not np.isfinite(scale0) or scale0 <= 0:
            raise FloatingPointError("invalid propagation start")
        y = raw_start / scale0
        state = ScaledVector(y.copy(), math.log(scale0))
    else:
        raise ValueError(normalisation)
    out = {}
    contractive = False
    rho = spectral_radius(T, 0)
    if isinstance(damping, (float, int)):
        contractive = float(damping) * rho < 1.0
    for k in range(1, max_steps + 1):
        query_y = state.finite()
        if not np.isfinite(query_y).all():
            query_y = state.unit
        a = damping_law(x, ops, damping, query_y, T, eta)
        if normalisation == "L1":
            y = _normalise((1.0 - a) * x + np.asarray(a) * (T @ y))
            state = ScaledVector(y.copy(), 0.0)
        else:
            state = _raw_step_scaled(state, x, T, a)
            y = state.unit
        if k in requested:
            out[k] = state.finite() if np.isfinite(state.log_scale) and state.log_scale < 700 \
                else state.unit.copy()
    if "FP" in requested and contractive:
        if normalisation == "L1":
            z, _ = phase_A(T, x, ops, damping=damping, steps="FP", y0=y, eta=eta)
        else:
            z = state.unit
            for _ in range(20000):
                nxt = _raw_step_scaled(state, x, T, damping_law(x, ops, damping, z, T, eta))
                if np.max(np.abs(nxt.unit - state.unit)) < 1e-12:
                    z, state = nxt.unit, nxt
                    break
                state, z = nxt, nxt.unit
        out["FP"] = z.copy()
    return out, {"rho": float(rho), "contractive": contractive, "max_steps": max_steps}


def run_schedule(ops, x, schedule: Schedule) -> dict:
    current = x.copy()
    snapshots = {}
    for block in schedule.blocks:
        T = ops[block.operator]
        start = current if block.start == "previous" else x
        if block.start.startswith("embed:"):
            current = x + float(block.start.split(":", 1)[1]) * np.maximum(start, 0)
        if block.operator in ("EW", "WT", "W") and block.normalisation == "L1":
            current, _ = phase_A(T, x, ops, block.operator, block.damping, block.steps,
                                 y0=start, eta=block.eta)
        else:
            traj, _ = phase_B_trajectory(T, x, ops, block.damping, start, (block.steps,),
                                         block.normalisation, block.eta)
            current = traj[block.steps]
        snapshots[block.name] = current.copy()
    names = schedule.snapshots or tuple(snapshots)
    return {name: snapshots[name] for name in names}


def fuse(members, method="rank_mean", weights=None, beta=None, k=60) -> np.ndarray:
    """Combine member vectors.  With two members and ``beta``: weights (1 - beta, beta)."""
    vals = list(members.values()) if isinstance(members, dict) else list(members)
    if not vals:
        raise ValueError("at least one member is required")
    R = np.column_stack([ranks_of(v) for v in vals])
    if beta is not None and len(vals) == 2:
        weights = np.array([1.0 - beta, beta])
    if weights is None:
        weights = np.full(len(vals), 1.0 / len(vals))
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    if method in ("rank_mean", "borda"):
        return R @ weights
    if method == "rrf":
        top = len(vals[0]) + 1 - R
        return (1.0 / (k + top)) @ weights
    raise ValueError(method)


# --------------------------------------------------------------- comparators
BASELINES = ("rule_EW", "rule_WT", "fixed085", "Pesc", "MP1", "MP2", "MPinf", "IRIE_exact",
             "IRIE_wc", "oracle:scalar", "oracle:scalar_WT", "heat_kernel:1", "heat_kernel:2",
             "heat_kernel:5", "heat_kernel:10", "TotalRank", "RAPr:0.5,0.5", "RAPr:1,1",
             "RAPr:2,1")


def baseline_scores(inst, yB=None, *, names=None, include_mc=False,
                    mc_k=(10, 20, 50, 100, 200, 500, 1000)) -> dict:
    """The comparator family.  Oracles are fitted on the independent sample yB only."""
    x, ops = inst.x, inst.ops()
    want = (lambda nm: True) if names is None else (lambda nm: nm in names)
    out = {}
    if want("rule_EW"):
        out["rule_EW"] = diffuse(ops["EW"], x, rule_alpha(ops["Pbar"], "EW"))[0]
    if want("rule_WT"):
        out["rule_WT"] = diffuse(ops["WT"], x, rule_alpha(ops["Pbar"], "WT"))[0]
    if want("fixed085"):
        out["fixed085"] = diffuse(ops["EW"], x, 0.85)[0]
    if want("Pesc"):
        out["Pesc"] = ops["Pesc"].copy()
    for nm, it in (("MP1", 1), ("MP2", 2), ("MPinf", None)):
        if want(nm):
            out[nm] = knz_mp(inst, iters=it)[0]
    if want("IRIE_exact"):
        out["IRIE_exact"] = irie_exact(inst)
    if want("IRIE_wc"):
        out["IRIE_wc"] = irie_wc(inst)
    if yB is not None and want("oracle:scalar"):
        ys, _ = scalar_sweep(ops["EW"], x)
        out["oracle:scalar"] = ys[:, int(np.argmax(corr_ranks(ranks_of(ys), ranks_of(yB))))]
    if yB is not None and want("oracle:scalar_WT"):
        yw, _ = scalar_sweep(ops["WT"], x)
        out["oracle:scalar_WT"] = yw[:, int(np.argmax(corr_ranks(ranks_of(yw), ranks_of(yB))))]
    for t in (1, 2, 5, 10):
        if want(f"heat_kernel:{t}"):
            out[f"heat_kernel:{t}"] = heat_kernel(ops["W"], x, float(t))
    if want("TotalRank"):
        out["TotalRank"] = total_rank(ops["EW"], x)
    for a, b in ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0)):
        if want(f"RAPr:{a:g},{b:g}"):
            out[f"RAPr:{a:g},{b:g}"] = rapr(ops["EW"], x, a, b)
    if include_mc:
        for k in mc_k:
            out[f"MC_{k}"] = mc_reach(inst, k, stable_seed(inst.name, "MC", k))
    return out


# ------------------------------------------------------------------ evaluation
def evaluate(inst, yA, yB, cfg, cache=None, return_vector=False):
    """Score vector (return_vector=True) or Spearman rho against yA of a configuration."""
    cache = {} if cache is None else cache
    ops = cache.setdefault("ops", search_operators(inst))
    x = inst.x
    kind = cfg["kind"]
    if kind == "irie":
        y = irie_exact(inst)
    elif kind == "baseline":
        y = baseline_scores(inst, yB, names=(cfg["baseline"],))[cfg["baseline"]]
    elif kind == "schedule":
        blocks = tuple(Block(**block) for block in cfg["blocks"])
        snaps = run_schedule(ops, x, Schedule(blocks, tuple(cfg["snapshots"])))
        y = fuse(snaps, cfg.get("fusion", "rank_mean"), beta=cfg.get("beta"), k=cfg.get("k", 60))
    elif kind == "A":
        y, _ = phase_A(ops[cfg["operator"]], x, ops, cfg["operator"], cfg["damping"], cfg["steps"])
    elif kind == "hybrid":
        a0 = cache.get("a0")
        if a0 is None:
            a0, _ = phase_A(ops["EW"], x, ops, "EW", "rule_EW", "FP")
            cache["a0"] = a0
        z = np.maximum(a0, 1e-300)
        z = z / np.mean(z / x)
        start = x + cfg.get("gamma", 1.0) * z
        operator = cfg.get("operator", "P")
        damping = cfg.get("damping", cfg.get("a", 0.9))
        traj, _ = phase_B_trajectory(ops[operator], x, ops, damping, start, (cfg["steps"],), "none")
        y = fuse([a0, traj[cfg["steps"]]], "rank_mean", beta=cfg.get("beta", 0.5))
    else:
        raise ValueError(kind)
    return y if return_vector else float(spearman(y, ranks_of(yA)))


# ------------------------------------------------------------ search stages
S2_OPERATORS = ("P", "Plog", "Prow", "EP", "W", "EW")
S2_DAMPINGS = tuple([0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99, 1.0]
                    + [f"spectral:{c}" for c in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)])
S2_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50)
S2_BETAS = (0.3, 0.4, 0.5, 0.6, 0.7)
S2_GAMMAS = (1.0, 4.0)
S3_BETAS = tuple(round(i / 20, 2) for i in range(21))
S3_GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0, 16.0)
S3_LAWS = ("rule_EW", "rule_WT", "D2:0.25", "D2:1", "D2:4", "D3:1", "D4:-0.25", "D4:0.25", "D4:1")

CTRL_LIN09 = {"name": "CTRL_lin0.9", "kind": "hybrid", "a": 0.9, "steps": 5, "beta": 0.5}


def stage_configs(stage: str, s2_top50: list[dict] | None = None) -> list[dict]:
    """Configurations of a search stage, in the order of the source study."""
    if stage == "s0":
        return [
            {"name": "rule_EW", "kind": "A", "operator": "EW", "damping": "rule_EW", "steps": "FP"},
            {"name": "rule_WT", "kind": "A", "operator": "WT", "damping": "rule_WT", "steps": "FP"},
            dict(CTRL_LIN09),
            {"name": "IRIE_exact", "kind": "irie"},
        ]
    if stage == "baselines":
        return [{"name": nm, "kind": "baseline", "baseline": nm} for nm in BASELINES]
    if stage == "s2":
        return [{"name": "S1", "kind": "hybrid", "operator": op, "damping": d,
                 "steps": t, "beta": b, "gamma": g}
                for op in S2_OPERATORS for d in S2_DAMPINGS for t in S2_STEPS
                for b in S2_BETAS for g in S2_GAMMAS]
    if stage == "s3":
        if s2_top50 is None:
            raise ValueError("stage s3 refines the 50 best stage-s2 configurations")
        out = []
        for base in s2_top50:
            for beta in S3_BETAS:
                for gamma in S3_GAMMAS:
                    out.append(dict(base, beta=beta, gamma=gamma))
                for law in S3_LAWS:
                    out.append(dict(base, beta=beta, gamma=4.0, damping=law))
        return out
    if stage == "s4":
        return [
            {"name": "S1_current", "kind": "schedule",
             "blocks": [{"operator": "EW", "damping": "rule_EW", "steps": "FP",
                         "normalisation": "L1", "name": "A"},
                        {"operator": "P", "damping": 0.9, "steps": 5,
                         "normalisation": "none", "name": "B"}],
             "snapshots": ["A", "B"], "fusion": "rank_mean", "beta": 0.5},
            {"name": "S3_alt", "kind": "schedule",
             "blocks": [{"operator": "EW", "damping": "rule_EW", "steps": 1,
                         "normalisation": "L1", "name": "A1"},
                        {"operator": "P", "damping": 0.9, "steps": 1,
                         "normalisation": "none", "name": "B1"},
                        {"operator": "EW", "damping": "rule_EW", "steps": 1,
                         "normalisation": "L1", "name": "A2"},
                        {"operator": "P", "damping": 0.9, "steps": 1,
                         "normalisation": "none", "name": "B2"}],
             "snapshots": ["A2", "B2"], "fusion": "rank_mean", "beta": 0.5},
        ]
    raise ValueError(stage)
