"""Reusable W1/MP2 members and rank-space fusion."""
from __future__ import annotations

import numpy as np

from w1.algorithm import W1, w1_scores
from w1.diffusion import knz_mp
from w1.metrics import ranks_of


def member_scores(inst, mp_iters: int = 2):
    """Return W1 and MP scores without fitting or using ground truth."""
    return w1_scores(inst, W1), knz_mp(inst, iters=mp_iters)[0]


def rank_fusion(w_scores, mp_scores, c):
    """Fuse ascending ranks, with c=0 equal to W1 and c=1 equal to MP."""
    rw = ranks_of(np.asarray(w_scores, float))
    rm = ranks_of(np.asarray(mp_scores, float))
    c = np.asarray(c, float)
    if c.ndim == 0:
        c = float(np.clip(c, 0.0, 1.0))
    else:
        c = np.clip(c, 0.0, 1.0)
    return (1.0 - c) * rw + c * rm
