"""One-off provenance step: pack the per-instance search results of the source study
(results/mixing/<stage>_dev/NNNN.csv.gz, one file per development instance) into
one matrix per stage, evidence/original/search/<stage>_dev.npz, and check that the
packing is lossless.

The source repository is needed only for this step:
    python scripts/convert_original_search.py /path/to/source/prop_damping
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "original" / "search"


def pack(src: Path, stage: str) -> Path:
    files = sorted((src / "results" / "mixing" / f"{stage}_dev").glob("*.csv.gz"))
    frames, rows = [], 0
    for f in files:
        df = pd.read_csv(f)
        rows = len(df)
        # Stage s3 re-generates identical configurations from different stage-s2 parents
        # (15,750 rows, 3,465 distinct configurations); duplicates carry identical values.
        spread = df.groupby("config_id").rho.agg(lambda v: v.max() - v.min()).max()
        if spread != 0:
            raise ValueError(f"{f}: duplicate configurations disagree")
        frames.append(df.drop_duplicates("config_id", keep="first").reset_index(drop=True))
    print(f"{stage}: {rows} rows per instance, {len(frames[0])} distinct configurations")
    order = list(dict.fromkeys(frames[0].config_id))
    inst = np.array([int(f.name[:4]) for f in files])
    rho = np.empty((len(order), len(files)))
    ms = np.empty((len(order), len(files)), dtype=np.float32)
    pos = {c: k for k, c in enumerate(order)}
    for j, df in enumerate(frames):
        if set(df.config_id) != set(order) or len(df) != len(order):
            raise ValueError(f"{files[j]}: configuration set differs")
        idx = df.config_id.map(pos).to_numpy()
        rho[idx, j] = df.rho.to_numpy()
        ms[idx, j] = df.ms_est.to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{stage}_dev.npz"
    np.savez_compressed(out, config_id=np.array(order), instance=inst, rho=rho, ms_est=ms)
    # lossless check: every stored value is recovered exactly
    with np.load(out) as d:
        back = {c: k for k, c in enumerate(d["config_id"])}
        for j, df in enumerate(frames):
            k = df.config_id.map(back).to_numpy()
            if not np.array_equal(d["rho"][k, j], df.rho.to_numpy()):
                raise AssertionError(f"lossy packing in {files[j]}")
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1])
    for stage in ("s0", "baselines", "s2", "s3", "s4"):
        print(pack(src, stage))
