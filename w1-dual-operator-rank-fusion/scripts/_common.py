"""Shared paths and settings of the evidence scripts."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
ORIG = ROOT / "evidence" / "original"
REPRO = ROOT / "evidence" / "reproduced"
EXTERNAL = ROOT / "external" / "escape_probability"

# Panels: development (selection only), fresh confirmation panels built after the
# freeze was planned, the earlier held-out panels, and the large-graph panel.
PRIMARY = ("test2", "scale2")
SECONDARY = ("prior2", "real2", "big")
EARLIER_HELDOUT = ("test", "prior")
CONFIRMATION = PRIMARY + SECONDARY + EARLIER_HELDOUT

# Comparators of the confirmation runs, in the order of the source study.
COMPARATORS = ("IRIE_exact", "rule_EW", "rule_WT", "MP2", "fixed085", "oracle:scalar")


def frozen() -> dict:
    return json.loads((ORIG / "selected_mixing.json").read_text())


def winners() -> dict:
    """{'W1': config, 'W3': config} from the frozen selection file."""
    return {w["name"]: w["config"] for w in frozen()["winners"]}


def out_dir(*parts) -> Path:
    p = REPRO.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")
