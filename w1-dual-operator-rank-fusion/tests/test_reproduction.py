"""Fast reproduction checks against the stored evidence (about one minute).

The full checks are the scripts in scripts/ (see README)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from w1.algorithm import W1, W3, w1_scores  # noqa: E402
from w1.hybrid import baseline_scores, config_id  # noqa: E402
from w1.instances import Instance, load_instance, make_instance, panel_paths  # noqa: E402
from w1.metrics import full_metrics, ranks_of, spearman  # noqa: E402

DATA = ROOT / "data"
ORIG = ROOT / "evidence" / "original"


def test_freeze_digest():
    import hashlib
    raw = (ORIG / "selected_mixing.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (ORIG / "selected_mixing.sha256").read_text().split()[0]


def test_development_values_of_w1():
    sel = json.loads((ORIG / "selected_mixing.json").read_text())
    w1_id = sel["winners"][0]["config_id"]
    with np.load(ORIG / "search" / "s3_dev.npz") as d:
        row = list(d["config_id"]).index(w1_id)
        stored = d["rho"][row]
        assert abs(stored.mean() - sel["winners"][0]["dev_rho"]) < 1e-15
    for j, path in list(enumerate(panel_paths(DATA, "dev")))[::45]:
        inst, yA, _, _ = load_instance(path)
        assert abs(spearman(w1_scores(inst, W1), ranks_of(yA)) - stored[j]) < 1e-12


def test_confirmation_values():
    for panel in ("test2", "scale2", "prior2", "real2", "big"):
        orig = pd.read_csv(ORIG / "confirm" / f"confirm_{panel}.csv.gz").set_index(["id", "method"])
        paths = panel_paths(DATA, panel)
        for path in (paths[:1] if panel != "big" else [paths[0]]):
            inst, yA, yB, _ = load_instance(path)
            rA = ranks_of(yA)
            got = {"W1": w1_scores(inst, W1), "W3": w1_scores(inst, W3)}
            got.update(baseline_scores(inst, yB, names=("IRIE_exact", "rule_EW", "oracle:scalar")))
            for name, s in got.items():
                want = orig.loc[(int(path.stem), name)]
                m = full_metrics(s, yA, rA)
                assert all(abs(m[k] - want[k]) < 1e-12 for k in ("rho", "tau", "topk", "ndcg")), (panel, name)


def test_instances_regenerate():
    for panel in ("dev", "test2", "scale2", "prior2", "real2", "big"):
        specs = json.loads((DATA / panel / "specs.json").read_text())
        path = panel_paths(DATA, panel)[0]
        inst, *_ = load_instance(path)
        new = make_instance(specs[0])
        assert np.array_equal(new.edges, inst.edges) and np.array_equal(new.p, inst.p) \
            and np.array_equal(new.x, inst.x)


def test_external_holdout_two_records():
    spec = importlib.util.spec_from_file_location("escape_lib", ROOT / "external" / "escape_probability" / "escape_lib.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    H = json.loads((ROOT / "external" / "escape_probability" / "holdout.json").read_text())
    src = pd.DataFrame(json.loads((ORIG / "external_holdout.json").read_text()))
    for rec in H["records"][:2]:
        old = E.make_instance(rec["topology"], 100, rec["lo"], rec["hi"], rec["seed"])
        gt = E.ground_truth(old, 3000, seed=5000 + rec["seed"])
        inst = Instance("x", old.n, np.stack([old.ei, old.ej], 1).astype(np.int64), old.p, old.x)
        want = src[(src.topology == rec["topology"]) & (src.lo == rec["lo"]) & (src.seed == rec["seed"])].iloc[0]
        assert abs(spearman(w1_scores(inst, W1), ranks_of(gt)) - want.W1) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
