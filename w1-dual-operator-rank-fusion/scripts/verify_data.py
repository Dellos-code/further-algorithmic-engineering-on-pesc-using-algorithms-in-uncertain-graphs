"""Regenerate the shipped instances from their specifications and compare.

Every instance (edges, probabilities, prior) is rebuilt from data/<panel>/specs.json
and compared bit for bit with the stored file.  The Monte-Carlo ground truth
(yA, yA2, yB; 2,000 worlds each) is recomputed for ``--gt`` instances per panel
(default 2; ``--gt all`` recomputes everything, which takes about an hour).

    python scripts/verify_data.py [--gt N|all]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _common import DATA, out_dir, write_json
from w1.instances import ground_truth, load_instance, make_instance, panel_paths, spec_list

PANELS = ("dev", "test", "prior", "test2", "scale2", "prior2", "real2", "big")


def main(gt: str) -> dict:
    report = {}
    for panel in PANELS:
        specs = json.loads((DATA / panel / "specs.json").read_text())
        assert specs == spec_list(panel), f"{panel}: specification list differs"
        paths = panel_paths(DATA, panel)
        assert len(paths) == len(specs), f"{panel}: {len(paths)} files for {len(specs)} specs"
        same_instance = 0
        gt_checked, gt_same = 0, 0
        n_gt = len(paths) if gt == "all" else int(gt)
        gt_idx = set(np.linspace(0, len(paths) - 1, n_gt).round().astype(int).tolist()) if n_gt else set()
        for k, (path, spec) in enumerate(zip(paths, specs)):
            inst, yA, yB, yA2 = load_instance(path)
            new = make_instance(spec)
            ok = (new.name == inst.name and np.array_equal(new.edges, inst.edges)
                  and np.array_equal(new.p, inst.p) and np.array_equal(new.x, inst.x))
            same_instance += ok
            if k in gt_idx:
                g = ground_truth(new)
                gt_checked += 1
                gt_same += (np.array_equal(g["yA"], yA) and np.array_equal(g["yB"], yB)
                            and np.array_equal(g["yA2"], yA2))
        report[panel] = dict(instances=len(paths), regenerated_identical=int(same_instance),
                             ground_truth_checked=gt_checked, ground_truth_identical=int(gt_same))
        print(panel, report[panel], flush=True)
    write_json(out_dir("data") / "regeneration.json", report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="2")
    main(ap.parse_args().gt)
