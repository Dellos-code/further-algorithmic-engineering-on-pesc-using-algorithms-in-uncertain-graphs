#!/usr/bin/env bash
# Regenerates every result of evidence/reproduced/ and both documents.
# Approximate single-core times are in brackets; steps are independent of each
# other except where stated, and each can be rerun alone.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python tests/test_algorithm.py                 # [10 s]  unit tests
python tests/test_reproduction.py              # [10 s]  spot checks against the stored evidence
python scripts/verify_data.py --gt 2           # [5 min] regenerate all 984 instances (+2 ground truths per panel)
python scripts/verify_freeze.py                # [5 s]   digest, W1 = best development configuration, bootstrap
python scripts/run_search.py                   # [20 min] re-evaluate 355 search configurations on all development instances
python scripts/run_confirm.py                  # [1-2 h] W1, W3 and comparators on the 7 held-out panels
python scripts/run_external_holdout.py         # [3 min] the earlier study's 100-instance hold-out
python scripts/analyze.py                      # [1 min] statistics (needs run_confirm.py)
python scripts/bench_cost.py                   # [15 min] exploratory: W1 vs Monte Carlo (machine-specific timings)
python scripts/ablations.py                    # [15 min] exploratory: sensitivity analyses
python scripts/make_doc_assets.py              # [10 s]  macros and tables of the evidence report, README tables
for doc in docs/specification/W1_specification_en docs/specification/W1_specification_gr \
           docs/evidence/W1_evidence_en docs/evidence/W1_evidence_gr; do
  (cd "$(dirname "$doc")" && for _ in 1 2 3; do xelatex -interaction=nonstopmode "$(basename "$doc").tex" >/dev/null; done)
done
