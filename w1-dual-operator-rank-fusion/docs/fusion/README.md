# Fusion continuation

Run from the repository root with the pinned environment:

```bash
python -m venv .venv
.venv/bin/pip install -r w1-dual-operator-rank-fusion/requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python w1-dual-operator-rank-fusion/tests/test_fusion.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python w1-dual-operator-rank-fusion/scripts/fusion_search_real.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python w1-dual-operator-rank-fusion/scripts/fusion_fit.py
```

The fit script uses only the existing `dev` panel and `yB` for selection. It
writes a resumable JSON freeze candidate under `results/fusion/`. Real-network
acquisition is recorded separately; synthetic evidence is never labelled as
real-network evidence.
