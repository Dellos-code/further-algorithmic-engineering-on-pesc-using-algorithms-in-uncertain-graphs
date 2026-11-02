# Mixing study S0 audit

The audit was run with one BLAS thread (`OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) on the existing panels only.
Each panel contains 180 instances. No instance in `test2`, `scale2`, `prior2`,
`real2`, or `big` was evaluated.

| panel | `CTRL_lin0.9` | `rule_EW` | `rule_WT` | `IRIE_exact` |
|---|---:|---:|---:|---:|
| dev | 0.9037 | 0.8571 | 0.8104 | 0.8800 |
| test | 0.9006 | 0.8442 | 0.8113 | 0.8960 |
| prior | 0.9016 | 0.8568 | 0.8554 | 0.8906 |
| real | 0.9005 | 0.8402 | 0.8169 | 0.8475 |
| scale | 0.8863 | 0.8320 | 0.7594 | 0.8434 |

The required `CTRL_lin0.9` and `rule_EW` reproduction targets match to four
decimals. The vendored IRIE implementation agrees with the pinned external
implementation on the checked instance to absolute tolerance `1e-12`.
