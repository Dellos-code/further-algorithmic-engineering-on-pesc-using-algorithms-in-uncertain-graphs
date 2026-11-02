# Mixing-strategy study status

This study is additive to the published implementation. Immutable assets under
`results/`, `data/`, `paper/`, `figures/`, `figures_gr/`, `logs/`, and the
checkpoint/documentation files are not modified.

## 2026-10-31 — scaffold and S0 audit

Created branch `exp/mixing-study` and added the general mixing engine,
canonical configuration IDs, overflow-safe raw propagation, fusion methods,
baseline ports, and the staged runner scaffolding. No fresh-panel method has
been evaluated.

S0 completed on the existing panels with 4 configurations and 180 instances
per panel. Mean Spearman values were:

| panel | CTRL_lin0.9 | rule_EW | rule_WT | IRIE_exact |
|---|---:|---:|---:|---:|
| dev | 0.9037 | 0.8571 | 0.8104 | 0.8800 |
| test | 0.9006 | 0.8442 | 0.8113 | 0.8960 |
| prior | 0.9016 | 0.8568 | 0.8554 | 0.8906 |
| real | 0.9005 | 0.8402 | 0.8169 | 0.8475 |
| scale | 0.8863 | 0.8320 | 0.7594 | 0.8434 |

The required targets match to four decimals. The current-hybrid per-instance
values match the published controls on the checked dev instances exactly.
The vendored IRIE port agrees with the pinned external implementation to
`1e-12` on the checked instance. No fresh panel has been evaluated.

Fresh ground truth generation completed for `test2` (180 instances),
`scale2` (90), and `prior2` (150). The repository lacks the four required
`data_real/*.npz` snowball sources; `build_panels_v2.py` therefore omits those
unavailable specifications rather than substituting other graphs. Classic
`real2` specifications remain available.

The full `(edges, p, prior)` duplicate audit found three unavoidable classic
weighted-cascade repeats: that scheme is deterministic from the degree
sequence, so changing its probability seed cannot change the instance. They
are retained and explicitly reported rather than silently altering the
published probability scheme.

## 2026-10-31 — S2/S3/S4 and freeze

S2 evaluated 16,800 configurations on all 180 dev instances. S3 evaluated the
top-50 S2 survivors with the full beta/gamma refinement and dynamic-law
variants, 15,750 configuration rows per dev instance. S4 evaluated two
schedule configurations. The S3 winner was `EP`, constant `a=0.7`, three
Phase-B steps, excess embedding `gamma=16`, and rank-fusion `beta=0.7`
(mean dev rho 0.9133). The best S4 schedule was 0.9029 and therefore did not
qualify for W2. A 2,000-resample dev bootstrap gave top-1 frequencies 0.274
for W1 and 0.184 for W3. The freeze contains W1 and W3 only; no confirmatory
panel has been evaluated.

Primary confirmation completed on `test2` and `scale2`. Mean Spearman values
were W1/W3 = 0.9064/0.9072 on `test2` and 0.8650/0.8651 on `scale2`;
`IRIE_exact` was 0.8841 and 0.8302, respectively. W1 versus IRIE had
Wilcoxon p-values 0.00028 and 0.0130 (Holm-adjusted 0.00085 and 0.0389).
Against `oracle:scalar`, the paired mean differences were +0.00547 on test2
and -0.01256 on scale2, so the preregistered non-inferiority margin is not
met on scale2.

Secondary confirmation means were W1/W3 = 0.9183/0.9162 on `prior2`,
0.9625/0.9638 on the available classic-only `real2`, and 0.9096/0.9099 on
`big`. On the vendored 100-instance external holdout, W1/W3 scored
0.9310/0.9289 versus 0.9142 for the pinned IRIE implementation and 0.8940
for the external paper's calibrated escape-scaled rule.

On the fixed test2 timing subset, W1/W3 median end-to-end times were 2.14/2.16
ms (machine-specific). MC-10/20/50/100/200 achieved mean rho
0.738/0.809/0.903/0.944/0.966 at median times 0.24/0.26/0.37/0.52/0.90
ms; these measurements are stored in `runtime_test2.json`.
