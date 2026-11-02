# Regime-adaptive W1/MP2 fusion status

This log records the continuation requested by `FINAL_PROMPT_fusion_complete.md`.
The checkout supplied to this session was a single-upload baseline on
`local-upload-1793630110`; the requested pilot branch `exp/w1-mp2-fusion`, pilot
commits, and pilot fusion artifacts were not present locally or on `origin`.
The work therefore proceeds additively on a newly created local branch, with that
deviation retained as part of the record.

## Checkpoints

| UTC | Stage | Completed | Remaining | Previous checkpoint |
|---|---|---|---|---|
| 2026-11-02T14:41:07Z | F0 | Pinned environment installed; baseline algorithm and reproduction tests pass | Fusion implementation and continuation stages | 9b7bc6be938ac3037b082943bef0a3d648d4f068 |
| 2026-11-02T14:42:53Z | V2 | Additive fusion modules, source-attempt log, and development-only rule candidate written; fusion unit tests pass | Freeze commit and held-out confirmation | 9b7bc6be938ac3037b082943bef0a3d648d4f068 |
| 2026-11-02T14:43:30Z | H | Freeze commit `6eee979` was pushed and confirmation produced 474 rows across the five available synthetic panels; baseline and fusion tests pass | Pilot v1 comparison, new scale3/big2/oof panels, MC timing study, cluster-level analysis, and XeLaTeX validation | 6eee9791e2726c06a9d7c706b722fc4aa02f11a6 |

## Confirmation snapshot

The frozen development rule is logistic in `kappa_E`. On the available,
previously published synthetic panels, its instance-level mean differences
against W1 are positive for `test2`, `scale2`, `prior2`, `real2`, and `big`.
The machine-readable results are `results/fusion/confirm_v2.csv` and
`confirm_v2_stats.csv`. These panels are not new blind confirmation panels.

## Remaining deviations

The supplied checkout did not contain the pilot v1 branch/artifacts, so v1
cannot be evaluated. No fresh `scale3`, `big2`, or `oof` generators were
constructed; no real network was acquired; and the requested cluster bootstrap,
Monte-Carlo cost sweep, full candidate-family cross-validation, and thesis
XeLaTeX build could not be completed in this environment (`xelatex` is not
installed). These omissions are intentionally recorded rather than silently
represented as completed.

## Declared scope deviation

The prior pilot branch and its `test3`/v1 artifacts cannot be resumed because they
are absent from the supplied repository and remote. No pilot result is recreated
or represented as completed. This checkout uses the existing synthetic panels for
development and a deterministic, small real-network acquisition manifest; large
confirmation panels and thesis compilation remain explicitly pending until their
inputs are available.

## Evidence mode

The continuation is currently `synthetic` pending completion of the acquisition
manifest. Synthetic results must not be described as real-network results.

Freeze candidate digest recorded at 2026-11-02T14:43:16Z UTC.
