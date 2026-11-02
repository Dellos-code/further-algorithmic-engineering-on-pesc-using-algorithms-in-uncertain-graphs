# Provenance of `evidence/original/`

Everything in this folder was produced by the source study and copied without
modification, except where stated.

* Source repository: `github.com/Dellos-code/self-consistent-proportional-per-node-damping-for-reachability-ranking-in-stochastic-graphs`
* Source commit: `6900c4d5d25f56b15d3a6fb25c135801540883f2` (merge of the mixing study, 2026-11-01)
* Source folder: `prop_damping_final/prop_damping/`

## Chronology recorded in the source history

| commit | time (UTC) | content |
|---|---|---|
| `aed22da` | 2026-10-31 17:04:05 | development search complete; fresh panels built |
| `ac265f7` | 2026-10-31 17:04:14 | **FREEZE**: `selected_mixing.json` (created 17:03:41) and its SHA-256 |
| `2ed1a1c` | 2026-10-31 17:30:16 | confirmation results on the held-out panels |
| `d1c04dd` | 2026-10-31 17:30:54 | cost-matched Monte-Carlo timing (9 instances) |

## Files

| file | source path | note |
|---|---|---|
| `selected_mixing.json`, `selected_mixing.sha256` | `results/mixing/` | the frozen selection; the digest is checked by `scripts/verify_freeze.py` |
| `baselines_frozen.json` | `results/mixing/` | specification of the reproduction targets of the audit stage |
| `configs.jsonl.gz` | `results/mixing/configs.jsonl` | gzip-compressed; every configuration evaluated, keyed by its SHA-1 identifier |
| `search/<stage>_dev.npz` | `results/mixing/<stage>_dev/NNNN.csv.gz` | **repacked** by `scripts/convert_original_search.py`: one matrix per stage (configurations x 180 development instances) of Spearman rho, plus the per-evaluation cost estimate `ms_est`; the packing was checked to be lossless. Stage s3 files contain 15,750 rows per instance but only 3,465 distinct configurations (identical configurations generated from different stage-s2 parents, with identical values); the matrix keeps each configuration once |
| `stage_tables/*.csv` | `results/mixing/*top*.csv`, `docs/mixing/s3_dev_summary.csv` | ranked stage summaries |
| `confirm/confirm_<panel>.csv.gz` | `results/mixing/` | per-instance metrics of the confirmation runs |
| `confirm/confirm_<panel>_stats.csv`, `*.friedman.txt` | `results/mixing/` | paired statistics; **the columns `t` and `l` hold losses and ties respectively** (the reproduced statistics use the names `losses` and `ties`) |
| `external_holdout.json` | `results/mixing/` | W1, W3 and comparators on the earlier study's 100-instance hold-out; no script of the source repository produced this file, it is reproduced exactly by `scripts/run_external_holdout.py` |
| `runtime_test2.json` | `results/mixing/` | timings of the source study on 9 instances; superseded by the exploratory cost study of this project |
| `STATUS_MIXING.md`, `S0_AUDIT.md` | `STATUS_MIXING.md`, `docs/mixing/S0_AUDIT.md` | the study's own log |

The specification lists of the data panels are copied from `data/<panel>/specs.json`.
For `real2` the source list declares 120 instances, of which only the first 24
(classic networks) were built, because the snowball-sampled source graphs were
unavailable; `data/real2/specs.json` holds those 24 and the declared list is kept as
`data/real2/specs_declared.json`.

`external/escape_probability/` holds `escape_lib.py`, `holdout.json` and `README.md`
of the earlier escape-probability study (commit `c5377f4`), vendored unmodified by the
source study and copied here unmodified.
