# Fusion continuation errata

The continuation prompt requires an errata record before new computation. The
named pilot files (`confirm_test3.stats.csv`, `confirm_test2.stats.csv`,
`split.json`, `frozen_fusion.json`, and commit `d4fded8`) are not present in this
checkout, and the requested pilot branch is not advertised by `origin`. Their
specific numerical defects therefore cannot be independently recomputed here.

The following absence is itself material evidence: the supplied repository is a
single-upload baseline containing the original W1 evidence, but no pilot fusion
status log or v1 continuation artifacts. Consequently, no claims about pilot
blindness, placeholder freeze times, pilot statistics, or pilot model selection
are made. New v2 artifacts use new filenames and are documented as a continuation
from this incomplete baseline.
