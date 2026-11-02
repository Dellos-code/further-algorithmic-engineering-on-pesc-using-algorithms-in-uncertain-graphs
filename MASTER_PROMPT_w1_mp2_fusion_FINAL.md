# MASTER PROMPT — Regime-adaptive rank fusion of W1 and two-iteration message passing (MP2)

You are a research engineer continuing a diploma-thesis study on reachability ranking in
uncertain graphs. Work **only additively**, keep a written log, and stop at every gate
marked **STOP** until its condition is satisfied. Everything you produce (code, data,
results, documents) must be reproducible from the repository by a stated command.

---

## 0. Repository, branch and immutable assets

* Repository: `Dellos-code/further-algorithmic-engineering-on-pesc-using-algorithms-in-probabilistic-graphs`
* Base: the default branch (`local-upload-1793576992`, commit `4e806ed`), which contains the self-contained W1 project in `w1-dual-operator-rank-fusion/` and the W1 documents at the top level.
* Work only in this repository. The other repositories of the author (named below only as sources) are read-only references: never push to them.
* Create and work on branch **`exp/w1-mp2-fusion`**. Open a draft pull request into the default branch at the end; do not merge it.
* Package root: `PKG = w1-dual-operator-rank-fusion/`. Read `PKG/README.md`, `PKG/docs/specification/W1_specification_en.pdf` and `PKG/docs/evidence/W1_evidence_en.pdf` first; run `python PKG/tests/test_algorithm.py` and `python PKG/tests/test_reproduction.py` and **STOP** if either fails.
* **Immutable** (never modify, move or regenerate): everything that exists in `PKG/data/`, `PKG/evidence/`, `PKG/docs/`, `PKG/external/`, the existing files of `PKG/w1/`, `PKG/scripts/` and `PKG/tests/`, `PKG/reproduce.sh`, the licences, and the PDF/TeX files at the top level. New code goes into new files (a new subpackage `PKG/fusion/`, new scripts `PKG/scripts/fusion_*.py`, new tests `PKG/tests/test_fusion.py`); new outputs go into `PKG/results/fusion/`, `PKG/data/<new panel>/`, `PKG/docs/fusion/`, `PKG/thesis/fusion/`. Keep the existing licences (MIT for code, CC BY 4.0 for documents and data) for everything you add.
* Single-threaded numerics for every run that produces numbers: `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.
* Never use ground truth `yA` for any fitting or selection; oracles and fits use `yB` (or development data only).

### Members of the fusion (reuse, do not re-implement)

* **W (W1)**: `w1.algorithm.w1_scores(inst, w1.algorithm.W1)` (the reference implementation; bit-identical to the frozen configuration of `PKG/evidence/original/selected_mixing.json` evaluated by `w1.hybrid.evaluate`). Verify first that it reproduces the stored W1 values of `PKG/evidence/original/confirm/confirm_test2.csv.gz` to 1e-12 on 20 instances; **STOP** if not.
* **M (MP2)**: `w1.diffusion.knz_mp(inst, iters=2)[0]` (Karrer–Newman–Zdeborová message passing in membership variables, two iterations). Verify against the stored `MP2` rows of the same file.
* Comparators: `w1.hybrid.baseline_scores` (IRIE, rule_EW, rule_WT, MP1/MP2/MPinf, oracle:scalar, …), `w1.hybrid.CTRL_LIN09` via `w1.hybrid.evaluate`, Monte Carlo `w1.instances.mc_reach`.
* Ranks and metrics: `w1.metrics.ranks_of`, `spearman`, `full_metrics` against `yA`. Instances and ground truth: `w1.instances` (`make_synthetic`, `spec_list`, `ground_truth`, `load_instance`, `stable_seed`).

---

## 1. Research question and why it is worth asking

On the held-out panels of the mixing study, W1 is the best deployable method on average, but
MP2 is better at high edge probabilities (e.g. `test2` R5: MP2 0.815 vs W1 0.740; `big` R4:
0.923 vs 0.848) and much worse at low ones (R1: about 0.50 vs 0.985). The two members fail in
opposite regimes. The question is whether a **regime-adaptive rank fusion**

    F_c = c · r(M) + (1 − c) · r(W),        c = g(φ(G, p, x)) ∈ [0, 1],

with a fixed, deterministic function `g` of a graph-level covariate `φ`, improves on W1, on
MP2, and — the decisive control — on the **best constant c** fusion.

Do not assume that `g` is linear, monotone, smooth or even non-constant: these are to be
established from data. Fusion is on ranks, not on scores (the two members live on different
scales: MP2 returns giant-component membership probabilities, W1 returns rank sums).

### Primary hypotheses (confirmatory, on held-out data only)

* **H1** mean ρ(F_g) > mean ρ(W1)
* **H2** mean ρ(F_g) > mean ρ(F_{c₀}), where c₀ is the best constant c on the fitting data
* **H3** mean ρ(F_g) > mean ρ(MP2)

### Secondary

* regret of F_g against the per-instance oracle c (chosen on `yB`, scored on `yA`) and against the per-instance oracle choice between W1 and MP2 (hard gate);
* the hard gate `c ∈ {0, 1}` (threshold on φ) as a special case of `g`;
* MP-k with k ∈ {1, 3, 5, ∞} in place of MP2 (only if pre-declared in the freeze; MP2 stays primary).

### Exploratory (label as such everywhere)

* **Per-vertex physical mixture**: `c_i = μ_i` (the converged message-passing probability that vertex i belongs to the giant component), i.e. `F_i = μ_i r(M)_i + (1 − μ_i) r(W)_i`, and variants. This uses the physics of percolation instead of a fitted `g` and has no fitted parameter; report it, but it is not a primary hypothesis.

---

## 2. The covariate φ

### 2.1 Primary covariate (pre-specified): mean edge probability over the **existing** edges

    p̄_E = (1/m) Σ_{(i,j)∈E} p_ij

**Non-edges are not included as p = 0.** Including them gives p̄_all = 2Σp / (n(n−1)) =
(mean expected degree)/(n − 1), the expected density of the graph: it scales as 1/n on sparse
graphs, is close to zero for every real network, and measures size and sparsity rather than
the reliability of the existing connections. Its only useful content — how many expected
neighbours a vertex has — is captured directly and without the 1/n distortion by the
expected-degree covariates below.

### 2.2 Challenger covariates (the data decide; see §5)

Whether a graph is in the regime where MP2 wins is a percolation question: a giant component
forms when the expected branching exceeds one, which depends on degrees and their
heterogeneity as well as on p. A mean over edges alone cannot see this (p̄_E = 0.3 is
subcritical on a path and supercritical on a dense graph). Compute for every instance:

| name | definition | cost |
|---|---|---|
| `pbar_E` | mean p over existing edges (primary) | O(m) |
| `dbar` | mean expected degree 2Σp/n | O(m) |
| `Pesc_bar` | mean escape probability (already computed by W1) | O(m) |
| `rho_P` | spectral radius of the symmetric matrix P (Lanczos) | O(m · it) |
| `rho_NB` | spectral radius of the p-weighted non-backtracking matrix on the 2m directed edges (power iteration) — the message-passing percolation threshold is `rho_NB = 1` | O(m · it) |
| `S_MP` | x-weighted giant-component membership Σ_i x_i μ_i from converged message passing | O(m · it) |
| `kappa_E` | p̄_E · (⟨k²⟩ − ⟨k⟩)/⟨k⟩ (configuration-model branching factor) | O(m) |

All covariates must be computable from (G, p, x) alone, in O(m · iterations), with no ground
truth. Record their cost: the fused method must remain cheap.

---

## 3. Data — real networks first

### 3.1 Real networks (primary evidence)

Aim for **at least 24 distinct real networks** from at least 5 domains (biological,
collaboration/social, communication, infrastructure such as power grids and roads, web or
citation, technological). Each network must be downloaded by a committed script with a
recorded URL, licence and SHA-256 of the raw file; the raw files themselves may be cached
outside git if large (`PKG/data_real_fusion/`, git-ignored) as long as the script rebuilds
them. Suitable public sources: SNAP, KONECT, Netzschleuder (graph-tool), Network Repository,
the gnn-benchmark `npz` collection, STRING.

Priority within the real networks:
1. **Native probabilities** (strongly preferred): protein–protein interaction networks with
   confidence scores (e.g. STRING combined score / 1000, restricted to one organism and a
   score threshold that you declare), and other uncertain-graph benchmarks with published
   edge probabilities (e.g. co-authorship networks with p = 1 − exp(−w/μ) for w joint papers,
   declare μ). Cite the definition used.
2. **Real topology with assigned probabilities**: uniform ranges R1–R5 (as in the existing
   panels), weighted cascade 1/max(deg_i, deg_j), trivalency {0.1, 0.01, 0.001}, and an
   exponential scheme; each network gets several schemes so that p̄_E spans its full range.

Graph sizes: connected (or largest-component) subgraphs of 500, 2,000 and, where affordable,
10,000 vertices, extracted by a snowball sampler with recorded seeds (port `snowball`, `subgraph_edges` and `load_npz_graph` from `prop_damping_lib.py` of the repository `self-consistent-proportional-per-node-damping-for-reachability-ranking-in-stochastic-graphs` into `PKG/fusion/`, unchanged, and cite the source); at least
two priors per instance (uniform and Dirichlet(1)). Ground truth exactly as in
`w1.instances.ground_truth`: two independent Monte-Carlo samples of 2,000 worlds (`yA`, `yB`) with
seeds from `stable_seed(inst.name, "A"/"B")`.

**Search thoroughly before giving up on real data.** If a source fails, try the others and
their mirrors before concluding that real networks are unavailable: the official sites
(SNAP, KONECT, Netzschleuder, Network Repository, STRING), GitHub raw files and release
assets (e.g. `shchur/gnn-benchmark`), Zenodo and Figshare records, Hugging Face datasets,
Python packages that ship or fetch network data (`networkx` built-ins, `torch_geometric`,
`ogb`, `scikit-network`, `igraph`, `graph-tool`'s collection), and conda packages. Log every
attempt in `PKG/results/fusion/data_search_log.md` (source, URL or package, date, outcome, error
message). Use whatever real networks you obtain, however few, including the classic networks
bundled with `networkx` (karate club, Les Misérables, Davis southern women, Florentine
families). Only when this search is exhausted, apply the fallback of §3.4 and continue; do not
stop.

### 3.2 Synthetic panels (secondary evidence and exploration in the `real` mode; see §3.4)

* `dev` (existing, 180 instances): may be used for exploration and fitting.
* The existing confirmation panels (`test`, `prior`, `test2`, `scale2`, `prior2`, `real2`, `big`) **are no longer blind for this question** — their MP2-versus-W1 behaviour by regime has already been inspected. Use them only for fitting or exploration, never for confirmation.
* Build **fresh** synthetic confirmation panels with new seeds (`rep` values never used before; record them): `test3` (n = 100, the 9 families × R1–R5 × 2 priors × 2 reps), `scale3` (n = 900, × 1 rep), `big2` (n ∈ {2000, 5000}, 5 families, R1–R5). Check, by hashing (edges, p, x), that no new instance coincides with any instance in `PKG/data/`.

### 3.3 Splits — by network, not by instance

Instances of the same network are strongly correlated. Split **real networks** (not instances)
into **FIT** (≈ 60 %) and **HOLD** (≈ 40 %), stratified by domain and by size, with a recorded
seed, before computing any member on HOLD. Every instance of a HOLD network (all sizes,
schemes, priors) is held out. Write the split to `PKG/results/fusion/split.json` and commit it
**before** any HOLD computation. **STOP** until this commit exists.

### 3.4 Fallback when real networks cannot be obtained

Let R be the number of distinct real networks obtained after the search of §3.1. Decide the
evidence mode from R, write it to `PKG/results/fusion/evidence_mode.json` together with the search
log, and commit both **before** the split:

| R | mode | primary evidence | real networks |
|---|---|---|---|
| R ≥ 24 | `real` | real HOLD networks (§3.3) | as planned |
| 8 ≤ R < 24 | `mixed` | real HOLD networks **and** the synthetic HOLD panels of this section, each with its own Holm family | split as in §3.3; report the reduced power |
| R < 8 | `synthetic` | the synthetic HOLD panels of this section | all of them are used as an extra held-out, descriptive set; never for fitting |

In the `mixed` and `synthetic` modes, extend the synthetic data so that they stand in for real
networks as well as possible:

* **Generators with realistic structure**, in addition to the nine existing families:
  Chung–Lu and configuration models with power-law degrees (exponents 2.1–3.0), the LFR
  benchmark (power-law degrees and communities), Holme–Kim (scale-free with clustering),
  random geometric graphs (infrastructure-like, spatial), R-MAT/Kronecker graphs,
  forest-fire graphs, and degree-corrected stochastic block models. Record every parameter.
* **Probability schemes**: the uniform ranges R1–R5, weighted cascade, trivalency
  {0.1, 0.01, 0.001}, an exponential scheme, and probabilities correlated with degree
  (high-degree pairs more and less reliable), so that every covariate of §2 spans its full
  range.
* **Sizes** 500, 2,000 and, where affordable, 10,000 vertices; priors uniform and
  Dirichlet(1); ground truth as in §3.1.
* **Split by generator family, not only by seed.** The synthetic FIT set uses some generator
  families; the synthetic HOLD set contains new seeds of those families **and at least three
  families that never appear in FIT** (for example LFR, random geometric and R-MAT). Report
  the out-of-family HOLD results separately: they are the closest substitute for transfer to
  unseen real networks.
* In model selection (§5.3), grouped cross-validation leaves out one generator family at a
  time; the real-network CV score, if any real networks were obtained, is still reported.

In every mode, state the mode prominently in `PKG/STATUS_FUSION.md`, in the pull-request summary
and in the abstract and limitations of both thesis chapters (e.g. "evidence from synthetic
graphs only; R real networks were obtainable"), and do not describe synthetic results as
results on real networks.

---

## 4. Per-instance quantities (FIT data only during development)

For every FIT instance compute and store (one row per instance, one file per panel):

* ρ of W1, MP2, MP1, MP3, MP5, MP∞, IRIE, CTRL (the paper's configuration), the oracle scalar damping;
* the curve ρ(c) of F_c for c ∈ {0, 0.025, …, 1} (41 values), evaluated on `yA`, **and** the same curve on `yB`;
* `c*_B` = argmax over the grid of ρ_yB(c) (fit sample) and the flat region {c : ρ_yB(c) ≥ max − 0.002};
* all covariates of §2;
* single-thread timings of W1, MP2 and the covariates.

Plot (exploration only): c*_B and the flat regions against each covariate, coloured by
domain/family; ρ(c) curves for representative instances.

---

## 5. Model of g, selection and the freeze

### 5.1 Candidate families for g (do not presuppose the form)

1. constant c₀;
2. hard gate: c = 1[φ > τ];
3. clipped linear in φ (or in log φ);
4. logistic: c = σ(a (φ − τ));
5. isotonic (monotone, non-parametric) with at most 5 steps;
6. natural cubic spline with ≤ 3 interior knots, clipped to [0, 1];
7. two-covariate logistic (only φ pairs from §2.2; at most 3 parameters).

### 5.2 Fit the objective, not c*

ρ(c) is flat near its maximum, so c* is a noisy target and least squares on c* is the wrong
objective. Fit every g by **maximising the pooled mean ρ_yB(g(φ_i)) over FIT instances**
(interpolate the 41-point curves linearly), with a small grid or derivative-free optimiser.
Report the least-squares-on-c* fit only as a diagnostic.

### 5.3 Selection with protection against overfitting

* Nested, **grouped** cross-validation over FIT: leave-one-network-out for real data (all instances of a network form one group); leave-one-family-out for synthetic data.
* Score = held-out-fold mean ρ_yA of F_g. Use the **one-standard-error rule**: among (covariate, family) pairs within one SE of the best CV score choose the one with the fewest parameters; ties are broken in favour of the pre-specified `pbar_E`, then of the cheaper covariate.
* In the `real` mode the real-network CV score is primary and the synthetic CV score may only break exact ties; in the `mixed` mode both are reported and a choice must be within one SE of the best on both; in the `synthetic` mode the leave-one-generator-family-out score is primary (§3.4).
* Report learning curves (CV score versus number of FIT networks) to show whether more data would change the choice.
* Check stability: refit the chosen g on 200 network-level bootstrap resamples of FIT and report the spread of g; if the chosen covariate changes in more than half of the resamples, report that the choice is unstable.

### 5.4 Freeze

Write `PKG/results/fusion/frozen_fusion.json` containing: the member definitions, the covariate, the
family and all parameters of g, the constant c₀ (best constant on FIT), the hard-gate τ, the
list of HOLD networks and fresh synthetic panels, the hypotheses H1–H3 and the statistical
plan of §6, and the UTC time. Commit it with its SHA-256 (`frozen_fusion.sha256`) in a commit
whose message starts with `FREEZE:`. **STOP**: no member, covariate or ground truth of a HOLD
network or of `test3`/`scale3`/`big2` may be computed before this commit. The log must show
the order.

---

## 6. Confirmation

* Evaluate on the primary HOLD set of the evidence mode (§3.4) and on `test3`, `scale3`, `big2` (secondary): F_g, F_{c₀}, hard gate, W1, MP2, MP∞, IRIE, CTRL, the oracle scalar damping, the per-instance oracle c, and Monte Carlo with 50/100/200/500 worlds (accuracy and single-thread time).
* Primary family (Holm across the three contrasts, separately for each primary HOLD set of the evidence mode): H1, H2, H3. Paired two-sided Wilcoxon signed-rank tests on per-instance ρ, zero differences dropped.
* Because instances are clustered within networks, also report **network-level** results: the per-network mean ρ, a Wilcoxon test over networks, and a **cluster (network) bootstrap** 95 % interval of the mean difference (20,000 resamples). The network-level test is the one quoted in abstracts.
* Secondary metrics: Kendall τ_b, top-10 % overlap, NDCG@10 %.
* Break down by domain, by size and by bins of the chosen covariate; report where F_g loses to either member.
* Report everything, including null and negative results, with the same prominence.

---

## 7. Code, outputs and documentation

* New modules (suggested): `PKG/fusion/members.py`, `PKG/fusion/covariates.py`, `PKG/fusion/gfamilies.py`, `PKG/fusion/real_data.py`, `PKG/fusion/generators.py`; scripts `PKG/scripts/fusion_build_real.py`, `fusion_build_panels.py`, `fusion_run.py` (staged runner), `fusion_select.py` (CV, freeze), `fusion_confirm.py`, `fusion_analyze.py` (statistics with `w1.stats`; columns `wins`, `losses`, `ties`), `fusion_make_assets.py`.
* Tests in `tests/test_fusion.py`: F_0 = W1 and F_1 = MP2 in rank order; covariates on hand-checked toy graphs (path, star, triangle, two components); `rho_NB = p(k − 1)` on a k-regular graph with uniform p (and `rho_NB = 0` on a tree, where the non-backtracking matrix is nilpotent); determinism; reproduction of stored W1/MP2 values.
* A status log `PKG/STATUS_FUSION.md` updated at every stage, with counts, timings and anything unexpected.
* A GitHub Actions workflow `.github/workflows/fusion.yml` (at the repository root) that runs the tests and a 5-minute smoke version of the pipeline.
* `PKG/docs/fusion/README.md` with the exact reproduction commands and expected runtimes, and a short section added to a **new** file `PKG/README_FUSION.md` (do not edit the existing README files).

### Thesis chapter (English and Greek)

`PKG/thesis/fusion/fusion_en.tex` and `PKG/thesis/fusion/fusion_gr.tex`: method, covariates (including
the reasoning of §2.1 on non-edges), data and split, model selection, freeze, confirmation,
limitations. Requirements:
* every number is a LaTeX macro generated by `make_fusion_assets.py` from `PKG/results/fusion/`; no number typed by hand;
* compile with XeLaTeX (FreeSerif with `RawFeature={-ccmp}`, babel `[english,main=greek]` for the Greek file), three passes, **0 errors, 0 overfull and 0 underfull boxes**;
* the documents are pure academic texts: no acknowledgement of method, tooling or assistance section;
* Greek terminology: no transliterations of English words in Greek script; give the English term in parentheses at first mention (e.g. «μετάδοση μηνυμάτων (message passing)», «συνδυασμός τάξεων (rank fusion)», «μαντείο (oracle)», «διασταυρωμένη επικύρωση (cross-validation)», «γιγάντια συνιστώσα (giant component)», «πύλη (gate)»); use «βαθμολογία», never «σκορ»; «αναμενόμενη προσβάσιμη μάζα», «πιθανότητα διαφυγής», «αποκομμένος γραμμικός (clipped-linear)».

---

## 8. Stages and gates

| stage | content | gate |
|---|---|---|
| F0 | reproduce stored W1 and MP2 values; commit | **STOP** on any mismatch > 1e-12 |
| F1 | exhaustive real-network search with its log; acquisition scripts, licences, checksums; evidence mode (§3.4); FIT/HOLD split committed | **STOP** until `evidence_mode.json` and `split.json` are committed; if real networks are scarce, continue with the fallback of §3.4 instead of stopping |
| F2 | FIT instances and ground truth; per-instance quantities (§4); exploration plots | — |
| F3 | model selection (§5.1–5.3), stability, learning curves | — |
| F4 | `FREEZE:` commit with digest | **STOP**: nothing on HOLD or fresh panels before this |
| F5 | HOLD and fresh synthetic panels built and evaluated; confirmation statistics | — |
| F6 | exploratory per-vertex mixture; cost study | label as exploratory |
| F7 | thesis chapter EN/GR, docs, workflow, draft PR with a summary of H1–H3 at network level | 0 errors / 0 overfull / 0 underfull |

At the end, reply with: the evidence mode and the number R of real networks obtained (with the search log), the frozen g (covariate, family, parameters), the network-level and
instance-level results of H1–H3 with intervals and Holm-adjusted p-values, the regret against
the oracles, the cost relative to W1, and a list of every deviation from this prompt.
