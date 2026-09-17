# Reproducing the paper experiments

The supported paper pipeline is `paper_runner.py` → `generate_paper_figures.py`.
It uses MovieLens-25M, 50 repeats, actual sequential trajectories, and the original
100,000–1,000,000 horizon grid. Both the arm-count and target-gap sweeps fix
T=1,000,000. The dataset size does not set the learner horizon.

```sh
python -m pip install -r requirements.txt
python -m unittest test_experiments -v
python paper_runner.py
python validate_xu_run.py
python generate_paper_figures.py
python theory_alignment_runner.py --no-reuse
python generate_theory_figures.py
```

Run these commands from this directory. Place the official `ratings.csv` and
`movies.csv` in `ml-25m/`. The complete official ZIP has MD5
`6b51fb2759a8657d3bfcbfc42b592ada` (verified for the downloaded archive).
Dataset: <https://grouplens.org/datasets/movielens/25m/>.

Results and their provenance manifest are written to `results/ml-25m/paper/`.
The manifest records input and output SHA-256 hashes, source hashes, selected movie
IDs/counts/means, seeds, configurations, and runtime. A partial CSV is never used by
the figure generator. `python paper_runner.py --smoke --repeats 2` writes a separate
small run; smoke results cannot be published as 50-repeat paper figures.

Figures are generated into both this directory's `fig/` and the parent paper's
`fig/`. The three former plot entry points delegate to the same generator so they
cannot silently restore different styles or consume uncertified legacy CSVs.
Compile the paper from its parent directory with:

```sh
latexmk -pdf -outdir=output/pdf -interaction=nonstopmode -halt-on-error 0-main.tex
```

Legacy runners and CSV files are retained for historical comparison. They are not
the source of the revised paper figures. In particular, old `certificate` rows
must not be interpreted as simulated success; allocation-only mode now leaves
selection counts and ratios missing.

## Near-boundary theory experiment

`theory_alignment_runner.py` adds an independent experiment under
`results/ml-25m/theory-alignment-fixed-k/`; it does not overwrite the original MovieLens
comparisons or known-target gap diagnostics. The new target is
Bernoulli((11/669)*sqrt(100000/T)) in both the clean log and online phase. Its true
mean is used only to generate rewards, never supplied to the attacker. The attack
retains `max(0, clean_mean - 2*beta)` and the continuous threshold minimizer,
upward rounding, and one extra target sample. With five clean observations and the
original sigma=0.5, delta=0.05, this confidence floor happens to be zero.

The horizons are 100,000, 250,000, 1,000,000, 2,500,000, and 10,000,000. Each curve
fixes K at one of 5, 10, 15, 20, and 25. All have Delta_K=o(S_T/T), with the same MovieLens competitor-selection rule, bounded rewards,
five clean observations per arm, sigma=0.5, and delta=0.05. Every configuration has
50 independent repeats; clean random numbers are shared across horizons at the
same K. The current run contains 2,500 trajectories, regenerated with
`--no-reuse` and the original seed rule. The previous ten-repeat results and
their sources are archived under the parent paper's
`output/reviews/repeats-50-baseline-20260910/`.

`generate_theory_figures.py` validates the completed manifest and trajectories,
then writes `ucb_theory_alignment.tex` and `ts_theory_alignment.tex` to both figure
directories. Both cost panels compare the five fixed arm counts, dividing cost by the
learner's own Lambda. Both allocation panels fix K=10 and show the complementary
shares n_K/C and sum_(i<K) n_i/C, with y-axis limits 0.3 and 0.7 and ticks only at
1/3 and 2/3; their panel-specific
legends identify target and aggregate non-target budgets.
The references 1, 2/3 and 1/3 are asymptotic: Lambda is not a finite-horizon lower
bound, and TS has no allocation necessity claim. Online success is recorded
separately from budget allocation. All plotted means and sample-SD bands use
the 50 per-configuration repeats.

For a quick isolated run and the added validation tests:

```sh
python -m unittest test_theory_simulation test_theory_figures -v
python theory_alignment_runner.py --smoke --repeats 2 --no-reuse
```

Smoke results are stored separately and rejected by the paper figure generator.
The original simulation's default reward path and random streams are unchanged;
the optional target_mean parameter generates an exact Bernoulli probability
without approximating it by a finite empirical array.

## Epsilon-greedy experiment

This independent controlled Bernoulli experiment uses no MovieLens files. The
entire warm-start log is injected (`N_i^0=0`), with target rewards one and
non-target rewards zero. The learner uses `epsilon_s=min(1,K/s)` on its internal
clock, starting online at `T0=C`. Competitor means are equally spaced in
`[0.2,0.8]`; the target mean is `r_T^eg/log(T)>0`, so its gap is
`o(r_T^eg)`. Reward means are used only by the environment, not the allocation.

```sh
python -m unittest test_epsilon_greedy_experiment -v
python epsilon_greedy_experiment.py
python -m unittest test_epsilon_greedy_comparison -v
python epsilon_greedy_comparison.py
python generate_epsilon_greedy_figures.py
```

The horizon and arm-count grids match the near-boundary figures above, with 50
independent sequential trajectories per configuration. Each online decision is
simulated, including greedy maximization, uniform exploration, and Bernoulli
feedback. The runner records greedy mistakes separately from exploratory pulls;
it does not force the target based on the theorem.

The near-boundary trajectories and provenance are saved under
`results/synthetic/epsilon-greedy/`. The separate comparison runner writes
`results/synthetic/epsilon-greedy-comparison/`: it uses ten horizon points from
100,000 to 1,000,000 at `K=10`, and `K=5,10,15,20,30,40,50` at `T=1,000,000`.
Each method has 50 repeats per configuration. The shared configuration is
simulated once and reused in both panels. Both
methods receive the same Bernoulli distributions. Clipped Suppression has no
offline log, initializes with an online target pull followed by the other arms,
and sets every non-target reward to zero. Its cost counts every non-target pull,
including zero-to-zero replacements. Greedy ties use the first arm index for
both methods. Total target ratios include our injected target observations;
online target ratios use only the remaining online horizon.

Use `--smoke --repeats 2` for an isolated quick run. The figure generator checks
completed manifests and saved counts, then writes `eg_cost_experiments.tex`,
`eg_ratio_experiments.tex`, and `eg_theory_alignment.tex` to both figure
directories. These match the UCB/TS cost, ratio, and theory panels. The theory
figure evaluates the exact allocation on ten horizon points, normalizing cost
by `2*sqrt(T*(K-1)*B)` and showing both budget shares with limit `1/2`.
These deterministic quantities have no sampling bands; stochastic comparison
statistics use sample SD. Additional allocation points do not claim extra
online simulations. The `1:1` split is a construction property, not an
allocation necessity result.

## UCB comparison with Xu et al. (2021)

`ucb_xu_comparison.py` runs a separate target-gap sweep for Figure 4. Both attacks
use the same anytime UCB index `mean + sqrt(log(t)/N)`, corresponding to the
exploration setting `sigma=1/3`. The existing UCB/TS/epsilon-greedy experiments
remain unchanged. Competitors use the same MovieLens rewards, while the target
always returns its known reward `Delta_K`. Thus the exact target feedback floor
is valid without a concentration assumption; `sigma=1/3` is not asserted to be
a sub-Gaussian bound for all MovieLens arms.

We retain Xu's prescribed A.2 two-phase budgets, whose formulas contain
`log(T)`, while applying the schedule to the common `log(t)` learner. The source
paper uses a fixed-horizon learner; this comparison does not assume that its
notation is a typo. Xu starts without offline observations; Ours retains five
clean observations per arm and the rounded threshold allocation. Both are run
for 50 repeats at `K=10`, `T=1,000,000`. The sweep uses the previous 14 gap
multipliers and plots the normalized gap `Delta_K/(S_T/T)`.

```sh
python ucb_xu_comparison.py
python generate_ucb_xu_figure.py
```

Results and a separate provenance manifest are saved under
`results/ml-25m/ucb-xu-gap/`. Undefined or horizon-exceeding Xu budgets are stored
as infeasible, with no fabricated selection ratios. Figure 4 shows our injection
cost versus Xu's scheduled corruption rounds, and the three ratio curves used in
Figure 2. Missing Xu points are not connected across infeasible configurations.

## Figure contract

Captions use title case and the pattern `Topic for learner.` Panel titles name
the metric and horizontal variable, for example `Attack Cost vs. Horizon T` or
`Target-arm Selection Ratio vs. Number of Arms K`. Long paired titles break
before `vs.` on both panels. Selection-ratio captions use the complete phrase
`Target-arm Selection Ratio`. Fixed sweep parameters, omitted-point rules, metric definitions, and
uncertainty-band descriptions belong in the experiment text.

| Figure | Panel (a) | Panel (b) |
|---|---|---|
| 1 / UCB costs | Clipped Suppression, Total Attack Cost vs T | Same two series vs K |
| 2 / UCB ratios | N_K/T, N_K^on/H, Clipped Suppression N_K/T vs T | Same three series vs K |
| 3 / UCB near-boundary | C/Lambda vs T, fixed K=5,10,15,20,25 | Target and aggregate non-target budget shares vs T, fixed K=10 |
| 4 / UCB versus Xu | Ours and Xu attack costs vs Delta_K/(S_T/T) | Ours N_K/T, Ours N_K^on/H, Xu N_K/T vs Delta_K/(S_T/T) |
| 5–6 / TS costs and ratios | Original cost/ratio series vs T | Same series vs K |
| 7 / TS near-boundary | C/Lambda_ts vs T, fixed K=5,10,15,20,25 | Target and aggregate non-target budget shares vs T, fixed K=10 |
| 8 / Epsilon-greedy costs | Clipped Suppression, Total Attack Cost vs T | Same two series vs K |
| 9 / Epsilon-greedy ratios | N_K/T, N_K^on/H, Clipped Suppression N_K/T vs T | Same three series vs K |
| 10 / Epsilon-greedy near-boundary | C/(2 sqrt(T(K-1)B)) vs T, fixed K=5,10,15,20,25 | Target and aggregate non-target budget shares vs T, fixed K=10; both approach 1/2 |

The former UCB/TS gap diagnostics remain preserved separately. Figure 4 uses the
new UCB-only comparison, not those earlier results; the shared paper figure
generator does not produce gap figures.

`Clipped Suppression` returns zero on every non-target online pull and leaves target rewards unchanged, for UCB, TS, and epsilon-greedy. It has no suppression-margin parameter.
Replaying all 700 UCB baseline records (50 repeats per configuration) under this rule preserved every stored metric and both UCB figures; see `../output/reviews/zero-suppression-baseline-20260911/replay-report.json`.
This baseline replaces the vague label `Heuristic`. All main cost curves
use the threshold construction without displaying “Threshold” in their legend.
Direct is omitted from all paper figures; its stored results remain available for numerical diagnostics. Cost axes share the
label `Attack Cost`; ratio axes share `Target-arm Ratio`. All axes use linear scales; cost axes start at zero. `STYLE` in
`generate_paper_figures.py` is the sole palette/marker/line-style registry. All curves are solid and use only the user's six RGB definitions
(`myblue`, `myorange`, `mygreen`, `mypurple`, `myred`, `mybrown`). Green is reserved
so blue and green do not share a panel. Distinct markers identify curves independently
of color. Markers have transparent interiors so curves remain visible through
them, and all axis grid lines are disabled. Bands represent sample SD (`ddof=1`), not standard errors or
confidence intervals.

## What the comparison does and does not establish

Our cost counts injected observations. Clipped Suppression counts every online
non-target suppression round, including 0-to-0 replacements; equivalently its
cost is the sum of online non-target pulls. Xu et al. (2021) counts all scheduled
corrupted rounds, including zero-valued rewards that stay zero. These retain
their explicit definitions and are not identical attacker capabilities.

The main arm environments are Bernoulli distributions calibrated from MovieLens,
not chronological replay or a contextual recommendation evaluation. The gap
sweep replaces only the target distribution with the known deterministic reward
mu_K and explicitly supplies mu_K as its exact target lower bound to both
Threshold and Direct. There is no statistical uncertainty in these target
observations. Gaussian TS decision noise remains included. This is a synthetic
perturbation with known target information; the stochastic MovieLens sweeps
continue to estimate their target lower bounds from the clean log. Selection of a minimum-positive-mean movie
does not establish the asymptotic near-boundary condition as T tends to infinity.

See `EXPERIMENT_AUDIT.md` for the formula audit, corrections, and remaining limits.

The Direct search for the deterministic gap sweep uses the same original grid
(4097 points followed by five 1025-point refinements) with an exact integer
minimization of the unimodal TS target envelope. `direct_search.py` documents the
monotonicity argument; validation compares allocations with the original search.
Historical conservative-bound gap results are archived with their manifest in
`results/ml-25m/paper/history/conservative-gap-20260907/`.

The stored Xu et al. (2021) results use the historical `Two-phase` method key.
These diagnostic comparisons use **the original learner settings and budget
formulas**, without a new adaptation proof:

- UCB: mean + sqrt(log(T)/N), initialization in index order; the bounded-reward
  model allows the same deterministic target feedback mu_K as our gap diagnostic.
  Appendix A.2 supplies C1=K*ceil(log(T)/mu_K^2) and
  C2=ceil(mu_K*C1/(1-mu_K)). C1 is rounded to a multiple of K.
- TS: Beta(sum+1,N-sum+1), starting directly from Beta(1,1) priors. The target
  generates Bernoulli(mu_K) rewards; competitors remain MovieLens-calibrated
  Bernoulli arms. Appendix A.4 supplies the phase lengths with integer rounding.
  The original finite-K waiting term can make this budget nonmonotone in mu_K.

`xu_budget.py` implements the original formulas; `xu_simulation.py` implements
only the original learners. Our UCB and Gaussian TS, including our deterministic
target feedback, stay as specified above. This comparison therefore mixes learner
specifications (and TS target-feedback models); it does not isolate attack choice
under identical learners. Original sources:
[paper Sections 5.1/5.3](https://proceedings.neurips.cc/paper/2021/file/be315e7f05e9f13629031915fe87ad44-Paper.pdf),
[supplement A.2/A.4](https://proceedings.neurips.cc/paper_files/paper/2021/file/be315e7f05e9f13629031915fe87ad44-Supplemental.pdf).

Every scheduled round counts, including 0-to-0. At mu_K=0 the formulas are undefined. Budgets reaching T and the
TS prescription with nonpositive phase-one lower count bound are omitted with
explicit reasons in the CSV. No clipping or fitted constants repair these cases.

The CSV saves both phase lengths, formula parameters, learner/feedback settings,
actual phase counts, and the post-attack target fraction. `validate_xu_run.py`
checks stage counts, source hashes, and stability of non-Xu records. It also runs
a separate implementation check using the original paper's Table 1 setting:
K=2, T=50,000, means (0.9,0.8), C1=34, C2=66. Those example phase lengths are not
reused for the MovieLens gap sweep.

The withdrawn adapted-budget version, including its removed proof, is archived
under `results/ml-25m/paper/history/derived-xu-budgets-20260907/`.

## Online baseline initialization

Clipped Suppression and Xu et al. (2021) receive no offline warm-start log.
Both start with zero counts and sums and run T online rounds. Clipped initializes
by pulling the target once and then the non-targets in index order. Original Xu
UCB initializes in index order; original Xu TS samples from Beta(1,1) immediately.
Attacks apply from round one. Both baseline total target ratios use denominator T;
the separately recorded Xu post-attack ratio uses T-C1-C2. Our offline attack
retains five clean observations per arm. `rerun_online_baselines.py` is a historical
migration script; use `paper_runner.py` for current results.
