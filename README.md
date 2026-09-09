# Reproducing the paper experiments

The supported paper pipeline is `paper_runner.py` → `generate_paper_figures.py`.
It uses MovieLens-25M, ten repeats, actual sequential trajectories, and the original
100,000–1,000,000 horizon grid. Both the arm-count and target-gap sweeps fix
T=1,000,000. The dataset size does not set the learner horizon.

```sh
python -m pip install -r requirements.txt
python -m unittest test_experiments -v
python paper_runner.py
python validate_xu_run.py
python generate_paper_figures.py
python theory_alignment_runner.py
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
small run; smoke results cannot be published as ten-repeat paper figures.

Figures are generated into both this directory's `fig/` and the parent paper's
`fig/`. The three former plot entry points delegate to the same generator so they
cannot silently restore different styles or consume uncertified legacy CSVs.
Compile the paper from its parent directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error 0-main.tex
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
ten independent repeats; clean random numbers are shared across horizons at the
same K. The 500 CSV rows contain 100 verified K=10 trajectories reused from
`results/ml-25m/theory-alignment/` and 400 newly simulated trajectories. The old
data are preserved; the new manifest records reuse provenance and source hashes.
Use `--no-reuse` to regenerate all trajectories.

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
the ten per-configuration repeats.

For a quick isolated run and the added validation tests:

```sh
python -m unittest test_theory_simulation test_theory_figures -v
python theory_alignment_runner.py --smoke --repeats 2
```

Smoke results are stored separately and rejected by the paper figure generator.
The original simulation's default reward path and random streams are unchanged;
the optional target_mean parameter generates an exact Bernoulli probability
without approximating it by a finite empirical array.

## Figure contract

Captions are short titles. Experimental settings, cost/ratio definitions, and
uncertainty-band descriptions belong in the main text and TS appendix.

| Figure | Panel (a) | Panel (b) |
|---|---|---|
| 1 / UCB costs | Clipped Suppression, Total Attack Cost vs T | Same two series vs K |
| 2 / UCB ratios | N_K/T, N_K^on/H, Clipped Suppression N_K/T vs T | Same three series vs K |
| 3 / UCB near-boundary | C/Lambda vs T, fixed K=5,10,15,20,25 | Target and aggregate non-target budget shares vs T, fixed K=10 |
| 4–5 / TS costs and ratios | Original cost/ratio series vs T | Same series vs K |
| 6 / TS near-boundary | C/Lambda_ts vs T, fixed K=5,10,15,20,25 | Target and aggregate non-target budget shares vs T, fixed K=10 |

The UCB and TS gap figures and experimental discussion are omitted from the
paper. The diagnostic CSVs and implementations are preserved; the paper figure
generator no longer produces gap figures.

`Clipped Suppression` replaces the vague label `Heuristic`. All main cost curves
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
