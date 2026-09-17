# Bounded Bandit Attack Experiments

Code and saved results for the paper's UCB, Gaussian TS, and epsilon-greedy experiments.
Run commands from this directory.

## Setup

```sh
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py'
```

For UCB/TS and the Xu comparison, download [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/)
and place `ratings.csv` and `movies.csv` in `ml-25m/`. Raw datasets are excluded from Git.
Epsilon-greedy uses synthetic Bernoulli rewards and needs no dataset.

## Reproduce experiments

Each configuration uses 50 repeats. These commands run sequential simulations;
large horizons can take substantial time.

```sh
# UCB/TS cost and target-selection comparisons
python paper_runner.py
python validate_xu_run.py

# UCB/TS cost scaling and budget allocation
python theory_alignment_runner.py --no-reuse

# Epsilon-greedy comparisons and allocation
python epsilon_greedy_comparison.py
python epsilon_greedy_experiment.py

# UCB versus Xu et al. (2021), varying the target gap
python ucb_xu_comparison.py
```

For a quick check, use `--smoke --repeats 2` with a runner (`--no-reuse` for the theory runner).
Smoke outputs are separate and cannot be used for paper figures.

## Generate figures from saved results

```sh
python generate_paper_figures.py
python generate_theory_figures.py
python generate_epsilon_greedy_figures.py
python generate_ucb_xu_figure.py
```

Generators validate completed result manifests and write TikZ files to `fig/`
and the parent paper's `fig/`. They do not rerun simulations.

| Experiment | Saved results |
|---|---|
| UCB/TS comparisons | `results/ml-25m/paper/` |
| UCB/TS allocation | `results/ml-25m/theory-alignment-fixed-k/` |
| Epsilon-greedy comparisons | `results/synthetic/epsilon-greedy-comparison/` |
| Epsilon-greedy allocation | `results/synthetic/epsilon-greedy/` |
| UCB versus Xu | `results/ml-25m/ucb-xu-gap/` |

Manifests record seeds, configuration, and source/data hashes. Other saved result
files are historical diagnostics, not inputs to the current paper figures.

## Reading the results

- Our cost counts injected observations. Clipped Suppression returns zero on every
  non-target pull and counts these rounds. Xu's cost counts scheduled corrupted rounds.
- Total target selection uses `N_K/T`; our online ratio uses `N_K^on/(T-T0)`.
  Budget shares describe injections, not online selections.
- Shaded bands show sample standard deviation across repeats. Epsilon-greedy's
  deterministic allocation curves have no sampling bands.
- UCB/TS use numerical threshold minimization followed by integer rounding and
  certificate checks. The UCB-Xu gap experiment supplies a known deterministic
  target reward; ordinary stochastic runs estimate its lower bound from the clean log.
- The UCB-Xu comparison uses a common `mean + sqrt(log(t)/N)` learner with
  `sigma=1/3` and Xu's prescribed two-phase budgets. Infeasible budgets are omitted.

`mlrunner.py` supplies data loading and clean logs. `bandit.py` and `simulation.py`
implement UCB/TS; the `epsilon_greedy_*` modules implement EG. `direct_search.py`,
`xu_budget.py`, and `xu_simulation.py` remain dependencies of the saved-result pipeline.
Historical methodological notes are in `EXPERIMENT_AUDIT.md`.
