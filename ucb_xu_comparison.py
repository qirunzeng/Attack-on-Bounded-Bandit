"""Independent gap comparison using one anytime UCB learner for both attacks.

Xu's original A.2 phase budgets are applied to the common log(t) learner.
This adaptation does not claim the guarantee for its original log(T) learner.
"""
import argparse
from contextlib import contextmanager
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import platform
import time

import numba
import numpy as np
from numba import njit

import bandit
import mlrunner
from paper_runner import GAPS, sha256
import xu_budget


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/ml-25m/ucb-xu-gap'
VERSION = 'ucb-xu-gap-v1'
K = 10
T = 1_000_000
EXPLORATION_PARAMETER = 1 / 3
DELTA = .05
N0 = 5
REPEATS = 50
SEED = 20260911
METHODS = ['Ours', 'Xu2021']


def configuration_grid(horizon=T, arms=K, multipliers=GAPS):
    scale = horizon**(2 / 3) * (arms * math.log(horizon))**(1 / 3) / horizon
    grid = [dict(multiplier=float(value), Delta_K=float(value * scale)) for value in multipliers]
    if any(not math.isfinite(c['Delta_K']) or not 0 <= c['Delta_K'] < 1 for c in grid):
        raise ValueError('The target gaps must lie in [0, 1)')
    return grid


@contextmanager
def allocation_settings(arms, exploration_parameter=EXPLORATION_PARAMETER):
    """Do not contaminate the existing runner's globals or allocation cache."""
    settings = dict(K=arms, target_arm=arms, sigma=exploration_parameter,
                    delta=DELTA, N0_i=N0, r_l=0., r_u=1., R=1.)
    previous = {name: getattr(bandit, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(bandit, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(bandit, name, value)


@lru_cache(maxsize=None)
def allocation(clean_sum, horizon, target_mean, exploration_parameter=EXPLORATION_PARAMETER):
    """A separate cache includes the exploration parameter and exact floor."""
    clean_mean = np.asarray(clean_sum, dtype=float) / N0
    with allocation_settings(len(clean_mean), exploration_parameter):
        n, z, floor, _ = bandit._ucb_counts_for_T(
            clean_mean, horizon, known_target_mean=target_mean)
        if N0 * len(n) + int(n.sum()) < horizon:
            bandit._check_ucb_cutoffs(clean_mean, horizon, n, z, floor)
    return tuple(int(v) for v in n), z, floor


@njit(cache=True)
def trajectory(initial_counts, initial_sums, rewards, offsets, horizon,
               rng, phase1, phase2, intervene, exploration_parameter):
    counts, sums = initial_counts.copy(), initial_sums.copy()
    arms = len(counts)
    start = int(counts.sum())
    online = np.zeros(arms, dtype=np.int64)
    stages = np.zeros((3, arms), dtype=np.int64)
    modifications, magnitude = 0, 0.
    for step in range(horizon - start):
        internal_time = start + step + 1
        missing = -1
        for i in range(arms):
            if counts[i] == 0:
                missing = i
                break
        if missing >= 0:
            arm = missing
        else:
            arm, best = 0, -math.inf
            for i in range(arms):
                value = (sums[i] / counts[i] + 3 * exploration_parameter *
                         math.sqrt(math.log(internal_time) / counts[i]))
                if value > best:
                    arm, best = i, value
        raw = rewards[rng.integers(offsets[arm], offsets[arm + 1])]
        stage = 0 if intervene and step < phase1 else (
            1 if intervene and step < phase1 + phase2 else 2)
        observed = 0. if stage == 0 else (1. if arm == arms - 1 else 0.) if stage == 1 else raw
        modifications += int(observed != raw)
        magnitude += abs(observed - raw)
        sums[arm] += observed
        counts[arm] += 1
        online[arm] += 1
        stages[stage, arm] += 1
    return online, modifications, magnitude, stages[0], stages[1], stages[2]


def simulate(clean_sum, n, reward_arrays, horizon, seed, *, phase1=0, phase2=0,
             intervene=False, exploration_parameter=EXPLORATION_PARAMETER):
    clean_sum, n = np.asarray(clean_sum, dtype=float), np.asarray(n)
    arms = len(reward_arrays)
    if (clean_sum.shape != (arms,) or n.shape != (arms,) or arms < 2 or
            not np.issubdtype(n.dtype, np.integer) or np.any(n < 0)):
        raise ValueError('Invalid arm counts or clean-log dimensions')
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError('horizon must be a positive integer')
    if any(np.asarray(a).ndim != 1 or not len(a) or not np.all(np.isfinite(a)) or
           np.any((np.asarray(a) < 0) | (np.asarray(a) > 1)) for a in reward_arrays):
        raise ValueError('Reward arrays must contain bounded feedback in [0, 1]')
    if not math.isfinite(exploration_parameter) or exploration_parameter <= 0:
        raise ValueError('The exploration parameter must be positive')
    if (not isinstance(phase1, (int, np.integer)) or not isinstance(phase2, (int, np.integer))
            or phase1 < 0 or phase2 < 0 or phase1 + phase2 >= horizon):
        raise ValueError('The intervention must end before the terminal time')
    if intervene:
        if np.any(clean_sum) or np.any(n):
            raise ValueError('Xu has no offline log')
        counts, sums = np.zeros(arms, dtype=np.int64), np.zeros(arms)
    else:
        if phase1 or phase2 or not np.all(np.isfinite(clean_sum)) or np.any((clean_sum < 0) | (clean_sum > N0)):
            raise ValueError('Invalid offline clean log or intervention phases')
        counts, sums = np.full(arms, N0, dtype=np.int64) + n, clean_sum.copy()
        sums[-1] += n[-1]
    if int(counts.sum()) >= horizon:
        raise ValueError('Allocation leaves no online horizon')
    rewards = np.concatenate(reward_arrays)
    offsets = np.r_[0, np.cumsum([len(a) for a in reward_arrays])]
    return trajectory(counts, sums, rewards, offsets, horizon, np.random.default_rng(seed),
                      phase1, phase2, intervene, exploration_parameter)


def run_case(metadata, repeat, config, horizon=T):
    arrays = list(metadata['selected_reward_arrays'])
    arms, gap = len(arrays), config['Delta_K']
    arrays[-1] = np.array([gap])
    clean_rng = np.random.default_rng(mlrunner.warm_start_seed(arms, repeat))
    clean_sum = np.array([clean_rng.choice(a, size=N0, replace=True).sum() for a in arrays])
    # Record the exact known constant, avoiding summation roundoff in the floor.
    clean_sum[-1] = N0 * gap
    n_values, z, floor = allocation(tuple(clean_sum), horizon, gap)
    budget = xu_budget.budget('UCB', arms, horizon, gap)
    rows = []
    for method_id, method in enumerate(METHODS):
        offline = method == 'Ours'
        n = np.asarray(n_values, dtype=np.int64) if offline else np.zeros(arms, dtype=np.int64)
        clean = clean_sum if offline else np.zeros(arms)
        start = N0 * arms + int(n.sum()) if offline else 0
        cost = int(n.sum()) if offline else budget['total']
        p1, p2 = (0, 0) if offline else (budget['C1'], budget['C2'])
        feasible = start < horizon if offline else budget['feasible']
        seed = int(np.random.SeedSequence([SEED, repeat, horizon, arms,
            round(config['multiplier'] * 100), method_id]).generate_state(1)[0])
        row = dict(dataset='MovieLens-25M', setting=VERSION, learner='UCB', method=method,
            T=horizon, K=arms, repeat=repeat, Delta_K=gap, multiplier=config['multiplier'],
            sigma=EXPLORATION_PARAMETER, exploration_parameter=EXPLORATION_PARAMETER,
            delta=DELTA, N0_i=N0 if offline else 0,
            cost='' if cost is None else cost,
            cost_definition='injected samples' if offline else 'scheduled corrupted rounds',
            T0=start, H=horizon-start,
            target_ratio='', online_ratio='', post_attack_target_ratio='', post_attack_H='',
            allocation_json=json.dumps(n.tolist()), clean_sum_json=json.dumps(clean.tolist()),
            online_counts_json='', phase1_counts_json='', phase2_counts_json='', post_attack_counts_json='',
            C1='' if p1 is None else p1, C2='' if p2 is None else p2,
            budget_log_json='' if offline else json.dumps(budget, sort_keys=True),
            status='simulated' if feasible else 'infeasible',
            infeasible_reason='' if feasible else ('allocation leaves no online horizon' if offline else budget['reason']),
            seed=seed, clean_seed=mlrunner.warm_start_seed(arms, repeat) if offline else '',
            z_star=z if offline else '', mu_minus_K=floor if offline else '',
            modifications='', magnitude='', trajectory_seconds='')
        if feasible:
            started = time.monotonic()
            online, modifications, magnitude, first, second, post = simulate(
                clean, n, arrays, horizon, seed, phase1=p1, phase2=p2, intervene=not offline)
            elapsed = time.monotonic() - started
            post_H = horizon - start if offline else horizon - p1 - p2
            assert int(online.sum()) == horizon-start and int(post.sum()) == post_H > 0
            assert int(first.sum()) == p1 and int(second.sum()) == p2
            if not offline:
                # Validate the observed phase structure; never substitute it
                # for the sequential learner's decisions or returned counts.
                assert p1 % arms == 0 and np.all(first == p1 // arms)
                assert np.all(second[:-1] == 1) and second[-1] == p2 - arms + 1
                assert np.all(post[:-1] == 0) and post[-1] == post_H
            row.update(target_ratio=((N0 if offline else 0)+int(n[-1])+int(online[-1]))/horizon,
                online_ratio=float(online[-1]/(horizon-start)),
                post_attack_target_ratio=float(post[-1]/post_H), post_attack_H=post_H,
                online_counts_json=json.dumps(online.tolist()),
                phase1_counts_json=json.dumps(first.tolist()), phase2_counts_json=json.dumps(second.tolist()),
                post_attack_counts_json=json.dumps(post.tolist()),
                modifications=int(modifications), magnitude=float(magnitude), trajectory_seconds=elapsed)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--repeats', type=int, help='Default: 50, or 2 for smoke')
    args = parser.parse_args()
    repeats = args.repeats if args.repeats is not None else (2 if args.smoke else REPEATS)
    if repeats < 2 or mlrunner.MOVIELENS_DATASET != '25m':
        raise ValueError('Use MovieLens-25M and at least two repeats')
    horizon = 100_000 if args.smoke else T
    multipliers = [0., .1, 1.] if args.smoke else GAPS
    configs = configuration_grid(horizon, K, multipliers)
    out = DATA / 'smoke' if args.smoke else DATA
    csv_path, manifest_path = out/'comparison_results.csv', out/'manifest.json'
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing results in {out}')
    sources = ['ucb_xu_comparison.py', 'test_ucb_xu_comparison.py', 'bandit.py',
               'mlrunner.py', 'paper_runner.py', 'xu_budget.py']
    hashes = {name: sha256(ROOT/name) for name in sources}
    ratings_hash, movies_hash = sha256(mlrunner.RATINGS_FILE), sha256(mlrunner.MOVIES_FILE)
    metadata = mlrunner.load_movielens(K)
    simulate(np.zeros(K), np.zeros(K, dtype=np.int64), [np.zeros(1)]*K,
             20, SEED, phase1=10, phase2=5, intervene=True)
    manifest = dict(version=VERSION, dataset='MovieLens-25M', completed=False, status='running',
        smoke=args.smoke, repeats=repeats, T=horizon, K=K, multipliers=multipliers,
        Delta_grid=[c['Delta_K'] for c in configs], configurations=configs, methods=METHODS,
        delta=DELTA, sigma=EXPLORATION_PARAMETER, exploration_parameter=EXPLORATION_PARAMETER,
        exploration_parameter_scope='Learner exploration coefficient; not a sub-Gaussian assumption on MovieLens competitors',
        learner_index='mean + sqrt(log(t)/N)', internal_time='t = total observed counts + 1, including the attacked warm start',
        tie_breaking='first arm index', initialization='Xu: online arms in index order; Ours: all arms initialized by attacked warm start',
        N0_i=N0, seed=SEED, target_information='known deterministic target reward; exact lower bound equals mu_K',
        target_feedback='deterministic Delta_K in clean and online rewards',
        xu_budget_version=xu_budget.VERSION, xu_budget_source=xu_budget.SOURCE, xu_budget_source_section='A.2',
        xu_schedule_adaptation='Original A.2 budgets applied to common anytime UCB; original fixed-horizon learner is not simulated',
        xu_stage_validation='Actual counts checked: phase1=C1/K per arm; phase2 one per non-target and C2-K+1 target; post-intervention target only',
        source_sha256=hashes, ratings_sha256=ratings_hash, movies_sha256=movies_hash,
        selected_arms=dict(movie_ids=metadata['selected_movie_ids'], means=metadata['mu'].tolist(),
                           counts=metadata['selected_counts'].tolist()),
        results_file=csv_path.name, python=platform.python_version(), numpy=np.__version__,numba=numba.__version__)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    started, count = time.monotonic(), 0
    with csv_path.open('w', newline='') as handle:
        writer = None
        for config in configs:
            for repeat in range(repeats):
                rows = run_case(metadata, repeat, config, horizon)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
                writer.writerows(rows); count += len(rows)
            handle.flush()
            print(f"Delta_K={config['Delta_K']:.6g}: {2*repeats} method records completed",flush=True)
    if hashes != {name: sha256(ROOT/name) for name in sources}:
        raise RuntimeError('Sources changed during simulation; results remain incomplete')
    manifest.update(completed=True,status='completed',rows=count,results_sha256=sha256(csv_path),
                    elapsed_seconds=time.monotonic()-started)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved {count} method records to {csv_path}',flush=True)


if __name__ == '__main__':
    main()
