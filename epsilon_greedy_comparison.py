"""Compare epsilon-greedy injection with online clipped suppression.

The shared horizon/arm configuration is simulated once per method and repeat.
The original boundary experiment and its recorded trajectories remain unchanged.
"""
import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numba
import numpy as np
from numba import njit

from epsilon_greedy_experiment import (
    DATASET, SEED, bernoulli_means, design, sha256, simulate)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/synthetic/epsilon-greedy-comparison'
VERSION = 'epsilon-greedy-comparison-v1'
HORIZON_GRID = list(range(100_000, 1_000_001, 100_000))
ARM_GRID = [5, 10, 15, 20, 30, 40, 50]
HORIZON_K = 10
ARM_T = 1_000_000
METHODS = ['Ours', 'Clipped Suppression']


def configurations(horizon_grid=HORIZON_GRID, arm_grid=ARM_GRID,
                   horizon_K=HORIZON_K, arm_T=ARM_T):
    return list(dict.fromkeys([(T, horizon_K) for T in horizon_grid] +
                             [(arm_T, K) for K in arm_grid]))


@njit(cache=True)
def diagnostic_trajectory(allocation, means, T, c, suppress, rng):
    """Exact sequential learner, recording genuine and learner-visible feedback."""
    K = len(means)
    counts = allocation.copy()
    totals = np.zeros(K)
    totals[-1] = allocation[-1]
    online = np.zeros(K, dtype=np.int64)
    exploratory = np.zeros(K, dtype=np.int64)
    initialization = np.zeros(K, dtype=np.int64)
    raw_sums = np.zeros(K)
    observed_sums = np.zeros(K)
    greedy_non_target = 0
    modified_rewards = 0
    for s in range(int(allocation.sum()) + 1, T + 1):
        missing = -1
        if counts[-1] == 0:
            missing = K - 1
        else:
            for i in range(K):
                if counts[i] == 0:
                    missing = i
                    break
        if missing >= 0:
            arm = missing
            initialization[arm] += 1
        elif rng.random() < min(1.0, c * K / s):
            arm = rng.integers(0, K)
            exploratory[arm] += 1
        else:
            best, arm = -1.0, 0
            for i in range(K):
                value = totals[i] / counts[i]
                if value > best:
                    best, arm = value, i
            greedy_non_target += int(arm != K - 1)
        raw = 1.0 if rng.random() < means[arm] else 0.0
        observed = 0.0 if suppress and arm != K - 1 else raw
        modified_rewards += int(raw != observed)
        raw_sums[arm] += raw
        observed_sums[arm] += observed
        totals[arm] += observed
        counts[arm] += 1
        online[arm] += 1
    return (online, exploratory, greedy_non_target, initialization,
            raw_sums, observed_sums, modified_rewards)


def simulate_diagnostics(allocation, means, T, c, seed, suppress=False):
    allocation = np.asarray(allocation)
    means = np.asarray(means, dtype=float)
    if (allocation.ndim != 1 or len(allocation) < 2 or
            means.shape != allocation.shape):
        raise ValueError('allocation and means must be equally sized arm vectors')
    if (not np.issubdtype(allocation.dtype, np.integer) or
            np.any(allocation < 0)):
        raise ValueError('allocation must contain nonnegative integers')
    if not np.all(np.isfinite(means)) or np.any((means < 0) | (means > 1)):
        raise ValueError('Bernoulli means must lie in [0, 1]')
    if not isinstance(T, (int, np.integer)) or int(allocation.sum()) >= T:
        raise ValueError('T must leave a nonempty online horizon')
    if not math.isfinite(c) or c < 0:
        raise ValueError('c must be finite and nonnegative')
    return diagnostic_trajectory(allocation.astype(np.int64), means, T, c,
                                 suppress, np.random.default_rng(seed))


def run_case(T, K, repeat, method, c=1.0, delta=0.05, seed=SEED):
    if method not in METHODS:
        raise ValueError('Unknown comparison method')
    attack = design(T, K, c, delta)
    means = bernoulli_means(T, K, attack['r_eg'])
    offline = method == 'Ours'
    n = attack['allocation'] if offline else np.zeros(K, dtype=np.int64)
    trajectory_seed = int(np.random.SeedSequence(
        [seed, T, K, repeat, METHODS.index(method), 701]).generate_state(1)[0])
    start = time.monotonic()
    if offline:
        online, exploratory, greedy_non_target = simulate(n, means, T, c, trajectory_seed)
        initialization = np.zeros(K, dtype=np.int64)
        raw = observed = None  # The frozen simulator does not return feedback sums.
        modified = 0
    else:
        result = simulate_diagnostics(n, means, T, c, trajectory_seed, suppress=True)
        online, exploratory, greedy_non_target, initialization, raw, observed, modified = result
    runtime = time.monotonic() - start
    T0 = int(n.sum())
    H = T - T0
    non_target = int(online[:-1].sum())
    assert int(online.sum()) == H > 0
    assert non_target == (int(exploratory[:-1].sum()) +
                          int(initialization[:-1].sum()) + greedy_non_target)
    cost = T0 if offline else non_target
    return dict(
        dataset=DATASET, setting=VERSION, learner='epsilon-greedy', method=method,
        T=T, K=K, repeat=repeat, c=c, delta=delta,
        target_mean=float(means[-1]), r_eg=attack['r_eg'],
        boundary_ratio=float(means[-1] / attack['r_eg']), B=attack['B'],
        cost=cost, C=cost, T0=T0, H=H, nK=int(n[-1]),
        total_target_ratio=float((int(n[-1]) + int(online[-1])) / T),
        online_ratio=float(online[-1] / H),
        non_target_online_pulls=non_target,
        greedy_non_target_pulls=int(greedy_non_target),
        exploratory_non_target_pulls=int(exploratory[:-1].sum()),
        initialization_non_target_pulls=int(initialization[:-1].sum()),
        modified_reward_count=int(modified),
        target_reward_sum='' if raw is None else float(raw[-1]),
        cost_definition=('injected samples' if offline else
                         'suppressed non-target rounds (including 0->0)'),
        allocation_json=json.dumps(n.tolist()),
        online_counts_json=json.dumps(online.tolist()),
        exploratory_counts_json=json.dumps(exploratory.tolist()),
        initialization_counts_json=json.dumps(initialization.tolist()),
        reward_sums_json='' if raw is None else json.dumps(raw.tolist()),
        observed_reward_sums_json='' if observed is None else json.dumps(observed.tolist()),
        means_json=json.dumps(means.tolist()), seed=trajectory_seed,
        trajectory_seconds=runtime, status='simulated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--repeats', type=int,
                        help='Default: fifty per configuration, or two for smoke')
    args = parser.parse_args()
    repeats = args.repeats if args.repeats is not None else (2 if args.smoke else 50)
    if repeats < 1:
        raise ValueError('At least one repeat is required')
    horizons = [10_000, 20_000] if args.smoke else HORIZON_GRID
    arms = [5, 10] if args.smoke else ARM_GRID
    arm_T = horizons[-1] if args.smoke else ARM_T
    cases = configurations(horizons, arms, HORIZON_K, arm_T)
    out = DATA / 'smoke' if args.smoke else DATA
    results_path, manifest_path = out / 'comparison_results.csv', out / 'manifest.json'
    if results_path.exists() or manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite experiment artifacts in {out}')
    for T, K in cases:
        design(T, K)
    for method in METHODS:
        run_case(10_000, 5, 0, method)
    sources = ['epsilon_greedy_experiment.py', 'test_epsilon_greedy_experiment.py',
               'epsilon_greedy_comparison.py', 'test_epsilon_greedy_comparison.py']
    manifest = dict(
        version=VERSION, dataset=DATASET, smoke=args.smoke, completed=False,
        status='running', horizon_grid=horizons, arm_grid=arms,
        horizon_K=HORIZON_K, arm_T=arm_T, repeats=repeats,
        configurations=[dict(T=T, K=K) for T, K in cases],
        methods=METHODS, seed=SEED, c=1.0, delta=0.05, N0_i=0,
        results_file=results_path.name,
        non_target_means='numpy.linspace(0.2, 0.8, K-1)',
        target_mean='sqrt((K-1)*B/T)/log(T); same means for both methods',
        baseline='No offline log; initialize target first then other arms in index order; replace each non-target reward with zero',
        baseline_cost='All non-target online pulls, including initialization and zero-to-zero replacements',
        initialization='Online actions, neither exploratory nor greedy; baseline T0=0,H=T',
        feedback_diagnostics='Baseline only: genuine online reward_sums and observed_reward_sums after suppression. Ours leaves these and target_reward_sum blank because the frozen simulator returns counts only',
        tie_breaking='First index maximizing the empirical mean; target has the last index',
        shared_configuration='Each (T,K,repeat,method) stored once and reused across figure panels',
        source_sha256={name: sha256(ROOT / name) for name in sources},
        python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, numba=numba.__version__)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    start = time.monotonic()
    rows = 0
    with results_path.open('w', newline='') as handle:
        writer = None
        for T, K in cases:
            for repeat in range(repeats):
                for method in METHODS:
                    row = run_case(T, K, repeat, method)
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    rows += 1
            handle.flush()
            print(f'T={T}, K={K}: {2 * repeats} trajectories completed', flush=True)
    if any(sha256(ROOT / name) != digest
           for name, digest in manifest['source_sha256'].items()):
        raise RuntimeError('Experiment sources changed during simulation; results remain incomplete')
    manifest.update(completed=True, status='completed', rows=rows,
                    results_sha256=sha256(results_path),
                    elapsed_seconds=time.monotonic() - start)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {rows} trajectories to {results_path}', flush=True)


if __name__ == '__main__':
    main()
