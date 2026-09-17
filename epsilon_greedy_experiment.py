"""Sequential epsilon-greedy trajectories with a fully synthetic warm start.

The allocation uses only the horizon, arm count, exploration schedule and
confidence level. True means are used only by the Bernoulli environment.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numba
import numpy as np
import scipy
from numba import njit
from scipy.special import digamma


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/synthetic/epsilon-greedy'
VERSION = 'epsilon-greedy-boundary-v1'
DATASET = 'Controlled Bernoulli'
T_GRID = [100_000, 250_000, 1_000_000, 2_500_000, 10_000_000]
K_GRID = [5, 10, 15, 20, 25]
SEED = 20260910


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_exploration_per_arm(T, K, c=1.0):
    """Sum min(1,cK/s)/K over s=1,...,T without allocating T entries."""
    if not isinstance(T, (int, np.integer)) or T < 1:
        raise ValueError('T must be a positive integer')
    if not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError('K must be an integer at least two')
    if not math.isfinite(c) or c < 0:
        raise ValueError('c must be finite and nonnegative')
    saturated = min(T, math.floor(c * K))
    return float((saturated + c * K *
                  (digamma(T + 1) - digamma(saturated + 1))) / K)


def ceil_sqrt_ratio(numerator, denominator=1):
    """Round sqrt(numerator/denominator) upward using exact integer checks."""
    root = math.isqrt(numerator // denominator)
    return root + int(root * root * denominator < numerator)


def design(T, K, c=1.0, delta=0.05):
    """The paper's count allocation; no environment means enter this function."""
    mean = expected_exploration_per_arm(T, K, c)
    if not math.isfinite(delta) or not 0 < delta < 0.5:
        raise ValueError('delta must lie in (0, 1/2)')
    log_failure = math.log((K - 1) / delta)
    B = math.ceil(mean + math.sqrt(2 * mean * log_failure) + log_failure / 3)
    allocation = np.full(K, ceil_sqrt_ratio(T * B, K - 1), dtype=np.int64)
    allocation[-1] = ceil_sqrt_ratio(T * (K - 1) * B) + 1
    cost = int(allocation.sum())
    if cost >= T:
        raise ValueError(f'Allocation leaves no online horizon: C={cost}, T={T}')
    return dict(B=B, allocation=allocation, cost=cost,
                expected_exploration_per_arm=mean,
                r_eg=math.sqrt((K - 1) * B / T))


def bernoulli_means(T, K, r_eg):
    """A positive target approaching the epsilon-greedy boundary."""
    means = np.empty(K)
    means[:-1] = np.linspace(0.2, 0.8, K - 1)
    means[-1] = r_eg / math.log(T)
    if not 0 < means[-1] < 1:
        raise ValueError('The controlled target mean must lie in (0, 1)')
    return means


@njit(cache=True)
def trajectory(allocation, means, T, c, rng):
    """Make every exploration/greedy decision from current empirical means."""
    K = len(allocation)
    counts = allocation.copy()
    totals = np.zeros(K)
    totals[-1] = allocation[-1]  # All injected target rewards equal one.
    online = np.zeros(K, dtype=np.int64)
    exploratory = np.zeros(K, dtype=np.int64)
    greedy_non_target = 0
    T0 = int(allocation.sum())
    for s in range(T0 + 1, T + 1):
        if rng.random() < min(1.0, c * K / s):
            arm = rng.integers(0, K)
            exploratory[arm] += 1
        else:
            best, arm = -1.0, 0
            for i in range(K):
                value = totals[i] / counts[i]
                if value > best:  # First maximizer, matching numpy.argmax.
                    best, arm = value, i
            greedy_non_target += int(arm != K - 1)
        reward = 1.0 if rng.random() < means[arm] else 0.0
        totals[arm] += reward
        counts[arm] += 1
        online[arm] += 1
    return online, exploratory, greedy_non_target


def simulate(allocation, means, T, c, seed):
    allocation = np.asarray(allocation)
    means = np.asarray(means, dtype=float)
    if (allocation.ndim != 1 or len(allocation) < 2 or
            means.shape != allocation.shape):
        raise ValueError('allocation and means must be equally sized arm vectors')
    if (not np.issubdtype(allocation.dtype, np.integer) or
            np.any(allocation <= 0)):
        raise ValueError('Every arm needs a positive integer allocation')
    if not np.all(np.isfinite(means)) or np.any((means < 0) | (means > 1)):
        raise ValueError('Bernoulli means must lie in [0, 1]')
    if not isinstance(T, (int, np.integer)) or int(allocation.sum()) >= T:
        raise ValueError('T must leave a nonempty online horizon')
    if not math.isfinite(c) or c < 0:
        raise ValueError('c must be finite and nonnegative')
    return trajectory(allocation.astype(np.int64), means, T, c,
                      np.random.default_rng(seed))


def run_case(T, K, repeat, c=1.0, delta=0.05, seed=SEED):
    attack = design(T, K, c, delta)
    n, B, cost = attack['allocation'], attack['B'], attack['cost']
    means = bernoulli_means(T, K, attack['r_eg'])
    trajectory_seed = int(np.random.SeedSequence(
        [seed, T, K, repeat]).generate_state(1)[0])
    start = time.monotonic()
    online, exploratory, greedy_non_target = simulate(
        n, means, T, c, trajectory_seed)
    runtime = time.monotonic() - start
    H = T - cost
    non_target = int(online[:-1].sum())
    assert int(online.sum()) == H > 0
    assert non_target == int(exploratory[:-1].sum()) + greedy_non_target
    return dict(
        dataset=DATASET, setting=VERSION, learner='epsilon-greedy', method='Ours',
        T=T, K=K, repeat=repeat, c=c, delta=delta,
        target_mean=float(means[-1]), r_eg=attack['r_eg'],
        boundary_ratio=float(means[-1] / attack['r_eg']), B=B,
        expected_exploration_per_arm=attack['expected_exploration_per_arm'],
        expected_non_target_exploratory_pulls=(K - 1) * (
            attack['expected_exploration_per_arm'] -
            expected_exploration_per_arm(cost, K, c)),
        cost=cost, C=cost, nK=int(n[-1]), target_cost=int(n[-1]),
        non_target_cost=int(n[:-1].sum()),
        target_budget_fraction=float(n[-1] / cost),
        non_target_budget_fraction=float(n[:-1].sum() / cost),
        normalized_cost=cost / (2 * math.sqrt(T * (K - 1) * B)),
        cost_over_sqrt_TKlogT=cost / math.sqrt(T * K * math.log(T)),
        H=H, T0=cost, xi=(K - 1) * B,
        non_target_online_pulls=non_target,
        greedy_non_target_pulls=int(greedy_non_target),
        exploratory_non_target_pulls=int(exploratory[:-1].sum()),
        success=int(non_target <= (K - 1) * B),
        individual_exploration_bound=int(np.all(exploratory[:-1] <= B)),
        all_greedy_target=int(greedy_non_target == 0),
        online_ratio=float(online[-1] / H),
        allocation_json=json.dumps(n.tolist()),
        online_counts_json=json.dumps(online.tolist()),
        exploratory_counts_json=json.dumps(exploratory.tolist()),
        means_json=json.dumps(means.tolist()), seed=trajectory_seed,
        trajectory_seconds=runtime, status='simulated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true',
                        help='Use a separate smoke directory and two small horizons')
    parser.add_argument('--repeats', type=int, default=None,
                        help='Replicates per configuration (default: 50, or 2 for smoke)')
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 2 if args.smoke else 50
    if args.repeats < 1:
        raise ValueError('At least one repeat is required')
    t_grid = [10_000, 25_000] if args.smoke else T_GRID
    k_grid = [5, 10] if args.smoke else K_GRID
    out = DATA / 'smoke' if args.smoke else DATA
    results_path, manifest_path = out / 'epsilon_greedy_results.csv', out / 'manifest.json'
    if results_path.exists() or manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite experiment artifacts in {out}')
    # Compile before measuring trajectories or creating incomplete artifacts.
    simulate(np.array([2, 3]), np.array([0.4, 0.02]), 12, 1.0, 1)
    for K in k_grid:
        for T in t_grid:
            design(T, K)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    manifest = dict(
        version=VERSION, dataset=DATASET, completed=False, status='running',
        smoke=args.smoke, T_grid=t_grid, K_grid=k_grid, repeats=args.repeats,
        configurations=[dict(T=T, K=K) for K in k_grid for T in t_grid],
        seed=SEED, c=1.0, delta=0.05, N0_i=0,
        results_file=results_path.name,
        non_target_means='numpy.linspace(0.2, 0.8, K-1)',
        target_mean='sqrt((K-1)*B/T)/log(T); positive Bernoulli target',
        schedule='epsilon_s=min(1,c*K/s), s=T0+1,...,T; T0=C',
        allocation='nK=ceil(sqrt(T*(K-1)*B))+1; ni=ceil(sqrt(T*B/(K-1)))',
        B='ceil(sum_{s=1}^T epsilon_s/K + sqrt(2*sum*log((K-1)/delta)) + log((K-1)/delta)/3)',
        rewards='Independent Bernoulli rewards; all injected target rewards 1, others 0',
        simulation='Actual sequential decisions; NumPy PCG64; first-max greedy ties',
        confidence='The theorem guarantees probability at least 1-2*delta',
        source_sha256={name: sha256(ROOT / name) for name in
                       ['epsilon_greedy_experiment.py', 'test_epsilon_greedy_experiment.py']},
        python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    rows = 0
    with results_path.open('w', newline='') as handle:
        writer = None
        for K in k_grid:
            for T in t_grid:
                for repeat in range(args.repeats):
                    row = run_case(T, K, repeat)
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    rows += 1
                handle.flush()
                print(f'K={K}, T={T}: {args.repeats} trajectories completed', flush=True)
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
