"""Controlled near-boundary trajectories for the paper's cost/allocation limits.

MovieLens competitors are unchanged. Only the target is resampled as Bernoulli
with mean mu_ML * sqrt(100000 / T), in both the clean log and online phase.
The attacker sees the clean log, never this mean. Original experiments are kept.
"""
import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

import bandit
import mlrunner
from paper_runner import allocation, sha256
from simulation import simulate

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/ml-25m/theory-alignment-fixed-k'
LEGACY_DATA = ROOT / 'results/ml-25m/theory-alignment'
VERSION = 'near-boundary-fixed-k-v2'
T_GRID = [100_000, 250_000, 1_000_000, 2_500_000, 10_000_000]
K_GRID = [5, 10, 15, 20, 25]
SEED = 20260908


def configurations(grid):
    return [('fixed', T, K) for K in K_GRID for T in grid]


def reusable_results(source_hashes, ratings_hash, movies_hash):
    """Reuse the unchanged K=10 trajectories only after full validation."""
    if not (LEGACY_DATA / 'manifest.json').exists():
        return {}, None
    from generate_theory_figures import read_results
    rows, manifest = read_results(LEGACY_DATA, allow_legacy=True)
    for name in ['bandit.py', 'mlrunner.py', 'paper_runner.py', 'simulation.py']:
        if manifest['source_sha256'][name] != source_hashes[name]:
            raise ValueError(f'Cannot reuse trajectories after changing {name}; use --no-reuse')
    if (manifest['ratings_sha256'] != ratings_hash or
            manifest['movies_sha256'] != movies_hash or manifest['seed'] != SEED):
        raise ValueError('Cannot reuse trajectories with changed data or seeds; use --no-reuse')
    cache = {}
    for row in rows:
        if row['regime'] == 'fixed' and int(row['K']) == 10:
            key = (int(row['T']), 10, int(row['repeat']))
            cache.setdefault(key, []).append(dict(row, setting=VERSION))
    provenance = dict(directory=str(LEGACY_DATA.relative_to(ROOT)),
                      manifest_sha256=sha256(LEGACY_DATA / 'manifest.json'),
                      results_sha256=manifest['results_sha256'],
                      source_sha256=manifest['source_sha256'])
    return cache, provenance


def target_mean(base_mean, T):
    return base_mean * math.sqrt(T_GRID[0] / T)


def cost_scale(learner, K, T):
    coefficient = (9 * .5**2 * math.log(T) if learner == 'UCB' else
                   2 * math.log(math.pi**2 * K * T**2 / (3 * .05)))
    return 3 * ((K - 1) * coefficient * T**2 / 4)**(1/3)


def run_case(metadata, repeat, regime, T):
    K = len(metadata['mu'])
    probability = target_mean(float(metadata['mu'][-1]), T)
    clean_sum, clean_mean = mlrunner.sample_clean_warm_start(metadata, repeat, K)
    # Shared uniforms across T give matched target logs without conditioning on
    # unobserved outcomes; each configuration still has five i.i.d. rewards.
    rng = np.random.default_rng(np.random.SeedSequence([SEED, repeat, K, 17]))
    clean_sum[-1] = np.count_nonzero(rng.random(5) < probability)
    clean_mean[-1] = clean_sum[-1] / 5
    design_mean = clean_mean.copy()
    bandit.K, bandit.target_arm = K, K
    bandit.sigma, bandit.delta, bandit.N0_i = .5, .05, 5
    bandit.r_l, bandit.r_u, bandit.R = 0., 1., 1.
    # This cache canonicalization leaves the default confidence floor unchanged.
    if design_mean[-1] <= 2 * bandit._beta(5):
        design_mean[-1] = 0.
    rows = []
    for learner_id, learner in enumerate(['UCB', 'TS']):
        n, z, lower, _ = allocation(learner, 'Ours', K, T, tuple(design_mean))
        assert lower == bandit._target_mean_lower_bound(clean_mean)
        H = T - 5*K - int(n.sum())
        sim_seed = int(np.random.SeedSequence(
            [SEED, repeat, K, T, learner_id, 101]).generate_state(1)[0])
        start = time.monotonic()
        counts, _, _ = simulate(clean_sum, n, metadata['selected_reward_arrays'],
                                 T, learner, 0, sim_seed, target_mean=probability)
        assert int(counts.sum()) == H > 0
        scale = cost_scale(learner, K, T)
        st = T**(2/3)*(K*math.log(T))**(1/3)
        rows.append(dict(
            dataset='MovieLens-25M', setting=VERSION, regime=regime,
            learner=learner, method='Ours', repeat=repeat, T=T, K=K,
            target_mean=probability, boundary_ratio=probability/(st/T),
            base_target_mean=float(metadata['mu'][-1]),
            target_feedback='Bernoulli; same mean in clean and online rewards',
            lower_bound_source='clean log: max(0, empirical mean - 2 beta)',
            mu_minus_K=lower, z_star=z, cost=int(n.sum()),
            Lambda_T=scale, normalized_cost=float(n.sum()/scale),
            target_budget_fraction=float(n[-1]/n.sum()),
            target_cost=int(n[-1]), non_target_avg=float(n[:-1].mean()),
            H=H, T0=T-H, non_target_online_pulls=int(counts[:-1].sum()),
            zero_non_target_pulls=int(counts[:-1].sum() == 0),
            online_ratio=float(counts[-1]/H), status='simulated', seed=sim_seed,
            clean_sum_json=json.dumps(clean_sum.tolist()),
            allocation_json=json.dumps(n.tolist()),
            online_counts_json=json.dumps(counts.tolist()),
            trajectory_seconds=time.monotonic()-start))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--no-reuse', action='store_true',
                        help='Rerun K=10 instead of reusing verified legacy trajectories')
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError('At least two repeats are required for sample SD')
    if mlrunner.MOVIELENS_DATASET != '25m':
        raise ValueError('This pipeline requires MovieLens-25M')
    grid = T_GRID[:1] if args.smoke else T_GRID
    out = DATA / 'smoke' if args.smoke else DATA
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT/'bandit.py', ROOT/'mlrunner.py',
               ROOT/'paper_runner.py', ROOT/'simulation.py', ROOT/'generate_theory_figures.py']
    source_hashes = {p.name: sha256(p) for p in sources}
    start = time.monotonic()
    config_grid = configurations(grid)
    ratings_hash, movies_hash = sha256(mlrunner.RATINGS_FILE), sha256(mlrunner.MOVIES_FILE)
    reuse, reuse_provenance = ({}, None) if args.no_reuse else reusable_results(
        source_hashes, ratings_hash, movies_hash)
    metadata = {K: mlrunner.load_movielens(K) for K in K_GRID}
    partial = out/'theory_results.partial.csv'
    all_rows = []
    reused_rows = 0
    with partial.open('w', newline='') as f:
        writer = None
        for repeat in range(args.repeats):
            for regime, T, K in config_grid:
                if (T, K, repeat) in reuse:
                    rows = reuse[T, K, repeat]
                    reused_rows += len(rows)
                    origin = 'reused'
                else:
                    rows = run_case(metadata[K], repeat, regime, T)
                    origin = 'simulated'
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader()
                writer.writerows(rows); f.flush(); all_rows.extend(rows)
                print(f'repeat {repeat+1}/{args.repeats}, {origin}, T={T}, K={K}: '
                      + ', '.join(f"{r['learner']} C/Lambda={float(r['normalized_cost']):.4f}, "
                                  f"target share={float(r['target_budget_fraction']):.4f}, "
                                  f"non-target pulls={r['non_target_online_pulls']}"
                                  for r in rows), flush=True)
    if {p.name: sha256(p) for p in sources} != source_hashes:
        raise RuntimeError('Sources changed during the run; results remain partial')
    final = out/'theory_results.csv'; partial.replace(final)
    manifest = dict(version=VERSION, dataset='MovieLens-25M', repeats=args.repeats,
                    smoke=args.smoke, rows=len(all_rows), T_grid=grid,
                    K_grid=K_GRID,
                    configurations=[dict(regime=r,T=T,K=K) for r,T,K in config_grid],
                    sigma=.5, delta=.05, N0_i=5, seed=SEED,
                    target_mean_rule='mu_ML * sqrt(100000 / T)',
                    arm_count_rule='fixed K in K_grid',
                    target_information='unknown mean; clean-log confidence floor only',
                    allocation='continuous threshold minimizer; ceil and one extra target sample',
                    simulation='actual sequential UCB and Gaussian TS, Bernoulli target',
                    selected_arms={str(K):dict(movie_ids=m['selected_movie_ids'],
                        means=m['mu'].tolist(), counts=m['selected_counts'].tolist())
                        for K,m in metadata.items()},
                    source_sha256=source_hashes, results_sha256=sha256(final),
                    ratings_sha256=ratings_hash, movies_sha256=movies_hash,
                    reused_trajectories=reused_rows,
                    simulated_trajectories=len(all_rows)-reused_rows,
                    reuse_provenance=reuse_provenance,
                    python=platform.python_version(), runtime_seconds=time.monotonic()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Wrote {len(all_rows)} rows to {final}',flush=True)


if __name__ == '__main__':
    main()
