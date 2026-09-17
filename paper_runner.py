"""Reproduce the six paper figures from MovieLens-25M, with real trajectories."""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from tqdm import tqdm

import bandit
import mlrunner
import xu_budget
import xu_simulation
from simulation import simulate

ROOT = Path(__file__).resolve().parent
K_GRID = [5, 10, 15, 20, 30, 40, 50]
GAPS = [0., .1, .2, .3, .4, .5, .75, 1., 1.5, 2., 3., 4., 5., 6.]
T_FIXED = 1_000_000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=None)
def allocation(learner, method, K, T, clean_means, known_target_mean=None):
    bandit.K = K
    bandit.target_arm = K
    means = np.asarray(clean_means)
    if method == 'Direct':
        fn = bandit._ucb_direct_counts_for_T if learner == 'UCB' else bandit._ts_direct_counts_for_T
        if known_target_mean is None:
            n, z, lower, epsilon, log = fn(means, T)
        else:
            from direct_search import search
            lower = bandit._target_mean_lower_bound(means, known_target_mean)
            n, z, stages = search(means, T, 5, lower, learner == 'TS')
            log = {'search': 'same grid; exact integer TS target envelope', 'z_stages': stages.tolist()}
        check = bandit._check_ucb_direct_cutoffs if learner == 'UCB' else bandit._check_ts_direct_cutoffs
    else:
        fn = bandit._ucb_counts_for_T if learner == 'UCB' else bandit._ts_counts_for_T
        n, z, lower, epsilon = fn(means, T, known_target_mean=known_target_mean)
        log = {}
        check = bandit._check_ucb_cutoffs if learner == 'UCB' else bandit._check_ts_cutoffs
    if n is None or n.sum() + 5 * K >= T:
        raise ValueError(f'Infeasible {learner}/{method} allocation at K={K}, T={T}')
    check(means, T, n, z, lower)
    return n, z, lower, json.dumps(log, sort_keys=True)


def run_case(metadata, repeat, sweep, T, gap=None, include_direct=True, baseline_only=False):
    K = len(metadata['mu'])
    clean_sum, clean_mean = mlrunner.sample_clean_warm_start(metadata, repeat, K)
    arrays = list(metadata['selected_reward_arrays'])
    st = T**(2/3) * (K * math.log(T))**(1/3)
    if gap is not None:
        delta_k = gap * st / T
        arrays[-1] = np.array([delta_k])
        clean_sum[-1] = 5 * delta_k
        clean_mean[-1] = delta_k
    else:
        delta_k = float(metadata['mu'][-1])
    # Only the lower confidence bound of the clean target enters either
    # construction. Canonicalize values giving the identical zero bound.
    design_mean = clean_mean.copy()
    bandit.K, bandit.target_arm = K, K
    if gap is None and design_mean[-1] <= 2 * bandit._beta(5):
        design_mean[-1] = 0.
    methods = ['Ours', 'Direct', 'Two-phase'] if sweep == 'gap' else ['Ours', 'Clipped Suppression']
    if not include_direct:
        methods = [m for m in methods if m != 'Direct']
    rows = []
    for learner_id, learner in enumerate(['UCB', 'TS']):
        for method_id, method in enumerate(methods):
            offline = method in {'Ours', 'Direct'}
            if baseline_only and offline:
                continue
            n = np.zeros(K, dtype=np.int64)
            z = lower = ''
            log = ''
            phase = 0
            phase_two = 0
            budget_info = {}
            budget = 0
            if method in {'Ours', 'Direct'}:
                n, z, lower, log = allocation(learner, method, K, T, tuple(design_mean),
                                                known_target_mean=delta_k if gap is not None else None)
            elif method == 'Two-phase':
                budget_info = xu_budget.budget(learner, K, T, delta_k)
                phase = budget_info['C1'] or 0
                phase_two = budget_info['C2'] or 0
                budget = math.inf if budget_info['total'] is None else budget_info['total']
            initial_per_arm = 5 if offline else 0
            initial_sum = clean_sum if offline else np.zeros(K)
            H = int(T - initial_per_arm * K - n.sum())
            feasible = budget < H
            # Stable, distinct streams for each learner, method, repeat and case.
            sim_seed = int(np.random.SeedSequence([2026, repeat, K, T, learner_id,
                            method_id, 0 if gap is None else round(gap * 100)]).generate_state(1)[0])
            row = dict(dataset=mlrunner.DATASET_LABEL, sweep=sweep, learner=learner,
                       method=method, repeat=repeat, T=T, K=K,
                       multiplier='' if gap is None else gap, Delta_K=delta_k,
                       H=H, T0=T-H, S_T=st, seed=sim_seed,
                       status='simulated' if feasible else 'infeasible',
                       cost='', cost_definition='', target_ratio='', online_ratio='',
                       target_cost=int(n[-1]), non_target_avg=float(n[:-1].mean()),
                       online_counts_json='', allocation_json=json.dumps(n.tolist()),
                       clean_sum_json=json.dumps(initial_sum.tolist()), z_star=z,
                       mu_minus_K=lower, search_log_json=log,
                       phase1_length=phase if budget_info else '',
                       phase2_length=phase_two if budget_info else '',
                       budget_log_json=json.dumps(budget_info, sort_keys=True) if budget_info else '',
                       phase1_counts_json='', phase2_counts_json='', post_attack_counts_json='',
                       post_attack_target_ratio='')
            if feasible:
                attack = 1 if method == 'Clipped Suppression' else 2 if method == 'Two-phase' else 0
                if attack == 2:
                    online, modifications, magnitude, p1, p2, post = xu_simulation.simulate(
                        arrays, T, learner, sim_seed, phase, phase_two, delta_k)
                else:
                    online, modifications, magnitude, p1, p2, post = simulate(
                        initial_sum, n, arrays, T, learner, attack, sim_seed, phase,
                        warm_start=offline, phase_two=phase_two, diagnostics=True)
                assert int(online.sum()) == H
                row.update(online_counts_json=json.dumps(online.tolist()),
                           target_ratio=(initial_per_arm + int(n[-1]) + int(online[-1])) / T,
                           online_ratio=float(online[-1] / H))
                # A suppression action costs one even when its input is zero.
                row['cost'] = int(n.sum()) if attack == 0 else int(budget) if attack == 2 else int(online[:-1].sum())
                if attack == 2:
                    assert p1.sum() == phase and p2.sum() == phase_two
                    assert post.sum() == T - budget
                    row.update(phase1_counts_json=json.dumps(p1.tolist()),
                               phase2_counts_json=json.dumps(p2.tolist()),
                               post_attack_counts_json=json.dumps(post.tolist()),
                               post_attack_target_ratio=float(post[-1] / post.sum()))
            elif math.isfinite(budget):
                row['cost'] = int(budget)
            row['cost_definition'] = ('injected samples' if method in {'Ours', 'Direct'} else
                                      'scheduled corrupted rounds' if method == 'Two-phase' else
                                      'suppressed non-target rounds (including 0->0)')
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--smoke', action='store_true', help='Small separate validation run; never used in paper figures')
    args = parser.parse_args()
    if mlrunner.MOVIELENS_DATASET != '25m':
        raise ValueError('Paper experiments require MovieLens-25M')
    if args.repeats < 2:
        raise ValueError('At least two repeats are needed for sample standard deviations')
    out = mlrunner.RESULTS_DIR / ('smoke' if args.smoke else 'paper')
    out.mkdir(parents=True, exist_ok=True)
    instances = {K: mlrunner.load_movielens(K) for K in K_GRID}
    tasks = [('horizon', T, 10, None) for T in mlrunner.T_GRID]
    tasks += [('arms', T_FIXED, K, None) for K in K_GRID]
    tasks += [('gap', T_FIXED, 10, gap) for gap in GAPS]
    if args.smoke:
        tasks = [('horizon', 100_000, 10, None), ('arms', T_FIXED, 5, None), ('gap', T_FIXED, 10, 0.)]
    # Write a partial file while running, then publish it atomically with a
    # manifest only after every repeat is complete.
    partial = out / 'paper_results.partial.csv'
    start = time.time()
    count = 0
    with partial.open('w', newline='') as f:
        writer = None
        for repeat in range(args.repeats):
            for sweep, T, K, gap in tqdm(tasks, desc=f'Repeat {repeat+1}/{args.repeats}'):
                rows = run_case(instances[K], repeat, sweep, T, gap)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=rows[0])
                    writer.writeheader()
                writer.writerows(rows)
                f.flush()
                count += len(rows)
    final = out / 'paper_results.csv'
    partial.replace(final)
    selected = {}
    for K, data in instances.items():
        selected[K] = dict(movie_ids=data['selected_movie_ids'],
                           counts=data['selected_counts'].tolist(), means=data['mu'].tolist(),
                           titles=[data['movie_titles'].get(i, '') for i in data['selected_movie_ids']])
    manifest = dict(dataset=mlrunner.DATASET_LABEL, repeats=args.repeats, rows=count,
                    T_grid=mlrunner.T_GRID, K_grid=K_GRID, gaps=GAPS, T_fixed=T_FIXED,
                    sigma=.5, delta=.05, N0_i=5, seed=2026, smoke=args.smoke,
                    gap_target_information='known deterministic target reward; exact lower bound equals mu_K',
                    xu_budget_version=xu_budget.VERSION,
                    xu_settings='original UCB (log T, coefficient 1) and Beta-Bernoulli TS; budgets from original A.2/A.4',
                    runtime_seconds=time.time()-start, python=platform.python_version(),
                    ratings_sha256=sha256(mlrunner.RATINGS_FILE),
                    movies_sha256=sha256(mlrunner.MOVIES_FILE),
                    results_sha256=sha256(final), selected_arms=selected,
                    simulation='sequential UCB and Gaussian TS; PCG64; Numba; no certificate counts',
                    baseline_initialization='no offline log; Clipped: target then non-targets; Xu UCB: index order; Xu TS: Beta(1,1) from round 1; attacks active from round 1',
                    clipped_suppression_feedback='all non-target rewards replaced by zero; targets unchanged',
                    source_sha256={p.name:sha256(p) for p in [ROOT/'bandit.py', ROOT/'mlrunner.py', ROOT/'simulation.py', ROOT/'direct_search.py', ROOT/'xu_budget.py', ROOT/'xu_simulation.py', Path(__file__)]})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Wrote {count} rows to {final}', flush=True)


if __name__ == '__main__':
    main()
