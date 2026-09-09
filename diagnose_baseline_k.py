"""Independently replay K=30/40/50 baseline trajectories as zero suppression.

Check exact saved counts/costs and, for UCB, that its requested threshold
stays negative so zero suppression is equivalent on these trajectories.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

import mlrunner


@njit(cache=True)
def replay(clean, rewards, offsets, T, ts, rng):
    K = len(clean)
    counts = np.zeros(K, dtype=np.int64)
    sums = clean.copy()
    positives = np.zeros(K, dtype=np.int64)
    max_threshold = -math.inf
    for t in range(1, T + 1):
        if t <= K:
            arm = K - 1 if t == 1 else t - 2
        else:
            indices = sums / counts
            for i in range(K):
                if ts:
                    indices[i] += rng.normal() / math.sqrt(counts[i])
                else:
                    indices[i] += 1.5 * math.sqrt(math.log(t) / counts[i])
            arm = np.argmax(indices)
        raw = rewards[rng.integers(offsets[arm], offsets[arm + 1])]
        positives[arm] += int(raw)
        if arm == K - 1:
            sums[arm] += raw
        elif not ts:
            nk = counts[-1]
            beta = math.sqrt(.5 / nk * math.log(math.pi**2 * K * nk**2 / .15))
            threshold = sums[-1] / nk - 2 * beta - .01
            max_threshold = max(max_threshold, threshold)
        counts[arm] += 1
    return counts, positives, max_threshold


def main():
    out = Path(__file__).resolve().parent / 'results/ml-25m/paper'
    path = out / 'paper_results.csv'
    manifest = json.loads((out / 'manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['results_sha256']
    rows = [r for r in csv.DictReader(path.open()) if r['sweep'] == 'arms'
            and r['method'] == 'Clipped Suppression' and int(r['K']) in (30, 40, 50)]
    details = []
    for K in (30, 40, 50):
        data = mlrunner.load_movielens(K)
        assert data['selected_movie_ids'] == manifest['selected_arms'][str(K)]['movie_ids']
        arrays = data['selected_reward_arrays']
        rewards = np.concatenate(arrays)
        offsets = np.r_[0, np.cumsum([len(a) for a in arrays])]
        for row in [r for r in rows if int(r['K']) == K]:
            assert int(row['T0']) == 0 and not any(json.loads(row['clean_sum_json']))
            counts, positives, threshold = replay(
                np.array(json.loads(row['clean_sum_json'])), rewards, offsets,
                int(row['T']), row['learner'] == 'TS', np.random.default_rng(int(row['seed'])))
            np.testing.assert_array_equal(counts, json.loads(row['online_counts_json']))
            assert counts[:-1].sum() == int(row['cost'])
            if row['learner'] == 'UCB':
                assert threshold < 0
            details.append(dict(learner=row['learner'], K=K, repeat=int(row['repeat']),
                cost=int(counts[:-1].sum()), changed_observations=int(positives[:-1].sum()),
                non_target_pulls=int(counts[:-1].sum()),
                target_pulls=int(counts[-1]),
                non_target_weighted_mu=float(counts[:-1] @ data['mu'][:-1] / counts[:-1].sum()),
                ucb_max_threshold=float(threshold) if row['learner'] == 'UCB' else None))
        print(f'K={K}: all 20 trajectories and costs match exactly', flush=True)
    summary = []
    for learner in ('UCB', 'TS'):
        for K in (30, 40, 50):
            ds = [r for r in details if r['learner'] == learner and r['K'] == K]
            summary.append(dict(learner=learner, K=K,
                cost_mean=float(np.mean([r['cost'] for r in ds])),
                cost_sd=float(np.std([r['cost'] for r in ds], ddof=1)),
                non_target_pulls_mean=float(np.mean([r['non_target_pulls'] for r in ds])),
                weighted_mu_mean=float(np.mean([r['non_target_weighted_mu'] for r in ds]))))
    result = dict(results_sha256=manifest['results_sha256'],
        verified_trajectories=len(details), summary=summary, repeats=details,
        explanation='Cost counts every suppressed non-target round, including 0->0. '
                    'All 60 saved costs equal non-target pulls exactly; '
                    'changed observations are retained as a separate diagnostic.')
    (out / 'baseline_k_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
