import json
import math
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import bandit
import generate_paper_figures
from simulation import simulate


class ExperimentTests(unittest.TestCase):
    def test_zero_to_zero_suppression_rounds_have_cost(self):
        import paper_runner
        metadata = dict(mu=np.zeros(2), selected_reward_arrays=[np.zeros(3), np.zeros(3)])
        with patch.object(paper_runner, 'allocation', return_value=(np.zeros(2, dtype=int), 0., 0., '')):
            rows = paper_runner.run_case(metadata, 0, 'arms', 120)
        for row in rows:
            if row['method'] == 'Clipped Suppression':
                counts = json.loads(row['online_counts_json'])
                self.assertGreater(row['cost'], 0)
                self.assertEqual(row['cost'], sum(counts[:-1]))
                self.assertEqual(row['cost_definition'], 'suppressed non-target rounds (including 0->0)')
                self.assertEqual(row['T0'], 0)
                self.assertEqual(row['H'], 120)
                self.assertEqual(sum(counts), 120)
                self.assertEqual(row['target_ratio'], counts[-1]/120)

    def test_online_initialization_and_attack_clock(self):
        arrays = [np.array([0., 1., 1.]), np.array([0., 0., 1.])]
        for learner in ['UCB', 'TS']:
            for attack in [1, 2]:
                for T in [1, 2, 3, 120]:
                    actual, changes, magnitude, p1, p2, post = simulate(np.zeros(2), np.zeros(2, dtype=int),
                        arrays, T, learner, attack, 42, phase=3, phase_two=5,
                        warm_start=False, diagnostics=True)
                    counts, sums = np.zeros(2, dtype=int), np.zeros(2)
                    rng = np.random.default_rng(42)
                    modifications = 0
                    total = 0.
                    expected_phases = np.zeros((3, 2), dtype=int)
                    for t in range(1, T+1):
                        if t <= 2:
                            arm = [1, 0][t-1]
                        else:
                            means = sums/counts
                            index = (means + 1.5*np.sqrt(math.log(t)/counts) if learner == 'UCB'
                                     else means + np.array([rng.normal(), rng.normal()])/np.sqrt(counts))
                            arm = int(np.argmax(index))
                        raw = arrays[arm][rng.integers(0, len(arrays[arm]))]
                        reward = raw
                        if attack == 1 and arm == 0:
                            if learner == 'TS':
                                reward = 0.
                            else:
                                nk = counts[-1]
                                beta = math.sqrt(.5/nk*math.log(math.pi**2*2*nk**2/.15))
                                threshold = sums[-1]/nk - 2*beta - .01
                                reward = max(0., raw-max(0., sums[arm]+raw-threshold*(counts[arm]+1)))
                        elif attack == 2 and t <= 8:
                            reward = float(t > 3 and arm == 1)
                        if attack == 2:
                            expected_phases[0 if t <= 3 else 1 if t <= 8 else 2, arm] += 1
                        modifications += reward != raw
                        total += abs(reward-raw)
                        counts[arm] += 1
                        sums[arm] += reward
                    np.testing.assert_array_equal(actual, counts)
                    self.assertEqual(changes, modifications)
                    self.assertAlmostEqual(magnitude, total)
                    np.testing.assert_array_equal(np.array([p1, p2, post]), expected_phases)
        with self.assertRaises(ValueError):
            simulate(np.ones(2), np.zeros(2, dtype=int), arrays, 120, 'UCB', 1, 42, warm_start=False)

    def test_xu_budget_uses_full_online_horizon(self):
        import paper_runner
        # Budget 74 leaves six online rounds at T=80. A fictitious five-per-arm
        # warm start would incorrectly reject it against H=70.
        T, K = 80, 2
        st = T**(2/3)*(K*math.log(T))**(1/3)
        gap = math.sqrt(K*math.log(T)/36.5)*T/st
        metadata = dict(mu=np.zeros(K), selected_reward_arrays=[np.zeros(3)]*K)
        # Isolate the horizon check from the sufficient-budget derivation.
        with patch.object(paper_runner.xu_budget, 'budget', return_value=dict(C1=40, C2=34, total=74)):
            rows = paper_runner.run_case(metadata, 0, 'gap', T, gap, baseline_only=True)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row['status'], 'simulated')
            self.assertEqual(row['cost'], 74)
            self.assertEqual(row['H'], T)
            self.assertEqual(sum(json.loads(row['online_counts_json'])), T)

    def test_xu_original_learners_against_independent_reference(self):
        import xu_simulation
        from xu_budget import budget
        K, T, mu = 3, 200, .3
        arrays = [np.array([0.,1.,1.]), np.array([1.]), np.array([mu])]
        for learner in ['UCB','TS']:
            self.assertFalse(budget(learner,K,T,0.)['feasible'])
            for seed in [10,42,89]:
                actual = xu_simulation.simulate(arrays,T,learner,seed,12,17,mu)
                counts, sums = np.zeros(K,dtype=int), np.zeros(K)
                stages = np.zeros((3,K),dtype=int)
                rng = np.random.default_rng(seed)
                changes, magnitude = 0, 0.
                for step in range(T):
                    if learner=='UCB' and step<K:
                        arm = step
                    elif learner=='UCB':
                        arm = int(np.argmax(sums/counts+np.sqrt(math.log(T)/counts)))
                    else:
                        arm = int(np.argmax([rng.beta(sums[i]+1,counts[i]-sums[i]+1) for i in range(K)]))
                    raw = float(rng.random()<mu) if learner=='TS' and arm==K-1 else arrays[arm][rng.integers(0,len(arrays[arm]))]
                    stage = 0 if step<12 else 1 if step<29 else 2
                    reward = 0. if stage==0 else float(arm==K-1) if stage==1 else raw
                    changes += raw!=reward
                    magnitude += abs(raw-reward)
                    counts[arm] += 1; sums[arm] += reward; stages[stage,arm] += 1
                np.testing.assert_array_equal(actual[0],counts)
                self.assertEqual(actual[1],changes)
                self.assertAlmostEqual(actual[2],magnitude)
                np.testing.assert_array_equal(np.array(actual[3:]),stages)
        # The original finite-K TS expression can lose a positive count bound.
        invalid = budget('TS',10,1000000,.3)
        self.assertFalse(invalid['feasible'])
        self.assertIsNone(invalid['total'])
        self.assertLessEqual(invalid['n1'],0)

    def test_known_deterministic_mean_changes_cost_without_changing_random_default(self):
        low = np.array([.8]*9+[0.])
        high = np.array([.8]*9+[.5])
        for allocate in [bandit._ucb_counts_for_T, bandit._ts_counts_for_T]:
            n0, _, lower0, _ = allocate(low, 200_000)
            n_random, _, lower_random, _ = allocate(high, 200_000)
            n_exact, _, lower_exact, _ = allocate(high, 200_000, known_target_mean=.5)
            np.testing.assert_array_equal(n0, n_random)
            self.assertEqual(lower_random, 0.)
            self.assertEqual(lower_exact, .5)
            self.assertLess(n_exact.sum(), n_random.sum())
        with self.assertRaises(ValueError):
            bandit._ucb_counts_for_T(high, 200_000, known_target_mean=.7)

    def test_fast_ts_envelope_matches_exhaustive_integer_minimum(self):
        from direct_search import target_min, ts_value
        for mu in [0., .2, .8]:
            for n in [0, 1, 30, 100]:
                for non in [0, 15]:
                    last = 200-2*5-non-n-1
                    expected = min(ts_value(m,n,non,200,3,5,mu,.05)
                                   for m in range(5,last+1))
                    actual = target_min(n,non,200,3,5,mu,True,.5,.05)
                    self.assertAlmostEqual(actual, expected, places=14)

    def test_plotter_rejects_certificate_and_incomplete_repeats(self):
        row = dict(dataset='MovieLens-25M', sweep='horizon', learner='UCB', method='Ours',
                   T=100, K=2, H=70, T0=30, multiplier='', repeat=0, status='certificate',
                   allocation_json='[10, 10]', online_counts_json='[0, 70]', cost=20,
                   online_ratio=1., target_ratio=.85, non_target_avg=10, target_cost=10)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for state, expected_error in [('certificate', 'Unsimulated'), ('simulated', 'Missing or duplicate')]:
                row['status'] = state
                path = out/'paper_results.csv'
                with path.open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=row)
                    w.writeheader()
                    w.writerow(row)
                manifest = dict(dataset='MovieLens-25M', smoke=False, repeats=10, rows=1,
                                results_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                (out/'manifest.json').write_text(json.dumps(manifest))
                with patch.object(generate_paper_figures, 'DATA', out):
                    with self.assertRaisesRegex(ValueError, expected_error):
                        generate_paper_figures.read_results()

    def test_trajectories_against_independent_numpy_reference(self):
        arrays = [np.array([0., 1., 1.]), np.array([0., 0., 1.])]
        for learner in ['UCB', 'TS']:
            for attack in [0, 1, 2]:
                for n in [np.array([0, 0]), np.array([7, 13])]:
                    sums = np.array([3., 1.])
                    actual, changes, magnitude = simulate(sums, n, arrays, 1200, learner, attack, 42, 50)
                    counts = 5 + n.copy()
                    sums[-1] += n[-1]
                    T0 = int(counts.sum())
                    expected = np.zeros(2, dtype=int)
                    rng = np.random.default_rng(42)
                    modifications = 0
                    total = 0.
                    for step in range(1200-T0):
                        means = sums / counts
                        if learner == 'UCB':
                            value = means + 1.5 * np.sqrt(math.log(T0+step+1) / counts)
                        else:
                            value = means + np.array([rng.normal(), rng.normal()]) / np.sqrt(counts)
                        arm = int(np.argmax(value))
                        raw = arrays[arm][rng.integers(0, len(arrays[arm]))]
                        reward = raw
                        if attack == 1 and arm != 1:
                            nk = counts[-1]
                            beta = math.sqrt(.5/nk * math.log(math.pi**2 * 2 * nk**2 / .15))
                            gap = .01 if learner == 'UCB' else 4*math.exp(min(counts[arm]+1,20)) + math.sqrt(8*math.log(math.pi**2*2/.15))
                            threshold = means[-1] - 2*beta - gap
                            alpha = min(raw, max(0., sums[arm]+raw-threshold*(counts[arm]+1)))
                            reward = raw-alpha
                        elif attack == 2 and step < 100:
                            reward = float(step >= 50 and arm == 1)
                        modifications += reward != raw
                        total += abs(reward-raw)
                        counts[arm] += 1
                        sums[arm] += reward
                        expected[arm] += 1
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(changes, modifications)
                    self.assertAlmostEqual(magnitude, total)

    def test_certificate_has_no_empirical_success(self):
        previous = bandit.simulate_online
        bandit.simulate_online = False
        try:
            for fn in [bandit.UCB_fixed_T, bandit.TS_fixed_T]:
                clean = np.array([4.]*9+[0.])
                row = fn(clean, clean/5, 100_000)
                self.assertIsNone(row['target_online_ratio'])
                self.assertTrue(all(v is None for v in row['online_counts'].values()))
        finally:
            bandit.simulate_online = previous

    def test_threshold_certificates_and_clock(self):
        clean = np.array([4., 3., 2., 4., 3., 4., 5., 3., 2., 0.])
        for allocate, check in [(bandit._ucb_counts_for_T, bandit._check_ucb_cutoffs),
                                (bandit._ts_counts_for_T, bandit._check_ts_cutoffs)]:
            n, z, lower, _ = allocate(clean/5, 100_000)
            self.assertTrue(np.all(n >= 0))
            check(clean/5, 100_000, n, z, lower)
            online, _, _ = simulate(clean, n, [np.array([0., 1.])]*9+[np.array([0.])],
                                     100_000, 'TS' if allocate == bandit._ts_counts_for_T else 'UCB', 0, 10)
            self.assertEqual(int(online.sum())+int(n.sum())+50, 100_000)
            self.assertEqual(int(online[:-1].sum()), 0)


if __name__ == '__main__':
    unittest.main()
