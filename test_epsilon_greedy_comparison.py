"""Independent sequential tests for clipped epsilon-greedy suppression."""
import json
import unittest

import numpy as np

from epsilon_greedy_comparison import (
    ARM_GRID, ARM_T, HORIZON_GRID, HORIZON_K, METHODS,
    configurations, run_case, simulate_diagnostics)
from epsilon_greedy_experiment import design, simulate


def reference(allocation, means, T, c, seed, suppress):
    counts = np.array(allocation, dtype=int)
    totals = np.zeros(len(counts))
    totals[-1] = counts[-1]
    online = np.zeros(len(counts), dtype=int)
    exploratory = np.zeros(len(counts), dtype=int)
    initialization = np.zeros(len(counts), dtype=int)
    raw_sums, observed_sums = np.zeros(len(counts)), np.zeros(len(counts))
    rng = np.random.default_rng(seed)
    greedy_errors, modified = 0, 0
    for s in range(int(counts.sum()) + 1, T + 1):
        missing = np.flatnonzero(counts == 0)
        if len(missing):
            arm = len(counts) - 1 if counts[-1] == 0 else int(missing[0])
            initialization[arm] += 1
        elif rng.random() < min(1, c * len(counts) / s):
            arm = int(rng.integers(len(counts)))
            exploratory[arm] += 1
        else:
            arm = int(np.argmax(totals / counts))
            greedy_errors += int(arm != len(counts) - 1)
        raw = float(rng.random() < means[arm])
        observed = 0. if suppress and arm != len(counts) - 1 else raw
        modified += int(raw != observed)
        raw_sums[arm] += raw
        observed_sums[arm] += observed
        totals[arm] += observed
        counts[arm] += 1
        online[arm] += 1
    return online, exploratory, greedy_errors, initialization, raw_sums, observed_sums, modified


class ComparisonTests(unittest.TestCase):
    def test_baseline_matches_sequential_reference_without_assuming_success(self):
        failures_seen = False
        for target in [0., .07, 1.]:
            for c in [0., 1., 500.]:
                for seed in [0, 11, 203]:
                    with self.subTest(target=target, c=c, seed=seed):
                        means = [.2, .8, target]
                        got = simulate_diagnostics([0, 0, 0], means, 300, c, seed, True)
                        expected = reference([0, 0, 0], means, 300, c, seed, True)
                        for a, b in zip(got, expected):
                            np.testing.assert_array_equal(a, b)
                        self.assertEqual(sum(got[0]), 300)
                        np.testing.assert_array_equal(got[3], [1, 1, 1])
                        np.testing.assert_array_equal(got[5][:-1], [0, 0])
                        self.assertEqual(got[6], sum(got[4][:-1]))
                        failures_seen |= got[2] > 0
        self.assertTrue(failures_seen)

    def test_initialization_is_online_target_first_and_counts_zero_replacements(self):
        for T, expected_counts in [(1, [0, 0, 1]), (2, [1, 0, 1]), (3, [1, 1, 1])]:
            got = simulate_diagnostics([0, 0, 0], [0, 0, 0], T, 1, 17, True)
            np.testing.assert_array_equal(got[0], expected_counts)
            np.testing.assert_array_equal(got[3], expected_counts)
            self.assertEqual(sum(got[1]), 0)
        got = simulate_diagnostics([0, 0, 0], [0, 0, 0], 100, 0, 17, True)
        self.assertEqual(sum(got[0][:-1]), 99)
        self.assertEqual(got[6], 0)  # Suppression still costs 99, including zero-to-zero.
        self.assertEqual(got[2], 97)

    def test_ours_diagnostics_match_frozen_simulator_and_independent_feedback(self):
        for allocation in [[1, 1, 1], design(10_000, 3)['allocation']]:
            for means in [[1, 1, 0], [.2, .8, .03]]:
                for c in [0., 1., 100_000.]:
                    got = simulate_diagnostics(allocation, means, 10_000, c, 17)
                    original = simulate(allocation, means, 10_000, c, 17)
                    expected = reference(allocation, means, 10_000, c, 17, False)
                    for a, b in zip(got[:3], original):
                        np.testing.assert_array_equal(a, b)
                    for a, b in zip(got, expected):
                        np.testing.assert_array_equal(a, b)
                    self.assertEqual(sum(got[3]), 0)
                    self.assertEqual(got[6], 0)

    def test_full_grid_deduplicates_common_configuration(self):
        cases = configurations()
        self.assertEqual(len(cases), 16)
        self.assertEqual(len(set(cases)), 16)
        self.assertEqual(cases.count((ARM_T, HORIZON_K)), 1)
        self.assertEqual({T for T, K in cases if K == HORIZON_K}, set(HORIZON_GRID))
        self.assertEqual({K for T, K in cases if T == ARM_T}, set(ARM_GRID))

    def test_row_ratios_costs_and_repeat_reproducibility(self):
        for method in METHODS:
            a = run_case(10_000, 5, 3, method)
            b = run_case(10_000, 5, 3, method)
            for field in a:
                if field != 'trajectory_seconds':
                    self.assertEqual(a[field], b[field])
            online = json.loads(a['online_counts_json'])
            n = json.loads(a['allocation_json'])
            self.assertEqual(a['H'], sum(online))
            self.assertEqual(a['T0'], sum(n))
            self.assertEqual(a['H'] + a['T0'], a['T'])
            self.assertEqual(a['total_target_ratio'], (n[-1] + online[-1]) / a['T'])
            self.assertEqual(a['online_ratio'], online[-1] / a['H'])
            self.assertEqual(a['cost'], sum(n) if method == 'Ours' else sum(online[:-1]))
            if method == 'Clipped Suppression':
                self.assertEqual(a['target_reward_sum'], json.loads(a['reward_sums_json'])[-1])
                self.assertEqual(a['T0'], 0)
                self.assertEqual(a['H'], a['T'])
                self.assertLessEqual(a['modified_reward_count'], a['cost'])
            else:
                self.assertEqual(a['target_reward_sum'], '')
                self.assertEqual(a['reward_sums_json'], '')
                self.assertEqual(a['observed_reward_sums_json'], '')


if __name__ == '__main__':
    unittest.main()
