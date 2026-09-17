"""Independent sequential checks of the epsilon-greedy experiment."""
import math
import unittest

import numpy as np

from epsilon_greedy_experiment import (
    K_GRID, T_GRID, bernoulli_means, ceil_sqrt_ratio, design,
    expected_exploration_per_arm, run_case, simulate)


def sequential_reference(allocation, means, T, c, seed):
    """Independent NumPy implementation sharing only the documented RNG stream."""
    counts = np.array(allocation, dtype=int)
    totals = np.zeros(len(counts))
    totals[-1] = counts[-1]
    online = np.zeros(len(counts), dtype=int)
    exploratory = np.zeros(len(counts), dtype=int)
    greedy_non_target = 0
    rng = np.random.default_rng(seed)
    for time in range(int(counts.sum()) + 1, T + 1):
        explore = rng.random() < min(1, c * len(counts) / time)
        arm = int(rng.integers(len(counts))) if explore else int(np.argmax(totals / counts))
        exploratory[arm] += int(explore)
        greedy_non_target += int(not explore and arm != len(counts) - 1)
        totals[arm] += float(rng.random() < means[arm])
        counts[arm] += 1
        online[arm] += 1
    return online, exploratory, greedy_non_target


class EpsilonGreedyTests(unittest.TestCase):
    def test_exact_sequential_reference_including_underbudget_attacks(self):
        cases = [([1, 1], [1., 0.]),
                 ([3, 5, 11], [.4, .8, .03]),
                 ([9, 4, 2], [0., 1., 1.]),
                 ([1, 1, 1], [1., 1., 0.])]
        found_greedy_failure = False
        for allocation, means in cases:
            for c in [0., 1., 300.]:
                for seed in [0, 11, 203]:
                    with self.subTest(allocation=allocation, c=c, seed=seed):
                        actual = simulate(allocation, means, 300, c, seed)
                        expected = sequential_reference(allocation, means, 300, c, seed)
                        for first, second in zip(actual, expected):
                            np.testing.assert_array_equal(first, second)
                        self.assertEqual(int(actual[0].sum()), 300 - sum(allocation))
                        if c == 0:
                            self.assertEqual(int(actual[1].sum()), 0)
                        if c == 300:
                            np.testing.assert_array_equal(actual[0], actual[1])
                            self.assertEqual(actual[2], 0)
                        found_greedy_failure |= actual[2] > 0
        self.assertTrue(found_greedy_failure,
                        'An underbudget attack must exercise genuine greedy non-target decisions')

    def test_schedule_sum_matches_explicit_internal_rounds(self):
        for T in [1, 7, 25, 319, 10_000]:
            for K in [2, 5, 25]:
                for c in [0., .01, .17, 1., 2.5, 100.]:
                    expected = sum(min(1, c * K / s) / K for s in range(1, T + 1))
                    self.assertAlmostEqual(expected_exploration_per_arm(T, K, c),
                                           expected, delta=2e-11)

    def test_integer_root_rounding_and_allocation_products(self):
        for denominator in [1, 2, 7, 24]:
            for numerator in [0, 1, 4, 17, 144, 10**18 + 1]:
                root = ceil_sqrt_ratio(numerator, denominator)
                self.assertGreaterEqual(root * root * denominator, numerator)
                if root:
                    self.assertLess((root - 1)**2 * denominator, numerator)
        for T in T_GRID:
            for K in K_GRID:
                a = design(T, K)
                n, B, C = a['allocation'], a['B'], a['cost']
                continuous = 2 * math.sqrt(T * (K - 1) * B)
                self.assertGreater(C, continuous)
                self.assertLessEqual(C, continuous + K + 1)
                self.assertLess(C, T)
                self.assertTrue(np.all(n[:-1] * n[-1] > T * B))
                means = bernoulli_means(T, K, a['r_eg'])
                self.assertGreater(means[-1], 0)
                self.assertAlmostEqual(means[-1] / a['r_eg'], 1 / math.log(T))
                log_failure = math.log((K - 1) / .05)
                mean = a['expected_exploration_per_arm']
                self.assertEqual(B, math.ceil(mean + math.sqrt(2 * mean * log_failure)
                                               + log_failure / 3))

    def test_nonempty_horizon_and_input_guards(self):
        result = simulate([2, 3], [.4, .2], 6, 1, 17)
        self.assertEqual(int(result[0].sum()), 1)
        for T in [4, 5]:
            with self.assertRaises(ValueError):
                simulate([2, 3], [.4, .2], T, 1, 17)
        for means in [[np.nan, 0], [1.1, .2], [-.1, .2]]:
            with self.assertRaises(ValueError):
                simulate([2, 3], means, 6, 1, 17)
        for allocation in [[0, 1], [1.5, 2], [-1, 2]]:
            with self.assertRaises(ValueError):
                simulate(allocation, [.4, .2], 6, 1, 17)
        for T, K in [(1, 2), (10, 25)]:
            with self.assertRaisesRegex(ValueError, 'no online horizon'):
                design(T, K)

    def test_trajectory_row_accounting_and_repeat_reproducibility(self):
        first = run_case(10_000, 5, 2)
        second = run_case(10_000, 5, 2)
        for key in first:
            if key != 'trajectory_seconds':
                self.assertEqual(first[key], second[key])
        self.assertEqual(first['H'] + first['T0'], first['T'])
        self.assertEqual(first['target_cost'] + first['non_target_cost'], first['cost'])
        self.assertEqual(first['non_target_online_pulls'],
                         first['greedy_non_target_pulls'] + first['exploratory_non_target_pulls'])
        self.assertAlmostEqual(first['target_budget_fraction'] +
                               first['non_target_budget_fraction'], 1)
        self.assertEqual(first['success'], 1)
        self.assertEqual(first['all_greedy_target'], 1)


if __name__ == '__main__':
    unittest.main()
