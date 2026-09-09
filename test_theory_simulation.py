"""Sequential checks for controlled target rewards in the theory experiments."""
import math
import unittest

import numpy as np

from simulation import simulate, trajectory


def sequential_reference(clean, allocation, arrays, horizon, learner, attack,
                         seed, warm_start, target_mean):
    """A NumPy learner with explicit feedback history, independent of the JIT loop."""
    k = len(arrays)
    counts = np.full(k, 5 if warm_start else 0) + allocation
    initial = counts.copy()
    totals = clean.copy()
    totals[-1] += allocation[-1]
    rng = np.random.default_rng(seed)
    modified, magnitude = 0, 0.0
    for time in range(int(counts.sum()) + 1, horizon + 1):
        missing = np.flatnonzero(counts == 0)
        if len(missing):
            arm = k - 1 if counts[-1] == 0 else missing[0]
        else:
            indices = totals / counts
            if learner == 'UCB':
                indices += 1.5 * np.sqrt(np.log(time) / counts)
            else:
                indices += rng.standard_normal(k) / np.sqrt(counts)
            arm = int(np.argmax(indices))
        if arm == k - 1 and target_mean is not None:
            raw = float(rng.uniform() < target_mean)
        else:
            raw = arrays[arm][rng.integers(len(arrays[arm]))]
        observed = raw
        if attack == 1 and arm != k - 1:
            if learner == 'TS':
                observed = 0.0
            else:
                count = counts[-1]
                radius = np.sqrt(.5 * np.log(math.pi**2 * k * count**2 / .15) / count)
                ceiling = totals[-1] / count - 2 * radius - .01
                # Project the largest feedback satisfying the suppression
                # ceiling onto the legal interval [0, raw].
                observed = np.clip(ceiling * (counts[arm] + 1) - totals[arm], 0.0, raw)
        modified += raw != observed
        magnitude += abs(raw - observed)
        counts[arm] += 1
        totals[arm] += observed
    return counts - initial, modified, magnitude


class ControlledTargetTests(unittest.TestCase):
    arrays = [np.array([0., .2, .8, 1.]), np.array([.1, .5, .9]),
              np.array([.35, .65])]

    def test_learners_and_attacks_match_sequential_reference(self):
        scenarios = [(True, 0, np.array([7, 3, 21])),
                     (True, 0, np.zeros(3, dtype=int)),
                     (False, 1, np.zeros(3, dtype=int))]
        for learner in ['UCB', 'TS']:
            for probability in [None, 0., .27, 1.]:
                for warm, attack, allocation in scenarios:
                    clean = np.array([4., 2., 5. if probability == 1. else
                                      0. if probability == 0. else 2.]) if warm else np.zeros(3)
                    for seed in [11, 42, 203]:
                        with self.subTest(learner=learner, p=probability,
                                          warm=warm, attack=attack, seed=seed):
                            actual = simulate(clean, allocation, self.arrays, 360,
                                              learner, attack, seed, warm_start=warm,
                                              target_mean=probability)
                            expected = sequential_reference(clean, allocation, self.arrays,
                                                            360, learner, attack, seed,
                                                            warm, probability)
                            np.testing.assert_array_equal(actual[0], expected[0])
                            self.assertEqual(actual[1], expected[1])
                            self.assertAlmostEqual(actual[2], expected[2], places=12)
                            self.assertEqual(int(actual[0].sum()) + int(allocation.sum())
                                             + (15 if warm else 0), 360)

    def test_default_and_explicit_none_preserve_the_same_stream(self):
        clean, allocation = np.array([4., 2., 1.]), np.array([7, 3, 21])
        for learner in ['UCB', 'TS']:
            for attack in [0, 1, 2]:
                args = (clean, allocation, self.arrays, 360, learner, attack, 42)
                old_api = simulate(*args, phase=17, phase_two=23, diagnostics=True)
                explicit = simulate(*args, phase=17, phase_two=23,
                                    diagnostics=True, target_mean=None)
                for actual, expected in zip(explicit, old_api):
                    np.testing.assert_array_equal(actual, expected)

    def test_controlled_target_does_not_use_its_empirical_array(self):
        replacement = self.arrays[:-1] + [np.ones(23)]
        for learner in ['UCB', 'TS']:
            for probability in [0., .27, 1.]:
                for warm, attack in [(True, 0), (False, 1)]:
                    clean = np.array([4., 2., 0.]) if warm else np.zeros(3)
                    args = (360, learner, attack, 42)
                    one = simulate(clean, np.zeros(3, dtype=int), self.arrays, *args,
                                   warm_start=warm, target_mean=probability)
                    two = simulate(clean, np.zeros(3, dtype=int), replacement, *args,
                                   warm_start=warm, target_mean=probability)
                    for actual, expected in zip(one, two):
                        np.testing.assert_array_equal(actual, expected)

    def test_endpoint_rewards_are_exact_in_learner_statistics(self):
        for learner in [0, 1]:
            for probability in [0., 1.]:
                counts = np.array([7, 5, 4])
                totals = np.array([3., 2., 0.])
                result = trajectory(counts, totals, np.concatenate(self.arrays),
                                    np.array([0, 4, 7, 9]), 120, learner, 0,
                                    np.random.default_rng(42), target_mean=probability)
                self.assertGreater(result[0][-1], 0)
                self.assertEqual(totals[-1], probability * result[0][-1])

    def test_invalid_probability_is_rejected_before_simulation(self):
        for probability in [-1., -.01, 1.01, np.nan, np.inf, -np.inf]:
            with self.subTest(p=probability), self.assertRaisesRegex(ValueError, 'target_mean'):
                simulate(np.zeros(3), np.zeros(3, dtype=int), self.arrays,
                         120, 'UCB', 0, 42, target_mean=probability)


if __name__ == '__main__':
    unittest.main()
