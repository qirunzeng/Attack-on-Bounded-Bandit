"""Validation tests use synthetic records; they never create paper results."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import generate_theory_figures as figures


def fixture(version=figures.VERSION):
    if version == figures.LEGACY_VERSION:
        configs = [(r, T, 10 if r == 'fixed' else round(10 * (T / 100000)**.25))
                   for r in ['fixed', 'growing'] for T in figures.T_GRID]
    else:
        configs = [('fixed', T, K) for K in figures.K_GRID for T in figures.T_GRID]
    base = 11 / 669
    manifest = dict(
        version=version, dataset='MovieLens-25M', repeats=10, smoke=False,
        T_grid=figures.T_GRID,
        configurations=[dict(regime=r, T=T, K=K) for r, T, K in configs],
        sigma=.5, delta=.05, N0_i=5,
        simulation='actual sequential UCB and Gaussian TS, Bernoulli target',
        target_information='unknown mean; clean-log confidence floor only',
        target_mean_rule='mu_ML * sqrt(100000 / T)',
        selected_arms={str(K): dict(means=[.6] * (K - 1) + [base])
                       for _, _, K in configs})
    if version == figures.LEGACY_VERSION:
        manifest['growing_K_rule'] = 'round(10 * (T / 100000)**(1/4))'
    else:
        manifest['K_grid'] = figures.K_GRID.copy()
        manifest['arm_count_rule'] = 'fixed K in K_grid'
    rows = []
    for regime, T, K in configs:
        for learner in figures.LEARNERS:
            for repeat in range(10):
                n = [100 + repeat + K] * (K - 1) + [200 * (K - 1)]
                cost = sum(n)
                H = T - cost - 5 * K
                coefficient = (9 * .5**2 * math.log(T) if learner == 'UCB' else
                               2 * math.log(math.pi**2 * K * T**2 / (3 * .05)))
                scale = 3 * ((K - 1) * coefficient * T**2 / 4)**(1/3)
                probability = base * math.sqrt(100000 / T)
                st = T**(2/3) * (K * math.log(T))**(1/3)
                rows.append(dict(
                    dataset='MovieLens-25M', setting=version,
                    regime=regime, learner=learner, method='Ours', repeat=repeat,
                    T=T, K=K, target_mean=probability,
                    boundary_ratio=probability / (st / T), base_target_mean=base,
                    target_feedback='Bernoulli; same mean in clean and online rewards',
                    lower_bound_source='clean log: max(0, empirical mean - 2 beta)',
                    mu_minus_K=0., z_star=.1, cost=cost, Lambda_T=scale,
                    normalized_cost=cost / scale, target_budget_fraction=n[-1] / cost,
                    target_cost=n[-1], non_target_avg=sum(n[:-1]) / (K - 1),
                    H=H, T0=T-H, non_target_online_pulls=0, zero_non_target_pulls=1,
                    online_ratio=1., status='simulated', seed=repeat,
                    clean_sum_json=json.dumps([3] * (K - 1) + [0]),
                    allocation_json=json.dumps(n),
                    online_counts_json=json.dumps([0] * (K - 1) + [H]),
                    trajectory_seconds=.1))
    return manifest, rows


class TheoryFigureTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.path = Path(self.scratch.name)
        self.manifest, self.rows = fixture()

    def write(self):
        path = self.path / 'theory_results.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)
        self.manifest['rows'] = len(self.rows)
        self.manifest['results_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.path / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_complete_grid_and_both_figures(self):
        self.write()
        rows, _ = figures.read_results(self.path)
        self.assertEqual(len(rows), 500)
        outputs = figures.generate(rows)
        self.assertEqual(set(outputs), {'ucb_theory_alignment.tex', 'ts_theory_alignment.tex'})
        for learner in ['ucb', 'ts']:
            tex = outputs[f'{learner}_theory_alignment.tex']
            self.assertEqual(tex.count(r'\begin{axis}'), 2)
            self.assertIn(f'fig:{learner}-theory-alignment', tex)
            self.assertIn('xmode=normal, ymode=normal', tex)
            self.assertIn('asymptotic limits', tex)
            self.assertIn('0.666666666667', tex)
            self.assertNotIn('lower bound', tex)
            self.assertNotIn('Growing', tex)
            for K in figures.K_GRID:
                self.assertIn(f'$K={K}$', tex)
            cost = tex.split(r'\end{axis}')[0]
            self.assertEqual(cost.count('fill between'), 5)
            self.assertIn('ymin=0.3, ymax=0.7', tex)
            self.assertIn(r'yticklabels={$1/3$,$2/3$}', tex)
        self.assertIn(r'\Lambda_T^{\rm ts}', outputs['ts_theory_alignment.tex'])

    def test_missing_or_wrong_arm_count_grid(self):
        self.manifest.pop('K_grid')
        self.write()
        with self.assertRaisesRegex(ValueError, 'arm-count grid'):
            figures.read_results(self.path)
        self.manifest['K_grid'] = [5, 10, 15, 20]
        self.write()
        with self.assertRaisesRegex(ValueError, 'arm-count grid'):
            figures.read_results(self.path)
        self.manifest['K_grid'] = figures.K_GRID.copy()
        self.manifest['configurations'][0]['K'] = 10
        self.write()
        with self.assertRaisesRegex(ValueError, 'configuration grid'):
            figures.read_results(self.path)

    def test_wrong_row_arm_count_and_version(self):
        self.rows[0]['K'] = 6
        self.write()
        with self.assertRaisesRegex(ValueError, 'configuration'):
            figures.read_results(self.path)
        self.rows[0]['K'] = 5
        self.rows[0]['setting'] = figures.LEGACY_VERSION
        self.write()
        with self.assertRaisesRegex(ValueError, 'Mixed experimental settings'):
            figures.read_results(self.path)

    def test_legacy_data_are_still_validated(self):
        self.manifest, self.rows = fixture(figures.LEGACY_VERSION)
        self.write()
        rows, manifest = figures.read_results(self.path)
        self.assertEqual(len(rows), 200)
        self.assertEqual(manifest['version'], figures.LEGACY_VERSION)
        self.manifest['growing_K_rule'] = 'arbitrary'
        self.write()
        with self.assertRaisesRegex(ValueError, 'legacy regime'):
            figures.read_results(self.path)

    def test_allocation_panel_preserves_fixed_k10_only(self):
        _, legacy = fixture(figures.LEGACY_VERSION)
        for learner in figures.LEARNERS:
            self.assertEqual(figures.panel(self.rows, learner, 'allocation_fractions'),
                             figures.panel(legacy, learner, 'allocation_fractions'))
            fixed = [row for row in self.rows if row['K'] == 10]
            expected = figures.points(fixed, learner, 'fixed', 'target_budget_fraction')
            self.assertEqual(figures.points(self.rows, learner, 'fixed',
                                           'target_budget_fraction', K=10), expected)
            self.assertNotEqual(figures.points(self.rows, learner, 'fixed',
                                              'target_budget_fraction', K=5), expected)

    def test_hash_mismatch(self):
        self.write()
        with (self.path / 'theory_results.csv').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'provenance hash'):
            figures.read_results(self.path)

    def test_incomplete_and_duplicate_repeats(self):
        complete = copy.deepcopy(self.rows)
        self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, 'Incomplete results'):
            figures.read_results(self.path)
        self.rows = complete
        self.rows[-1]['repeat'] = 0
        self.write()
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            figures.read_results(self.path)

    def test_smoke_and_certificate_records_rejected(self):
        self.manifest['smoke'] = True
        self.write()
        with self.assertRaisesRegex(ValueError, 'non-smoke'):
            figures.read_results(self.path)
        self.manifest['smoke'] = False
        self.rows[0]['status'] = 'certificate'
        self.write()
        with self.assertRaisesRegex(ValueError, 'Non-simulated'):
            figures.read_results(self.path)
        self.rows[0]['status'] = 'simulated'
        self.manifest['simulation'] = 'certificate counts'
        self.write()
        with self.assertRaisesRegex(ValueError, 'actual sequential'):
            figures.read_results(self.path)

    def test_default_floor_and_learner_scale(self):
        self.rows[0]['mu_minus_K'] = self.rows[0]['target_mean']
        self.write()
        with self.assertRaisesRegex(ValueError, 'default target floor'):
            figures.read_results(self.path)
        self.rows[0]['mu_minus_K'] = 0
        self.rows[10]['Lambda_T'] = self.rows[0]['Lambda_T']
        self.write()
        with self.assertRaisesRegex(ValueError, 'theoretical cost scale'):
            figures.read_results(self.path)

    def test_realized_counts_and_success_indicator(self):
        row = self.rows[0]
        counts = json.loads(row['online_counts_json'])
        counts[0], counts[-1] = 1, counts[-1] - 1
        row['online_counts_json'] = json.dumps(counts)
        self.write()
        with self.assertRaisesRegex(ValueError, 'Success indicator'):
            figures.read_results(self.path)
        # Real failures remain valid observations when recorded consistently.
        row['non_target_online_pulls'], row['zero_non_target_pulls'] = 1, 0
        row['online_ratio'] = counts[-1] / row['H']
        self.write()
        figures.read_results(self.path)
        row['H'] -= 1
        self.write()
        with self.assertRaisesRegex(ValueError, 'online horizon'):
            figures.read_results(self.path)

    def test_first_point_is_one_shared_trajectory(self):
        self.manifest, self.rows = fixture(figures.LEGACY_VERSION)
        self.rows[100]['trajectory_seconds'] = .2
        self.write()
        with self.assertRaisesRegex(ValueError, 'reuse the same trajectory'):
            figures.read_results(self.path)

    def test_sample_standard_deviation(self):
        rows = [dict(learner='UCB', regime='fixed', T=100000, value=v) for v in [1., 3.]]
        self.assertEqual(figures.points(rows, 'UCB', 'fixed', 'value'),
                         [(100000, 2., math.sqrt(2.))])


if __name__ == '__main__':
    unittest.main()
