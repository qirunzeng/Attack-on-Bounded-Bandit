"""Figure validation uses temporary synthetic records, never paper results."""
import copy
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import tempfile
import unittest

import generate_epsilon_greedy_figures as figures


def fixture():
    configs = [dict(T=T, K=K) for K in figures.K_GRID for T in figures.T_GRID]
    source = figures.ROOT / 'epsilon_greedy_experiment.py'
    manifest = dict(version=figures.VERSION, dataset='Controlled Bernoulli',
                    completed=True, status='completed', smoke=False, repeats=50,
                    T_grid=figures.T_GRID, K_grid=figures.K_GRID, configurations=configs,
                    c=1., delta=.05, N0_i=0, results_file='epsilon_greedy_results.csv',
                    simulation='Actual sequential decisions; NumPy PCG64; first-max greedy ties',
                    schedule='epsilon_s=min(1,c*K/s), s=T0+1,...,T; T0=C',
                    source_sha256={source.name: hashlib.sha256(source.read_bytes()).hexdigest()})
    rows = []
    for config in configs:
        T, K = config['T'], config['K']
        expected = 1 + figures.harmonic(T) - figures.harmonic(K)
        confidence = math.log((K - 1) / .05)
        B = math.ceil(expected + math.sqrt(2 * expected * confidence) + confidence / 3)
        n = ([math.ceil(math.sqrt(T * B / (K - 1)))] * (K - 1)
             + [math.ceil(math.sqrt(T * (K - 1) * B)) + 1])
        C = sum(n)
        radius = math.sqrt((K - 1) * B / T)
        for repeat in range(50):
            pulls = repeat % 10
            counts = [pulls] * (K - 1) + [T - C - pulls * (K - 1)]
            exploration = [pulls] * K
            rows.append(dict(
                dataset='Controlled Bernoulli', setting=figures.VERSION,
                learner='epsilon-greedy', method='Ours', status='simulated',
                T=T, K=K, repeat=repeat, seed=len(rows), c=1., delta=.05,
                expected_exploration_per_arm=expected, B=B, C=C, cost=C, H=T-C, T0=C,
                expected_non_target_exploratory_pulls=(K-1)*(figures.harmonic(T)-figures.harmonic(C)),
                nK=n[-1], target_cost=n[-1], non_target_cost=sum(n[:-1]),
                target_budget_fraction=n[-1]/C, non_target_budget_fraction=sum(n[:-1])/C,
                normalized_cost=C/(2*math.sqrt(T*(K-1)*B)),
                cost_over_sqrt_TKlogT=C/math.sqrt(T*K*math.log(T)),
                r_eg=radius, target_mean=radius/math.log(T), boundary_ratio=1/math.log(T),
                xi=(K-1)*B, non_target_online_pulls=sum(counts[:-1]),
                exploratory_non_target_pulls=sum(exploration[:-1]),
                greedy_non_target_pulls=0, success=1, individual_exploration_bound=1,
                all_greedy_target=1, online_ratio=counts[-1]/(T-C), trajectory_seconds=.1,
                allocation_json=json.dumps(n), online_counts_json=json.dumps(counts),
                exploratory_counts_json=json.dumps(exploration),
                means_json=json.dumps([.2+.6*i/(K-2) for i in range(K-1)]
                                      + [radius/math.log(T)])))
    return manifest, rows


def comparison_fixture():
    configs = sorted({(T, 10) for T in figures.HORIZON_GRID} |
                     {(1_000_000, K) for K in figures.ARM_GRID})
    sources = ['epsilon_greedy_experiment.py', 'epsilon_greedy_comparison.py']
    manifest = dict(version=figures.COMPARISON_VERSION, dataset='Controlled Bernoulli',
                    completed=True, status='completed', smoke=False, repeats=50,
                    horizon_grid=figures.HORIZON_GRID, arm_grid=figures.ARM_GRID,
                    horizon_K=10, arm_T=1_000_000, c=1., delta=.05,
                    configurations=[dict(T=T, K=K) for T, K in configs],
                    results_file='comparison_results.csv',
                    source_sha256={name: hashlib.sha256((figures.ROOT / name).read_bytes()).hexdigest()
                                   for name in sources})
    rows = []
    for T, K in configs:
        B, target, competitor, cost = figures.allocation(T, K)
        radius = math.sqrt((K - 1) * B / T)
        for repeat in range(50):
            for method in ['Ours', 'Clipped Suppression']:
                offline = method == 'Ours'
                n = [competitor] * (K-1) + [target] if offline else [0] * K
                T0, H = sum(n), T-sum(n)
                qi = repeat if offline else repeat+10
                online = [qi] * (K-1) + [H-qi*(K-1)]
                initialization = [0 if offline else 1] * K
                exploration = [repeat] * K
                rewards = [qi//2] * (K-1) + [repeat+1]
                observed = [0] * (K-1) + [repeat+1]
                rows.append(dict(
                    dataset='Controlled Bernoulli', setting=figures.COMPARISON_VERSION,
                    learner='epsilon-greedy', method=method, status='simulated',
                    T=T, K=K, repeat=repeat, seed=len(rows), c=1., delta=.05, B=B,
                    r_eg=radius, target_mean=radius/math.log(T), boundary_ratio=1/math.log(T),
                    T0=T0, H=H, cost=cost if offline else qi*(K-1),
                    C=cost if offline else qi*(K-1), nK=n[-1],
                    allocation_json=json.dumps(n), online_counts_json=json.dumps(online),
                    exploratory_counts_json=json.dumps(exploration),
                    initialization_counts_json=json.dumps(initialization),
                    means_json=json.dumps([.2+.6*i/(K-2) for i in range(K-1)]
                                          + [radius/math.log(T)]),
                    non_target_online_pulls=qi*(K-1),
                    greedy_non_target_pulls=0 if offline else 9*(K-1),
                    exploratory_non_target_pulls=repeat*(K-1),
                    initialization_non_target_pulls=0 if offline else K-1,
                    modified_reward_count=0 if offline else sum(rewards[:-1]),
                    target_reward_sum='' if offline else rewards[-1],
                    reward_sums_json='' if offline else json.dumps(rewards),
                    observed_reward_sums_json='' if offline else json.dumps(observed),
                    cost_definition='injected samples' if offline else
                                    'suppressed non-target rounds (including 0->0)',
                    total_target_ratio=(n[-1]+online[-1])/T,
                    online_ratio=online[-1]/H, trajectory_seconds=.1))
    return manifest, rows


class EpsilonGreedyFigureTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.path = Path(self.scratch.name)
        self.manifest, self.rows = fixture()

    def write(self):
        path = self.path / self.manifest['results_file']
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)
        self.manifest['rows'] = len(self.rows)
        self.manifest['results_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.path / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_theory_grid_has_no_sampling_bands(self):
        self.write()
        rows, manifest = figures.read_results(self.path)
        self.assertEqual(len(rows), 1250)
        tex = figures.generate_theory(rows, manifest)
        self.assertEqual(tex.count(r'\begin{axis}'), 2)
        left, right = tex.split(r'\end{axis}', 1)
        self.assertNotIn('fill between', left)
        self.assertNotIn('fill between', right)
        self.assertIn('fig:eg-theory-alignment', tex)
        self.assertIn('ymin=0.95, ymax=1.05', left)
        self.assertIn('ymin=0.3, ymax=0.7', right)
        for K in figures.K_GRID:
            self.assertIn(f'$K={K}$', tex)
        self.assertNotIn('MovieLens', tex)
        self.assertNotIn('Growing', tex)
        self.assertIn(r'yticklabels={$1/2$}', tex)
        self.assertIn(r'Non-target $\sum_{i<K}n_i/\mathcal C$', tex)
        self.assertIn(r'\begin{figure}[H]', tex)
        cost_coordinates = re.findall(r'coordinates \{([^}]*)\};', left)
        self.assertEqual(len(cost_coordinates), 6)  # Reference plus five arm-count curves.
        self.assertTrue(all(len(re.findall(r'\(', entry)) == 10
                            for entry in cost_coordinates[1:]))
        split_axis = right.split(r'\begin{axis}', 1)[1].split(r'\end{axis}')[0]
        split_coordinates = re.findall(r'coordinates \{([^}]*)\};', split_axis)
        self.assertEqual(len(split_coordinates), 3)
        self.assertTrue(all(len(re.findall(r'\(', entry)) == 10
                            for entry in split_coordinates[1:]))

    def test_comparison_grid_and_three_figures(self):
        boundary_manifest, boundary_rows = self.manifest, self.rows
        self.manifest, self.rows = comparison_fixture()
        self.write()
        rows, manifest = figures.read_comparison_results(self.path)
        self.assertEqual(len(rows), 1600)
        boundary_manifest['results_sha256'] = 'test boundary provenance'
        outputs = figures.generate(boundary_rows, boundary_manifest, rows, manifest)
        self.assertEqual(set(outputs), {'eg_cost_experiments.tex', 'eg_ratio_experiments.tex',
                                        'eg_theory_alignment.tex'})
        for tex in outputs.values():
            self.assertEqual(tex.count(r'\begin{axis}'), 2)
            self.assertIn(r'\begin{figure}[H]', tex)
        costs = outputs['eg_cost_experiments.tex']
        self.assertEqual(costs.count('fill between'), 2)
        self.assertIn('Clipped Suppression', costs)
        ratios = outputs['eg_ratio_experiments.tex']
        self.assertEqual(ratios.count('fill between'), 6)
        self.assertIn(r'$N_K^{\on}/H$', ratios)
        self.assertIn(r'$N_K/T$', ratios)
        for sweep, grid in [('horizon', figures.HORIZON_GRID), ('arms', figures.ARM_GRID)]:
            for method in ['Ours', 'Clipped Suppression']:
                self.assertEqual([x for x, _, _ in figures.comparison_points(rows, sweep, method, 'cost')],
                                 grid)

    def test_comparison_metadata_counts_and_feedback_are_checked(self):
        original_manifest, original_rows = comparison_fixture()
        for key, value in [('smoke', True), ('completed', False), ('repeats', 9),
                           ('arm_grid', figures.ARM_GRID[:-1]),
                           ('source_sha256', {'epsilon_greedy_experiment.py': '0'*64})]:
            with self.subTest(metadata=key):
                self.manifest, self.rows = copy.deepcopy(original_manifest), copy.deepcopy(original_rows)
                self.manifest[key] = value
                self.write()
                with self.assertRaises(ValueError):
                    figures.read_comparison_results(self.path)
        for index, key, value in [(0, 'target_reward_sum', 0), (0, 'cost', 0),
                                  (0, 'total_target_ratio', 0), (0, 'repeat', 1),
                                  (1, 'cost', 0), (1, 'modified_reward_count', 0),
                                  (1, 'target_reward_sum', 0), (1, 'initialization_non_target_pulls', 0),
                                  (1, 'greedy_non_target_pulls', 0), (1, 'means_json', '[0]')]:
            with self.subTest(row=index, field=key):
                self.manifest, self.rows = copy.deepcopy(original_manifest), copy.deepcopy(original_rows)
                self.rows[index][key] = value
                self.write()
                with self.assertRaises(ValueError):
                    figures.read_comparison_results(self.path)
        self.manifest, self.rows = copy.deepcopy(original_manifest), copy.deepcopy(original_rows)
        self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, 'Incomplete comparison'):
            figures.read_comparison_results(self.path)
        self.manifest, self.rows = original_manifest, original_rows
        self.write()
        with (self.path / 'comparison_results.csv').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'provenance hash'):
            figures.read_comparison_results(self.path)

    def test_computed_cost_grid_matches_saved_costs(self):
        self.assertEqual(len(figures.COST_T_GRID), 10)
        self.assertTrue(set(figures.T_GRID) < set(figures.COST_T_GRID))
        for K in figures.K_GRID:
            computed = figures.deterministic_cost_points(self.rows, K)
            self.assertEqual([T for T, _, _ in computed], figures.COST_T_GRID)
            saved = {T: cost for T, cost, _ in figures.points(self.rows, K, 'cost')}
            self.assertTrue(all(sd == 0 and isinstance(cost, int) and 0 < cost < T
                                for T, cost, sd in computed))
            self.assertEqual({T: cost for T, cost, _ in computed if T in saved}, saved)
        self.rows[0]['cost'] += 1
        with self.assertRaisesRegex(ValueError, 'disagrees with saved trajectories'):
            figures.deterministic_cost_points(self.rows, 5)

    def test_smoke_incomplete_and_stale_sources_rejected(self):
        original = copy.deepcopy(self.manifest)
        for key, value in [('smoke', True), ('completed', False), ('repeats', 9),
                           ('T_grid', figures.T_GRID[:-1]),
                           ('source_sha256', {'epsilon_greedy_experiment.py': '0'*64})]:
            with self.subTest(key=key):
                self.manifest = copy.deepcopy(original)
                self.manifest[key] = value
                self.write()
                with self.assertRaises(ValueError):
                    figures.read_results(self.path)

    def test_counts_flags_budget_and_repeats_checked(self):
        original = copy.deepcopy(self.rows)
        for key, value in [('T0', 0), ('B', 0), ('target_mean', 0),
                           ('greedy_non_target_pulls', 1), ('all_greedy_target', 0),
                           ('non_target_online_pulls', 10), ('repeat', 1),
                           ('online_counts_json', '[0]'), ('status', 'certificate')]:
            with self.subTest(key=key):
                self.rows = copy.deepcopy(original)
                self.rows[0][key] = value
                self.write()
                with self.assertRaises(ValueError):
                    figures.read_results(self.path)

    def test_csv_hash_and_missing_row_checked(self):
        self.write()
        with (self.path / self.manifest['results_file']).open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'provenance hash'):
            figures.read_results(self.path)
        self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, 'Incomplete results'):
            figures.read_results(self.path)

    def test_harmonic_accuracy_and_no_spurious_deterministic_band(self):
        for n in [5, 25, 49, 50, 1000, 100000]:
            self.assertAlmostEqual(figures.harmonic(n),
                                   math.fsum(1/s for s in range(1, n+1)), places=12)
        with self.assertRaisesRegex(ValueError, 'deterministic budget varied'):
            figures.deterministic_curve([(100000, 100., .1)], 'total')


if __name__ == '__main__':
    unittest.main()
