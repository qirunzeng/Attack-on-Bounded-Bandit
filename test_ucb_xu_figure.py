"""Temporary synthetic fixtures test plotting validation, not attack success."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import generate_ucb_xu_figure as figures


def fixture():
    T, K = figures.T, figures.K
    radius = (K * math.log(T) / T)**(1/3)
    gaps = [g*radius for g in figures.MULTIPLIERS]
    manifest = dict(version=figures.VERSION, dataset='MovieLens-25M',
        completed=True, status='completed', smoke=False, repeats=50, T=T, K=K,
        multipliers=figures.MULTIPLIERS, Delta_grid=gaps, methods=figures.METHODS,
        configurations=[dict(multiplier=g, Delta_K=gap)
                        for g, gap in zip(figures.MULTIPLIERS, gaps)],
        sigma=1/3, exploration_parameter=1/3, delta=.05, N0_i=5,
        learner_index='mean + sqrt(log(t)/N)', tie_breaking='first arm index',
        target_information='known deterministic target reward; exact lower bound equals mu_K',
        target_feedback='deterministic Delta_K in clean and online rewards',
        xu_budget_version=figures.xu_budget.VERSION,
        xu_budget_source=figures.xu_budget.SOURCE, xu_budget_source_section='A.2',
        xu_schedule_adaptation=figures.ADAPTATION, results_file='comparison_results.csv')
    rows = []
    for multiplier, gap in zip(figures.MULTIPLIERS, gaps):
        info = figures.xu_budget.budget('UCB', K, T, gap)
        for repeat in range(50):
            for method in figures.METHODS:
                ours = method == 'Ours'
                clean = [1.+.02*(repeat % 5)]*(K-1)+[5*gap] if ours else [0]*K
                n, z = [0]*K, ''
                if ours:
                    z = max(.05, gap+.0001)
                    c1 = z+math.sqrt(math.log(T)/T)
                    c2 = math.sqrt(math.log(T))
                    n = [math.ceil(((c2+math.sqrt(c2*c2+4*c1*s))/(2*c1))**2-5)
                         for s in clean[:-1]] + [math.ceil(T*(z-gap)/(1-z))+1]
                T0 = 5*K+sum(n) if ours else 0
                H = T-T0
                feasible = ours or info['feasible']
                cost = sum(n) if ours else info['total']
                c1, c2 = (0, 0) if ours else (info['C1'], info['C2'])
                row = dict(dataset='MovieLens-25M', setting=figures.VERSION, learner='UCB',
                    method=method, T=T, K=K, repeat=repeat, multiplier=multiplier, Delta_K=gap,
                    sigma=1/3, exploration_parameter=1/3, delta=.05, N0_i=5 if ours else 0,
                    seed=len(rows), T0=T0, H=H, cost='' if cost is None else cost,
                    allocation_json=json.dumps(n), clean_sum_json=json.dumps(clean),
                    C1='' if c1 is None else c1, C2='' if c2 is None else c2,
                    budget_log_json='' if ours else json.dumps(info), z_star=z,
                    mu_minus_K=gap if ours else '',
                    status='simulated' if feasible else 'infeasible',
                    infeasible_reason='' if feasible else info['reason'],
                    cost_definition='injected samples' if ours else 'scheduled corrupted rounds',
                    online_counts_json='', phase1_counts_json='', phase2_counts_json='',
                    post_attack_counts_json='', target_ratio='', online_ratio='',
                    post_attack_target_ratio='', post_attack_H='', modifications='',
                    magnitude='', trajectory_seconds='')
                if feasible:
                    p1 = [0]*K if ours else [c1//K]*K
                    p2 = [0]*(K-1)+[c2]
                    q = 0 if ours else repeat % 2
                    post = [q]*(K-1)+[H-c1-c2-q*(K-1)]
                    online = [p1[i]+p2[i]+post[i] for i in range(K)]
                    row.update(online_counts_json=json.dumps(online),
                        phase1_counts_json=json.dumps(p1), phase2_counts_json=json.dumps(p2),
                        post_attack_counts_json=json.dumps(post), post_attack_H=H-c1-c2,
                        target_ratio=((5 if ours else 0)+n[-1]+online[-1])/T,
                        online_ratio=online[-1]/H, post_attack_target_ratio=post[-1]/(H-c1-c2),
                        modifications=c1+c2, magnitude=.4*c1+(1-gap)*c2, trajectory_seconds=.1)
                rows.append(row)
    return manifest, rows


class UCBXuFigureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.manifest, self.rows = fixture()
        source_hashes = {}
        for name in figures.SOURCES:
            path = self.path/name
            path.write_text('# Synthetic provenance fixture, not simulation code.\n')
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest['source_sha256'] = source_hashes
        root_patch = patch.object(figures, 'ROOT', self.path)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write(self):
        path = self.path/self.manifest['results_file']
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)
        self.manifest['rows'] = len(self.rows)
        self.manifest['results_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.path/'manifest.json').write_text(json.dumps(self.manifest))

    def test_complete_fixture_uses_normalized_gaps_and_sample_sd(self):
        self.write()
        rows, manifest = figures.read_results(self.path)
        self.assertEqual(len(rows), 1400)
        ours = figures.segments(rows, 'Ours', 'cost')
        xu = figures.segments(rows, 'Xu2021', 'cost')
        self.assertEqual([len(part) for part in ours], [14])
        self.assertEqual([len(part) for part in xu], [11])
        self.assertEqual([point[0] for point in ours[0]], manifest['multipliers'])
        selected = [float(row['cost']) for row in rows
                    if row['method'] == 'Ours' and float(row['multiplier']) == 0.]
        self.assertEqual(ours[0][0][2], statistics.stdev(selected))
        self.assertGreater(ours[0][0][2], 0.)
        self.assertTrue(all(std == 0 for _, _, std in xu[0]))
        tex = figures.generate(rows, manifest)
        self.assertEqual(tex.count(r'\begin{axis}'), 2)
        self.assertEqual(tex.count('fill between['), 5)
        self.assertIn(r'fig:ucb-xu-gap', tex)
        self.assertIn(r'xlabel={$\Delta_K/(S_T/T)$}', tex)
        self.assertIn('xmin=0, xmax=6,', tex)
        self.assertIn('xtick={0,1,2,3,4,5,6}', tex)
        self.assertIn('ymin=0.3, ymax=1,', tex)
        self.assertIn(r'Ours $N_K^{\on}/H$', tex)
        self.assertIn(r'Xu (2021) $N_K/T$', tex)
        self.assertIn(r'\caption{Comparison with \citet{xu2021observation} for UCB.}', tex)

    def test_interior_infeasibility_breaks_lines_and_bands(self):
        self.write()
        rows, _ = figures.read_results(self.path)
        # Exercise a future noncontiguous feasible grid; no simulated values
        # from the removed configuration may bridge the two curve segments.
        for row in rows:
            if row['method'] == 'Xu2021' and float(row['multiplier']) == 1.:
                row['status'] = 'infeasible'
        parts = figures.segments(rows, 'Xu2021', 'target_ratio')
        self.assertEqual([len(part) for part in parts], [4, 6])
        tex = figures.panel(rows, ratio=True)
        self.assertIn('name path=ucbxutarget_ratioxu0up', tex)
        self.assertIn('name path=ucbxutarget_ratioxu1up', tex)
        self.assertEqual(tex.count('fill between['), 4)

    def test_rejects_partial_smoke_and_wrong_learner(self):
        for field, value, message in [('completed', False, 'complete fifty-repeat'),
                                      ('smoke', True, 'complete fifty-repeat'),
                                      ('repeats', 10, 'complete fifty-repeat'),
                                      ('learner_index', 'mean + sqrt(log(T)/N)', 'learner')]:
            original = self.manifest[field]
            with self.subTest(field=field):
                self.manifest[field] = value
                self.write()
                with self.assertRaisesRegex(ValueError, message):
                    figures.read_results(self.path)
            self.manifest[field] = original

    def test_rejects_missing_and_duplicate_repeats(self):
        original = copy.deepcopy(self.rows)
        self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, 'Incomplete comparison results'):
            figures.read_results(self.path)
        self.rows = original
        self.rows[-1]['repeat'] = self.rows[-3]['repeat']
        self.write()
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            figures.read_results(self.path)

    def test_rejects_stale_csv_and_source(self):
        self.write()
        path = self.path/'comparison_results.csv'
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'provenance hash'):
            figures.read_results(self.path)
        self.write()
        (self.path/'ucb_xu_comparison.py').write_text('# Changed source\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            figures.read_results(self.path)

    def test_rejects_fabricated_infeasible_outcome(self):
        row = next(row for row in self.rows if row['status'] == 'infeasible')
        row['target_ratio'] = 0.
        self.write()
        with self.assertRaisesRegex(ValueError, 'Fabricated infeasible'):
            figures.read_results(self.path)

    def test_rejects_wrong_budget_counts_and_ratio(self):
        index = next(i for i, row in enumerate(self.rows)
                     if row['method'] == 'Xu2021' and row['status'] == 'simulated')
        original = copy.deepcopy(self.rows[index])
        for field, value, message in [('cost', 1, 'prescribed Xu cost'),
                                      ('online_counts_json', json.dumps([0]*figures.K), 'stage counts'),
                                      ('target_ratio', 0., 'target_ratio')]:
            with self.subTest(field=field):
                self.rows[index][field] = value
                self.write()
                with self.assertRaisesRegex(ValueError, message):
                    figures.read_results(self.path)
            self.rows[index] = copy.deepcopy(original)


if __name__ == '__main__':
    unittest.main()
