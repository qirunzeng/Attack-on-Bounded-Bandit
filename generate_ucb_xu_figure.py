"""Plot the separate UCB/Xu gap comparison from verified sequential runs."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from generate_paper_figures import curve, legend
import xu_budget

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/ml-25m/ucb-xu-gap'
VERSION = 'ucb-xu-gap-v1'
MULTIPLIERS = [0., .1, .2, .3, .4, .5, .75, 1., 1.5, 2., 3., 4., 5., 6.]
METHODS = ['Ours', 'Xu2021']
T, K, REPEATS = 1_000_000, 10, 50
SOURCES = {'ucb_xu_comparison.py', 'test_ucb_xu_comparison.py', 'bandit.py',
           'mlrunner.py', 'paper_runner.py', 'xu_budget.py'}
ADAPTATION = ('Original A.2 budgets applied to common anytime UCB; '
              'original fixed-horizon learner is not simulated')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name):
    number = float(value)
    require(math.isfinite(number) and number >= 0 and number.is_integer(),
            f'Invalid {name}: {value}')
    return int(number)


def close(actual, expected, name):
    require(math.isfinite(float(actual)) and
            math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12),
            f'Inconsistent {name}: {actual}; expected {expected}')


def vector(row, field):
    values = [integer(value, field) for value in json.loads(row[field])]
    require(len(values) == K, f'Invalid {field} dimensions')
    return values


def read_results(data=DATA):
    """Validate provenance, the full grid, finite budgets, and actual counts."""
    data = Path(data)
    manifest = json.loads((data / 'manifest.json').read_text())
    require(manifest.get('version') == VERSION and
            manifest.get('dataset') == 'MovieLens-25M' and
            manifest.get('completed') is True and manifest.get('status') == 'completed' and
            manifest.get('smoke') is False and manifest.get('repeats') == REPEATS,
            'Figure requires a complete fifty-repeat, non-smoke comparison')
    require(manifest['T'] == T and manifest['K'] == K and
            manifest['multipliers'] == MULTIPLIERS and manifest['methods'] == METHODS,
            'Unexpected comparison grid or methods')
    radius = (K * math.log(T) / T)**(1/3)
    gaps = [multiplier * radius for multiplier in MULTIPLIERS]
    require(len(manifest['Delta_grid']) == len(gaps) and
            len(manifest['configurations']) == len(gaps), 'Incomplete gap grid')
    for multiplier, gap, declared, config in zip(
            MULTIPLIERS, gaps, manifest['Delta_grid'], manifest['configurations']):
        close(declared, gap, 'gap grid')
        close(config['multiplier'], multiplier, 'configuration multiplier')
        close(config['Delta_K'], gap, 'configuration gap')
    for key, expected in [('sigma', 1/3), ('exploration_parameter', 1/3), ('delta', .05)]:
        close(manifest[key], expected, key)
    require(manifest['N0_i'] == 5 and manifest['learner_index'] == 'mean + sqrt(log(t)/N)' and
            manifest['tie_breaking'] == 'first arm index', 'Unexpected learner or initialization')
    require(manifest['target_information'] ==
            'known deterministic target reward; exact lower bound equals mu_K' and
            manifest['target_feedback'] == 'deterministic Delta_K in clean and online rewards',
            'Unexpected target information')
    require(manifest['xu_budget_version'] == xu_budget.VERSION and
            manifest['xu_budget_source'] == xu_budget.SOURCE and
            manifest['xu_budget_source_section'] == 'A.2' and
            manifest['xu_schedule_adaptation'] == ADAPTATION, 'Unexpected Xu budget provenance')
    require(SOURCES <= set(manifest['source_sha256']), 'Missing source provenance')
    for name, expected in manifest['source_sha256'].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file(), f'Invalid source path: {name}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f'Simulation source changed since this run: {name}')
    require(manifest['results_file'] == 'comparison_results.csv', 'Unexpected results filename')
    raw = (data / manifest['results_file']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest['results_sha256'],
            'Results do not match the provenance hash')
    rows = list(csv.DictReader(raw.decode().splitlines()))
    require(len(rows) == manifest['rows'] == len(gaps) * len(METHODS) * REPEATS,
            'Incomplete comparison results')
    groups = defaultdict(list)
    seeds = set()
    for row in rows:
        require(row['dataset'] == 'MovieLens-25M' and row['setting'] == VERSION and
                row['learner'] == 'UCB' and row['method'] in METHODS, 'Mixed comparison records')
        require(integer(row['T'], 'T') == T and integer(row['K'], 'K') == K,
                'Unexpected row configuration')
        multiplier = float(row['multiplier'])
        require(multiplier in MULTIPLIERS, 'Unexpected gap multiplier')
        gap = float(row['Delta_K'])
        close(gap, multiplier * radius, 'row target gap')
        for key, expected in [('sigma', 1/3), ('exploration_parameter', 1/3), ('delta', .05)]:
            close(row[key], expected, f'row {key}')
        groups[row['method'], multiplier].append(integer(row['repeat'], 'repeat'))
        seed = integer(row['seed'], 'seed')
        require(seed not in seeds, 'Duplicate trajectory seed')
        seeds.add(seed)
        ours = row['method'] == 'Ours'
        require(integer(row['N0_i'], 'N0_i') == (5 if ours else 0), 'Wrong warm-start size')
        n = vector(row, 'allocation_json')
        clean = json.loads(row['clean_sum_json'])
        require(len(clean) == K and all(math.isfinite(v) and 0 <= v <= 5 for v in clean),
                'Invalid clean reward sums')
        T0, H = integer(row['T0'], 'T0'), integer(row['H'], 'H')
        require(T0 + H == T and H > 0, 'No valid online horizon')
        if ours:
            require(T0 == 5*K + sum(n) and row['cost_definition'] == 'injected samples',
                    'Inconsistent offline cost or clock')
            require(integer(row['cost'], 'cost') == sum(n), 'Inconsistent injected cost')
            close(clean[-1], 5*gap, 'clean target sum')
            close(row['mu_minus_K'], gap, 'target lower bound')
            z = float(row['z_star'])
            require(gap <= z < 1 and n[-1] >= 1, 'Invalid chosen threshold')
            target_budget = T * (z-gap) / (1-z)
            require(n[-1]-1 >= target_budget-1e-7 and n[-1]-2 < target_budget+1e-7,
                    'Target budget does not match the threshold')
            for i in range(K-1):
                index = clean[i]/(5+n[i]) + math.sqrt(math.log(T)/(5+n[i]))
                require(index <= z + math.sqrt(math.log(T)/T) + 1e-10,
                        'Non-target budget violates its certificate')
            require(integer(row['C1'], 'C1') == integer(row['C2'], 'C2') == 0,
                    'Unexpected online corruption stages for Ours')
            require(row['status'] == 'simulated', 'Ours lacks actual trajectory results')
        else:
            require(T0 == 0 and H == T and not any(n) and not any(clean),
                    'Xu must have no offline log')
            info = xu_budget.budget('UCB', K, T, float(row['Delta_K']))
            require(json.loads(row['budget_log_json']) == info, 'Xu budget log disagrees with A.2')
            require(row['cost_definition'] == 'scheduled corrupted rounds', 'Wrong Xu cost definition')
            for field, expected in [('C1', info['C1']), ('C2', info['C2']), ('cost', info['total'])]:
                require(row[field] == '' if expected is None else integer(row[field], field) == expected,
                        f'Inconsistent prescribed Xu {field}')
            require(row['status'] == ('simulated' if info['feasible'] else 'infeasible'),
                    'Xu feasibility does not match the prescribed budget')
            if not info['feasible']:
                require(row['infeasible_reason'] == info['reason'], 'Missing infeasibility reason')
                require(all(row[field] == '' for field in
                            ['online_counts_json', 'phase1_counts_json', 'phase2_counts_json',
                             'post_attack_counts_json', 'target_ratio', 'online_ratio',
                             'post_attack_target_ratio']), 'Fabricated infeasible trajectory results')
                continue
        require(row['infeasible_reason'] == '', 'A simulated run has an infeasibility reason')
        counts = vector(row, 'online_counts_json')
        phases = [vector(row, field) for field in
                  ['phase1_counts_json', 'phase2_counts_json', 'post_attack_counts_json']]
        c1, c2 = integer(row['C1'], 'C1'), integer(row['C2'], 'C2')
        require(sum(counts) == H and [sum(stage) for stage in phases] == [c1, c2, H-c1-c2] and
                [sum(stage[i] for stage in phases) for i in range(K)] == counts,
                'Inconsistent online or stage counts')
        require(integer(row['post_attack_H'], 'post_attack_H') == H-c1-c2,
                'Inconsistent post-attack horizon')
        for field, expected in [('target_ratio', ((5 if ours else 0)+n[-1]+counts[-1])/T),
                                ('online_ratio', counts[-1]/H),
                                ('post_attack_target_ratio', phases[-1][-1]/(H-c1-c2))]:
            close(row[field], expected, field)
        modifications = integer(row['modifications'], 'modifications')
        magnitude = float(row['magnitude'])
        require(modifications <= c1+c2 and math.isfinite(magnitude) and
                0 <= magnitude <= modifications+1e-7, 'Invalid corruption diagnostics')
        require(math.isfinite(float(row['trajectory_seconds'])) and
                float(row['trajectory_seconds']) > 0, 'Missing actual trajectory runtime')
    expected_groups = {(method, multiplier) for method in METHODS for multiplier in MULTIPLIERS}
    require(set(groups) == expected_groups and
            all(sorted(repeats) == list(range(REPEATS)) for repeats in groups.values()),
            'Missing or duplicate configurations/repeats')
    return rows, manifest


def segments(rows, method, metric):
    """Keep infeasible grid points as breaks rather than joining across them."""
    grouped = defaultdict(list)
    for row in rows:
        if row['method'] == method:
            grouped[float(row['multiplier'])].append(row)
    result, current = [], []
    for multiplier in MULTIPLIERS:
        group = grouped.get(multiplier, [])
        if not group or any(row['status'] != 'simulated' for row in group):
            if current:
                result.append(current)
                current = []
            continue
        values = [float(row[metric]) for row in group]
        require(len(values) == REPEATS and all(math.isfinite(value) for value in values),
                'Cannot aggregate incomplete or non-finite trajectories')
        current.append((multiplier, statistics.mean(values), statistics.stdev(values)))
    if current:
        result.append(current)
    return result


def panel(rows, ratio=False):
    series = [('Ours', 'cost', 'total'), ('Xu2021', 'cost', 'xu')]
    if ratio:
        series = [('Ours', 'target_ratio', 'total'), ('Ours', 'online_ratio', 'online'),
                  ('Xu2021', 'target_ratio', 'xu')]
    plots = ''.join(curve(part, key, f'ucbxu{metric}{key}{number}', ratio=ratio)
                    for method, metric, key in series
                    for number, part in enumerate(segments(rows, method, metric)))
    ylabel = 'Target-arm Ratio' if ratio else 'Attack Cost'
    limits = 'ymin=0.3, ymax=1, ytick={0.3,0.5,0.7,0.9,1},' if ratio else 'ymin=0,'
    return rf'''\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.97\linewidth, height=4.1cm,
    xlabel={{$\Delta_K/(S_T/T)$}}, ylabel={{{ylabel}}},
    label style={{font=\small}}, tick label style={{font=\scriptsize}},
    tick align=outside, tick pos=left, axis on top, grid=none,
    xmode=normal, ymode=normal, xmin=0, xmax=6,
    xtick={{0,1,2,3,4,5,6}}, {limits}
]
{plots}
\end{{axis}}
\end{{tikzpicture}}'''


def generate(rows, manifest):
    cost_legend = legend([('total', 'Ours'), ('xu', 'Xu (2021)')])
    ratio_legend = (legend([('total', r'Ours $N_K/T$'), ('online', r'Ours $N_K^{\on}/H$')])
                    + legend([('xu', r'Xu (2021) $N_K/T$')]))
    generator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return rf'''% Generated by generate_ucb_xu_figure.py; verified sequential comparison.
% Results SHA256: {manifest['results_sha256']}
% Generator SHA256: {generator_hash}
\begin{{figure}}[t]
\centering
\begin{{minipage}}[t]{{0.49\linewidth}}
\centering
{panel(rows)}
\par\vspace{{2pt}}{{\small (a) Attack Cost\\vs. $\Delta_K/(S_T/T)$}}
{cost_legend}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\linewidth}}
\centering
{panel(rows, ratio=True)}
\par\vspace{{2pt}}{{\small (b) Target-arm Selection Ratio\\vs. $\Delta_K/(S_T/T)$}}
{ratio_legend}
\end{{minipage}}
\caption{{Comparison with \citet{{xu2021observation}} for UCB.}}
\label{{fig:ucb-xu-gap}}
\end{{figure}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    rows, manifest = read_results(args.data)
    content = generate(rows, manifest)
    for directory in [ROOT / 'fig', ROOT.parent / 'fig']:
        directory.mkdir(exist_ok=True)
        (directory / 'ucb_xu_gap.tex').write_text(content)
    print(f'Generated UCB/Xu comparison from {len(rows)} verified records in both figure directories.')


if __name__ == '__main__':
    main()
