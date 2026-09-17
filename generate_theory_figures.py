"""Generate the two theory-alignment figures from verified sequential runs."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from generate_paper_figures import curve, figure, legend

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/ml-25m/theory-alignment-fixed-k'
VERSION = 'near-boundary-fixed-k-v2'
LEGACY_VERSION = 'near-boundary-clean-log-v1'
T_GRID = [100_000, 250_000, 1_000_000, 2_500_000, 10_000_000]
K_GRID = [5, 10, 15, 20, 25]
K_STYLES = list(zip(K_GRID, ['total', 'heuristic', 'online', 'xu', 'direct']))
LEARNERS = ['UCB', 'TS']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    require(math.isfinite(float(actual)) and
            math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12),
            f'Inconsistent {name}: {actual}, expected {expected}')


def integer(value, name):
    number = float(value)
    require(math.isfinite(number) and number >= 0 and number.is_integer(),
            f'Invalid {name}: {value}')
    return int(number)


def read_results(data=DATA, *, allow_legacy=False):
    """Reject partial runs and check every displayed value against saved counts."""
    data = Path(data)
    manifest = json.loads((data / 'manifest.json').read_text())
    version = manifest['version']
    repeats = 10 if allow_legacy and version == LEGACY_VERSION else 50
    require((version == VERSION or (allow_legacy and version == LEGACY_VERSION)) and
            manifest['dataset'] == 'MovieLens-25M' and
            manifest['repeats'] == repeats and manifest['smoke'] is False,
            'Theory figures require a complete fifty-repeat, non-smoke run; legacy reading is explicit')
    require(manifest['simulation'] ==
            'actual sequential UCB and Gaussian TS, Bernoulli target',
            'Figures require actual sequential trajectories')
    require(manifest['target_information'] ==
            'unknown mean; clean-log confidence floor only',
            'Unexpected target information')
    require(manifest['target_mean_rule'] == 'mu_ML * sqrt(100000 / T)',
            'Unexpected near-boundary regime')
    require(manifest['T_grid'] == T_GRID, 'Incomplete horizon grid')
    close(manifest['sigma'], .5, 'sigma')
    close(manifest['delta'], .05, 'delta')
    require(manifest['N0_i'] == 5, 'Unexpected clean-log size')
    if version == VERSION:
        require(manifest.get('K_grid') == K_GRID, 'Incomplete arm-count grid')
        require(manifest.get('arm_count_rule') == 'fixed K in K_grid' and
                'growing_K_rule' not in manifest, 'Unexpected fixed-K regime')
        configs = [('fixed', T, K) for K in K_GRID for T in T_GRID]
    else:
        require(manifest.get('growing_K_rule') ==
                'round(10 * (T / 100000)**(1/4))', 'Unexpected legacy regime')
        configs = [(r, T, 10 if r == 'fixed' else round(10 * (T / T_GRID[0])**.25))
                   for r in ['fixed', 'growing'] for T in T_GRID]
    require(manifest['configurations'] ==
            [dict(regime=r, T=T, K=K) for r, T, K in configs],
            'Incomplete or inconsistent configuration grid')
    raw = (data / 'theory_results.csv').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest['results_sha256'],
            'Results do not match the provenance hash')
    rows = list(csv.DictReader(raw.decode().splitlines()))
    require(len(rows) == manifest['rows'] == repeats * len(configs) * len(LEARNERS),
            'Incomplete results')
    groups = defaultdict(list)
    for row in rows:
        require(row['dataset'] == 'MovieLens-25M' and row['setting'] == version and
                row['method'] == 'Ours', 'Mixed experimental settings')
        require(row['status'] == 'simulated', 'Non-simulated result cannot be plotted')
        require(row['target_feedback'] ==
                'Bernoulli; same mean in clean and online rewards' and
                row['lower_bound_source'] ==
                'clean log: max(0, empirical mean - 2 beta)',
                'Unexpected feedback or target-floor model')
        T, K = integer(row['T'], 'T'), integer(row['K'], 'K')
        learner, regime = row['learner'], row['regime']
        require(learner in LEARNERS and (regime, T, K) in configs,
                'Unexpected learner or configuration')
        groups[regime, T, K, learner].append(integer(row['repeat'], 'repeat'))
        n = [integer(v, 'injected count') for v in json.loads(row['allocation_json'])]
        counts = [integer(v, 'online count') for v in json.loads(row['online_counts_json'])]
        clean = [integer(v, 'clean Bernoulli sum') for v in json.loads(row['clean_sum_json'])]
        require(len(n) == len(counts) == len(clean) == K and max(clean) <= 5,
                'Invalid count-vector dimensions or clean rewards')
        H, cost = integer(row['H'], 'H'), integer(row['cost'], 'cost')
        require(H > 0 and cost > 0 and cost == sum(n) and n[-1] > 0 and
                sum(counts) == H and H + cost + 5 * K == T,
                'Inconsistent realized cost or online horizon')
        require(integer(row['T0'], 'T0') == cost + 5 * K,
                'Inconsistent attacked start time')
        require(integer(row['target_cost'], 'target cost') == n[-1],
                'Inconsistent target budget')
        close(row['non_target_avg'], sum(n[:-1]) / (K - 1), 'non-target budget')
        q = sum(counts[:-1])
        require(integer(row['non_target_online_pulls'], 'non-target pulls') == q and
                integer(row['zero_non_target_pulls'], 'zero-pull flag') == int(q == 0),
                'Success indicator does not match the trajectory')
        close(row['online_ratio'], counts[-1] / H, 'online target ratio')
        require(float(row['trajectory_seconds']) > 0,
                'Missing sequential-trajectory runtime')
        sigma, delta = float(manifest['sigma']), float(manifest['delta'])
        beta = math.sqrt(2 * sigma**2 / 5 * math.log(math.pi**2 * K * 25 / (3 * delta)))
        floor = max(0., clean[-1] / 5 - 2 * beta)
        close(row['mu_minus_K'], floor, 'default target floor')
        require(floor <= float(row['z_star']) < 1., 'Invalid threshold')
        base = float(manifest['selected_arms'][str(K)]['means'][-1])
        close(row['base_target_mean'], base, 'MovieLens target mean')
        probability = base * math.sqrt(T_GRID[0] / T)
        close(row['target_mean'], probability, 'target mean')
        st = T**(2/3) * (K * math.log(T))**(1/3)
        close(row['boundary_ratio'], probability / (st / T), 'normalized target gap')
        coefficient = (9 * sigma**2 * math.log(T) if learner == 'UCB' else
                       2 * math.log(math.pi**2 * K * T**2 / (3 * delta)))
        scale = 3 * ((K - 1) * coefficient * T**2 / 4)**(1/3)
        close(row['Lambda_T'], scale, 'theoretical cost scale')
        close(row['normalized_cost'], cost / scale, 'normalized cost')
        close(row['target_budget_fraction'], n[-1] / cost, 'target budget fraction')
    expected = {(r, T, K, learner) for r, T, K in configs for learner in LEARNERS}
    require(set(groups) == expected and
            all(sorted(observed) == list(range(repeats)) for observed in groups.values()),
            'Missing or duplicate configurations/repeats')
    if version == LEGACY_VERSION:
        first = {(row['regime'], row['learner'], int(row['repeat'])): row
                 for row in rows if int(row['T']) == T_GRID[0]}
        for learner in LEARNERS:
            for repeat in range(repeats):
                fixed = first['fixed', learner, repeat]
                growing = first['growing', learner, repeat]
                require({k: v for k, v in fixed.items() if k != 'regime'} ==
                        {k: v for k, v in growing.items() if k != 'regime'},
                        'Coincident first points must reuse the same trajectory')
    return rows, manifest


def points(rows, learner, regime, metric, K=None):
    grouped = defaultdict(list)
    for row in rows:
        if ((row['learner'], row['regime']) == (learner, regime) and
                (K is None or int(row['K']) == K)):
            if metric == 'non_target_budget_fraction':
                n = json.loads(row['allocation_json'])
                value = sum(n[:-1]) / sum(n)
            else:
                value = float(row[metric])
            grouped[int(row['T'])].append(value)
    return [(T, statistics.mean(values), statistics.stdev(values))
            for T, values in sorted(grouped.items())]


def panel(rows, learner, metric):
    is_cost = metric == 'normalized_cost'
    is_split = metric == 'allocation_fractions'
    reference = 1. if is_cost else 2 / 3
    ylabel = (r'$\mathcal C/\Lambda_T$' if learner == 'UCB' else
              r'$\mathcal C/\Lambda_T^{\rm ts}$') if is_cost else r'$n_K/\mathcal C$'
    if is_split:
        ylabel = 'Budget Share'
        values = {'target': points(rows, learner, 'fixed', 'target_budget_fraction', K=10),
                  'non_target': points(rows, learner, 'fixed', 'non_target_budget_fraction', K=10)}
        styles = [('target', 'target'), ('non_target', 'non_target')]
    else:
        values = {K: points(rows, learner, 'fixed', metric, K=K) for K in K_GRID}
        styles = K_STYLES
    low = min([reference] + [m - s for group in values.values() for _, m, s in group])
    high = max([reference] + [m + s for group in values.values() for _, m, s in group])
    padding = max(.008 if is_cost else .005, (high - low) * .12)
    ymin, ymax = low - padding, high + padding
    ticks = ''
    if is_split:
        ymin, ymax = .3, .7
        ticks = r'ytick={0.333333333333,0.666666666667}, yticklabels={$1/3$,$2/3$},'
    ticks_line = f'    {ticks}\n' if ticks else ''
    curves = ''.join(curve(values[series], key, learner + str(series) + metric)
                     for series, key in styles)
    references = [2/3, 1/3] if is_split else [reference]
    reference_lines = '\n'.join(
        rf'\addplot[expdirect, opacity=0.65, thin, solid, no marks, forget plot] coordinates {{(100000,{value:.12g}) (10000000,{value:.12g})}};'
        for value in references)
    return rf'''\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.97\linewidth, height=4.1cm,
    xlabel={{Horizon $T$}}, ylabel={{{ylabel}}},
    label style={{font=\small}}, tick label style={{font=\scriptsize}},
    tick align=outside, tick pos=left, axis on top, grid=none,
    xmode=normal, ymode=normal,
    xmin=100000, xmax=10000000,
    xtick={{100000,2500000,5000000,7500000,10000000}},
    scaled x ticks=base 10:-6, scaled y ticks=false,
    ymin={ymin:.12g}, ymax={ymax:.12g},
{ticks_line}    yticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=3}},
]
% Horizontal lines mark asymptotic limits.
{reference_lines}
{curves}
\end{{axis}}
\end{{tikzpicture}}'''


def generate(rows):
    outputs = {}
    for learner in LEARNERS:
        cost_legend = legend([(style, rf'$K={K}$') for K, style in K_STYLES])
        # Fit five entries in one panel without shrinking the legend font.
        cost_legend = cost_legend.replace(
            r'{\scriptsize\begin{tabular}',
            r'{\scriptsize\setlength{\tabcolsep}{1pt}\begin{tabular}'
        ).replace('(0,0)--(.48,0)', '(0,0)--(.30,0)').replace(
            'coordinates {(.24,0)}', 'coordinates {(.15,0)}')
        left = panel(rows, learner, 'normalized_cost') + cost_legend
        right = panel(rows, learner, 'allocation_fractions') + legend([
            ('target', r'Target $n_K/\mathcal C$'),
            ('non_target', r'Non-target $\sum_{i<K}n_i/\mathcal C$')])
        labels = [r'Normalized Attack Cost vs. Horizon $T$', r'Budget Share vs. Horizon $T$']
        outputs[f'{learner.lower()}_theory_alignment.tex'] = figure(
            left, right, labels, '',
            f'Near-Boundary Cost and Allocation for {learner}.',
            f'fig:{learner.lower()}-theory-alignment').replace(
                'generate_paper_figures.py from verified MovieLens-25M trajectories.',
                'generate_theory_figures.py from verified near-boundary trajectories.')
    return outputs


def main():
    rows, _ = read_results()
    outputs = generate(rows)
    for directory in [ROOT / 'fig', ROOT.parent / 'fig']:
        directory.mkdir(exist_ok=True)
        for name, content in outputs.items():
            (directory / name).write_text(content)
    print('Generated two theory-alignment figures in both figure directories.')


if __name__ == '__main__':
    main()
