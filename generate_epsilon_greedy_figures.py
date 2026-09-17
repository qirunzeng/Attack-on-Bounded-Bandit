"""Render the epsilon-greedy paper figure from complete, verified trajectories."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from generate_paper_figures import curve, figure, legend, line_style

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/synthetic/epsilon-greedy'
COMPARISON_DATA = ROOT / 'results/synthetic/epsilon-greedy-comparison'
VERSION = 'epsilon-greedy-boundary-v1'
COMPARISON_VERSION = 'epsilon-greedy-comparison-v1'
T_GRID = [100_000, 250_000, 1_000_000, 2_500_000, 10_000_000]
COST_T_GRID = [100_000, 250_000, 500_000, 1_000_000, 2_000_000,
               2_500_000, 4_000_000, 5_000_000, 7_500_000, 10_000_000]
K_GRID = [5, 10, 15, 20, 25]
K_STYLES = list(zip(K_GRID, ['total', 'heuristic', 'online', 'xu', 'direct']))
HORIZON_GRID = list(range(100_000, 1_000_001, 100_000))
ARM_GRID = [5, 10, 15, 20, 30, 40, 50]


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
            f'Inconsistent {name}: {actual}, expected {expected}')


def harmonic(n):
    """H_n, without an O(T) array or any dataset imports."""
    if n < 50:
        return math.fsum(1 / s for s in range(1, n + 1))
    inverse = 1 / n
    return (math.log(n) + 0.5772156649015328606 + inverse / 2
            - inverse**2 / 12 + inverse**4 / 120 - inverse**6 / 252
            + inverse**8 / 240)


def read_results(data=DATA):
    """Reject smoke, incomplete, stale, and internally inconsistent runs."""
    data = Path(data)
    manifest = json.loads((data / 'manifest.json').read_text())
    require(manifest['version'] == VERSION and
            manifest['dataset'] == 'Controlled Bernoulli' and
            manifest['completed'] is True and manifest['status'] == 'completed' and
            manifest['smoke'] is False and manifest['repeats'] == 50,
            'Paper figures require a complete fifty-repeat, non-smoke run')
    require(manifest['T_grid'] == T_GRID and manifest['K_grid'] == K_GRID,
            'Incomplete horizon or arm-count grid')
    configs = {(T, K) for K in K_GRID for T in T_GRID}
    require(len(manifest['configurations']) == len(configs) and
            {(entry['T'], entry['K']) for entry in manifest['configurations']} == configs,
            'Inconsistent configuration grid')
    close(manifest['c'], 1., 'exploration coefficient')
    close(manifest['delta'], .05, 'delta')
    require(manifest['N0_i'] == 0, 'The offline log must be entirely injected')
    require(manifest['simulation'] ==
            'Actual sequential decisions; NumPy PCG64; first-max greedy ties' and
            manifest['schedule'] == 'epsilon_s=min(1,c*K/s), s=T0+1,...,T; T0=C',
            'Unexpected simulation or internal exploration clock')
    sources = manifest['source_sha256']
    require('epsilon_greedy_experiment.py' in sources, 'Missing simulation provenance')
    for name, expected in sources.items():
        source = (ROOT / name).resolve()
        require(source.is_relative_to(ROOT) and source.is_file(),
                f'Invalid provenance source: {name}')
        require(hashlib.sha256(source.read_bytes()).hexdigest() == expected,
                f'Simulation source changed since this run: {name}')
    require(manifest['results_file'] == 'epsilon_greedy_results.csv',
            'Unexpected results filename')
    raw = (data / manifest['results_file']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest['results_sha256'],
            'Results do not match their provenance hash')
    rows = list(csv.DictReader(raw.decode().splitlines()))
    require(len(rows) == manifest['rows'] == 50 * len(configs), 'Incomplete results')
    groups = defaultdict(list)
    seeds = set()
    for row in rows:
        require(row['dataset'] == 'Controlled Bernoulli' and row['setting'] == VERSION and
                row['learner'] == 'epsilon-greedy' and row['method'] == 'Ours' and
                row['status'] == 'simulated', 'Mixed or non-simulated results')
        T, K = integer(row['T'], 'T'), integer(row['K'], 'K')
        require((T, K) in configs, 'Unexpected configuration')
        repeat = integer(row['repeat'], 'repeat')
        groups[T, K].append(repeat)
        seed = integer(row['seed'], 'seed')
        require(seed not in seeds, 'Repeated trajectory seed')
        seeds.add(seed)
        close(row['c'], 1., 'row exploration coefficient')
        close(row['delta'], .05, 'row delta')
        expected = 1 + harmonic(T) - harmonic(K)
        close(row['expected_exploration_per_arm'], expected, 'exploration expectation')
        log_confidence = math.log((K - 1) / .05)
        B = math.ceil(expected + math.sqrt(2 * expected * log_confidence)
                      + log_confidence / 3)
        require(integer(row['B'], 'B') == B, 'Incorrect exploration bound')
        n = [integer(v, 'injected count') for v in json.loads(row['allocation_json'])]
        counts = [integer(v, 'online count') for v in json.loads(row['online_counts_json'])]
        exploration = [integer(v, 'exploratory count')
                       for v in json.loads(row['exploratory_counts_json'])]
        means = json.loads(row['means_json'])
        require(len(n) == len(counts) == len(exploration) == len(means) == K,
                'Invalid arm-vector dimensions')
        target = math.ceil(math.sqrt(T * (K - 1) * B)) + 1
        competitor = math.ceil(math.sqrt(T * B / (K - 1)))
        require(n == [competitor] * (K - 1) + [target], 'Incorrect attack allocation')
        cost = integer(row['cost'], 'cost')
        H, T0 = integer(row['H'], 'H'), integer(row['T0'], 'T0')
        require(cost == sum(n) == T0 and T0 + H == T and H > 0 and
                sum(counts) == H, 'Inconsistent cost or online horizon')
        close(row['expected_non_target_exploratory_pulls'],
              (K - 1) * (harmonic(T) - harmonic(T0)),
              'post-warm-start exploration expectation')
        for key in ['C', 'cost']:
            require(integer(row[key], key) == cost, 'Inconsistent total budget')
        for key in ['nK', 'target_cost']:
            require(integer(row[key], key) == target, 'Inconsistent target budget')
        require(integer(row['non_target_cost'], 'non-target cost') == sum(n[:-1]),
                'Inconsistent non-target budget')
        for key, value in [('target_budget_fraction', target / cost),
                           ('non_target_budget_fraction', sum(n[:-1]) / cost),
                           ('normalized_cost', cost / (2 * math.sqrt(T * (K - 1) * B))),
                           ('cost_over_sqrt_TKlogT', cost / math.sqrt(T * K * math.log(T))),
                           ('online_ratio', counts[-1] / H)]:
            close(row[key], value, key)
        radius = math.sqrt((K - 1) * B / T)
        close(row['r_eg'], radius, 'boundary scale')
        close(row['target_mean'], radius / math.log(T), 'positive target mean')
        close(row['boundary_ratio'], 1 / math.log(T), 'normalized gap')
        require(0 < float(row['target_mean']) < 1, 'Target must have positive Bernoulli mean')
        for i, mean in enumerate(means[:-1]):
            close(mean, .2 + .6 * i / (K - 2), 'non-target Bernoulli mean')
        close(means[-1], float(row['target_mean']), 'target Bernoulli mean')
        require(all(e <= q for e, q in zip(exploration, counts)),
                'Exploratory counts exceed online counts')
        q, q_exp = sum(counts[:-1]), sum(exploration[:-1])
        q_greedy = q - q_exp
        for key, value in [('xi', (K - 1) * B),
                           ('non_target_online_pulls', q),
                           ('exploratory_non_target_pulls', q_exp),
                           ('greedy_non_target_pulls', q_greedy),
                           ('success', int(q <= (K - 1) * B)),
                           ('individual_exploration_bound', int(max(exploration[:-1]) <= B)),
                           ('all_greedy_target', int(q_greedy == 0))]:
            require(integer(row[key], key) == value, f'Inconsistent {key}')
        require(math.isfinite(float(row['trajectory_seconds'])) and
                float(row['trajectory_seconds']) > 0, 'Missing sequential runtime')
    require(set(groups) == configs and
            all(sorted(repeats) == list(range(50)) for repeats in groups.values()),
            'Missing or duplicate configurations/repeats')
    return rows, manifest


def points(rows, K, metric):
    grouped = defaultdict(list)
    for row in rows:
        if int(row['K']) == K:
            grouped[int(row['T'])].append(float(row[metric]))
    return [(T, statistics.mean(values), statistics.stdev(values))
            for T, values in sorted(grouped.items())]


def allocation(T, K):
    """Exact integer allocation from the known schedule and confidence level."""
    expected = 1 + harmonic(T) - harmonic(K)
    confidence = math.log((K - 1) / .05)
    B = math.ceil(expected + math.sqrt(2 * expected * confidence) + confidence / 3)
    target = math.isqrt(T * (K - 1) * B)
    target += int(target * target < T * (K - 1) * B) + 1
    competitor = math.isqrt(T * B // (K - 1))
    competitor += int(competitor * competitor * (K - 1) < T * B)
    cost = target + (K - 1) * competitor
    require(0 < cost < T, 'Computed allocation leaves no online horizon')
    return B, target, competitor, cost


def read_comparison_results(data=COMPARISON_DATA):
    """Check costs, ratios, initialization, and suppression on each trajectory."""
    data = Path(data)
    manifest = json.loads((data / 'manifest.json').read_text())
    require(manifest['version'] == COMPARISON_VERSION and
            manifest['dataset'] == 'Controlled Bernoulli' and
            manifest['completed'] is True and manifest['status'] == 'completed' and
            manifest['smoke'] is False and manifest['repeats'] == 50,
            'Comparison figures require a complete fifty-repeat, non-smoke run')
    require(manifest['horizon_grid'] == HORIZON_GRID and manifest['arm_grid'] == ARM_GRID and
            manifest['horizon_K'] == 10 and manifest['arm_T'] == 1_000_000,
            'Incomplete comparison sweep')
    close(manifest['c'], 1., 'comparison exploration coefficient')
    close(manifest['delta'], .05, 'comparison delta')
    configs = {(T, 10) for T in HORIZON_GRID} | {(1_000_000, K) for K in ARM_GRID}
    require(len(manifest['configurations']) == len(configs) and
            {(entry['T'], entry['K']) for entry in manifest['configurations']} == configs,
            'Inconsistent comparison configuration grid')
    sources = manifest['source_sha256']
    require('epsilon_greedy_experiment.py' in sources and
            'epsilon_greedy_comparison.py' in sources, 'Missing comparison provenance')
    for name, expected in sources.items():
        source = (ROOT / name).resolve()
        require(source.is_relative_to(ROOT) and source.is_file(),
                f'Invalid comparison provenance source: {name}')
        require(hashlib.sha256(source.read_bytes()).hexdigest() == expected,
                f'Comparison source changed since this run: {name}')
    require(manifest['results_file'] == 'comparison_results.csv', 'Unexpected comparison filename')
    raw = (data / manifest['results_file']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest['results_sha256'],
            'Comparison results do not match their provenance hash')
    rows = list(csv.DictReader(raw.decode().splitlines()))
    require(len(rows) == manifest['rows'] == len(configs) * 50 * 2, 'Incomplete comparison results')
    groups = defaultdict(list)
    for row in rows:
        require(row['dataset'] == 'Controlled Bernoulli' and
                row['setting'] == COMPARISON_VERSION and row['learner'] == 'epsilon-greedy' and
                row['method'] in {'Ours', 'Clipped Suppression'} and row['status'] == 'simulated',
                'Mixed or non-simulated comparison results')
        T, K = integer(row['T'], 'T'), integer(row['K'], 'K')
        require((T, K) in configs, 'Unexpected comparison configuration')
        groups[T, K, row['method']].append(integer(row['repeat'], 'repeat'))
        integer(row['seed'], 'seed')
        close(row['c'], 1., 'row exploration coefficient')
        close(row['delta'], .05, 'row delta')
        B, target, competitor, attack_cost = allocation(T, K)
        require(integer(row['B'], 'B') == B, 'Incorrect comparison exploration bound')
        radius = math.sqrt((K - 1) * B / T)
        close(row['r_eg'], radius, 'comparison boundary scale')
        close(row['target_mean'], radius / math.log(T), 'comparison positive target mean')
        close(row['boundary_ratio'], 1 / math.log(T), 'comparison normalized gap')
        means = json.loads(row['means_json'])
        require(len(means) == K, 'Invalid comparison mean-vector dimensions')
        for i, mean in enumerate(means[:-1]):
            close(mean, .2 + .6 * i / (K - 2), 'comparison non-target Bernoulli mean')
        close(means[-1], radius / math.log(T), 'comparison target Bernoulli mean')
        vectors = {}
        for key in ['allocation_json', 'online_counts_json', 'exploratory_counts_json',
                    'initialization_counts_json']:
            vectors[key] = [integer(v, key) for v in json.loads(row[key])]
            require(len(vectors[key]) == K, 'Invalid comparison count-vector dimensions')
        n, online, exploration, initialization = (
            vectors[key] for key in ['allocation_json', 'online_counts_json',
                                    'exploratory_counts_json', 'initialization_counts_json'])
        cost, T0, H = (integer(row[key], key) for key in ['cost', 'T0', 'H'])
        require(integer(row['C'], 'C') == cost and integer(row['nK'], 'nK') == n[-1],
                'Inconsistent comparison budget')
        require(H > 0 and T0 + H == T and sum(online) == H,
                'Inconsistent comparison online horizon')
        require(all(e + init <= q for e, init, q in zip(exploration, initialization, online)),
                'Exploration and initialization exceed online counts')
        q = sum(online[:-1])
        greedy = q - sum(exploration[:-1]) - sum(initialization[:-1])
        require(integer(row['non_target_online_pulls'], 'non-target pulls') == q and
                integer(row['greedy_non_target_pulls'], 'greedy non-target pulls') == greedy and
                integer(row['exploratory_non_target_pulls'], 'exploratory non-target pulls') == sum(exploration[:-1]) and
                integer(row['initialization_non_target_pulls'], 'initialization non-target pulls') == sum(initialization[:-1]),
                'Inconsistent comparison decision accounting')
        modified = integer(row['modified_reward_count'], 'modified reward count')
        if row['method'] == 'Ours':
            require(n == [competitor] * (K - 1) + [target] and cost == attack_cost == T0 and
                    initialization == [0] * K and modified == 0 and
                    row['cost_definition'] == 'injected samples' and
                    all(row[key] == '' for key in ['reward_sums_json', 'observed_reward_sums_json',
                                                  'target_reward_sum']),
                    'Inconsistent offline attack accounting')
        else:
            rewards = [integer(v, 'genuine reward sum') for v in json.loads(row['reward_sums_json'])]
            observed = [integer(v, 'observed reward sum') for v in json.loads(row['observed_reward_sums_json'])]
            require(len(rewards) == len(observed) == K and
                    all(seen <= genuine <= q for seen, genuine, q in zip(observed, rewards, online)),
                    'Invalid genuine or observed Bernoulli reward sums')
            require(integer(row['target_reward_sum'], 'target reward sum') == rewards[-1] == observed[-1],
                    'Target feedback was modified')
            require(n == [0] * K and T0 == 0 and H == T and initialization == [1] * K and
                    cost == q and observed[:-1] == [0] * (K - 1) and
                    modified == sum(rewards[:-1]) and
                    row['cost_definition'] == 'suppressed non-target rounds (including 0->0)',
                    'Inconsistent clipped-suppression accounting')
        close(row['total_target_ratio'], (n[-1] + online[-1]) / T, 'total target ratio')
        close(row['online_ratio'], online[-1] / H, 'online target ratio')
        require(math.isfinite(float(row['trajectory_seconds'])) and
                float(row['trajectory_seconds']) > 0, 'Missing comparison sequential runtime')
    require(set(groups) == {(T, K, method) for T, K in configs
                            for method in ['Ours', 'Clipped Suppression']} and
            all(sorted(repeats) == list(range(50)) for repeats in groups.values()),
            'Missing or duplicate comparison configurations/repeats')
    return rows, manifest


def deterministic_cost_points(rows, K):
    """Compute allocation costs on a denser grid; no additional trajectories."""
    values = [(T, allocation(T, K)[-1], 0) for T in COST_T_GRID]
    computed = {T: cost for T, cost, _ in values}
    sampled = points(rows, K, 'cost')
    require([T for T, _, _ in sampled] == T_GRID and
            all(sd == 0 and cost == computed[T] for T, cost, sd in sampled),
            'Computed allocation cost disagrees with saved trajectories')
    return values


def deterministic_curve(values, style):
    """A deterministic count has no sampling band."""
    require(all(sd == 0 for _, _, sd in values), 'A deterministic budget varied across repeats')
    coordinates = ' '.join(f'({T},{value:.12g})' for T, value, _ in values)
    return rf'\addplot[{line_style(style)}] coordinates {{{coordinates}}};' + '\n'


def panel(plots, ylabel, ymax):
    return rf'''\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.97\linewidth, height=4.1cm,
    xlabel={{Horizon $T$}}, ylabel={{{ylabel}}},
    label style={{font=\small}}, tick label style={{font=\scriptsize}},
    tick align=outside, tick pos=left, axis on top, grid=none,
    xmode=normal, ymode=normal,
    xmin=100000, xmax=10000000,
    xtick={{100000,2500000,5000000,7500000,10000000}},
    scaled x ticks=base 10:-6,
    ymin=0, ymax={ymax:.12g},
]
{plots}\end{{axis}}
\end{{tikzpicture}}'''


def provenance(content, manifest, source):
    header = ('% Generated by generate_epsilon_greedy_figures.py.\n'
              f'% {source}\n'
              f"% Results SHA-256: {manifest['results_sha256']}\n")
    for filename in ['generate_epsilon_greedy_figures.py', 'generate_paper_figures.py']:
        header += f'% {filename} SHA-256: {hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()}\n'
    return header + content.split('\n', 1)[1].replace(
        r'\begin{figure}[t]', r'\begin{figure}[H]', 1)


def theory_panel(plots, ylabel, ymin, ymax, ticks):
    return panel(plots, ylabel, ymax).replace(
        f'ymin=0, ymax={ymax:.12g},',
        f'ymin={ymin:.12g}, ymax={ymax:.12g}, scaled y ticks=false, {ticks}')


def generate_theory(rows, manifest):
    costs = {K: deterministic_cost_points(rows, K) for K in K_GRID}
    normalized = {K: [(T, cost / (2 * math.sqrt(T * (K - 1) * allocation(T, K)[0])), 0)
                      for T, cost, _ in values] for K, values in costs.items()}
    reference = (r'\addplot[expdirect, opacity=0.65, thin, solid, no marks, forget plot] '
                 r'coordinates {(100000,%s) (10000000,%s)};' + '\n')
    cost_plots = reference % ('1', '1') + ''.join(
        deterministic_curve(normalized[K], style) for K, style in K_STYLES)
    cost_legend = legend([(style, rf'$K={K}$') for K, style in K_STYLES])
    cost_legend = cost_legend.replace(
        r'{\scriptsize\begin{tabular}',
        r'{\scriptsize\setlength{\tabcolsep}{1pt}\begin{tabular}'
    ).replace('(0,0)--(.48,0)', '(0,0)--(.30,0)').replace(
        'coordinates {(.24,0)}', 'coordinates {(.15,0)}')
    left = theory_panel(cost_plots, r'$\mathcal C/(2\sqrt{T(K-1)B})$', .95, 1.05,
                        'ytick={0.95,1,1.05},') + cost_legend
    target, non_target = [], []
    for T, cost, _ in costs[10]:
        _, nK, ni, _ = allocation(T, 10)
        target.append((T, nK / cost, 0))
        non_target.append((T, 9 * ni / cost, 0))
    split_plots = (reference % ('0.5', '0.5') + deterministic_curve(target, 'target')
                   + deterministic_curve(non_target, 'non_target'))
    right = theory_panel(split_plots, 'Budget Share', .3, .7,
                         r'ytick={0.5}, yticklabels={$1/2$},')
    right += legend([('target', r'Target $n_K/\mathcal C$'),
                     ('non_target', r'Non-target $\sum_{i<K}n_i/\mathcal C$')])
    content = figure(left, right,
                     [r'Normalized Attack Cost vs. Horizon $T$', r'Budget Share vs. Horizon $T$'], '',
                     r'Near-Boundary Cost and Allocation for $\epsilon$-greedy.', 'fig:eg-theory-alignment')
    return provenance(content, manifest,
                      'Deterministic allocations on ten horizons; shared five horizons checked against saved trajectories.')


def comparison_points(rows, sweep, method, metric):
    grouped = defaultdict(list)
    for row in rows:
        if row['method'] != method:
            continue
        T, K = int(row['T']), int(row['K'])
        if (sweep == 'horizon' and K == 10) or (sweep == 'arms' and T == 1_000_000):
            grouped[T if sweep == 'horizon' else K].append(float(row[metric]))
    return [(x, statistics.mean(values), statistics.stdev(values))
            for x, values in sorted(grouped.items())]


def comparison_panel(rows, sweep, ratio=False):
    if ratio:
        series = [('Ours', 'total_target_ratio', 'total'),
                  ('Ours', 'online_ratio', 'online'),
                  ('Clipped Suppression', 'total_target_ratio', 'heuristic')]
    else:
        series = [('Clipped Suppression', 'cost', 'heuristic'), ('Ours', 'cost', 'total')]
    plots = ''
    for method, metric, style in series:
        values = comparison_points(rows, sweep, method, metric)
        plots += (deterministic_curve(values, style) if not ratio and method == 'Ours' else
                  curve(values, style, 'eg' + sweep + style + metric, ratio=ratio))
    xlabel = r'Horizon $T$' if sweep == 'horizon' else r'Number of Arms $K$'
    ylabel = 'Target-arm Ratio' if ratio else 'Attack Cost'
    options = 'xmode=normal, ymode=normal, ymin=0,'
    if ratio:
        options += 'ymax=1.06, ytick={0,0.25,0.5,0.75,1},'
    if sweep == 'horizon':
        options += ('xmin=100000, xmax=1000000, xtick={100000,400000,700000,1000000}, '
                    'scaled x ticks=base 10:-6,')
    else:
        options += 'xmin=5, xmax=50, xtick={5,10,20,30,40,50},'
    return rf'''\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.97\linewidth, height=4.1cm,
    xlabel={{{xlabel}}}, ylabel={{{ylabel}}},
    label style={{font=\small}}, tick label style={{font=\scriptsize}},
    tick align=outside, tick pos=left, axis on top, grid=none,
    {options}
]
{plots}\end{{axis}}
\end{{tikzpicture}}'''


def generate(rows, manifest, comparison_rows, comparison_manifest):
    costs = figure(comparison_panel(comparison_rows, 'horizon'),
                   comparison_panel(comparison_rows, 'arms'),
                   [r'Attack Cost vs. Horizon $T$', r'Attack Cost vs. Number of Arms $K$'],
                   legend([('heuristic', 'Clipped Suppression'), ('total', 'Total Attack Cost')]),
                   r'Attack Cost for $\epsilon$-greedy.', 'fig:eg_cost_experiments')
    ratios = figure(comparison_panel(comparison_rows, 'horizon', True),
                    comparison_panel(comparison_rows, 'arms', True),
                    [r'Target-arm Selection Ratio\\vs. Horizon $T$',
                     r'Target-arm Selection Ratio\\vs. Number of Arms $K$'],
                    legend([('total', r'$N_K/T$'), ('online', r'$N_K^{\on}/H$'),
                            ('heuristic', r'Clipped Suppression $N_K/T$')]),
                    r'Target-arm Selection Ratio for $\epsilon$-greedy.', 'fig:eg_ratio_experiments')
    source = 'Verified sequential controlled Bernoulli comparison; mean and sample standard deviation.'
    return {'eg_cost_experiments.tex': provenance(costs, comparison_manifest, source),
            'eg_ratio_experiments.tex': provenance(ratios, comparison_manifest, source),
            'eg_theory_alignment.tex': generate_theory(rows, manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--comparison-data', type=Path, default=COMPARISON_DATA)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    rows, manifest = read_results(args.data)
    comparison_rows, comparison_manifest = read_comparison_results(args.comparison_data)
    if args.check_only:
        print(f'Validated {len(rows)} boundary and {len(comparison_rows)} comparison trajectories.')
        return
    outputs = generate(rows, manifest, comparison_rows, comparison_manifest)
    for directory in [ROOT / 'fig', ROOT.parent / 'fig']:
        directory.mkdir(exist_ok=True)
        for name, content in outputs.items():
            (directory / name).write_text(content)
    print('Generated three epsilon-greedy figures in both figure directories.')


if __name__ == '__main__':
    main()
