"""Single source of labels, colors, markers, statistics and paper figure layout."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
import xu_budget

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/ml-25m/paper'
# User-specified palette. Blue and green never share a panel; green is reserved.
PALETTE = {
    'myblue': (31, 119, 180),
    'myorange': (255, 127, 14),
    'mygreen': (44, 160, 44),
    'mypurple': (148, 103, 189),
    'myred': (214, 39, 40),
    'mybrown': (140, 86, 75),
}
STYLE = {
    'total': ('myblue', 'o', 'solid'),
    'heuristic': ('myred', 'square', 'solid'),
    'online': ('mypurple', 'triangle', 'solid'),
    'direct': ('mybrown', 'x', 'solid'),
    'xu': ('myorange', 'diamond', 'solid'),
    'target': ('myred', 'triangle', 'solid'),
    'non_target': ('mypurple', 'square', 'solid'),
}


def read_results():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    raw = (DATA / 'paper_results.csv').read_bytes()
    if manifest['dataset'] != 'MovieLens-25M' or manifest['smoke'] or manifest['repeats'] != 50:
        raise ValueError('Paper figures require a complete fifty-repeat MovieLens-25M run')
    if hashlib.sha256(raw).hexdigest() != manifest['results_sha256']:
        raise ValueError('Results do not match the provenance manifest')
    rows = list(csv.DictReader(raw.decode().splitlines()))
    if len(rows) != manifest['rows']:
        raise ValueError('Incomplete results')
    groups = defaultdict(list)
    for r in rows:
        if r['dataset'] != 'MovieLens-25M':
            raise ValueError('Mixed datasets')
        key = (r['sweep'], r['learner'], r['method'], r['T'], r['K'], r['multiplier'])
        groups[key].append(int(r['repeat']))
        offline = r['method'] in {'Ours', 'Direct'}
        if not offline:
            assert int(r['T0']) == 0 and int(r['H']) == int(r['T'])
            assert not any(json.loads(r['clean_sum_json']))
            assert not any(json.loads(r['allocation_json']))
        if r['method'] == 'Two-phase':
            info = json.loads(r['budget_log_json'])
            assert info == xu_budget.budget(r['learner'], int(r['K']), int(r['T']), float(r['Delta_K']))
            assert r['status'] == ('simulated' if info['feasible'] else 'infeasible')
            assert int(r['phase1_length']) == (info['C1'] or 0)
            assert int(r['phase2_length']) == (info['C2'] or 0)
            if info['total'] is None:
                assert r['cost'] == ''
            else:
                assert float(r['cost']) == info['C1'] + info['C2']
        if r['status'] != 'simulated':
            if r['method'] != 'Two-phase':
                raise ValueError(f'Unsimulated attack: {r}')
            continue
        T, K, H = int(r['T']), int(r['K']), int(r['H'])
        n = json.loads(r['allocation_json'])
        counts = json.loads(r['online_counts_json'])
        initial_per_arm = 5 if offline else 0
        assert sum(n) + initial_per_arm*K + H == T
        assert sum(counts) == H
        assert math.isclose(float(r['online_ratio']), counts[-1]/H)
        assert math.isclose(float(r['target_ratio']), (initial_per_arm+n[-1]+counts[-1])/T)
        if r['method'] in {'Ours', 'Direct'}:
            assert float(r['cost']) == sum(n)
            assert math.isclose(float(r['non_target_avg'])*(K-1)+float(r['target_cost']), sum(n))
        elif r['method'] == 'Clipped Suppression':
            assert r['cost_definition'] == 'suppressed non-target rounds (including 0->0)'
            assert float(r['cost']) == sum(counts[:-1])
        elif r['method'] == 'Two-phase':
            phases = [json.loads(r[k]) for k in ['phase1_counts_json', 'phase2_counts_json', 'post_attack_counts_json']]
            assert [sum(p) for p in phases] == [info['C1'], info['C2'], T-info['C1']-info['C2']]
            assert [sum(p[i] for p in phases) for i in range(K)] == counts
            assert math.isclose(float(r['post_attack_target_ratio']), phases[2][-1]/sum(phases[2]))
    for key, repeats in groups.items():
        if sorted(repeats) != list(range(50)):
            raise ValueError(f'Missing or duplicate repeats: {key}')
    expected = {(s, l, m, str(T), str(K), str(float(g)) if g is not None else '')
                for s, tasks, methods in [
                    ('horizon', [(T,10,None) for T in manifest['T_grid']], ['Ours','Clipped Suppression']),
                    ('arms', [(manifest['T_fixed'],K,None) for K in manifest['K_grid']], ['Ours','Clipped Suppression']),
                    ('gap', [(manifest['T_fixed'],10,g) for g in manifest['gaps']], ['Ours','Direct','Two-phase'])]
                for T,K,g in tasks for l in ['UCB','TS'] for m in methods}
    if set(groups) != expected:
        raise ValueError('Result grid does not match the manifest')
    if manifest.get('xu_budget_version') != xu_budget.VERSION:
        raise ValueError('Rerun the Xu baseline with the original learner settings and budgets')
    if manifest.get('gap_target_information') != 'known deterministic target reward; exact lower bound equals mu_K':
        raise ValueError('Rerun the known-deterministic-target sweep before generating these figures')
    for row in rows:
        if row['sweep'] == 'gap' and row['method'] in {'Ours', 'Direct'}:
            if not math.isclose(float(row['mu_minus_K']), float(row['Delta_K']), abs_tol=1e-12):
                raise ValueError('Gap design incorrectly uses a conservative target bound')
    if manifest.get('clipped_suppression_feedback') != 'all non-target rewards replaced by zero; targets unchanged':
        raise ValueError('Paper figures require verified zero-feedback Clipped Suppression results')
    return rows, manifest


def points(rows, sweep, learner, method, metric):
    grouped = defaultdict(list)
    xkey = {'horizon':'T', 'arms':'K', 'gap':'multiplier'}[sweep]
    for row in rows:
        if (row['sweep'], row['learner'], row['method']) != (sweep, learner, method):
            continue
        if row['status'] != 'simulated':
            continue
        if sweep == 'gap' and not 0 <= float(row[xkey]) <= 2:
            continue
        grouped[float(row[xkey])].append(float(row[metric]))
    return [(x, statistics.mean(y), statistics.stdev(y)) for x,y in sorted(grouped.items())]


def line_style(key):
    _, mark, dash = STYLE[key]
    return f'color=exp{key}, thick, {dash}, mark={mark}, mark size=2.4pt, mark options={{solid, fill=none}}'


def curve(values, key, name, ratio=False):
    def coords(which):
        result = []
        for x, mean, std in values:
            y = max(0., mean + which*std)
            if ratio:
                y = min(1., y)
            result.append(f'({x:.12g},{y:.12g})')
        return ' '.join(result)
    return rf'''
\addplot[name path={name}up, draw=none, forget plot] coordinates {{{coords(1)}}};
\addplot[name path={name}lo, draw=none, forget plot] coordinates {{{coords(-1)}}};
\addplot[exp{key}, fill opacity=0.12, draw=none, forget plot] fill between[of={name}up and {name}lo];
\addplot[{line_style(key)}] coordinates {{{coords(0)}}};
'''


def legend(items):
    entries = [rf'\tikz[baseline=-0.6ex]{{\draw[exp{key}, thick, {STYLE[key][2]}] (0,0)--(.48,0); \draw plot[mark={STYLE[key][1]}, mark size=2.4pt, mark options={{solid, fill=none, draw=exp{key}}}] coordinates {{(.24,0)}};}}\;{label}' for key,label in items]
    return '\n\\par\\vspace{2pt}\n{\\scriptsize\\begin{tabular}{'+'c'*len(items)+'}\n'+' & '.join(entries)+'\n\\end{tabular}}\n'


def panel(rows, learner, sweep, series, ratio=False):
    xlabel = {'horizon':r'Horizon $T$', 'arms':r'Number of Arms $K$',
              'gap':r'Target Gap $\Delta_K/(S_T/T)$'}[sweep]
    ylabel = 'Target-arm Ratio' if ratio else 'Attack Cost'
    options = 'xmode=normal, ymode=normal,'
    if ratio:
        options += 'ymin=0, ymax=1.06, ytick={0,0.25,0.5,0.75,1},'
    else:
        options += 'ymin=0,'
    if sweep == 'horizon':
        options += 'xmin=100000, xmax=1000000, xtick={100000,400000,700000,1000000}, scaled x ticks=base 10:-6,'
    elif sweep == 'arms':
        options += 'xmin=5, xmax=50, xtick={5,10,20,30,40,50},'
    else:
        options += 'xmin=0, xmax=2, xtick={0,0.5,1,1.5,2},'
    curves = ''.join(curve(points(rows,sweep,learner,method,metric),key,
                           learner+sweep+key+metric,ratio) for method,metric,key in series)
    return rf'''\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.97\linewidth, height=4.1cm,
    xlabel={{{xlabel}}}, ylabel={{{ylabel}}},
    label style={{font=\small}}, tick label style={{font=\scriptsize}},
    tick align=outside, tick pos=left, axis on top,
    grid=none,
    {options}
]
{curves}
\end{{axis}}
\end{{tikzpicture}}'''


def figure(left, right, labels, legend_tex, caption, ref):
    return rf'''% Generated by generate_paper_figures.py from verified MovieLens-25M trajectories.
\begin{{figure}}[t]
\centering
\begin{{minipage}}[t]{{0.49\linewidth}}
\centering
{left}
\par\vspace{{2pt}}{{\small (a) {labels[0]}}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\linewidth}}
\centering
{right}
\par\vspace{{2pt}}{{\small (b) {labels[1]}}}
\end{{minipage}}
{legend_tex}
\caption{{{caption}}}
\label{{{ref}}}
\end{{figure}}
'''


def main():
    rows, _ = read_results()
    outputs = {'experiment_style.tex':'% User-specified RGB palette; solid curves and shared semantic styles.\n'+
               ''.join(rf'\definecolor{{{name}}}{{RGB}}{{{",".join(map(str, rgb))}}}'+'\n'
                       for name,rgb in PALETTE.items())+
               ''.join(rf'\colorlet{{exp{k}}}{{{v[0]}}}'+'\n' for k,v in STYLE.items())}
    for learner in ['UCB','TS']:
        tag = learner.lower()
        costs = [('Clipped Suppression','cost','heuristic'),('Ours','cost','total')]
        ratios = [('Ours','target_ratio','total'),('Ours','online_ratio','online'),
                  ('Clipped Suppression','target_ratio','heuristic')]
        outputs[f'{tag}_cost_experiments.tex'] = figure(
            panel(rows,learner,'horizon',costs),panel(rows,learner,'arms',costs),
            [r'Attack Cost vs. Horizon $T$', r'Attack Cost vs. Number of Arms $K$'],
            legend([('heuristic','Clipped Suppression'),('total','Total Attack Cost')]),
            f'Attack Cost for {learner}.',
            f'fig:{tag}_cost_experiments')
        outputs[f'{tag}_ratio_experiments.tex'] = figure(
            panel(rows,learner,'horizon',ratios,True),panel(rows,learner,'arms',ratios,True),
            [r'Target-arm Selection Ratio\\vs. Horizon $T$',
             r'Target-arm Selection Ratio\\vs. Number of Arms $K$'],
            legend([('total',r'$N_K/T$'),('online',r'$N_K^{\on}/H$'),
                    ('heuristic',r'Clipped Suppression $N_K/T$')]),
            f'Target-arm Selection Ratio for {learner}.',
            f'fig:{tag}_ratio_experiments')
    for directory in [ROOT/'fig', ROOT.parent/'fig']:
        directory.mkdir(exist_ok=True)
        for name, content in outputs.items():
            (directory/name).write_text(content)
    print(f'Generated {len(outputs)-1} figures in both experiment and paper directories.')


if __name__ == '__main__':
    main()
