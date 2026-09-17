"""Audit sufficient-budget parameters, actual stage counts, and rerun stability."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from generate_paper_figures import DATA, ROOT, read_results


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows, manifest = read_results()
    for name, expected in manifest.get('reproduction_source_sha256', manifest['source_sha256']).items():
        assert sha(ROOT/name) == expected, f'Source changed: {name}'
    previous = DATA/'history/derived-xu-budgets-20260907/paper_results.csv'
    stable = 0
    if previous.exists():
        key = lambda r: tuple(r[k] for k in ['sweep','learner','method','repeat','T','K','multiplier'])
        old = {key(r): r for r in csv.DictReader(previous.open())}
        current = {key(r): r for r in rows}
        for row_key, before in old.items():
            if before['method'] == 'Two-phase':
                continue
            assert row_key in current, f'Missing historical record: {row_key}'
            row = current[row_key]
            assert all(row[k] == v for k, v in before.items()), row_key
            stable += 1
    grouped = defaultdict(list)
    for row in rows:
        if row['method'] == 'Two-phase':
            grouped[row['learner'], float(row['multiplier'])].append(row)
    summary = []
    for (learner, gap), group in sorted(grouped.items()):
        info = json.loads(group[0]['budget_log_json'])
        item = dict(learner=learner, normalized_gap=gap, mu=info['mu'],
                    C1=info['C1'], C2=info['C2'], status=group[0]['status'], repeats=len(group))
        if info['feasible']:
            failure = dict(post_attack=0)
            for row in group:
                post = json.loads(row['post_attack_counts_json'])
                failure['post_attack'] += sum(post[:-1])>0
            item.update(cost=info['C1']+info['C2'], observed_failures=failure,
                target_ratio_mean=statistics.mean(float(r['target_ratio']) for r in group),
                post_attack_target_ratio_min=min(float(r['post_attack_target_ratio']) for r in group),
                post_attack_target_ratio_mean=statistics.mean(float(r['post_attack_target_ratio']) for r in group))
        summary.append(item)
    # A separate implementation check in the original main-paper Table 1 setting.
    # This is not a universal budget prescription for the MovieLens sweep.
    import numpy as np
    from xu_simulation import simulate
    original_checks = []
    for learner in ['UCB','TS']:
        for seed in range(10):
            counts, _, _, p1, p2, post = simulate(
                [np.array([1.]*9+[0.]), np.array([1.]*8+[0.]*2)],
                50000, learner, seed, 34, 66, .8)
            assert p1.sum()==34 and p2.sum()==66 and post.sum()==49900
            original_checks.append(dict(learner=learner, seed=seed,
                target_ratio=float(counts[-1]/50000), post_attack_target_ratio=float(post[-1]/49900)))
    report = dict(results_sha256=manifest['results_sha256'],
        budget_source='Xu et al. (2021), supplementary A.2/A.4',
        validation_source_sha256=sha(__file__), total_rows=len(rows),
        simulated_rows=sum(r['status']=='simulated' for r in rows),
        unchanged_non_xu_records=stable, xu_budget_version=manifest['xu_budget_version'],
        cases=summary, original_table1_check=dict(K=2,T=50000,means=[.9,.8],C1=34,C2=66,runs=original_checks))
    (DATA/'xu_budget_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'cases','original_table1_check'}}, indent=2))
    for item in summary:
        if item['normalized_gap']<=2:
            print(json.dumps(item))


if __name__ == '__main__':
    main()
