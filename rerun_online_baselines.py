"""Replace baseline rows with full-T online runs; preserve offline results."""
import csv
import json
import shutil
import time
from pathlib import Path

import mlrunner
import paper_runner as runner

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/ml-25m/paper'


def main():
    path = OUT / 'paper_results.csv'
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert runner.sha256(path) == manifest['results_sha256']
    with path.open(newline='') as f:
        old = list(csv.DictReader(f))
    archive = OUT / 'history/baseline-warm-start-20260907'
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('paper_results.csv', 'manifest.json', 'baseline_k_diagnostic.json', 'baseline_k_diagnostic.log'):
        if (OUT / name).exists():
            shutil.copy2(OUT / name, archive / name)
    instances = {K: mlrunner.load_movielens(K) for K in manifest['K_grid']}
    cases = list(dict.fromkeys((int(r['repeat']), r['sweep'], int(r['T']), int(r['K']), r['multiplier']) for r in old))
    replacements = {}
    key = lambda r: (str(r['repeat']), r['sweep'], str(r['T']), str(r['K']), str(r['multiplier']), r['learner'], r['method'])
    started = time.time()
    for index, (repeat, sweep, T, K, gap) in enumerate(cases):
        rows = runner.run_case(instances[K], repeat, sweep, T,
                               None if gap == '' else float(gap), baseline_only=True)
        for row in rows:
            replacements[key(row)] = row
        if (index + 1) % 28 == 0:
            print(f'{index+1}/{len(cases)} cases, {time.time()-started:.1f}s', flush=True)
    updated = [replacements.get(key(r), r) for r in old]
    assert len(replacements) == 560
    for before, after in zip(old, updated):
        if before['method'] in {'Ours', 'Direct'}:
            assert before == after
        else:
            assert after['T0'] == 0 and after['H'] == after['T']
            assert int(before['seed']) == after['seed']
    partial = OUT / 'paper_results.online-baselines.csv'
    with partial.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=old[0])
        writer.writeheader()
        writer.writerows(updated)
    partial.replace(path)
    manifest['online_baseline_rerun'] = dict(parent_manifest=str(archive.relative_to(ROOT) / 'manifest.json'),
        parent_results_sha256=manifest['results_sha256'], replaced_rows=len(replacements),
        runtime_seconds=time.time()-started, script_sha256=runner.sha256(Path(__file__)),
        source_sha256={name:runner.sha256(ROOT / name) for name in manifest['source_sha256']})
    manifest['baseline_initialization'] = 'no offline log; first K online pulls: target then non-targets; attacks active from round 1'
    manifest['reproduction_source_sha256'] = manifest['online_baseline_rerun']['source_sha256']
    manifest['results_sha256'] = runner.sha256(path)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Updated {len(replacements)} baseline rows; 840 offline rows unchanged.', flush=True)


if __name__ == '__main__':
    main()
