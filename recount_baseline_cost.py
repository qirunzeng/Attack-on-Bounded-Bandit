"""Recount saved suppression actions, including 0->0, without resimulation."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/ml-25m/paper'
DEFINITION = 'suppressed non-target rounds (including 0->0)'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path, provenance = OUT / 'paper_results.csv', OUT / 'manifest.json'
    manifest = json.loads(provenance.read_text())
    assert digest(path) == manifest['results_sha256']
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    baseline = [r for r in rows if r['method'] == 'Clipped Suppression']
    if all(r['cost_definition'] == DEFINITION for r in baseline):
        print('Suppression-round costs already current.')
        return
    assert len(baseline) == 280
    assert all(r['cost_definition'] == 'modified online observations' for r in baseline)
    archive = OUT / 'history/changed-observation-cost-20260907'
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('paper_results.csv', 'manifest.json', 'baseline_k_diagnostic.json',
                 'baseline_k_diagnostic.log'):
        if (OUT / name).exists():
            shutil.copy2(OUT / name, archive / name)
    for row in baseline:
        assert row['status'] == 'simulated'
        counts = json.loads(row['online_counts_json'])
        assert sum(counts) == int(row['H'])
        row['cost'] = str(sum(counts[:-1]))
        row['cost_definition'] = DEFINITION
    partial = OUT / 'paper_results.recount.csv'
    with partial.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)
    manifest['postprocessing'] = dict(
        operation='Count every Clipped Suppression non-target round, including 0->0',
        parent_results_sha256=manifest['results_sha256'],
        parent_manifest=str(archive.relative_to(ROOT) / 'manifest.json'),
        changed_rows=len(baseline), changed_fields=['cost', 'cost_definition'],
        trajectories_reused=True, script_sha256=digest(Path(__file__)))
    # Keep source_sha256 as the original trajectory-generation provenance.
    manifest['reproduction_source_sha256'] = {
        name: digest(ROOT / name) for name in manifest['source_sha256']}
    manifest['results_sha256'] = digest(path)
    provenance.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Recounted {len(baseline)} baseline rows; original trajectories preserved.')


if __name__ == '__main__':
    main()
