"""Archive compact validated scenario results; keep large datasets ignored."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def archive(run, destination):
    if not (run/'COMPLETE').is_file():
        raise ValueError('Refusing to archive an incomplete run')
    cfg = json.loads((run/'configuration.json').read_text())
    conditions, versions = [], None
    for path in sorted(run.glob('seed_*/*/manifest.json')):
        m = json.loads(path.read_text())
        if hashlib.sha256((path.parent/'dataset.mat').read_bytes()).hexdigest() != m['dataset_sha256']:
            raise ValueError('Dataset hash mismatch')
        conditions.append(dict(seed=m['seed'], condition=m['condition'], fits=m['fits'],
                               skipped=m['skipped'], dataset_sha256=m['dataset_sha256'],
                               source_sha256=m['source_sha256'], samples=m['partition_samples']))
        versions = m['versions']
    destination.mkdir(parents=True, exist_ok=True)
    for path in (run/'figures').glob('*.png'):
        shutil.copy2(path, destination/path.name)
    for name in ('all_metrics.csv', 'pooled_metrics.csv', 'conditions.csv'):
        shutil.copy2(run/name, destination/name)
    rows = list(csv.DictReader((run/'all_metrics.csv').open()))
    record = dict(configuration=cfg, conditions_completed=len(conditions),
                  receiver_metric_rows=len(rows), versions=versions,
                  source_snapshot_sha256={str(p.relative_to(run/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (run/'source').rglob('*') if p.is_file()},
                  conditions=conditions)
    (destination/'validation_record.json').write_text(json.dumps(record, indent=2)+'\n')
    pooled = list(csv.DictReader((run/'pooled_metrics.csv').open()))
    selected = cfg.get('report_condition', conditions[0]['condition']['Condition'])
    example = [r for r in pooled if r['Condition'] == selected and int(r['user']) in (0, 1)]
    # A single user and aggregate row would duplicate multiuser results.
    if any(r['user'] == '0' for r in example):
        example = [r for r in example if r['user'] == '0']
    table = ['| Receiver | RMS EVM (%) | Bit errors | Bits tested | BER |',
             '|---|---:|---:|---:|---:|']
    for row in example:
        ber = f"{float(row['ber']):.5g}" if int(row['bit_errors']) else f"0 observed; upper95={float(row['ber_upper_95']):.3g}"
        table.append(f"| {row['method']} | {float(row['evm_percent']):.3f} | {row['bit_errors']} | {row['bit_count']} | {ber} |")
    text = (f"# Scenario {cfg['scenario']}: validated results\n\n"
            f"Completed {len(conditions)} independent-seed operating-point runs and "
            f"{len(rows)} MATLAB-verified receiver/user metric evaluations.\n\n"
            f"Representative condition: `{selected}`.\n\n" + '\n'.join(table) + '\n\n'
            'All fits use training data, checkpoint/ridge selection uses validation data, and test targets are used only for evaluation. '
            'Zero errors are finite-sample observations. Nominal binomial 95% intervals may be optimistic with correlated errors. '
            'The 3.8e-3 line is a literature reference, not an implemented FEC decoder.\n\n'
            'Large datasets, checkpoints, editable FIG and vector PDF exports remain in '
            f'`{run}`; source snapshots and hashes are recorded in `validation_record.json`. '
            'The recorded Git revision precedes this scenario commit because validation runs before committing.\n\n')
    for p in sorted(destination.glob('*.png')):
        text += f'![{p.stem.replace("_", " ")}]({p.name})\n\n'
    text += 'See [scenario equations and limitations](../../SCENARIOS.md).\n'
    (destination/'README.md').write_text(text)
    print(f'Archived {len(conditions)} runs in {destination}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    archive(a.run, a.output)
