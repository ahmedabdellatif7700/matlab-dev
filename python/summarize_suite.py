"""Index completed study reports and screen exploratory BER operating regions."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def classify(rows, target=3.8e-3):
    neural = [r for r in rows if 'NN' in r['method']]
    other = [r for r in rows if 'NN' not in r['method']]
    neural_met = [r['method'] for r in neural if float(r['ber_upper_95']) < target]
    other_met = [r['method'] for r in other if float(r['ber_upper_95']) < target]
    if other_met:
        status = 'non-neural receiver meets reference'
    elif neural_met and other and all(float(r['ber_lower_95']) > target for r in other):
        status = 'NN-only candidate in tested models'
    elif neural_met:
        status = 'NN meets; other receiver bounds inconclusive'
    elif all(float(r['ber_lower_95']) > target for r in rows):
        status = 'all tested receivers above reference'
    else:
        status = 'inconclusive at current bit count'
    return status, neural_met, other_met


def summarize(root):
    labels = {1:'RF driver', 2:'MZM / adaptive activation', 3:'DSB / SSB optical fading',
              4:'Four-user SC-FDMA', 5:'28 GHz OFDM'}
    regions, table, total_runs, total_rows = [], [], 0, 0
    for scenario in range(1, 6):
        folder = root/f'scenario{scenario}'
        with (folder/'pooled_metrics.csv').open() as stream:
            pooled = list(csv.DictReader(stream))
        with (folder/'all_metrics.csv').open() as stream:
            raw = list(csv.DictReader(stream))
        runs = len({(r['Seed'], r['Condition']) for r in raw})
        total_runs += runs; total_rows += len(raw)
        table.append(f'| [{scenario}: {labels[scenario]}](scenario{scenario}/README.md) | {runs} | {len(raw)} |')
        groups = defaultdict(list)
        for row in pooled:
            groups[(row['Condition'], row.get('user', '1'))].append(row)
        for (condition, user), rows in sorted(groups.items()):
            status, neural, other = classify(rows)
            regions.append(dict(scenario=scenario, condition=condition, user=user, status=status,
                                neural_meets='; '.join(neural), non_neural_meets='; '.join(other),
                                bit_count=rows[0]['bit_count']))
    with (root/'operating_regions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(regions[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(regions)
    candidates = [r for r in regions if r['status'] == 'NN-only candidate in tested models']
    text = ('# Validated millimeter-wave equalizer studies\n\n'
            f'Five completed studies contain **{total_runs} seeded operating-point runs** '
            f'and **{total_rows} receiver/user metric rows**. Each condition uses three independent replicate seeds; '
            'operating points share random draws within a seed for paired comparisons. '
            'Multiuser aggregate rows are not additional independent experiments.\n\n'
            '| Study and full figures | Runs | Metric rows |\n|---|---:|---:|\n' + '\n'.join(table) + '\n\n'
            'Each report contains held-out BER/EVM, constellation figures, numerical CSVs and source/dataset provenance. '
            'Large waveforms, saved models, vector PDFs and editable MATLAB FIGs remain in the ignored per-run dataset directories.\n\n'
            '## Where is an NN necessary?\n\n'
            '[operating_regions.csv](operating_regions.csv) screens every tested condition and user. '
            'A non-neural receiver counts as meeting the 3.8e-3 literature reference only when its nominal '
            '95% upper BER bound lies below it. An NN-only candidate requires a neural upper bound below '
            'the reference and **every evaluated non-neural lower bound above it**, including the untrained '
            'front end. Overlapping bounds remain inconclusive.\n\n'
            f'This screen found **{len(candidates)} NN-only candidates** across the listed condition/user combinations. '
            'This is exploratory evidence within the stated models, observation windows and training/complexity budgets, '
            'not universal NN necessity. The bounds are nominal binomial intervals: correlated errors and scanning many '
            'operating points limit statistical interpretation. Confirm a candidate with additional independent data '
            'before claiming a reproducible operational advantage.\n\n'
            '## Interpretation\n\n'
            '- The RF-driver study establishes a useful non-neural baseline; fifth-order Volterra is competitive.\n'
            '- MZM drive sweeps distinguish noise-limited, nonlinear and severely folded regimes. Multilevel output '
            'can lower EVM by pulling estimates toward discrete QAM levels, so BER must be examined alongside EVM.\n'
            '- DSB fading near a null can limit every receiver. SSB avoids that magnitude fading; it is not an NN effect.\n'
            '- The multiuser study gives all joint model families the same cross-user observations and also evaluates individual receivers. '
            'Joint and individual NN parameter budgets differ and are reported, rather than interpreted as a pure information-only experiment.\n'
            '- OFDM compares pilot estimates with a known-linear-shape diagnostic. Global nonlinear ICI is not generally recoverable '
            'from only three neighboring carriers. Failure of this small post-equalizer is not a limitation of all neural OFDM receivers.\n\n'
            'See [equations, controls, caveats and reproduction commands](../SCENARIOS.md). '
            'No experimental paper curve or optical-power calibration is claimed reproduced.\n')
    (root/'README.md').write_text(text)
    print(f'{total_runs} runs; {total_rows} metric rows; {len(candidates)} NN-only candidates.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reports', type=Path, default=Path(__file__).resolve().parents[1]/'matlab/results')
    summarize(ap.parse_args().reports)
