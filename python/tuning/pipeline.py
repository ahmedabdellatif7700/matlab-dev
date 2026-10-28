"""Validation-only search, locked four-case design, then fresh confirmation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
import numpy as np
import equalizer as models
from . import baselines
from config import network_catalog, RESEARCH
from tuning.data import load


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def design(run, only=None):
    if (run/'confirmation').exists():
        raise RuntimeError('Design locking is forbidden after confirmation data generation. Start a fresh run.')
    cfg=json.loads((run/'configuration.json').read_text())
    cfg.update(RESEARCH)
    cfg['network_catalog']=network_catalog()
    (run/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with (run/'candidates.csv').open() as stream:
        candidates=list(csv.DictReader(stream))
    all_rows=[]; selected=[]
    for candidate in candidates:
        if only and candidate['ID'] not in only: continue
        folder=run/'discovery'/f"seed_{cfg['discovery_seed']}"/candidate['ID']
        if (folder/'models.json').exists():
            infos=json.loads((folder/'models.json').read_text())
            if (folder/'extended_models.json').exists():
                extended=json.loads((folder/'extended_models.json').read_text())
                for i in extended:
                    i.setdefault('training_epochs',900); i.setdefault('training_patience',90); i.setdefault('training_polish',20)
                infos=[i for i in infos if not i['name'].startswith('NN extended')]+extended
            for info in infos:
                if info['name'].startswith('Volterra'):
                    degree=info['order']; dimension=2*cfg['equalizer_taps']; count=math.comb(dimension+degree,degree)-1
                    outputs=int(candidate['ModOrder']).bit_length()-1 if 'classifier' in info['name'] else 2
                    info['multiplications']=count-dimension+outputs*count
            (folder/'models.json').write_text(json.dumps(infos,indent=2)+'\n')
            for info in infos:
                all_rows.append(dict(case=candidate['ID'],family=candidate['Family'],name=info['name'],neural=info['neural'],validation_ber=info['validation_ber'],multiplications=info['multiplications'],parameters=info['parameters'],nonlinear_evaluations=info['nonlinear_evaluations']))
            continue
        d=load(folder/'dataset.mat',phase='discovery')
        ideal=d['ideal_symbols'].ravel(); labels=d['ideal_bits']
        fitted=baselines.fit_all(d['X_train'],d['Y_train'],d['bits_train'],d['X_validation'],
                                d['Y_validation'],d['bits_validation'],ideal,labels,
                                degrees=cfg['polynomial_orders'],bit_degrees=cfg['bit_polynomial_orders'],
                                kernel_sizes=cfg['kernel_centers'])
        with (folder/'classical_models.pkl').open('wb') as stream: pickle.dump(fitted,stream)
        infos=[dict(m['info'],neural=False) for m in fitted]
        for k,spec in enumerate(network_catalog()):
            model=models.train_bit_equalizer(d['X_train'],d['bits_train'],d['X_validation'],d['bits_validation'],spec,
                               seed=42,epochs=cfg['nn_epochs'],patience=cfg['nn_patience'],
                               polish=cfg['nn_polish_steps'])
            logits=models.predict_bits(model,d['X_validation'])
            info=dict(model['info'],name=f"NN {spec['activation']} {spec['widths']}",neural=True,
                      validation_ber=float(np.mean((logits>=0)!=d['bits_validation'])),catalog_index=k)
            models.save_checkpoint(model,folder/f'nn_{k}.pt')
            np.savetxt(folder/f'nn_{k}_history.csv',model['history'],delimiter=',',
                       header='step,training_bce,validation_bce,validation_ber,phase',comments='')
            infos.append(info)
            print(candidate['ID'],info['name'],f"validation BER={info['validation_ber']:.6g}",flush=True)
        (folder/'models.json').write_text(json.dumps(infos,indent=2)+'\n')
        for info in infos:
            all_rows.append(dict(case=candidate['ID'],family=candidate['Family'],name=info['name'],
                                 neural=info['neural'],validation_ber=info['validation_ber'],
                                 multiplications=info['multiplications'],parameters=info['parameters'],
                                 nonlinear_evaluations=info['nonlinear_evaluations']))
        print('Completed candidate',candidate['ID'],flush=True)
    if only:
        return
    write_csv(run/'discovery_models.csv',all_rows)
    families=list(dict.fromkeys(c['Family'] for c in candidates))
    for family in families:
        options=[]
        for candidate in [c for c in candidates if c['Family']==family]:
            folder=run/'discovery'/f"seed_{cfg['discovery_seed']}"/candidate['ID']
            infos=json.loads((folder/'models.json').read_text())
            classical=[i for i in infos if not i['neural']]
            for neural in [i for i in infos if i['neural']]:
                if neural['validation_ber'] > cfg.get('max_nn_validation_ber',3.8e-3):
                    continue
                frontier=[b['validation_ber'] for b in classical if b['multiplications'] < cfg['required_cost_ratio']*neural['multiplications']]
                frontier_gap=min(frontier)-neural['validation_ber']
                if frontier_gap < cfg.get('min_validation_frontier_gap',1e-4):
                    continue
                derived_target=(neural['validation_ber']+min(frontier))/2
                targets=sorted(set([*cfg['ber_targets'],derived_target]))
                for target in targets:
                    if neural['validation_ber'] > target:
                        continue
                    matching=[b for b in classical if b['validation_ber'] <= target]
                    cost=min([b['multiplications'] for b in matching],default=math.inf)
                    ratio=cost/max(neural['multiplications'],1)
                    # Require a substantial arithmetic gap rather than a
                    # one-operation technicality; confirmation remains decisive.
                    if ratio < cfg['required_cost_ratio']:
                        continue
                    cheaper=[b for b in classical if b['multiplications'] <= neural['multiplications']]
                    gap=min(b['validation_ber'] for b in cheaper)-neural['validation_ber']
                    if gap <= 0:
                        continue
                    option=dict(condition=candidate, neural=neural, ber_target=target,
                                target_origin='reference grid' if target in cfg['ber_targets'] else 'validation frontier midpoint',
                                validation_cost_ratio=None if math.isinf(ratio) else ratio,
                                validation_cheaper_baseline_gap=gap,validation_frontier_gap=frontier_gap,
                                classical=[dict(name=b['name'],spec=b) for b in classical])
                    # Prefer a robust gap, then a smaller NN, without consulting tests.
                    options.append((frontier_gap,-neural['multiplications'],target,option))
        if options:
            selected.append(max(options,key=lambda o:o[:3])[3])
    lock=dict(configuration=cfg,selected=selected,complete=len(selected)==4,
              disclosure='Conditions and hyperparameters selected on discovery validation only; no discovery test variables exist.',
              discovery_table_sha256=hashlib.sha256((run/'discovery_models.csv').read_bytes()).hexdigest())
    (run/'selection.json').write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n')
    print('Locked',len(selected),'of four families',flush=True)


