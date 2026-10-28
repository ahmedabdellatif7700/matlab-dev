"""Evaluate locked architectures on independent seeds without test tuning."""
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.io import savemat
from . import models, baselines
from .data import load


def confirm(run):
    raw=(run/'selection.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==(run/'selection.sha256').read_text().strip()
    lock=json.loads(raw); cfg=lock['configuration']
    if not lock['complete']:
        raise RuntimeError('Four discovery cases have not qualified; do not claim four successes.')
    rows=[]
    for case in lock['selected']:
        for seed in cfg['confirmation_seeds']:
            folder=run/'confirmation'/f'seed_{seed}'/case['condition']['ID']
            if (folder/'metrics.json').exists():
                rows.extend(json.loads((folder/'metrics.json').read_text()))
                continue
            d=load(folder/'dataset.mat',phase='confirmation')
            ideal=d['ideal_symbols'].ravel(); labels=d['ideal_bits']
            classical=baselines.fit_all(d['X_train'],d['Y_train'],d['bits_train'],d['X_validation'],d['Y_validation'],d['bits_validation'],ideal,labels,degrees=cfg['polynomial_orders'],bit_degrees=cfg['bit_polynomial_orders'],kernel_sizes=cfg['kernel_centers'])
            spec={k:case['neural'][k] for k in ['widths','activation','lr','weight_decay']}
            neural=models.train(d['X_train'],d['bits_train'],d['X_validation'],d['bits_validation'],spec,seed=seed,epochs=case['neural'].get('training_epochs',cfg['nn_epochs']),patience=case['neural'].get('training_patience',cfg['nn_patience']),polish=case['neural'].get('training_polish',cfg['nn_polish_steps']))
            models.save(neural,folder/'neural.pt')
            with (folder/'classical.pkl').open('wb') as stream: pickle.dump(classical,stream)
            predictions=[models.predict(neural,d['X_test'])>=0]
            infos=[dict(neural['info'],name='Selected NN')]
            for m in classical:
                predictions.append(np.concatenate([baselines.predict(m,x,ideal,labels) for x in np.array_split(d['X_test'],max(1,len(d['X_test'])//2048))]))
                infos.append(m['info'])
            truth=d['bits_test']; nn_error=np.mean(predictions[0]!=truth,axis=1)
            # Contiguous symbol blocks preserve local error dependence. Paired
            # bootstrap compares methods on exactly the same received symbols.
            rng=np.random.default_rng(seed); block=128
            for info,pred in zip(infos,predictions):
                err=np.mean(pred!=truth,axis=1); count=int(np.sum(pred!=truth)); n=truth.size
                difference=err-nn_error
                blocks=np.array([part.mean() for part in np.array_split(difference,max(1,len(err)//block))])
                bootstrap=blocks[rng.integers(len(blocks),size=(2000,len(blocks)))].mean(axis=1)
                lo,hi=np.quantile(bootstrap,[.025,.975])
                # Hard-decision EVM is explicitly distinct from analog regression EVM.
                lookup={tuple(b):s for b,s in zip(labels,ideal)}
                decided=np.asarray([lookup[tuple(b)] for b in pred.astype(int)])
                target=d['Y_test'][:,0]+1j*d['Y_test'][:,1]
                rows.append(dict(case=case['condition']['ID'],family=case['condition']['Family'],seed=seed,method=info['name'],ber=count/n,bit_errors=count,bits=n,hard_decision_evm_percent=100*np.sqrt(np.mean(abs(decided-target)**2)/np.mean(abs(target)**2)),multiplications=info['multiplications'],parameters=info['parameters'],nonlinear_evaluations=info['nonlinear_evaluations'],paired_ber_difference_low=lo,paired_ber_difference_high=hi,ber_target=case['ber_target']))
            savemat(folder/'decisions.mat',dict(truth=truth,nn=predictions[0],predictions=np.stack(predictions),names=np.array([i['name'] for i in infos],dtype=object)))
            (folder/'metrics.json').write_text(json.dumps(rows[-len(infos):],indent=2)+'\n')
            print('Confirmed',case['condition']['ID'],seed,'NN BER',rows[-len(infos)]['ber'],flush=True)
            with (run/'confirmation_metrics.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    with (run/'confirmation_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
