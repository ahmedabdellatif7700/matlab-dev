"""Q8.24 LMS FFE versus supplied forward/loss/backward/SGD ANN stages.

Consumes independent MATLAB train/validation/test files. This is an untimed
algorithm model; new dimensions and tanh are not claimed as validated RTL.
"""
import copy
import csv
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import loadmat
from config import ADAPTIVE
from ffe_equalizer import Q824LMS
from hardware.training import memory_map as mm, fixedpoint as fp
from hardware.training.controller import TrainingEngine, PHASES
from hardware.training.activation import create_activation


class Q824ANN:
    def __init__(self, inputs, hidden, activation, lr_q24, seed):
        self.inputs=inputs; self.hidden=hidden; mm.configure(inputs,hidden)
        self.engine=TrainingEngine(create_activation(activation),int(lr_q24))
        self.ram=mm.new_ram(); rng=np.random.default_rng(seed)
        w0=rng.uniform(-np.sqrt(6/(inputs+hidden)),np.sqrt(6/(inputs+hidden)),inputs*hidden)
        w1=rng.uniform(-np.sqrt(6/(hidden+2)),np.sqrt(6/(hidden+2)),hidden*2)
        mm.load_parameters(self.ram,np.r_[w0,np.zeros(hidden),w1,np.zeros(2)])
        self.engine.init_ram(self.ram)

    def activate(self):
        mm.configure(self.inputs,self.hidden)

    def step(self, x, target):
        self.ram[mm.A0:mm.A0+self.inputs]=x
        self.ram[mm.T:mm.T+2]=target
        for phase in PHASES:
            self.engine.run_phase(self.ram,phase)

    def predict(self, x):
        self.activate(); ram=self.ram.copy(); result=[]
        for row in x:
            ram[mm.A0:mm.A0+self.inputs]=row
            self.engine.run_phase(ram,'fwd'); result.append(ram[mm.A2:mm.A2+2])
        return np.asarray(result,dtype=np.int32)

    def state(self):
        return self.ram.copy()

    def restore(self, state):
        self.ram=state.copy()


def dataset(path):
    d=loadmat(path,struct_as_record=False,squeeze_me=False)
    m=d['metadata'][0,0]; scale=float(m.normalization_scale.item())
    target=np.asarray(d['Y_q8_24'],dtype=np.int32)
    n=int(m.modulation_order.item())
    if not np.array_equal(d['Y'],target.astype(np.float64)/fp.ONE):
        raise ValueError('Y is not the exact decoded Q8.24 target')
    return dict(raw=d, target=target, scale=scale, modulation=n)


def features(data,taps):
    d=data['raw']; raw=d['rx_iq']; idx=d['target_indices'].ravel().astype(int)-1
    if idx[0]-taps+1<0:
        raise ValueError('Insufficient prefix history')
    x=np.column_stack([raw[idx-k] for k in range(taps)])/data['scale']
    return [[fp.to_q(float(v)) for v in row] for row in x]


def metrics(prediction,target,scale,modulation):
    p=np.asarray(prediction,dtype=np.float64)/fp.ONE
    y=np.asarray(target,dtype=np.float64)/fp.ONE
    levels=int(round(np.sqrt(modulation)))
    def gray(v):
        i=np.clip(np.floor((v*scale+levels)/2),0,levels-1).astype(np.uint8)
        return i^(i>>1)
    xor=np.bitwise_xor(gray(p),gray(y))
    errors=int(np.unpackbits(xor.reshape(-1,1),axis=1).sum())
    bits=len(y)*int(np.log2(modulation)); ber=errors/bits; z=1.96
    center=(ber+z*z/(2*bits))/(1+z*z/bits)
    radius=z*np.sqrt(ber*(1-ber)/bits+z*z/(4*bits**2))/(1+z*z/bits)
    return dict(evm_percent=100*np.linalg.norm(p-y)/np.linalg.norm(y),ber=ber,
                bit_errors=errors,bits=bits,ber_lower=max(0,center-radius),ber_upper=center+radius)


def write_csv(path,rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def fit(model,x,y,vx,vy,scale,modulation,epochs,tag,curves):
    best=float('inf'); best_state=None; best_epoch=0; fp.stats.clear()
    started=time.monotonic()
    for epoch in range(1,epochs+1):
        if isinstance(model,Q824ANN): model.activate()
        for a,b in zip(x,y): model.step(a,b)
        score=metrics(model.predict(vx),vy,scale,modulation)
        curves.append(dict(**tag,epoch=epoch,validation_evm_percent=score['evm_percent'],validation_ber=score['ber']))
        if score['evm_percent']<best:
            best=score['evm_percent']; best_state=model.state(); best_epoch=epoch
    model.restore(best_state)
    return dict(validation_evm_percent=best,best_epoch=best_epoch,updates=epochs*len(x),
                selected_updates=best_epoch*len(x),training_samples=len(x),
                training_seconds=time.monotonic()-started,saturations=int(sum(fp.stats.values())))


def compare(run,only=None):
    run=Path(run).resolve()
    with (run/'conditions.csv').open() as f: conditions=list(csv.DictReader(f))
    if only:
        available={c['condition'] for c in conditions}
        if not set(only).issubset(available): raise ValueError('Unknown comparison condition')
        conditions=[c for c in conditions if c['condition'] in only]
    out=run/('python_results_'+'_'.join(c['condition'] for c in conditions) if only else 'python_results')
    settings=copy.deepcopy(ADAPTIVE)
    if (out/'settings.json').exists():
        if json.loads((out/'settings.json').read_text())!=settings:
            raise ValueError('Existing experiment has different settings; use a new run directory')
    out.mkdir(exist_ok=True); (out/'models').mkdir(exist_ok=True)
    (out/'candidate_checkpoints').mkdir(exist_ok=True)
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    validation=[]; curves=[]; results=[]; selected=[]
    for condition in conditions:
        source=Path(condition['dataset_folder'])
        with (source/'manifest.csv').open() as f: manifest=list(csv.DictReader(f))
        powers=sorted({float(r['received_power_dbm']) for r in manifest})
        for power in powers:
            files={r['split']:source/r['file'] for r in manifest if float(r['received_power_dbm'])==power}
            train=dataset(files['train']); val=dataset(files['validation'])
            y=train['target'].tolist(); vy=val['target']; scale=train['scale']; modulation=train['modulation']
            candidates=[]
            for taps in settings['lms_taps']:
                for rate in settings['lms_learning_rates']:
                    candidates.append(dict(receiver='LMS_FFE',taps=taps,hidden=0,activation='linear',learning_rate=rate))
            for a in settings['ann_candidates']:
                candidates.append(dict(receiver='ANN',taps=2,**a))
            winners={}
            for index,spec in enumerate(candidates):
                tag=dict(condition=condition['condition'],received_power_dbm=power,candidate=index,**spec)
                x=features(train,spec['taps']); vx=features(val,spec['taps'])
                rate=fp.to_q(spec['learning_rate'])
                if spec['receiver']=='LMS_FFE':
                    model=Q824LMS(2*spec['taps'],rate); group='LMS_FFE'
                else:
                    model=Q824ANN(2*spec['taps'],spec['hidden'],spec['activation'],rate,settings['initialization_seed'])
                    group=f"ANN_{spec['activation']}_{spec['hidden']}"
                checkpoint=out/'candidate_checkpoints'/f"{condition['condition']}_{power:g}_{index}.npz"
                if checkpoint.exists():
                    saved=np.load(checkpoint,allow_pickle=False)
                    row=json.loads(str(saved['row_json'].item()))
                    state=saved['state_q8_24'].astype(np.int64).tolist()
                    if group=='LMS_FFE':
                        width=2*spec['taps']; state=[ [state[:width],state[width:2*width]],state[2*width:] ]
                    model.restore(state)
                    curves.extend(json.loads(str(saved['curves_json'].item())))
                    summary={k:row[k] for k in ('validation_evm_percent','best_epoch','updates','selected_updates','training_samples','training_seconds','saturations')}
                else:
                    first_curve=len(curves)
                    summary=fit(model,x,y,vx,vy,scale,modulation,settings['epochs'],tag,curves)
                    row=dict(**tag,learning_rate_q8_24=rate,**summary)
                    state=model.state()
                    if group=='LMS_FFE': state=[v for weights in state[0] for v in weights]+state[1]
                    np.savez_compressed(checkpoint,state_q8_24=np.asarray(state,dtype=np.int32),
                        row_json=json.dumps(row),curves_json=json.dumps(curves[first_curve:]))
                validation.append(row)
                write_csv(out/'validation_candidates.csv',validation)
                write_csv(out/'learning_curves.csv',curves)
                if group not in winners or summary['validation_evm_percent']<winners[group][0]['validation_evm_percent']:
                    winners[group]=(row,model)
                print(f"{condition['condition']} {power:g} dBm candidate {index+1}/{len(candidates)}: val EVM {summary['validation_evm_percent']:.3f}%",flush=True)
            # Test is loaded only after every candidate at this operating point is selected.
            test=dataset(files['test']); target=test['target']; primary_ann=min(
                ((k,v) for k,v in winners.items() if k.startswith('ANN')),key=lambda kv:kv[1][0]['validation_evm_percent'])[0]
            raw=np.asarray(features(test,1),dtype=np.int32)
            raw_metrics=metrics(raw,target,scale,modulation)
            # Gray bit distances must reproduce MATLAB's unequalized bit counts.
            mat_expected=float(test['raw']['metadata'][0,0].unequalized_ber.item())
            if abs(raw_metrics['ber']-mat_expected)>1/(len(target)*np.log2(modulation)):
                raise ValueError('Python BER demapping differs from MATLAB')
            results.append(dict(condition=condition['condition'],received_power_dbm=power,receiver='unequalized',primary=True,**raw_metrics))
            for group,(row,model) in winners.items():
                prediction=model.predict(features(test,row['taps']))
                result=metrics(prediction,target,scale,modulation)
                primary=group=='LMS_FFE' or group==primary_ann
                results.append(dict(condition=condition['condition'],received_power_dbm=power,receiver=group,primary=primary,**result))
                selected.append(dict(**row,receiver_group=group,primary=primary,parameter_count=(2*(2*row['taps'])+2 if group=='LMS_FFE' else 2*row['taps']*row['hidden']+row['hidden']+2*row['hidden']+2)))
                state=model.state()
                if group=='LMS_FFE': state=[v for r in state[0] for v in r]+state[1]
                np.savez_compressed(out/'models'/f"{condition['condition']}_{power:g}_{group}.npz",state_q8_24=np.asarray(state,dtype=np.int32),predictions_q8_24=prediction)
            write_csv(out/'validation_candidates.csv',validation); write_csv(out/'learning_curves.csv',curves)
            write_csv(out/'test_metrics.csv',results); write_csv(out/'selected_parameters.csv',selected)
            print(f"FINISHED {condition['condition']} {power:g} dBm",flush=True)
    report(out)
    return out


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    with (out/'test_metrics.csv').open() as f: rows=list(csv.DictReader(f))
    conditions=list(dict.fromkeys(r['condition'] for r in rows))
    for metric,label,log in [('ber','BER',True),('evm_percent','EVM (%)',False)]:
        fig,axes=plt.subplots(1,len(conditions),figsize=(5*len(conditions),4),squeeze=False)
        for ax,condition in zip(axes[0],conditions):
            subset=[r for r in rows if r['condition']==condition]
            for receiver in dict.fromkeys(r['receiver'] for r in subset):
                data=sorted([r for r in subset if r['receiver']==receiver],key=lambda r:float(r['received_power_dbm']))
                x=[float(r['received_power_dbm']) for r in data]
                y=[float(r[metric]) if float(r[metric])>0 or not log else float(r['ber_upper']) for r in data]
                ax.plot(x,y,'o-',label=receiver)
            if log: ax.set_yscale('log')
            ax.set_title(condition); ax.set_xlabel('ROP at PD (dBm)'); ax.set_ylabel(label); ax.grid(True,alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out/f'{metric}_comparison.png',dpi=180); fig.savefig(out/f'{metric}_comparison.pdf'); plt.close(fig)


def combine_reports(run):
    """Collect independently run conditions; no model selection uses test scores."""
    import shutil
    run=Path(run).resolve(); out=run/'python_results'
    with (run/'conditions.csv').open() as f: conditions=list(csv.DictReader(f))
    out.mkdir(exist_ok=True); (out/'models').mkdir(exist_ok=True)
    merged={key:[] for key in ('validation_candidates','learning_curves','test_metrics','selected_parameters')}
    settings=None
    for condition in conditions:
        child=run/('python_results_'+condition['condition'])
        current=json.loads((child/'settings.json').read_text())
        if settings is None: settings=current
        if current!=settings: raise ValueError('Condition training settings differ')
        for key in merged:
            with (child/(key+'.csv')).open() as f: merged[key].extend(csv.DictReader(f))
        for model in (child/'models').glob('*.npz'): shutil.copy2(model,out/'models'/model.name)
    for key,rows in merged.items(): write_csv(out/(key+'.csv'),rows)
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    report(out)
    return out
