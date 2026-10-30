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


def fit(model,x,y,vx,vy,scale,modulation,epochs,tag,curves,schedule=None):
    best=float('inf'); best_state=None; best_epoch=0; fp.stats.clear()
    started=time.monotonic()
    for epoch in range(1,epochs+1):
        if schedule:
            factor=next(factor for start,factor in reversed(schedule) if epoch>=start)
            rate=fp.to_q(tag['learning_rate']*factor)
            if isinstance(model,Q824ANN): model.engine.lr_q24=rate
            else: model.lr_q24=rate
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


def compare(run,only=None,settings=None):
    run=Path(run).resolve()
    with (run/'conditions.csv').open() as f: conditions=list(csv.DictReader(f))
    if only:
        available={c['condition'] for c in conditions}
        if not set(only).issubset(available): raise ValueError('Unknown comparison condition')
        conditions=[c for c in conditions if c['condition'] in only]
    out=run/('python_results_'+'_'.join(c['condition'] for c in conditions) if only else 'python_results')
    settings=copy.deepcopy(ADAPTIVE if settings is None else settings)
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
                candidates.append(dict(receiver='ANN',**{'taps':2,**a}))
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
                    summary=fit(model,x,y,vx,vy,scale,modulation,settings['epochs'],tag,curves,
                                settings.get('learning_rate_schedule'))
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
    report_constellations(out,conditions)
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
    report_learning_curves(out)


def report_learning_curves(out):
    """Show validation convergence and selected checkpoints, never test tuning."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    if not (out/'selected_parameters.csv').exists(): return
    with (out/'selected_parameters.csv').open() as f:
        selected=[r for r in csv.DictReader(f) if r['primary']=='True']
    with (out/'learning_curves.csv').open() as f: curves=list(csv.DictReader(f))
    for condition in dict.fromkeys(r['condition'] for r in selected):
        models=[r for r in selected if r['condition']==condition]
        powers=sorted({float(r['received_power_dbm']) for r in models})
        fig,axes=plt.subplots(1,len(powers),figsize=(5*len(powers),4),squeeze=False)
        for ax,power in zip(axes[0],powers):
            for model in (r for r in models if float(r['received_power_dbm'])==power):
                history=sorted([r for r in curves if r['condition']==condition and
                    float(r['received_power_dbm'])==power and r['candidate']==model['candidate']],
                    key=lambda r:int(r['epoch']))
                label='LMS FFE' if model['receiver']=='LMS_FFE' else f"ANN {2*int(model['taps'])}-{model['hidden']}-2 {model['activation']}"
                line,=ax.plot([int(r['epoch']) for r in history],
                    [float(r['validation_evm_percent']) for r in history],label=label)
                ax.plot(int(model['best_epoch']),float(model['validation_evm_percent']),
                        '*',color=line.get_color(),markersize=11)
            ax.set(title=f'{power:g} dBm',xlabel='Training epoch',ylabel='Validation EVM (%)')
            ax.grid(True,alpha=.3); ax.legend(fontsize=8)
        fig.suptitle(condition.replace('_',' ')+' | stars: selected validation checkpoints')
        fig.tight_layout()
        for ext in ['png','pdf']:
            fig.savefig(out/f'validation_convergence_{condition}.{ext}',dpi=180)
        plt.close(fig)


def report_constellations(out, conditions):
    """Plot aligned held-out samples from saved inference, without retraining."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); destination=out/'constellations'; destination.mkdir(exist_ok=True)
    with (out/'test_metrics.csv').open() as f: rows=list(csv.DictReader(f))
    records=[]
    for condition in conditions:
        name=condition['condition']; source=Path(condition['dataset_folder'])
        with (source/'manifest.csv').open() as f:
            tests=[r for r in csv.DictReader(f) if r['split']=='test']
        for item in tests:
            power=float(item['received_power_dbm']); test=dataset(source/item['file'])
            point=[r for r in rows if r['condition']==name and float(r['received_power_dbm'])==power]
            ann=next(r['receiver'] for r in point if r['receiver'].startswith('ANN') and r['primary']=='True')
            receivers=['unequalized','LMS_FFE',ann]
            predictions=[np.asarray(features(test,1),dtype=np.int32)]
            for receiver in receivers[1:]:
                with np.load(out/'models'/f'{name}_{power:g}_{receiver}.npz',allow_pickle=False) as saved:
                    predictions.append(saved['predictions_q8_24'].copy())
            scores=[]
            for receiver,prediction in zip(receivers,predictions):
                if prediction.shape!=test['target'].shape:
                    raise ValueError('Constellation inference/target lengths differ')
                score=metrics(prediction,test['target'],test['scale'],test['modulation'])
                recorded=next(r for r in point if r['receiver']==receiver)
                if score['bit_errors']!=int(recorded['bit_errors']) or not np.isclose(
                        score['evm_percent'],float(recorded['evm_percent']),rtol=1e-12,atol=1e-12):
                    raise ValueError('Constellation predictions do not reproduce recorded test metrics')
                scores.append(score)
            # Use the same chronological subset and common axes for all receivers.
            shown=min(8192,len(test['target']))
            iq=[p.astype(np.float64)*test['scale']/fp.ONE for p in predictions]
            levels=np.arange(1-int(np.sqrt(test['modulation'])),int(np.sqrt(test['modulation'])),2)
            ideal_i,ideal_q=np.meshgrid(levels,levels)
            limit=1.08*max(float(np.max(np.abs(p))) for p in iq)
            limit=max(limit,1.15*float(np.max(np.abs(levels))))
            fig,axes=plt.subplots(1,3,figsize=(12,5))
            for ax,receiver,p,score in zip(axes,receivers,iq,scores):
                ax.scatter(p[:shown,0],p[:shown,1],s=2,alpha=.22,color='tab:blue',rasterized=True)
                ax.scatter(ideal_i,ideal_q,s=40,marker='x',color='black',linewidths=1,label='Ideal QAM')
                label={'unequalized':'Unequalized','LMS_FFE':'LMS FFE'}.get(receiver,receiver.replace('_',' '))
                ax.set_title(f"{label}\nBER {score['ber']:.3g} | EVM {score['evm_percent']:.2f}%")
                ax.set(xlabel='In-phase (I)',ylabel='Quadrature (Q)',xlim=(-limit,limit),ylim=(-limit,limit))
                ax.set_aspect('equal'); ax.grid(True,alpha=.2)
            axes[0].legend(loc='upper left',fontsize=8)
            fig.suptitle(f"{name.replace('_',' ')} | ROP {power:g} dBm | {test['modulation']}-QAM")
            fig.text(.5,.025,f'First {shown:,} identical held-out symbols shown; metrics use all {len(test["target"]):,} test symbols. '
                     'Common axes; no additional gain/phase fitting.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.15,1,.94))
            stem=f'{name}_{power:g}_dbm'
            fig.savefig(destination/(stem+'.png'),dpi=180)
            fig.savefig(destination/(stem+'.pdf')); plt.close(fig)
            records.append(dict(condition=name,received_power_dbm=power,ann_receiver=ann,
                displayed_symbols=shown,test_symbols=len(test['target']),first_test_row=1,
                filename=stem+'.png'))
    write_csv(destination/'manifest.csv',records)


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
    report_constellations(out,conditions)
    return out


def compare_previous(run, previous_run):
    """Evaluate frozen pre-tuning models on the new, common test sequences."""
    run=Path(run).resolve(); previous=Path(previous_run).resolve()/'python_results'
    current=run/'python_results'; destination=current/'before_after'
    destination.mkdir(exist_ok=True); (destination/'models').mkdir(exist_ok=True)
    with (run/'conditions.csv').open() as f: conditions=list(csv.DictReader(f))
    with (previous/'selected_parameters.csv').open() as f: parameters=list(csv.DictReader(f))
    with (current/'test_metrics.csv').open() as f: tuned=list(csv.DictReader(f))
    rows=[]
    for condition in conditions:
        name=condition['condition']; source=Path(condition['dataset_folder'])
        with (source/'manifest.csv').open() as f:
            tests=[r for r in csv.DictReader(f) if r['split']=='test']
        for item in tests:
            power=float(item['received_power_dbm']); test=dataset(source/item['file'])
            tag=dict(condition=name,received_power_dbm=power,primary=True)
            point=[r for r in tuned if r['condition']==name and float(r['received_power_dbm'])==power]
            for row in point:
                if row['receiver']=='unequalized' or row['primary']=='True':
                    row=row.copy()
                    if row['receiver']!='unequalized':
                        row['receiver']='tuned_'+('LMS_FFE' if row['receiver']=='LMS_FFE' else 'ANN')
                    rows.append(row)
            old=[p for p in parameters if p['condition']==name and float(p['received_power_dbm'])==power and p['primary']=='True']
            if len(old)!=2: raise ValueError('Expected frozen previous LMS and primary ANN')
            for p in old:
                group=p['receiver_group']; taps=int(p['taps']); rate=int(p['learning_rate_q8_24'])
                with np.load(previous/'models'/f'{name}_{power:g}_{group}.npz',allow_pickle=False) as saved:
                    state=saved['state_q8_24'].astype(np.int64).tolist()
                if group=='LMS_FFE':
                    model=Q824LMS(2*taps,rate); width=2*taps
                    model.restore([[state[:width],state[width:2*width]],state[2*width:]])
                    receiver='previous_LMS_FFE'
                else:
                    model=Q824ANN(2*taps,int(p['hidden']),p['activation'],rate,0)
                    model.restore(state); receiver='previous_ANN'
                prediction=model.predict(features(test,taps))
                score=metrics(prediction,test['target'],test['scale'],test['modulation'])
                rows.append(dict(**tag,receiver=receiver,**score))
                np.savez_compressed(destination/'models'/f'{name}_{power:g}_{receiver}.npz',
                                    predictions_q8_24=prediction)
    write_csv(destination/'test_metrics.csv',rows)
    (destination/'provenance.json').write_text(json.dumps(dict(
        previous_run=str(previous_run),current_run=str(run),
        previous_weights_retrained=False,test_sequences='Current run test splits for both versions'),indent=2)+'\n')
    report(destination)
    return destination
