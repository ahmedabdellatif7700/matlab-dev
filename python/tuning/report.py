"""Generate reviewable tables and BER/resource figures from confirmation."""
import csv
import json
import shutil
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def report(run, out):
    out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((run/'selection.json').read_text())
    collected=[]
    for selected in lock['selected']:
        for seed in lock['configuration']['confirmation_seeds']:
            path=run/'confirmation'/f'seed_{seed}'/selected['condition']['ID']/'metrics.json'
            if not path.exists(): raise RuntimeError(f'Missing confirmation: {path}')
            collected.extend(json.loads(path.read_text()))
    table(run/'confirmation_metrics.csv',collected)
    with (run/'confirmation_metrics.csv').open() as f: metrics=list(csv.DictReader(f))
    conditions=[]; parameters=[]; summaries=[]
    baseline_parameters=[]; fitted_parameters=[]
    import equalizer as eq
    for selected in lock['selected']:
        for seed in lock['configuration']['confirmation_seeds']:
            folder=run/'confirmation'/f'seed_{seed}'/selected['condition']['ID']
            with (folder/'classical.pkl').open('rb') as stream: classical=pickle.load(stream)
            for model in classical:
                info=model['info']; fit=info.get('fit',{})
                baseline_parameters.append(dict(case=selected['condition']['ID'],seed=seed,method=info['name'],order=info.get('order',''),ridge=info.get('ridge',''),centers=info.get('centers',''),bandwidth=info.get('bandwidth',''),optimizer_iterations=fit.get('iterations',''),optimizer_converged=fit.get('converged',''),validation_ber=info['validation_ber'],multiplications=info['multiplications'],parameters=info['parameters']))
            model=eq.load_checkpoint(folder/'neural.pt'); info=model['info']
            learned={}
            for name,value in model['net'].state_dict().items():
                if 'centers' in name or 'raw_slopes' in name: learned[name]=value.tolist()
            fitted_parameters.append(dict(case=selected['condition']['ID'],data_seed=seed,selected_initialization_seed=info['seed'],best_step=info['best_step'],adam_epochs=info['epochs_completed'],validation_bce=info['validation_bce'],parameters=info['parameters'],activation_coefficients=json.dumps(learned)))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for selected,ax in zip(lock['selected'],axes.ravel()):
        c=selected['condition']; spec=selected['neural']; cid=c['ID']
        cfg=lock['configuration']
        conditions.append(dict(c,carrier_GHz=cfg['rf_carrier_hz']/1e9,symbol_rate_Gbaud=cfg['symbol_rate_hz']/1e9,rrc_rolloff=cfg['rrc_rolloff'],samples_per_symbol=cfg['samples_per_symbol'],tap_offsets=';'.join(map(str,cfg['tap_offsets'])),ber_target=selected['ber_target'],target_origin=selected.get('target_origin','reference grid')))
        parameters.append(dict(case=cid,widths=str(spec['widths']),activation=spec['activation'],inputs=spec['input_size'],output_bits=spec['output_bits'],parameters=spec['parameters'],multiplications=spec['multiplications'],nonlinear_evaluations=spec['nonlinear_evaluations'],loss='bitwise BCE',validation_selected_restarts=lock['configuration'].get('nn_restarts',3),optimizer=spec.get('optimizer','adamw')+' then LBFGS',batch_size=spec.get('batch_size',512),learning_rate=spec['lr'],weight_decay=spec['weight_decay'],epochs_max=spec.get('training_epochs',lock['configuration']['nn_epochs']),training_symbols=lock['configuration']['confirmation_counts'][0],validation_symbols=lock['configuration']['confirmation_counts'][1]))
        rows=[r for r in metrics if r['case']==cid]
        methods=list(dict.fromkeys(r['method'] for r in rows)); grouped=[]
        for name in methods:
            values=[r for r in rows if r['method']==name]
            errors=sum(int(r['bit_errors']) for r in values); bits=sum(int(r['bits']) for r in values)
            p=errors/bits; n=bits; z=1.96
            center=(p+z*z/(2*n))/(1+z*z/n); radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            item=dict(case=cid,method=name,ber=p,bit_errors=errors,bits=bits,nominal_wilson_low=max(0,center-radius),nominal_wilson_high=center+radius,multiplications=int(values[0]['multiplications']),parameters=int(values[0]['parameters']),nonlinear_evaluations=int(values[0]['nonlinear_evaluations']),mean_hard_decision_evm_percent=np.mean([float(r['hard_decision_evm_percent']) for r in values]))
            grouped.append(item)
        nn=grouped[0]; target=selected['ber_target']
        cheaper=[r for r in grouped[1:] if r['multiplications']<=nn['multiplications']]
        matching=[r for r in grouped[1:] if r['ber']<=target]
        ratio=min([r['multiplications']/nn['multiplications'] for r in matching],default=np.inf)
        for item in grouped:
            item['ber_target']=target
            item['nn_efficiency_condition_met']=bool(nn['ber']<=target and all(r['ber']>target for r in cheaper) and ratio>=lock['configuration']['required_cost_ratio'])
        summaries.extend(grouped)
        for r in grouped:
            neural=r['method']=='Selected NN'
            ax.scatter(r['multiplications'],max(r['ber'],.5/r['bits']),s=65 if neural else 25,c='crimson' if neural else 'steelblue',marker='*' if neural else 'o')
            if neural or r['method']=='Complex FFE' or r is min(grouped[1:],key=lambda r:r['ber']): ax.annotate(r['method'],(r['multiplications'],max(r['ber'],.5/r['bits'])),fontsize=7,xytext=(4,4),textcoords='offset points')
        ax.axhline(target,color='gray',linestyle='--'); ax.set(xscale='log',yscale='log',title=cid,xlabel='Real multiplications / decided symbol',ylabel='BER (zero errors shown at 0.5 / tested bits)'); ax.grid(alpha=.2)
    fig.savefig(out/'ber_vs_arithmetic.png',dpi=180); plt.close(fig)
    table(out/'baseline_parameters.csv',baseline_parameters)
    table(out/'fitted_neural_parameters.csv',fitted_parameters)
    table(out/'conditions.csv',conditions); table(out/'neural_parameters.csv',parameters); table(out/'comparison.csv',summaries)
    for name in ['confirmation_metrics.csv','discovery_models.csv','selection.json','configuration.json','candidates.csv']:
        shutil.copy2(run/name,out/name)
    if (run/'protocol_history.md').exists(): shutil.copy2(run/'protocol_history.md',out/'protocol_history.md')
    passed=sum(next(r for r in summaries if r['case']==s['condition']['ID'])['nn_efficiency_condition_met'] for s in lock['selected'])
    (out/'README.md').write_text(f'''# Four-family independent confirmation\n\n{passed}/4 selected conditions meet the declared arithmetic/BER criterion on pooled confirmation data. See comparison.csv for every classical control, including those that match or beat the NN at greater cost. This is a bounded receiver comparison, not proof that an impairment requires a neural network.\n\n- conditions.csv: physical link and preselected targets.\n- neural_parameters.csv: locked network architecture and training parameters.\n- discovery_models.csv: architecture and activation ablations on validation only.\n- comparison.csv: pooled BER, nominal Wilson intervals, hard-decision EVM and arithmetic.\n- confirmation_metrics.csv: individual seed results and paired contiguous-block bootstrap differences relative to NN.\n\nAll methods see the same two complex samples. Volterra regression includes all real monomials through degree 7, with shared-term multiplication accounting; polynomial bit classifiers through degree 5 and Gaussian RBF bit classifiers with up to 256 centers control for nonlinear decision boundaries and the bit objective. Hyperparameters use training/validation exclusively. NN architectures/conditions are frozen before generating fresh confirmation waveforms.\n\nArithmetic counts exclude additions and fused affine normalization; activation evaluations are reported separately. Counts are not measured latency, hardware power or fixed-point performance. Classical fitting/inference uses float64 and neural inference float32; parameter counts are not byte counts. EVM here is explicitly hard-decision EVM, not continuous equalizer reconstruction EVM. Wilson intervals assume independent bits; use seed spread and paired symbol-block intervals for dependence-aware comparisons. Adaptive activation is an ablation, not an established novelty or guaranteed improvement.\n\nThe optical stage is a small-signal DSB model after a phenomenological RF PA; it is not a calibrated large-signal optical transmitter. MZM cases use an isolated narrowband RF-fundamental model. Echoes are controlled one-symbol stress channels. Conditions are simulation hypotheses inspired by the supplied papers, not reproduced measurements.\n''')
    print('Confirmation criteria met:',passed,'/4',flush=True)


def case_reports(run):
    """Readable conditions, network tables and one report per locked case."""
    out=run/'report'; lock=json.loads((run/'selection.json').read_text())
    def read(name):
        with (out/name).open() as f: return list(csv.DictReader(f))
    comparison=read('comparison.csv'); curve=read('snr_sweep.csv')
    conditions=read('conditions.csv'); networks=read('neural_parameters.csv')
    cfg=lock['configuration']
    names=['case1_mzm16','case2_mzm64_memory','case3_rfpa16_memory','case4_rfpa64_dsb']
    overview=['# Four locked 60 GHz equalizer cases','',
              f"All links use {cfg['symbol_rate_hz']/1e9:g} Gbaud single-carrier Gray QAM, RRC rolloff {cfg['rrc_rolloff']}, {cfg['samples_per_symbol']} waveform samples/symbol and the same {len(cfg['tap_offsets'])} complex equalizer samples. Training/validation/test payloads are independent. Confirmation uses {len(cfg['confirmation_seeds'])} seeds and {cfg['confirmation_counts'][2]:,} test symbols per seed. Networks predict Gray bits; EVM is explicitly hard-decision EVM.",'',
              '| Case | QAM | MZM drive | PA IBO / AMPM | Echo | DSB fiber | Reference SNR |',
              '|---|---:|---:|---|---|---|---:|']
    network_lines=['','| Case | Real network dimensions | Activation | Parameters | Multiplications | Activation evaluations |',
                   '|---|---|---|---:|---:|---:|']
    result_lines=['','| Case | NN BER | FFE BER | Best classical BER within NN multiplication budget | Classical/NN cost at locked target | Target criterion |',
                  '|---|---:|---:|---|---:|---|']
    for index,(selected,directory) in enumerate(zip(lock['selected'],names),1):
        cid=selected['condition']['ID']; c=next(x for x in conditions if x['ID']==cid); n=next(x for x in networks if x['case']==cid)
        rows=[r for r in comparison if r['case']==cid]; nn=next(r for r in rows if r['method']=='Selected NN')
        classical=[r for r in rows if r['method']!='Selected NN']; cheaper=[r for r in classical if int(r['multiplications'])<=int(nn['multiplications'])]
        best=min(cheaper,key=lambda r:float(r['ber'])); ffe=next(r for r in classical if r['method']=='Complex FFE')
        matching=[r for r in classical if float(r['ber'])<=selected['ber_target']]
        ratio=min([int(r['multiplications'])/int(nn['multiplications']) for r in matching],default=np.inf)
        passed=nn['nn_efficiency_condition_met']=='True'
        mzm=c['MZM_drive'] if c['MZM']=='1' else '—'
        pa=f"{c['Backoff_dB']} dB / {c['AMPM_deg']}°" if c['PA']=='1' else '—'
        echo=complex(float(c['Echo_real']),float(c['Echo_imag']))
        echo_text=f'{abs(echo):.2g} ∠ {np.angle(echo,deg=True):.0f}°; 1 symbol' if echo else '—'
        optical=f"{c['Fiber_km']} km" if c['Optical']=='1' else '—'
        overview.append(f"| [{index}]({directory}/README.md) | {c['ModOrder']} | {mzm} | {pa} | {echo_text} | {optical} | {c['Reference_SNR_dB']} dB |")
        dims=' → '.join(map(str,[int(n['inputs']),*selected['neural']['widths'],int(n['output_bits'])]))
        network_lines.append(f"| {index} | {dims} | {n['activation']} | {n['parameters']} | {n['multiplications']} | {n['nonlinear_evaluations']} |")
        result_lines.append(f"| {index} | {float(nn['ber']):.3g} | {float(ffe['ber']):.3g} | {best['method']}: {float(best['ber']):.3g} | {ratio:.2f}× | {'Pass' if passed else 'Not met'} |")
        folder=out/directory; folder.mkdir(exist_ok=True)
        for filename in ['comparison.csv','confirmation_metrics.csv','baseline_parameters.csv','fitted_neural_parameters.csv','neural_parameters.csv','snr_sweep.csv']:
            subset=[r for r in read(filename) if r['case']==cid]
            table(folder/filename,subset)
        table(folder/'conditions.csv',[c])
        table(folder/'discovery_models.csv',[r for r in read('discovery_models.csv') if r['family']==c['Family']])
        (folder/'design.json').write_text(json.dumps(selected,indent=2)+'\n')
        fig,ax=plt.subplots(figsize=(6.5,4.2),layout='constrained')
        sub=[r for r in curve if r['case']==cid]
        for method in dict.fromkeys(r['method'] for r in sub):
            snrs=sorted(set(float(r['reference_snr_db']) for r in sub if r['method']==method)); ys=[]
            for snr in snrs:
                points=[r for r in sub if r['method']==method and float(r['reference_snr_db'])==snr]
                ys.append(max(.5,sum(int(r['bit_errors']) for r in points))/sum(int(r['bits']) for r in points))
            ax.semilogy(snrs,ys,'o-',label=method,markersize=4)
        ax.axhline(3.8e-3,color='gray',linestyle=':',label='Reference 3.8e-3')
        ax.set(title=cid,xlabel='Reference receiver SNR (dB)',ylabel='BER'); ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.savefig(folder/'ber_vs_snr.png',dpi=180); plt.close(fig)
        (folder/'README.md').write_text(f'''# Case {index}: {c['Family']}

Condition: {c['ModOrder']}-QAM; MZM drive {mzm}; PA {pa}; echo {echo_text}; DSB fiber {optical}; reference SNR {c['Reference_SNR_dB']} dB.

| Network | Activation | Parameters | Real multiplications | Activation evaluations |
|---|---|---:|---:|---:|
| {dims} | {n['activation']} | {n['parameters']} | {n['multiplications']} | {n['nonlinear_evaluations']} |

Gray-bit BCE training uses {n['optimizer']}, learning rate {n['learning_rate']}, weight decay {n['weight_decay']}, minibatches of {n['batch_size']}, up to {n['epochs_max']} epochs and LBFGS polishing. {n['validation_selected_restarts']} restarts are selected by validation BCE. See neural_parameters.csv and fitted_neural_parameters.csv for configuration and actual fitted run details.

Pooled NN BER: **{float(nn['ber']):.6g}**; FFE: {float(ffe['ber']):.6g}; best tested classical receiver within the NN multiplication budget: **{best['method']}**, BER {float(best['ber']):.6g}. Locked BER target: {selected['ber_target']:.6g} ({selected.get('target_origin','reference grid')}). Target/cost criterion: **{'met' if passed else 'not met'}**; cheapest tested classical receiver meeting it costs {ratio:.2f}× the NN's explicit real multiplications.

![Frozen receiver BER/SNR curves](ber_vs_snr.png)

Curves use frozen receivers on fresh payloads, not per-point retraining. Higher-cost classical receivers remain in comparison.csv. Nominal Wilson intervals assume independent bits; confirmation_metrics.csv also gives paired contiguous-symbol-block uncertainty and seed results. Counts exclude additions and fusible normalization; activation evaluations are separate, so multiplication savings do not establish lower latency or power. Classical inference is float64, neural inference float32. These results concern the tested receiver catalog and observation window, not universal necessity of neural processing or experimental reproduction of the papers. Activation ablations are in discovery_models.csv and do not establish novelty or a test-confirmed activation contribution.
''')
    overview += network_lines + ['', 'The final layer contains Gray-bit logits. Per-case learning rate, optimizer, batch size, weight decay, epoch budget and validation-selected restart count are in neural_parameters.csv; actual fitted run details are in fitted_neural_parameters.csv.']+result_lines+['',
        'All tested classical receivers, including higher-cost matches, are retained in comparison.csv; baseline_parameters.csv reports ridge/bandwidth and optimizer convergence. These are explicit multiplication savings with separately reported activation costs, not measured hardware speed or power. The criterion is descriptive for this bounded catalog and locked validation-selected BER targets; it is not a claim that conventional methods can never solve these channels.',
        '', '![BER and multiplication count](ber_vs_arithmetic.png)', '', '![Frozen receiver BER/SNR curves](ber_vs_snr.png)', '',
        'Targets and architectures were locked before generating the confirmation data. protocol_history.md records the validation-only search extensions. Hard-decision EVM and raw counts are in the CSVs. Activation comparisons in discovery_models.csv are validation ablations, not demonstrated mathematical novelty or independent test confirmation of an activation contribution.']
    (out/'README.md').write_text('\n'.join(overview)+'\n')
