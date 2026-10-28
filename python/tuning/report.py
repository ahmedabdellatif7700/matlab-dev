"""Generate reviewable tables and BER/resource figures from confirmation."""
import csv
import json
import shutil
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
    with (run/'confirmation_metrics.csv').open() as f: metrics=list(csv.DictReader(f))
    conditions=[]; parameters=[]; summaries=[]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for selected,ax in zip(lock['selected'],axes.ravel()):
        c=selected['condition']; spec=selected['neural']; cid=c['ID']
        conditions.append(dict(c,carrier_GHz=60,symbol_rate_Gbaud=1,rrc_rolloff=.25,samples_per_symbol=8,tap_offsets='0;-1',ber_target=selected['ber_target'],target_origin=selected.get('target_origin','reference grid')))
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
    table(out/'conditions.csv',conditions); table(out/'neural_parameters.csv',parameters); table(out/'comparison.csv',summaries)
    for name in ['confirmation_metrics.csv','discovery_models.csv','selection.json','configuration.json','candidates.csv']:
        shutil.copy2(run/name,out/name)
    passed=sum(next(r for r in summaries if r['case']==s['condition']['ID'])['nn_efficiency_condition_met'] for s in lock['selected'])
    (out/'README.md').write_text(f'''# Four-family independent confirmation\n\n{passed}/4 selected conditions meet the declared arithmetic/BER criterion on pooled confirmation data. See comparison.csv for every classical control, including those that match or beat the NN at greater cost. This is a bounded receiver comparison, not proof that an impairment requires a neural network.\n\n- conditions.csv: physical link and preselected targets.\n- neural_parameters.csv: locked network architecture and training parameters.\n- discovery_models.csv: architecture and activation ablations on validation only.\n- comparison.csv: pooled BER, nominal Wilson intervals, hard-decision EVM and arithmetic.\n- confirmation_metrics.csv: individual seed results and paired contiguous-block bootstrap differences relative to NN.\n\nAll methods see the same two complex samples. Volterra regression includes all real monomials through degree 7, with shared-term multiplication accounting; polynomial bit classifiers through degree 5 and Gaussian RBF bit classifiers with up to 256 centers control for nonlinear decision boundaries and the bit objective. Hyperparameters use training/validation exclusively. NN architectures/conditions are frozen before generating fresh confirmation waveforms.\n\nArithmetic counts exclude additions and fused affine normalization; activation evaluations are reported separately. Counts are not measured latency, hardware power or fixed-point performance. Classical fitting/inference uses float64 and neural inference float32; parameter counts are not byte counts. EVM here is explicitly hard-decision EVM, not continuous equalizer reconstruction EVM. Wilson intervals assume independent bits; use seed spread and paired symbol-block intervals for dependence-aware comparisons. Adaptive activation is an ablation, not an established novelty or guaranteed improvement.\n\nThe optical stage is a small-signal DSB model after a phenomenological RF PA; it is not a calibrated large-signal optical transmitter. MZM cases use an isolated narrowband RF-fundamental model. Echoes are controlled one-symbol stress channels. Conditions are simulation hypotheses inspired by the supplied papers, not reproduced measurements.\n''')
    print('Confirmation criteria met:',passed,'/4',flush=True)
