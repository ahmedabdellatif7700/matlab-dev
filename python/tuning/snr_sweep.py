"""Frozen receiver SNR generalization, never a second tuning stage."""
import csv,json,pickle
import numpy as np
from scipy.io import loadmat
import equalizer as models
from . import baselines
from .data import labels as target_bits
from .report import table
import matplotlib.pyplot as plt


def evaluate(run):
    lock=json.loads((run/'selection.json').read_text()); rows=[]
    for selected in lock['selected']:
        cid=selected['condition']['ID']
        for seed in lock['configuration']['confirmation_seeds']:
            folder=run/'confirmation'/f'seed_{seed}'/cid
            neural=models.load_checkpoint(folder/'neural.pt')
            with (folder/'classical.pkl').open('rb') as f: classical=pickle.load(f)
            ideal_data=loadmat(folder/'dataset.mat',variable_names=['ideal_symbols','ideal_bits'])
            ideal=ideal_data['ideal_symbols'].ravel(); labels=ideal_data['ideal_bits']
            chosen=[m for m in classical if m['info']['name'] in ['Complex FFE','Volterra regression 3','Volterra regression 5']]
            best=min(classical,key=lambda m:m['info']['validation_ber'])
            if all(best is not m for m in chosen): chosen.append(best)
            for file in sorted((run/'snr_sweep'/f'seed_{seed}'/cid).glob('offset_*.mat')):
                d=loadmat(file,variable_names=['X','Y','reference_snr_db']); truth=target_bits(d['Y'],ideal,labels)
                decisions=[('Selected NN',models.predict_bits(neural,d['X'])>=0)]
                for m in chosen:
                    pred=np.concatenate([baselines.predict(m,x,ideal,labels) for x in np.array_split(d['X'],max(1,len(d['X'])//2048))])
                    decisions.append((m['info']['name'],pred))
                for name,pred in decisions:
                    rows.append(dict(case=cid,seed=seed,reference_snr_db=float(d['reference_snr_db'].item()),method=name,bit_errors=int(np.sum(pred!=truth)),bits=truth.size,ber=float(np.mean(pred!=truth))))
            print('Frozen SNR sweep',cid,seed,flush=True)
    table(run/'report'/'snr_sweep.csv',rows)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for selected,ax in zip(lock['selected'],axes.ravel()):
        cid=selected['condition']['ID']; sub=[r for r in rows if r['case']==cid]
        for name in dict.fromkeys(r['method'] for r in sub):
            snrs=sorted(set(r['reference_snr_db'] for r in sub if r['method']==name)); ber=[]
            for snr in snrs:
                values=[r for r in sub if r['method']==name and r['reference_snr_db']==snr]
                errors=sum(r['bit_errors'] for r in values); n=sum(r['bits'] for r in values)
                ber.append(max(errors,.5)/n)
            ax.semilogy(snrs,ber,'o-',label=name,linewidth=2 if name=='Selected NN' else 1,markersize=3)
        ax.axhline(3.8e-3,color='gray',linestyle=':',label='Reference 3.8e-3')
        ax.set(title=cid,xlabel='Reference receiver SNR (dB)',ylabel='BER'); ax.grid(alpha=.25); ax.legend(fontsize=6)
    fig.suptitle('Frozen receivers: SNR generalization on fresh payloads')
    fig.savefig(run/'report'/'ber_vs_snr.png',dpi=180); plt.close(fig)
