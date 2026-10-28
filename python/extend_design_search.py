import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tuning import models
from tuning.data import load
import numpy as np
import argparse
p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); args=p.parse_args()
cfg=json.loads((args.run/'configuration.json').read_text())
root=args.run/'discovery'/f"seed_{cfg['discovery_seed']}"
for cid in ['mzm64_mu090','mzm64_mu105','mzm64_mu120']:
 folder=root/cid
 if (folder/'extended_models.json').exists(): continue
 d=load(folder/'dataset.mat'); infos=[]
 for k,(widths,activation) in enumerate([([7],'tanh'),([8],'silu'),([8],'softsign'),([2,8],'tanh'),([3,6],'tanh'),([8],'learned_multilevel')]):
  spec=dict(widths=widths,activation=activation,lr=.003,weight_decay=1e-4)
  m=models.train(d['X_train'],d['bits_train'],d['X_validation'],d['bits_validation'],spec,seed=42,epochs=900,patience=90,polish=20)
  info=dict(m['info'],name=f'NN extended {activation} {widths}',neural=True,training_epochs=900,training_patience=90,training_polish=20,validation_ber=float(np.mean((models.predict(m,d['X_validation'])>=0)!=d['bits_validation'])),catalog_index=100+k)
  models.save(m,folder/f'nn_extended_{k}.pt');infos.append(info)
  print(cid,info['name'],info['validation_ber'],info['multiplications'],flush=True)
 (folder/'extended_models.json').write_text(json.dumps(infos,indent=2)+'\n')
