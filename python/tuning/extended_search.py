"""Focused validation-only 64-QAM architecture ablation."""
import json
import numpy as np
import equalizer as models
from .data import load
from config import EXTENDED_NETWORKS, EXTENDED_TRAINING, NN_LEARNING_RATE, NN_WEIGHT_DECAY, NN_BATCH_SIZE, NN_OPTIMIZER

def search(run):
    cfg=json.loads((run/'configuration.json').read_text())
    root=run/'discovery'/f"seed_{cfg['discovery_seed']}"
    for cid in ['mzm64_mu090','mzm64_mu105','mzm64_mu120']:
     folder=root/cid
     if (folder/'extended_models.json').exists(): continue
     d=load(folder/'dataset.mat'); infos=[]
     for k,(widths,activation) in enumerate(EXTENDED_NETWORKS):
      spec=dict(widths=widths,activation=activation,lr=NN_LEARNING_RATE,weight_decay=NN_WEIGHT_DECAY,batch_size=NN_BATCH_SIZE,optimizer=NN_OPTIMIZER)
      m=models.train_bit_equalizer(d['X_train'],d['bits_train'],d['X_validation'],d['bits_validation'],spec,seed=42,**EXTENDED_TRAINING)
      info=dict(m['info'],name=f'NN extended {activation} {widths}',neural=True,training_epochs=EXTENDED_TRAINING['epochs'],training_patience=EXTENDED_TRAINING['patience'],training_polish=EXTENDED_TRAINING['polish'],validation_ber=float(np.mean((models.predict_bits(m,d['X_validation'])>=0)!=d['bits_validation'])),catalog_index=100+k)
      models.save_checkpoint(m,folder/f'nn_extended_{k}.pt');infos.append(info)
      print(cid,info['name'],info['validation_ber'],info['multiplications'],flush=True)
     (folder/'extended_models.json').write_text(json.dumps(infos,indent=2)+'\n')
