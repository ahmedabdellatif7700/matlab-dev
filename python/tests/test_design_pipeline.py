"""Small end-to-end protocol check; synthetic data is never research evidence."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tuning.confirmation import confirm
from tuning.report import report


class PipelineTest(unittest.TestCase):
    def test_frozen_confirmation_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ideal=np.array([-1-1j,-1+1j,1-1j,1+1j])
            bits=np.array([[0,0],[0,1],[1,0],[1,1]])
            cfg=dict(confirmation_seeds=[9001],polynomial_orders=[1,2],bit_polynomial_orders=[1,2],kernel_centers=[4],nn_epochs=2,nn_patience=2,nn_polish_steps=0,required_cost_ratio=1.5,confirmation_counts=[64,32,32])
            selected=[]
            rng=np.random.default_rng(7)
            for k in range(4):
                cid=f'synthetic{k}'; folder=root/'confirmation'/'seed_9001'/cid; folder.mkdir(parents=True)
                d=dict(ideal_symbols=ideal,ideal_bits=bits)
                for split,n in [('train',64),('validation',32),('test',32)]:
                    z=ideal[rng.integers(4,size=n)]; y=np.column_stack([z.real,z.imag]).astype('float32')
                    d['Y_'+split]=y; d['X_'+split]=np.column_stack([y,y])+rng.normal(0,.01,(n,4)).astype('float32')
                savemat(folder/'dataset.mat',d)
                selected.append(dict(condition=dict(ID=cid,Family=cid),neural=dict(widths=[4],activation='tanh',lr=.003,weight_decay=1e-4,input_size=4,output_bits=2,parameters=30,multiplications=24,nonlinear_evaluations=4),ber_target=.01))
            lock=dict(complete=True,configuration=cfg,selected=selected)
            raw=json.dumps(lock).encode(); (root/'selection.json').write_bytes(raw)
            (root/'selection.sha256').write_text(hashlib.sha256(raw).hexdigest())
            for name in ['configuration.json','candidates.csv','discovery_models.csv']:
                (root/name).write_text('{}' if name.endswith('json') else 'synthetic\n')
            confirm(root); report(root,root/'report')
            self.assertTrue((root/'report'/'comparison.csv').exists())
            self.assertEqual(len(list(root.rglob('decisions.mat'))),4)
            # Reusing the frozen design resumes rather than retraining/tuning.
            original=(root/'confirmation_metrics.csv').read_bytes()
            confirm(root)
            self.assertEqual(original,(root/'confirmation_metrics.csv').read_bytes())

if __name__=='__main__': unittest.main()
