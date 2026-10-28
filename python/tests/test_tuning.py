import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.io import savemat
import torch
from tuning.models import BitEqualizer, Multilevel
from tuning.data import load
from tuning.baselines import nearest_bits, logistic, bce


class TuningTests(unittest.TestCase):
    def test_discovery_never_reads_test_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'dataset.mat'
            ideal=np.array([-1-1j,-1+1j,1-1j,1+1j])
            bits=np.array([[0,0],[0,1],[1,0],[1,1]])
            y=np.column_stack([ideal.real,ideal.imag]).astype(np.float32)
            savemat(p,dict(X_train=y,Y_train=y,X_validation=y,Y_validation=y,X_test=np.array([np.nan]),ideal_symbols=ideal,ideal_bits=bits))
            data=load(p)
            self.assertNotIn('X_test',data)
            np.testing.assert_array_equal(data['bits_train'],bits)
    def test_cartesian_gray_slicing(self):
        ideal=np.array([-3-3j,-3+3j,3-3j,3+3j]); labels=np.array([[0,0],[0,1],[1,0],[1,1]])
        np.testing.assert_array_equal(nearest_bits(np.column_stack([ideal.real,ideal.imag]),ideal,labels),labels)
    def test_learned_activation_has_finite_gradients(self):
        net=BitEqualizer(6,4,[9,7],'learned_multilevel')
        loss=net(torch.randn(16,6)).square().mean(); loss.backward()
        for p in net.parameters(): self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(net(torch.randn(3,6)).shape,(3,4))
    def test_logistic_learns_separable_bits(self):
        x=np.linspace(-2,2,100)[:,None]; phi=np.column_stack([x,np.ones(100)])
        y=(x>0).astype(float); weights,_=logistic(phi,y,1e-4)
        self.assertLess(bce(phi@weights,y),.1)
        np.testing.assert_array_equal(phi@weights>=0,y)

if __name__=='__main__': unittest.main()
