import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_suite import ResearchNet, StableMultilevel, fit_nn
from activations.activation import MultilevelSigmoid
from scenario_benchmark import PolynomialRegressor


class ResearchSuiteTests(unittest.TestCase):
    def test_multiuser_polynomial_outputs_and_cost(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(500, 4))
        y = np.column_stack([x[:, k%4]+.05*(k+1)*x[:, 0]*x[:, 1] for k in range(8)])
        model = PolynomialRegressor(2).fit(x[:300], y[:300], x[300:400], y[300:400], [0])
        np.testing.assert_allclose(model.predict(x[400:]), y[400:], atol=1e-12)
        self.assertEqual(model.parameter_count, 120)
        self.assertEqual(model.multiplications, 122)

    def test_stable_multilevel_value_and_extreme_gradient(self):
        x = torch.linspace(-8, 8, 1001)
        torch.testing.assert_close(StableMultilevel(levels=8)(x), MultilevelSigmoid(levels=8)(x))
        extreme = torch.tensor([-1e4, 1e4], requires_grad=True)
        StableMultilevel(levels=8)(extreme).sum().backward()
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_adaptive_alpha_gradient_and_multioutput(self):
        torch.manual_seed(1)
        model = ResearchNet(6, 12, 8, 'adaptive_alpha', 4)
        x = torch.randn(20, 6)
        loss = model(x).square().mean()
        loss.backward()
        self.assertEqual(model(x).shape, (20, 8))
        self.assertTrue(torch.isfinite(model.activation.alpha.grad))
        self.assertNotEqual(model.activation.alpha.grad.item(), 0)

    def test_reproducible_training_and_validation_checkpoint(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, 4)).astype(np.float32)
        y = np.column_stack((x[:, 0]+.2*x[:, 1]**2, x[:, 1])).astype(np.float32)
        cfg = dict(nn_hidden_size=12, nn_learning_rate=.01, alpha_learning_rate=.003,
                   nn_epochs=30, nn_patience=10, nn_batch_size=128, nn_lbfgs_steps=2)
        a = fit_nn(x[:200], y[:200], x[200:], y[200:], cfg, 42, 'sigmoid', 16)
        b = fit_nn(x[:200], y[:200], x[200:], y[200:], cfg, 42, 'sigmoid', 16)
        np.testing.assert_array_equal(a[3], b[3])
        with torch.no_grad():
            prediction = a[0](torch.from_numpy((x[200:]-a[1])/a[2])).numpy()
        self.assertAlmostEqual(np.mean((prediction-y[200:])**2), a[4]['best_validation_mse'], places=6)
        self.assertLess(a[4]['best_validation_mse'], .05)
