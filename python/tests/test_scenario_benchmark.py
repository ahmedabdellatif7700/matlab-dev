"""Numerical checks for cross-term coverage, model export, and Gray-label BER."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenario_benchmark import PolynomialRegressor, bit_error_interval, metrics


class ScenarioBenchmarkTests(unittest.TestCase):
    def test_volterra_recovers_joint_iq_and_temporal_polynomial(self):
        rng = np.random.default_rng(10)
        x = rng.uniform(-1, 1, (900, 4))
        # Includes I*Q^2 and a product between different temporal taps.
        y = np.column_stack((x[:, 0] + .4*x[:, 0]*x[:, 1]**2 + .3*x[:, 2]*x[:, 3],
                             x[:, 1] - .2*x[:, 0]*x[:, 2]*x[:, 3] + .1))
        linear = PolynomialRegressor(1).fit(x[:500], y[:500], x[500:700], y[500:700], [0])
        cubic = PolynomialRegressor(3).fit(x[:500], y[:500], x[500:700], y[500:700], [0, 1e-3])
        np.testing.assert_allclose(cubic.predict(x[700:]), y[700:], atol=1e-12)
        self.assertGreater(np.mean((linear.predict(x[700:]) - y[700:])**2), 1e-3)
        self.assertEqual(cubic.parameter_count, 70)
        self.assertEqual(cubic.ridge, 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.npz'
            cubic.save(path)
            with np.load(path) as saved:
                z = (x[700:] - saved['input_mean']) / saved['input_scale']
                phi = np.prod(z[:, None, :] ** saved['powers'][None, :, :], axis=2)
                prediction = ((phi - saved['feature_mean']) / saved['feature_scale']) @ saved['weights'] + saved['target_mean']
            np.testing.assert_allclose(prediction, y[700:], atol=1e-12)

    def test_bit_labels_and_zero_error_bounds(self):
        # MATLAB labels are passed with the constellation, never re-created by
        # assuming a different convention in Python.
        ideal = np.array([-1-1j, -1+1j, 1-1j, 1+1j])
        bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        reference = np.repeat(ideal, 100)
        predicted = reference.copy()
        predicted[0] = ideal[1]  # Exactly one changed bit, not one symbol / N.
        result = metrics(predicted, reference, ideal, bits)
        self.assertEqual(result['bit_errors'], 1)
        self.assertEqual(result['bit_count'], 800)
        self.assertEqual(result['ber'], 1/800)
        low, high = bit_error_interval(0, 800)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, 1 - .025**(1/800))
        self.assertGreater(high, 0)


if __name__ == '__main__':
    unittest.main()
